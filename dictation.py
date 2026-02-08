import enum
import time
import threading
import queue
import argparse
import platform
import subprocess
import pyaudio
import numpy as np
from faster_whisper import WhisperModel
from faster_whisper.vad import get_speech_timestamps, VadOptions
from pynput import keyboard
from transitions import Machine



if platform.system() == 'Windows':
    import winsound
    def playsound(s, wait=True):
        # SND_ASYNC winsound cannot play asynchronously from memory
        winsound.PlaySound(s, winsound.SND_MEMORY)
    def loadwav(filename):
        with open(filename, "rb") as f:
            data = f.read()
        return data
else:
    import soundfile as sf
    import sounddevice # or pygame.mixer, py-simple-audio
    sounddevice.default.samplerate = 44100
    def playsound(s, wait=True):
        sounddevice.play(s) # samplerate=16000
        if wait:
            sounddevice.wait()
    def loadwav(filename):
        data, fs = sf.read(filename, dtype='float32')
        return data


# ---------------------------------------------------------------------------
# Batch-mode classes (original behaviour, kept for --batch-mode fallback)
# ---------------------------------------------------------------------------

class SpeechTranscriber:
    def __init__(self, callback, model_size='base', device='cpu', compute_type="int8", language=None):
        self.callback = callback
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.language = language

    def transcribe(self, event):
        print('Transcribing...')
        audio = event.kwargs.get('audio', None)
        if audio is not None:
            # Force language if specified, otherwise auto-detect
            if self.language:
                segments, info = self.model.transcribe(audio, beam_size=5, language=self.language)
                print("Using forced language: '%s'" % self.language)
            else:
                segments, info = self.model.transcribe(audio, beam_size=5)
                print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
            self.callback(segments=segments)
        else:
            self.callback(segments=[])

class Recorder:
    def __init__(self, callback):
        self.callback = callback
        self.recording = False

    def start(self, language=None):
        print('Recording ...')
        thread = threading.Thread(target=self._record_impl, args=())
        thread.start()

    def stop(self):
        print('Done recording.')
        self.recording = False

    def _record_impl(self):
        self.recording = True

        frames_per_buffer = 1024
        p = pyaudio.PyAudio()
        stream = p.open(format            = pyaudio.paInt16,
                        channels          = 1,
                        rate              = 16000,
                        frames_per_buffer = frames_per_buffer,
                        input             = True)
        frames = []

        while self.recording:
            data = stream.read(frames_per_buffer)
            frames.append(data)

        stream.stop_stream()
        stream.close()
        p.terminate()

        audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
        audio_data_fp32 = audio_data.astype(np.float32) / 32768.0

        self.callback(audio=audio_data_fp32)


# ---------------------------------------------------------------------------
# Streaming-mode classes (new VAD-based pipeline)
# ---------------------------------------------------------------------------

class StreamingRecorder:
    """Records audio and uses Silero VAD to split on speech pauses.

    Two internal threads:
      - Recording thread: reads audio frames from PyAudio into a shared buffer.
      - VAD thread: periodically checks the buffer for completed speech segments
        and pushes them onto the transcription queue.
    """

    SAMPLE_RATE = 16000
    FRAMES_PER_BUFFER = 1024

    def __init__(self, transcription_queue, silence_ms=1000,
                 auto_stop_silence_s=10, on_auto_stop=None):
        self.transcription_queue = transcription_queue
        self.silence_ms = silence_ms
        self.auto_stop_silence_s = auto_stop_silence_s
        self.on_auto_stop = on_auto_stop
        self.recording = False

        # Shared buffer (protected by lock)
        self._buffer_lock = threading.Lock()
        self._raw_frames = []  # list of int16 byte strings

        # VAD options tuned for streaming dictation
        self.vad_options = VadOptions(
            threshold=0.5,
            min_speech_duration_ms=250,
            min_silence_duration_ms=silence_ms,
            speech_pad_ms=200,
        )

        # Track how many samples we already processed in the VAD thread
        self._vad_processed_samples = 0

    def start(self):
        """Start recording + VAD threads."""
        self.recording = True
        self._raw_frames = []
        self._vad_processed_samples = 0

        self._rec_thread = threading.Thread(target=self._record_loop, daemon=True)
        self._vad_thread = threading.Thread(target=self._vad_loop, daemon=True)
        self._rec_thread.start()
        self._vad_thread.start()

    def stop(self):
        """Stop recording and flush remaining audio."""
        print('Done recording.')
        self.recording = False
        self._rec_thread.join(timeout=3)
        self._vad_thread.join(timeout=3)

        # Flush: transcribe any remaining audio that VAD hasn't sent yet
        remaining = self._get_all_audio_fp32()
        if remaining is not None and len(remaining) > self._vad_processed_samples:
            leftover = remaining[self._vad_processed_samples:]
            if len(leftover) > self.SAMPLE_RATE * 0.1:  # at least 100ms
                print('[flush] Sending remaining %.1fs audio' % (len(leftover) / self.SAMPLE_RATE))
                self.transcription_queue.put(leftover)

        # Sentinel to signal end of stream
        self.transcription_queue.put(None)

    def _record_loop(self):
        """PyAudio recording loop — runs in its own thread."""
        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.SAMPLE_RATE,
            frames_per_buffer=self.FRAMES_PER_BUFFER,
            input=True,
        )

        while self.recording:
            data = stream.read(self.FRAMES_PER_BUFFER, exception_on_overflow=False)
            with self._buffer_lock:
                self._raw_frames.append(data)

        stream.stop_stream()
        stream.close()
        p.terminate()

    def _vad_loop(self):
        """Periodically check accumulated audio for completed speech segments."""
        # We need enough audio context for VAD to detect silence after speech.
        # Check every 300ms.
        while self.recording:
            time.sleep(0.3)
            if self._check_and_send_chunks():
                # auto-stop was triggered
                break

    def _get_all_audio_fp32(self):
        """Convert all buffered frames to float32 numpy array."""
        with self._buffer_lock:
            if not self._raw_frames:
                return None
            raw = b''.join(self._raw_frames)

        audio_int16 = np.frombuffer(raw, dtype=np.int16)
        return audio_int16.astype(np.float32) / 32768.0

    def _check_and_send_chunks(self):
        """Run VAD on accumulated audio and send completed speech chunks.

        Returns True if auto-stop was triggered, False otherwise.
        """
        audio = self._get_all_audio_fp32()
        if audio is None or len(audio) < self.SAMPLE_RATE * 0.5:
            return False  # need at least 500ms of audio

        audio_end = len(audio)
        audio_duration_s = audio_end / self.SAMPLE_RATE

        timestamps = get_speech_timestamps(audio, self.vad_options, sampling_rate=self.SAMPLE_RATE)

        # Determine trailing silence duration from the audio buffer
        if timestamps:
            last_speech_end_s = timestamps[-1]['end'] / self.SAMPLE_RATE
            trailing_silence_s = audio_duration_s - last_speech_end_s
        else:
            # No speech detected at all — entire buffer is silence
            trailing_silence_s = audio_duration_s

        # Auto-stop if trailing silence exceeds threshold
        if (self.auto_stop_silence_s and self.on_auto_stop
                and trailing_silence_s > self.auto_stop_silence_s):
            print('[VAD] %.0fs trailing silence — auto-stopping' % trailing_silence_s)
            # Fire callback in separate thread to avoid deadlock
            # (App._auto_stop() joins the VAD thread, which is us)
            threading.Thread(target=self.on_auto_stop, daemon=True).start()
            return True

        if not timestamps:
            return False

        # A speech segment is "complete" if there is enough silence after it.
        # We consider a segment complete if its end + silence_duration is before
        # the end of the current audio buffer (meaning silence has been observed).
        silence_samples = int(self.silence_ms * self.SAMPLE_RATE / 1000)

        for ts in timestamps:
            seg_start = ts['start']
            seg_end = ts['end']

            # Skip segments we've already sent
            if seg_end <= self._vad_processed_samples:
                continue

            # Only send if there's confirmed silence after this segment
            if seg_end + silence_samples <= audio_end:
                # This segment has enough trailing silence — it's complete
                chunk = audio[seg_start:seg_end]
                if len(chunk) > self.SAMPLE_RATE * 0.1:  # at least 100ms
                    duration = len(chunk) / self.SAMPLE_RATE
                    print('[VAD] Speech chunk: %.1fs (samples %d-%d)' % (duration, seg_start, seg_end))
                    self.transcription_queue.put(chunk)

                # Update processed marker to after this segment
                self._vad_processed_samples = max(self._vad_processed_samples, seg_end)

        return False


class TranscriptionWorker:
    """Loads Whisper model once and transcribes audio chunks from a queue."""

    def __init__(self, model_size='base', device='cpu', compute_type='int8', language=None):
        print('Loading Whisper model: %s (device=%s, compute=%s)' % (model_size, device, compute_type))
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.language = language

    def transcribe_chunk(self, audio_fp32):
        """Transcribe a single audio chunk. Returns text string."""
        if self.language:
            segments, info = self.model.transcribe(audio_fp32, beam_size=5, language=self.language)
        else:
            segments, info = self.model.transcribe(audio_fp32, beam_size=5)
            print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

        text = ''
        for segment in segments:
            seg_text = segment.text
            if text == '' and seg_text.startswith(' '):
                seg_text = seg_text[1:]
            text += seg_text

        return text


# ---------------------------------------------------------------------------
# Keyboard output (shared between batch and streaming mode)
# ---------------------------------------------------------------------------

class KeyboardReplayer():
    def __init__(self, callback=None):
        self.callback = callback
        self.kb = keyboard.Controller()

    def _get_active_window_class(self):
        """Detect active window class via xprop (returns lowercase)."""
        try:
            wid = subprocess.check_output(
                ['xdotool', 'getactivewindow'], timeout=2
            ).decode().strip()
            xprop = subprocess.check_output(
                ['xprop', '-id', wid, 'WM_CLASS'], timeout=2
            ).decode().strip().lower()
            # xprop output: wm_class(string) = "kitty", "kitty"
            if '"kitty"' in xprop:
                return 'kitty'
            # Extract second value (class name)
            parts = xprop.split('"')
            return parts[3] if len(parts) >= 4 else parts[1] if len(parts) >= 2 else ''
        except:
            return ''

    def _type_via_clipboard(self, text):
        """Inject text via xclip clipboard + Ctrl+Shift+V paste.
        Bypasses Kitty XTest shift bug (#2261) on AZERTY keyboards."""
        p = subprocess.Popen(['xclip', '-selection', 'clipboard'],
                             stdin=subprocess.PIPE)
        p.communicate(text.encode('utf-8'))
        time.sleep(0.05)
        self.kb.press(keyboard.Key.ctrl)
        self.kb.press(keyboard.Key.shift)
        self.kb.press('v')
        self.kb.release('v')
        self.kb.release(keyboard.Key.shift)
        self.kb.release(keyboard.Key.ctrl)

    def _type_via_pynput(self, text):
        """Original character-by-character typing via pynput (works in GUI apps)."""
        for element in text:
            try:
                self.kb.type(element)
                time.sleep(0.0025)
            except:
                pass

    def type_text(self, text):
        """Type a text string using the appropriate method for the active window."""
        if not text:
            return
        print(text)
        wm_class = self._get_active_window_class()
        if wm_class == 'kitty':
            print('[clipboard mode - kitty detected]')
            self._type_via_clipboard(text)
        else:
            print('[pynput mode - %s]' % (wm_class or 'unknown'))
            self._type_via_pynput(text)

    def replay(self, event):
        """Batch-mode interface (used by transitions state machine)."""
        print('Typing transcribed words...')
        segments = event.kwargs.get('segments', [])
        text = ''
        for segment in segments:
            segment_text = segment.text
            if text == '' and segment_text.startswith(' '):
                segment_text = segment_text[1:]
            text += segment_text

        if text:
            self.type_text(text)

        print('')
        if self.callback:
            self.callback()


class KeyListener():
    def __init__(self, callback, key):
        self.callback = callback
        self.key = key
    def run(self):
        with keyboard.GlobalHotKeys({self.key : self.callback}) as h:
            h.join()


class DoubleKeyListener():
    def __init__(self, activate_callback, deactivate_callback, key=keyboard.Key.cmd_r):
        self.activate_callback = activate_callback
        self.deactivate_callback = deactivate_callback
        self.key = key
        self.last_press_time = 0

    def on_press(self, key):
        if key == keyboard.Key.ctrl_r:
            current_time = time.time()
            is_dbl_click = current_time - self.last_press_time < 0.5
            self.last_press_time = current_time
            if is_dbl_click:
                return self.activate_callback()
            else:
                return self.deactivate_callback()

    def on_release(self, key):
        pass
    def run(self):
        with keyboard.Listener(
                on_press=self.on_press,
                on_release=self.on_release) as listener:
            listener.join()


def parse_args():
    parser = argparse.ArgumentParser(description='Dictation app powered by Faster whisper')
    parser.add_argument('-m', '--model-name', type=str, default='base',
                        help='''\
 Size of the model to use
 (tiny, tiny.en, base, base.en, small, small.en, medium, medium.en, large-v1, large-v2, or large).
 A path to a converted model directory, or a CTranslate2-converted Whisper model ID from the Hugging Face Hub.
 When a size or a model ID is configured, the converted model is downloaded from the Hugging Face Hub.
 Default: base.''')
    parser.add_argument('-k', '--key-combo', type=str,
                        help='''\
 Specify the key combination to toggle the app.
 
 See https://pynput.readthedocs.io/en/latest/keyboard.html#pynput.keyboard.Key for a list of keys supported.
 
 Examples: <cmd_l>+<alt>+x , <ctrl>+<alt>+a. Note on windows, the winkey is specified using <cmd>.
 
 Default: <win>+z on Windows (see below for MacOS and Linux defaults).''')
    parser.add_argument('-d', '--double-key', type=str,
                        help='''\
 If key-combo is not set, on macOS/linux the default behavior is double tapping a key to start recording.
 Tap the same key again to stop recording.
 
 On MacOS the key is Right Cmd and on Linux the key is Right Super (Right Win Key)
 
 You can set to a different key for double triggering.
 
 ''')
    parser.add_argument('-t', '--max-time', type=int, default=30,
                        help='''\
 Specify the maximum recording time in seconds.
 The app will automatically stop recording after this duration.
 Default: 30 seconds.''')
    parser.add_argument('-v', '--device', type=str, default='cpu',
                        help='''\
 By default we use 'cpu' for inference.
 If you have supported GPU with proper driver and libraries installed, you can set it to 'auto' or 'cuda'.''')
 
    parser.add_argument('-c', '--compute-type', type=str, default='int8',
                        help='''\
If your GPU stack supports it, you can set compute-type to 'float32' or 'float16' to improve accuracy. Default 'int8' ''')

    parser.add_argument('-l', '--language', type=str, default=None,
                        help='''\
Force a specific language for transcription (e.g., 'nl' for Dutch, 'en' for English).
This improves accuracy especially for short audio fragments where auto-detection can fail.
If not specified, language will be auto-detected.
Common codes: nl (Dutch), en (English), fr (French), de (German), es (Spanish).''')

    parser.add_argument('--silence-ms', type=int, default=1000,
                        help='''\
Silence duration in milliseconds before splitting a speech chunk (streaming mode).
Lower values give faster feedback but may split mid-sentence.
Default: 1000 (1 second).''')

    parser.add_argument('--auto-stop-silence', type=int, default=10,
                        help='''\
Automatically stop recording after this many seconds of silence (streaming mode).
Set to 0 to disable auto-stop. Default: 10 seconds.''')

    parser.add_argument('--batch-mode', action='store_true',
                        help='''\
Use original batch mode: record all audio first, then transcribe, then type.
By default, streaming mode with VAD is used for real-time feedback.''')

    args = parser.parse_args()
    return args


# ---------------------------------------------------------------------------
# Batch-mode App (original transitions state machine)
# ---------------------------------------------------------------------------

class States(enum.Enum):
    READY        = 1
    RECORDING    = 2
    TRANSCRIBING = 3
    REPLAYING    = 4


transitions = [
    {'trigger':'start_recording'     ,'source': States.READY        ,'dest': States.RECORDING    },
    {'trigger':'finish_recording'    ,'source': States.RECORDING    ,'dest': States.TRANSCRIBING },
    {'trigger':'finish_transcribing' ,'source': States.TRANSCRIBING ,'dest': States.REPLAYING    },
    {'trigger':'finish_replaying'    ,'source': States.REPLAYING    ,'dest': States.READY        },
]


class BatchApp():
    """Original batch-mode app using transitions state machine."""

    def __init__(self, args):
        m = Machine(states=States, transitions=transitions, send_event=True, ignore_invalid_triggers=True, initial=States.READY)

        self.m = m
        self.args = args
        self.recorder    = Recorder(m.finish_recording)
        self.transcriber = SpeechTranscriber(m.finish_transcribing, args.model_name, args.device, args.compute_type, args.language)
        self.replayer    = KeyboardReplayer(m.finish_replaying)
        self.timer = None

        m.on_enter_RECORDING(self.recorder.start)
        m.on_enter_TRANSCRIBING(self.transcriber.transcribe)
        m.on_enter_REPLAYING(self.replayer.replay)

        # https://freesound.org/people/leviclaassen/sounds/107786/
        # https://freesound.org/people/MATRIXXX_/
        self.SOUND_EFFECTS = {
            "start_recording": loadwav("assets/granted-04.wav"),
            "finish_recording": loadwav("assets/beepbeep.wav")
        }

    def beep(self, k, wait=True):
        playsound(self.SOUND_EFFECTS[k], wait=wait)

    def start(self):
        if self.m.is_READY():
            self.beep("start_recording")
            if self.args.max_time:
                self.timer = threading.Timer(self.args.max_time, self.timer_stop)
                self.timer.start()
            self.m.start_recording()
            return True

    def stop(self):
        if self.m.is_RECORDING():
            self.recorder.stop()
            if self.timer is not None:
                self.timer.cancel()
            self.beep("finish_recording", wait=False)
            return True

    def timer_stop(self):
        print('Timer stop')
        self.stop()

    def toggle(self):
        return self.start() or self.stop()

    def run(self):
        def normalize_key_names(keyseqs, parse=False):
            k = keyseqs.replace('<win>', '<cmd>').replace('<win_r>', '<cmd_r>').replace('<win_l>', '<cmd_l>').replace('<super>', '<cmd>').replace('<super_r>', '<cmd_r>').replace('<super_l>', '<cmd_l>')
            if parse:
                k = keyboard.HotKey.parse(k)[0]
            print('Using key:', k)
            return k

        if (platform.system() != 'Windows' and not self.args.key_combo) or self.args.double_key:
            key = self.args.double_key or (platform.system() == 'Linux' and '<ctrl_r>') or '<cmd_r>'
            keylistener= DoubleKeyListener(self.start, self.stop, normalize_key_names(key, parse=True))
            self.m.on_enter_READY(lambda *_: print("Double tap ", key, " to start recording. Tap again to stop recording"))
        else:
            key = self.args.key_combo or '<win>+z'
            keylistener= KeyListener(self.toggle, normalize_key_names(key))
            self.m.on_enter_READY(lambda *_: print("Press ", key, " to start/stop recording."))
        self.m.to_READY()
        keylistener.run()


# ---------------------------------------------------------------------------
# Streaming-mode App (new VAD-based pipeline)
# ---------------------------------------------------------------------------

class App():
    """Streaming dictation app with VAD-based chunking.

    Architecture: 4 threads connected by queues.

      [Recording Thread] -> audio frames -> [VAD Thread] -> speech chunks
                                             (inside StreamingRecorder)
          -> transcription_queue -> [Transcription Thread]
          -> typing_queue -> [Typing Thread]

    Sentinel flow: recorder.stop() puts None on transcription_queue
                   -> transcription thread puts None on typing_queue
                   -> typing thread plays stop beep and sets active=False
    """

    def __init__(self, args):
        self.args = args
        self.active = False

        # Queues connecting the pipeline stages
        self.transcription_queue = queue.Queue()
        self.typing_queue = queue.Queue()

        # Workers
        self.transcription_worker = TranscriptionWorker(
            args.model_name, args.device, args.compute_type, args.language
        )
        self.replayer = KeyboardReplayer()
        self.recorder = None  # created fresh each session
        self.timer = None

        # Sound effects
        self.SOUND_EFFECTS = {
            "start_recording": loadwav("assets/granted-04.wav"),
            "finish_recording": loadwav("assets/beepbeep.wav")
        }

    def beep(self, k, wait=True):
        playsound(self.SOUND_EFFECTS[k], wait=wait)

    def start(self):
        """Begin a streaming dictation session."""
        if self.active:
            return None

        self.active = True

        # Drain any leftover items from previous session
        for q in (self.transcription_queue, self.typing_queue):
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break

        self.beep("start_recording")
        print('Recording (streaming mode) ...')

        # Create fresh recorder
        self.recorder = StreamingRecorder(
            self.transcription_queue,
            silence_ms=self.args.silence_ms,
            auto_stop_silence_s=self.args.auto_stop_silence or None,
            on_auto_stop=self._auto_stop,
        )
        self.recorder.start()

        # Start transcription and typing threads
        self._transcription_thread = threading.Thread(
            target=self._transcription_loop, daemon=True
        )
        self._typing_thread = threading.Thread(
            target=self._typing_loop, daemon=True
        )
        self._transcription_thread.start()
        self._typing_thread.start()

        # Safety timer
        if self.args.max_time:
            self.timer = threading.Timer(self.args.max_time, self.timer_stop)
            self.timer.start()

        return True

    def stop(self):
        """Stop the current dictation session."""
        if not self.active:
            return None

        print('Stopping ...')
        self.beep("finish_recording", wait=False)  # immediate feedback
        if self.timer is not None:
            self.timer.cancel()
            self.timer = None

        # Stop recorder — this flushes remaining audio and sends sentinel
        self.recorder.stop()
        return True

    def _auto_stop(self):
        """Auto-stop: double beep to distinguish from manual stop."""
        if not self.active:
            return None

        print('Auto-stopping (silence timeout) ...')
        self.beep("finish_recording", wait=True)
        self.beep("finish_recording", wait=False)
        if self.timer is not None:
            self.timer.cancel()
            self.timer = None

        self.recorder.stop()
        return True

    def timer_stop(self):
        print('Timer stop')
        self._auto_stop()

    def toggle(self):
        return self.start() or self.stop()

    def _transcription_loop(self):
        """Consumer: takes audio chunks from transcription_queue, transcribes, puts text on typing_queue."""
        while True:
            chunk = self.transcription_queue.get()
            if chunk is None:
                # Sentinel: end of stream
                self.typing_queue.put(None)
                break

            try:
                t0 = time.time()
                text = self.transcription_worker.transcribe_chunk(chunk)
                elapsed = time.time() - t0
                audio_dur = len(chunk) / StreamingRecorder.SAMPLE_RATE
                print('[transcribe] %.1fs audio -> %.1fs processing: "%s"' % (audio_dur, elapsed, text.strip()))
                if text.strip():
                    self.typing_queue.put(text)
            except Exception as e:
                print('[transcribe] Error: %s' % e)

    def _typing_loop(self):
        """Consumer: takes text from typing_queue and types it."""
        first_chunk = True
        while True:
            text = self.typing_queue.get()
            if text is None:
                # Sentinel: end of stream (beep already played in stop())
                self.active = False
                print('Session ended.')
                break

            # Strip leading space from first chunk only
            if first_chunk:
                if text.startswith(' '):
                    text = text[1:]
                first_chunk = False
            else:
                # Add space between chunks if the text doesn't start with punctuation
                if text and text[0] not in ' .,;:!?\'")-]}>':
                    text = ' ' + text

            self.replayer.type_text(text)

    def run(self):
        def normalize_key_names(keyseqs, parse=False):
            k = keyseqs.replace('<win>', '<cmd>').replace('<win_r>', '<cmd_r>').replace('<win_l>', '<cmd_l>').replace('<super>', '<cmd>').replace('<super_r>', '<cmd_r>').replace('<super_l>', '<cmd_l>')
            if parse:
                k = keyboard.HotKey.parse(k)[0]
            print('Using key:', k)
            return k

        auto_stop_info = ', auto-stop after %ds silence' % self.args.auto_stop_silence if self.args.auto_stop_silence else ', no auto-stop'
        print('Streaming dictation mode (chunk silence: %dms%s)' % (self.args.silence_ms, auto_stop_info))

        if (platform.system() != 'Windows' and not self.args.key_combo) or self.args.double_key:
            key = self.args.double_key or (platform.system() == 'Linux' and '<ctrl_r>') or '<cmd_r>'
            keylistener = DoubleKeyListener(self.start, self.stop, normalize_key_names(key, parse=True))
            print("Double tap ", key, " to start recording. Tap again to stop recording")
        else:
            key = self.args.key_combo or '<win>+z'
            keylistener = KeyListener(self.toggle, normalize_key_names(key))
            print("Press ", key, " to start/stop recording.")

        keylistener.run()


if __name__ == "__main__":
    args = parse_args()
    if args.batch_mode:
        print('Using batch mode (original behaviour)')
        BatchApp(args).run()
    else:
        App(args).run()
