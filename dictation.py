import enum
import os
import time
import threading
import queue
import argparse
import platform
import subprocess
import re
import pyaudio
import numpy as np
from faster_whisper import WhisperModel
from faster_whisper.vad import get_speech_timestamps, VadOptions, get_vad_model
from pynput import keyboard
from pynput import mouse
from transitions import Machine
from coverage_ledger import (CoverageLedger, diagnostic, MAX_SECONDS, RATE, audio_stats, silent_audio,
                             HELD_FOR_REWRITE)
import context_capture
import rewrite

# PortAudio paInputOverflowed: input samples were dropped by the driver.
INPUT_OVERFLOWED_ERRNO = -9981  # pyaudio.paInputOverflowed


# Session language detection (rewrite mode in the Shift language): the
# alternative language must reach this probability. Measured on large-v3:
# Dutch voice notes never gave English >= 0.3 on >= 3 s of audio, accented
# English gave >= 0.74 on 5 s.
DETECT_MIN_P = 0.5
# A chunk with less speech than this is decoded in the fallback language and
# the detection waits for the next chunk.
DETECT_MIN_VOICED_S = 1.0


# Sentinel: distinguishes "caller passed no language" (keep current) from an
# explicit language=None ("auto-detect this session"). Used by App.start /
# BatchApp.start so the momentary hotkey selection sets the language every
# session without the combo-mode toggle() path resetting a forced -l language.
_UNSET = object()

# ---------------------------------------------------------------------------
# Output hygiene: hallucination filter + horizontal-rule guard
# ---------------------------------------------------------------------------
#
# Whisper was trained on subtitled video, so a chunk without real speech does
# not decode to an empty string -- it decodes to the most likely *subtitle* for
# silence. On this machine (nl, large-v3) that is a small, stable set: a
# broadcaster ident, a sign-off, and a literal '***' filler. Measured over 60
# days of journal output: 93x '***', 29x 'TV Gelderland 2021', 33x a
# 'dank u/je wel' variant, nearly all on chunks shorter than two seconds.
#
# First line of defence is vad_filter on the transcribe call, so silence never
# reaches the decoder. This list is the second net, for chunks carrying just
# enough noise to pass VAD. Only a chunk that is *entirely* one of these
# phrases is dropped, never a phrase inside a real sentence, and every drop is
# printed so nothing disappears silently from the journal.
# These can also be literal dictation. Never reject them without audio evidence.
THANK_PHRASES = (
    'dank u wel', 'dank je wel', 'dankjewel',
    'dank u wel voor het kijken', 'dank je wel voor het kijken',
    'bedankt voor het kijken', 'dank u wel voor het luisteren',
    'thank you for watching', 'thanks for watching',
)

HALLUCINATION_PHRASES = (
    'tv gelderland 2021',
    'ondertiteling door de amara.org gemeenschap',
    'ondertiteld door de amara.org gemeenschap',
    'ondertiteling',
    'muziek',
    'applaus',
    # Subtitle sound captions (17/10: 'ZANG EN MUZIEK' on 10.5 s of silence).
    'zang',
    'zang en muziek',
    'geluid van machines',
    'muziek en zang',
    'muziek speelt',
    'rustige muziek',
    'spannende muziek',
    'vrolijke muziek',
    'gelach',
    'stilte',
    'ondertitels ingevoegd door de amara.org gemeenschap',
    'ondertiteling ingevoegd door de amara.org gemeenschap',
    'subtitles by the amara.org community',
)

# A run of three or more of these characters renders as a horizontal rule in
# Markdown (and as a table separator, setext heading or bullet elsewhere), so a
# dictated pause must never survive as an unbroken run.
_RULE_RUN_RE = re.compile(r'([*\-_=~#])\1{2,}')

# Dots are handled separately: they never form a Markdown rule on their own,
# but a dictated pause still gets auto-formatted into a dash by editors and
# chat clients. --ellipsis-style decides what a pause becomes.
_ELLIPSIS_RE = re.compile(r'\.{3,}|\u2026+')


def _phrase_key(text):
    """Normalise a chunk for comparison: lowercase, letters/digits only."""
    return re.sub(r'[^a-z0-9]+', ' ', text.lower()).strip()


_THANK_KEYS = frozenset(_phrase_key(p) for p in THANK_PHRASES)
_HALLUCINATION_KEYS = frozenset(_phrase_key(p) for p in HALLUCINATION_PHRASES)

# Families rather than single strings, still whole-chunk only:
# a broadcaster ident with any year, a subtitle credit line ("Ondertiteling
# door ...", "Subtitles by ..."), and a chunk made only of sound-caption words
# ("MUZIEK. ZANG.", "muziek en applaus").
_IDENT_RE = re.compile(r'^tv gelderland(?: \d{2,4})?$')
_CREDIT_RE = re.compile(r'^(?:ondertitel\w*|subtitle[sd]?)\b.*\b(?:door|by|amara|ingevoegd|gemeenschap|community)\b')
_SOUND_WORDS = frozenset(('muziek', 'zang', 'applaus', 'gelach', 'gejuich', 'stilte',
                          'music', 'applause', 'laughter', 'silence', 'singing'))
# "(muziek speelt)", "[Gelach]", "*zucht*": a short chunk wholly in caption brackets.
_BRACKETED_RE = re.compile(r'^[\[(\u266a*]+[^\[\]()]{1,40}[\])\u266a*]+[.!]?$')


def is_hallucination(text):
    """True when the whole chunk is a known silence artefact rather than speech."""
    stripped = text.strip()
    if not stripped:
        return False
    # A chunk of pure punctuation ('***', '...', '.') carries no words at all.
    if not re.search(r'[a-zA-Z0-9]', stripped):
        return True
    key = _phrase_key(stripped)
    if key in _HALLUCINATION_KEYS or _IDENT_RE.match(key):
        return True
    words = key.split()
    if _CREDIT_RE.match(key) and len(words) <= 10:
        return True
    if any(w in _SOUND_WORDS for w in words) and all(w in _SOUND_WORDS or w in ('en', 'and') for w in words):
        return True
    return bool(_BRACKETED_RE.match(stripped)) and len(words) <= 4


def break_rule_runs(text):
    """Space out '...', '***', '---' so a dictated pause cannot render as a line."""
    def space_out(match):
        run = match.group(0)
        return ' '.join(run)
    return _RULE_RUN_RE.sub(space_out, text)


def format_ellipsis(text, style='space'):
    """Render a dictated pause so no formatter can turn it into a line.

    space  -> '. . .'  (a visible pause, impossible to read as a rule)
    single -> '\u2026'      (one ellipsis character, same protection, tidier prose)
    keep   -> unchanged
    """
    if style == 'keep':
        return text
    replacement = '\u2026' if style == 'single' else '. . .'
    return _ELLIPSIS_RE.sub(replacement, text)


# Never teach the next decode a pause, subtitle artefact or repetition loop.
_PAUSE_RUN_RE = re.compile(r'(?:\.\s*){3,}|\u2026+|(?:[*_~#=\-]\s*){3,}')
_REPEAT_RE = re.compile(r'\b(.{2,80}?)(?:\s+\1){3,}(?=\W|$)', re.IGNORECASE)


def prompt_text(text):
    """Context is a hint, not an output archive. Omit suspect material."""
    if is_hallucination(text) or _phrase_key(text) in _THANK_KEYS or _REPEAT_RE.search(text):
        return ''
    return re.sub(r'\s+', ' ', _PAUSE_RUN_RE.sub(' ', text)).strip()


def suspect_decode(text, duration):
    """Bounded recovery trigger, not proof that speech was lost."""
    if not text.strip() or is_hallucination(text):
        return True
    if _PAUSE_RUN_RE.search(text) or _REPEAT_RE.search(text):
        return True
    return duration >= 8 and len(re.findall(r'\w+', text)) < duration / 2


# Peak below this on a silent range means nothing near speech level arrived.
SILENT_PEAK_DBFS = -50.0


def thanks_on_silence(text, audio):
    """Require both low energy and a low peak; soft real thanks are not captions."""
    return (_phrase_key(text) in _THANK_KEYS and audio is not None
            and silent_audio(audio) and audio['peak_dbfs'] < SILENT_PEAK_DBFS)


def transcript_discard_reason(text, audio, drop_phrases=True):
    """One acceptance gate for ordinary output and each stop-recovery window."""
    if drop_phrases and text.strip() and (
            is_hallucination(text)
            or thanks_on_silence(text, audio)):
        return 'known_silence_artifact'
    # WP11: exactly two identical multiword phrases on short, near-idle audio.
    # Zero energy alone is not speech absence: quiet real words must survive.
    words = _phrase_key(text).split()
    half = len(words) // 2
    if (audio and 1 <= audio['seconds'] <= 3 and audio['voiced_s'] == 0
            and audio['rms_dbfs'] <= -60 and 6 <= len(words) <= 12
            and len(words) % 2 == 0 and words[:half] == words[half:]):
        return 'silent_repeated_phrase'
    if text.strip() and audio and silent_audio(audio) and (
            suspect_decode(text, audio['seconds'])
            or audio['peak_dbfs'] < SILENT_PEAK_DBFS):
        return 'silent_audio'
    return None


def sanitize_transcript(text, drop_phrases=True, space_runs=True, ellipsis_style='space', audio=None):
    """Clean one transcribed chunk before it is typed.

    Returns '' when the chunk should not be typed at all.
    """
    if drop_phrases and (is_hallucination(text) or thanks_on_silence(text, audio)):
        print('[filter] Dropped silence artefact: %r' % text.strip())
        return ''
    cleaned = text
    if space_runs:
        cleaned = break_rule_runs(cleaned)
    cleaned = format_ellipsis(cleaned, ellipsis_style)
    if cleaned != text:
        print('[filter] Rewrote pause run: %r -> %r' % (text.strip(), cleaned.strip()))
    return cleaned


def log_cuda_oom_snapshot():
    """One read-only, time/output-bounded diagnostic; never obscure the OOM."""
    fields = dict(pid=os.getpid())
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory',
             '--format=csv,noheader'],
            capture_output=True, text=True, timeout=1, check=False)
        fields.update(status='ok' if result.returncode == 0 else 'query_failed',
                      returncode=result.returncode,
                      compute_apps=result.stdout[:8192], stderr=result.stderr[:512],
                      truncated=len(result.stdout) > 8192)
    except Exception as exc:
        fields.update(status='unavailable', error=type(exc).__name__)
    diagnostic('cuda_oom_snapshot', **fields)

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
    import sounddevice

    def playsound(s, wait=True):
        """Play sound on a dedicated OutputStream so concurrent beeps don't cancel each other."""
        done = threading.Event()
        pos = [0]

        def callback(outdata, frames, time_info, status):
            end = pos[0] + frames
            chunk = s[pos[0]:end]
            if len(chunk) < frames:
                outdata[:len(chunk)] = chunk
                outdata[len(chunk):] = 0
                done.set()
                raise sounddevice.CallbackStop
            outdata[:] = chunk
            pos[0] = end

        channels = s.shape[1] if s.ndim > 1 else 1
        stream = sounddevice.OutputStream(
            samplerate=44100, channels=channels,
            callback=callback, blocksize=1024,
            finished_callback=done.set
        )
        stream.start()
        if wait:
            done.wait()
            stream.close()
        else:
            # Fire-and-forget: clean up after playback finishes
            threading.Thread(target=lambda: (done.wait(), stream.close()), daemon=True).start()

    def loadwav(filename):
        data, fs = sf.read(filename, dtype='float32')
        return data


def tone(frequency, ms, volume=0.25, channels=2, rate=44100):
    """A short sine tone with soft edges, shaped like the loaded WAV files."""
    n = int(rate * ms / 1000)
    t = np.arange(n) / rate
    wave = np.sin(2 * np.pi * frequency * t) * volume
    edge = max(1, min(n // 4, int(rate * 0.005)))
    ramp = np.linspace(0.0, 1.0, edge)
    wave[:edge] *= ramp
    wave[-edge:] *= ramp[::-1]
    wave = wave.astype(np.float32)
    return np.repeat(wave[:, None], channels, axis=1) if channels > 1 else wave


def rewrite_sounds(start):
    """Rewrite-mode cues built from the start sound plus synthetic tones.

    start twice              rewrite mode, no reply context
    start twice + high pip   rewrite mode with context
    short mid tone           rewritten text pasted
    two low tones            rewrite failed or unavailable: raw text pasted
    """
    channels = start.shape[1] if getattr(start, 'ndim', 1) > 1 else 1
    def gap(ms):
        return tone(1, ms, volume=0.0, channels=channels)
    twice = np.concatenate([start, gap(60), start])
    failed = np.concatenate([tone(330, 120, channels=channels), gap(60),
                             tone(220, 200, channels=channels)])
    return {
        'rewrite_start': twice,
        'rewrite_start_context': np.concatenate([twice, gap(60), tone(1320, 90, channels=channels)]),
        'rewrite_done': tone(880, 80, channels=channels),
        'rewrite_failed': failed,
        'rewrite_unavailable': np.concatenate([failed, gap(80), start]),
    }


# ---------------------------------------------------------------------------
# Batch-mode classes (original behaviour, kept for --batch-mode fallback)
# ---------------------------------------------------------------------------

class SpeechTranscriber:
    def __init__(self, callback, model_size='base', device='cpu', compute_type="int8", language=None,
                 vad_filter=False):
        self.callback = callback
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.language = language
        # Same opt-in silence gate as streaming mode (see TranscriptionWorker).
        self.vad_filter = vad_filter

    def transcribe(self, event):
        print('Transcribing...')
        audio = event.kwargs.get('audio', None)
        if audio is not None:
            # Force language if specified, otherwise auto-detect
            if self.language:
                segments, info = self.model.transcribe(audio, beam_size=5, language=self.language,
                                                       vad_filter=self.vad_filter)
                print("Using forced language: '%s'" % self.language)
            else:
                segments, info = self.model.transcribe(audio, beam_size=5, vad_filter=self.vad_filter)
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

    SPEECH_FRAME = 512  # Silero VAD analyses audio in 512-sample (~32ms) frames

    def __init__(self, transcription_queue, silence_ms=1000,
                 auto_stop_silence_s=10, on_auto_stop=None,
                 min_chunk_s=2.5, max_chunk_s=10, eager_prob=0.35, ledger=None):
        self.transcription_queue = transcription_queue
        self.ledger = ledger
        self._buffer_offset = 0
        self.silence_ms = silence_ms
        self.auto_stop_silence_s = auto_stop_silence_s
        self.on_auto_stop = on_auto_stop
        self.recording = False
        self._stop_lock = threading.Lock()

        # Eager flush: don't wait for a full silence_ms pause during a long
        # turn. Once min_chunk_s of speech has piled up since the last cut,
        # split on the next micro-pause so transcription starts early instead
        # of after a 1s pause that may never come. max_chunk_s is the upper
        # bound — cut at the latest pause still within it. Setting
        # min_chunk_s to 0 disables eager flushing.
        self.min_chunk_samples = int(min_chunk_s * self.SAMPLE_RATE) if min_chunk_s else 0
        self.max_chunk_samples = int(max_chunk_s * self.SAMPLE_RATE) if max_chunk_s else 0

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

        # Eager split is chosen statistically: within the allowed window we cut
        # at the quietest moment (lowest Silero speech probability). If that
        # quietest point is below eager_prob it's a real pause → cut early; once
        # the window hits the ceiling we cut at the quietest point regardless,
        # so a fluent unbroken talker still gets cut on the best boundary
        # available rather than mid-word or not at all.
        self.eager_prob = eager_prob

        # Track how many samples we already processed in the VAD thread
        self._vad_processed_samples = 0

    def start(self):
        """Start recording + VAD threads."""
        self.recording = True
        self._raw_frames = []
        self._vad_processed_samples = 0
        self._buffer_offset = 0

        self._rec_thread = threading.Thread(target=self._record_loop, daemon=True)
        self._vad_thread = threading.Thread(target=self._vad_loop, daemon=True)
        self._rec_thread.start()
        self._vad_thread.start()

    def stop(self):
        """Stop recording and flush remaining audio."""
        with self._stop_lock:
            if not self.recording:
                return
            print('Done recording.')
            self.recording = False
            # The VAD writer must finish before the final snapshot/sentinel.
            # A timed-out join could previously enqueue audio AFTER the sentinel.
            self._rec_thread.join()
            self._vad_thread.join()

            remaining = self._get_all_audio_fp32()
            if remaining is not None and len(remaining) > self._vad_processed_samples:
                leftover = remaining[self._vad_processed_samples:]
                print('[flush] Sending remaining %.1fs audio' % (len(leftover) / self.SAMPLE_RATE))
                self._enqueue(leftover, True, self._vad_processed_samples)

            self.transcription_queue.put(None)

    def _record_loop(self):
        """PyAudio recording loop — runs in its own thread."""
        p = pyaudio.PyAudio()
        try:
            stream = p.open(
                format=pyaudio.paInt16, channels=1, rate=self.SAMPLE_RATE,
                frames_per_buffer=self.FRAMES_PER_BUFFER, input=True)
        except Exception as exc:
            p.terminate()
            if self.ledger:
                self.ledger.incident('capture_open_error', error=str(exc))
            threading.Thread(target=self.on_auto_stop, daemon=True).start()
            return

        if self.ledger:
            latency = time.time() - self.ledger.started
            diagnostic('capture_ready', start_latency_seconds=latency)
            if latency > 0.5:
                self.ledger.incident('capture_start_delay', seconds=latency)
        try:
            while self.recording:
                # Surface overflow instead of silently accepting missing input,
                # but keep recording: stopping on an overflow would lose far
                # more speech than the overflow itself.
                try:
                    data = stream.read(self.FRAMES_PER_BUFFER, exception_on_overflow=True)
                except OSError as exc:
                    if getattr(exc, 'errno', None) != INPUT_OVERFLOWED_ERRNO:
                        if self.ledger:
                            self.ledger.incident('capture_error', error=str(exc))
                        threading.Thread(target=self.on_auto_stop, daemon=True).start()
                        break
                    # Log only: a notification per overflow would flood the desktop.
                    diagnostic('capture_overflow', error=str(exc))
                    continue
                except Exception as exc:
                    if self.ledger:
                        self.ledger.incident('capture_error', error=str(exc))
                    threading.Thread(target=self.on_auto_stop, daemon=True).start()
                    break
                if self.ledger:
                    data = self.ledger.capture(data)
                with self._buffer_lock:
                    if data:
                        self._raw_frames.append(data)
                if self.ledger and self.ledger.samples >= MAX_SECONDS * RATE:
                    self.ledger.incident('duration_limit', seconds=MAX_SECONDS)
                    threading.Thread(target=self.on_auto_stop, daemon=True).start()
                    break
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()
        return

    def _enqueue(self, audio, boundary, start):
        if getattr(self, 'ledger', None):
            self.transcription_queue.put((audio.copy(), boundary, self._buffer_offset + start))
        else:
            self.transcription_queue.put((audio, boundary))

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
            seg_end = ts['end']

            # Skip segments we've already sent
            if seg_end <= self._vad_processed_samples:
                continue

            # Only send if there's confirmed silence after this segment
            if seg_end + silence_samples <= audio_end:
                # This segment has enough trailing silence — it's complete.
                # Start no earlier than what we already flushed (an eager
                # flush may have split inside this segment).
                # VAD decides WHEN to send, never which captured samples to
                # discard. Quiet speech before seg_start must reach Whisper too.
                chunk_start = self._vad_processed_samples
                chunk = audio[chunk_start:seg_end]
                if len(chunk):
                    duration = len(chunk) / self.SAMPLE_RATE
                    print('[VAD] Speech chunk: %.1fs (samples %d-%d)' % (duration, chunk_start, seg_end))
                    # Boundary: a full >=silence_ms pause followed this segment,
                    # a natural sentence end — flush the paste buffer here.
                    self._enqueue(chunk, True, chunk_start)

                # Update processed marker to after this segment
                self._vad_processed_samples = max(self._vad_processed_samples, seg_end)

        # Eager flush: a long turn with only short (<silence_ms) pauses never
        # trips the trailing-silence test above, so the buffer would grow
        # until the speaker finally pauses for real. Once min_chunk_samples of
        # audio sits unsent, split it on a micro-pause and send that head so
        # transcription starts now instead of waiting.
        if self.min_chunk_samples:
            unsent_len = audio_end - self._vad_processed_samples
            if unsent_len > self.min_chunk_samples:
                split = self._find_split_point(audio, self._vad_processed_samples, audio_end)
                if split is not None and split > self._vad_processed_samples:
                    chunk = audio[self._vad_processed_samples:split]
                    if len(chunk) > self.SAMPLE_RATE * 0.1:
                        duration = len(chunk) / self.SAMPLE_RATE
                        print('[VAD] Eager chunk: %.1fs (samples %d-%d)'
                              % (duration, self._vad_processed_samples, split))
                        # Not a boundary: only a micro-pause inside fluent speech.
                        # The paste buffer keeps accumulating these until it
                        # reaches paste_min_s, then transcribes them merged.
                        self._enqueue(chunk, False, self._vad_processed_samples)
                    self._vad_processed_samples = split

        self._trim_buffer()
        return False

    def _trim_buffer(self):
        """Drop fully-processed frames so VAD cost and memory stay flat over a
        long turn instead of growing with session length (the VAD re-scans the
        whole buffer every tick). Buffer and processed-marker shift together by
        the same whole-frame amount, so every index stays relative to the
        current buffer and the rest of the pipeline is unaffected.
        """
        drop_frames = self._vad_processed_samples // self.FRAMES_PER_BUFFER
        if drop_frames <= 0:
            return
        with self._buffer_lock:
            # Keep at least the last frame; never outrun the recording thread.
            drop_frames = min(drop_frames, len(self._raw_frames) - 1)
            if drop_frames <= 0:
                return
            del self._raw_frames[:drop_frames]
        self._vad_processed_samples -= drop_frames * self.FRAMES_PER_BUFFER
        self._buffer_offset = getattr(self, '_buffer_offset', 0) + drop_frames * self.FRAMES_PER_BUFFER

    def _speech_probs(self, audio):
        """Per-frame Silero speech probability for audio (one value per
        SPEECH_FRAME samples, ~32ms). Low = silence, high = speech. One forward
        pass — cheaper than running get_speech_timestamps repeatedly."""
        model = get_vad_model()
        frame = self.SPEECH_FRAME
        pad = (frame - len(audio) % frame) % frame
        padded = np.pad(audio, (0, pad)) if pad else audio
        return np.asarray(model(padded.reshape(1, -1)).squeeze(0)).reshape(-1)

    def _find_split_point(self, audio, start, hard_end):
        """Locate a split point inside audio[start:hard_end] statistically.

        Within the allowed range [min_chunk_samples, ceiling] pick the quietest
        moment (lowest smoothed speech probability) and cut there when either it
        is a genuine pause (prob <= eager_prob) or the window has reached the
        ceiling (then cut at the quietest point regardless — best boundary
        available beats waiting). Returns an absolute sample index, or None to
        keep waiting (still below the ceiling and no quiet-enough point yet).
        """
        window = audio[start:hard_end]
        win_len = len(window)
        if win_len < self.SAMPLE_RATE * 0.5:
            return None

        upper = self.max_chunk_samples or win_len
        frame = self.SPEECH_FRAME

        probs = self._speech_probs(window)
        if len(probs) >= 3:  # smooth ~96ms so a single noisy frame isn't a false dip
            probs = np.convolve(probs, np.ones(3) / 3, mode='same')

        lo = int(self.min_chunk_samples // frame)
        hi = int(min(win_len, upper) // frame)
        if hi <= lo:
            return None

        rel = int(np.argmin(probs[lo:hi]))
        k = lo + rel
        quietest = float(probs[k])
        at_ceiling = win_len >= upper

        if quietest <= self.eager_prob or at_ceiling:
            split = k * frame
            if split <= 0:
                return None
            print('[VAD] split @ %.1fs (quietest p=%.2f%s, window %.1fs)'
                  % ((start + split) / self.SAMPLE_RATE, quietest,
                     ', ceiling' if at_ceiling and quietest > self.eager_prob else '',
                     win_len / self.SAMPLE_RATE))
            return start + split
        return None


class TranscriptionWorker:
    """Loads Whisper model once and transcribes audio chunks from a queue."""

    # (fallback, alternative) while the session language is still open: detect
    # it on the first chunk with speech, restricted to these two (rewrite mode
    # in the Shift language). Set per session by the app.
    detect = None

    def __init__(self, model_size='base', device='cpu', compute_type='int8', language=None,
                 context_chars=500, vad_filter=False, filter_hallucinations=True,
                 space_rule_runs=True, ellipsis_style='space'):
        print('Loading Whisper model: %s (device=%s, compute=%s)' % (model_size, device, compute_type))
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.language = language
        # Rolling tail of text transcribed earlier in this session, fed to the
        # next chunk as initial_prompt so eager-flush chunks keep cross-seam
        # context (Whisper's prompt window is 224 tokens; the char cap stays
        # well under that). 0 disables.
        self.context_chars = context_chars
        self._context = ''
        # Off by default, deliberately. Gating the decoder with VAD does remove
        # the subtitle artefacts at the source, but it also strips silence from
        # the audio and so shifts segment boundaries: measured on 5 minutes of
        # noisy far-field speech, 3 of 10 blocks decoded differently (neither
        # version better). The chunking in this app is heavily tuned for
        # continuous dictation, so recognition quality wins over elegance --
        # the artefacts are removed after the fact by sanitize_transcript,
        # which cannot alter a single recognised word. Enable with
        # --transcribe-vad only if the phrase filter proves insufficient.
        self.vad_filter = vad_filter
        self.filter_hallucinations = filter_hallucinations
        self.space_rule_runs = space_rule_runs
        self.ellipsis_style = ellipsis_style

    def reset_context(self):
        """Forget the rolling context (new session or language switch)."""
        self._context = ''

    def detect_session_language(self, audio_fp32):
        """Fix self.language for this session from the first chunk.

        Only the two candidates count. The alternative needs probability
        >= DETECT_MIN_P and more than the fallback; otherwise (short or noisy
        audio, an error) the fallback is used. Journal: numbers only.
        """
        fallback, other = self.detect
        self.detect = None
        audio_s = round(len(audio_fp32) / 16000, 2)
        try:
            _, _, probs = self.model.detect_language(audio_fp32)
        except Exception as exc:
            self.language = fallback
            diagnostic('language_detected', language=fallback, reason='error',
                       error=type(exc).__name__, audio_s=audio_s)
            return
        p = dict(probs)
        scores = {fallback: round(p.get(fallback, 0.0), 3), other: round(p.get(other, 0.0), 3)}
        chosen = other if scores[other] >= DETECT_MIN_P and scores[other] > scores[fallback] else fallback
        self.language = chosen
        diagnostic('language_detected', language=chosen, probabilities=scores, audio_s=audio_s)

    def transcribe_chunk(self, audio_fp32, recovery=False):
        """Transcribe a single audio chunk. Returns text string."""
        if self.detect and not self.language \
                and (audio_stats(audio_fp32) or {}).get('voiced_s', 0) >= DETECT_MIN_VOICED_S:
            self.detect_session_language(audio_fp32)
        # Until detection ran (a chunk without enough speech), decode in the fallback.
        language = self.language or (self.detect[0] if self.detect else None)
        prompt = (prompt_text(self._context) or None) if self.context_chars and not recovery else None
        def decode(initial_prompt):
            options = dict(beam_size=5, initial_prompt=initial_prompt,
                           vad_filter=self.vad_filter)
            if language:
                options['language'] = language
            try:
                segments, info = self.model.transcribe(audio_fp32, **options)
                return ''.join(segment.text for segment in segments).lstrip()
            except Exception as exc:
                message = str(exc).lower()
                if 'cuda' in message and 'out of memory' in message:
                    log_cuda_oom_snapshot()
                raise

        duration = len(audio_fp32) / 16000
        text = decode(prompt)
        stats = audio_stats(audio_fp32)
        suspect = suspect_decode(text, duration) or thanks_on_silence(text, stats)
        if suspect and not recovery:
            # Reset BEFORE retry: even a failed retry must break the cascade.
            self.reset_context()
            print('[recovery] Suspect decode (%.1fs audio); retry once without prompt' % duration)
            try:
                retry = decode(None)
                # Never replace useful first-pass words with a known artefact
                # or an even shorter suspect result.
                if (re.search(r'\w', retry) and not is_hallucination(retry)
                        and (not suspect_decode(retry, duration)
                             or is_hallucination(text)
                             or len(re.findall(r'\w+', retry)) >= len(re.findall(r'\w+', text)))):
                    text = retry
                elif not re.search(r'\w', text):
                    text = retry
            except Exception as exc:
                print('[recovery] Retry failed; retaining first decode: %s' % exc)
            if suspect_decode(text, duration):
                print('[recovery] Unresolved suspect decode; stop ledger will check')

        self.last_raw = text
        # Recovery keeps captions here so the ledger's per-window acceptance
        # gate can log the refusal before clean_correction/joining loses it.
        text = sanitize_transcript(text,
                                   drop_phrases=self.filter_hallucinations and not recovery,
                                   space_runs=self.space_rule_runs,
                                   ellipsis_style=self.ellipsis_style,
                                   audio=stats)

        # A dropped artefact must not enter the rolling prompt either: feeding
        # 'TV Gelderland 2021' back as context makes the next chunk repeat it.
        context = prompt_text(text)
        if self.context_chars and context and not suspect_decode(text, duration):
            self._context = (prompt_text(self._context) + ' ' + context).strip()[-self.context_chars:]

        return text


# ---------------------------------------------------------------------------
# Keyboard output (shared between batch and streaming mode)
# ---------------------------------------------------------------------------

class KeyboardReplayer():
    def __init__(self, callback=None):
        self.callback = callback
        self.kb = keyboard.Controller()
        # Window-ID waar de dictatie begon — paste landt hier ook als focus tussendoor verschuift.
        self.target_wid = None

    def window_id(self):
        """Active window id, or None."""
        try:
            return subprocess.check_output(
                ['xdotool', 'getactivewindow'], timeout=2
            ).decode().strip()
        except Exception:
            return None

    def capture_target_window(self):
        """Save active window at recording-start so paste lands here, even if focus drifts."""
        self.target_wid = self.window_id()

    def _restore_target_window(self):
        """Activate the recording-start window before paste (no-op if not captured).

        Retries once if focus didn't actually land on the target (workspace switches,
        focus-stealing prevention, or slow WM compositors can make --sync return before
        focus is truly assigned). Without this, paste sometimes lands on the wrong window
        even though the clipboard is filled correctly.
        """
        if not self.target_wid:
            return False
        for attempt in range(2):
            try:
                subprocess.run(
                    ['xdotool', 'windowactivate', '--sync', self.target_wid],
                    timeout=2, check=False, stderr=subprocess.DEVNULL,
                )
                time.sleep(0.15)
                active = subprocess.check_output(
                    ['xdotool', 'getactivewindow'], timeout=2
                ).decode().strip()
                if active == self.target_wid:
                    return True
            except Exception:
                pass
        print('[warn] target window %s did not regain focus — paste may land elsewhere'
              % self.target_wid)
        return False

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

    # Terminals paste with Ctrl+Shift+V; GUI apps with Ctrl+V.
    TERMINAL_CLASSES = {'kitty', 'gnome-terminal', 'konsole', 'xterm',
                        'alacritty', 'terminator', 'st', 'wezterm', 'tilix',
                        'urxvt', 'rxvt'}

    def _set_clipboard(self, text):
        """Place text on the system clipboard AND into CopyQ history.

        Clipboard-first injection: doing this for every app guarantees the
        dictation lands in clipboard history (so it is recoverable even if the
        paste keystroke fails) and lets us paste atomically instead of typing
        character-by-character. CopyQ is preferred because it records history;
        xclip is the fallback when CopyQ is unavailable. Returns command receipts;
        success is not an application acknowledgement.

        Note: 'copyq copy' sets the clipboard but the monitor skips its own
        writes, so it never reaches history; 'copyq add' records history but
        leaves the clipboard untouched. Both are needed — add for recovery,
        copy for the paste.
        """
        result = dict(ok=False, path='none', copyq_add=None, copyq_copy=None, xclip=None)
        try:
            for action in ('add', 'copy'):
                command = subprocess.run(['copyq', action, '--', text], timeout=3,
                                         check=False, stderr=subprocess.DEVNULL)
                result['copyq_' + action] = command.returncode
                if command.returncode:
                    raise RuntimeError('copyq %s rc=%d' % (action, command.returncode))
            time.sleep(0.05)
            result.update(ok=True, path='copyq')
            return result
        except Exception as exc:
            result['copyq_error'] = str(exc)
            diagnostic('copyq_failure', **result)
        try:
            command = subprocess.run(['xclip', '-selection', 'clipboard'],
                                     input=text.encode('utf-8'), timeout=3, check=False)
            result['xclip'] = command.returncode
            result.update(ok=command.returncode == 0, path='xclip')
        except Exception as exc:
            result['xclip_error'] = str(exc)
        diagnostic('clipboard_fallback', **result)
        return result

    def _send_paste(self, terminal):
        """Send the paste shortcut: Ctrl+Shift+V in terminals, Ctrl+V elsewhere."""
        self.kb.press(keyboard.Key.ctrl)
        if terminal:
            self.kb.press(keyboard.Key.shift)
        self.kb.press('v')
        self.kb.release('v')
        if terminal:
            self.kb.release(keyboard.Key.shift)
        self.kb.release(keyboard.Key.ctrl)

    def _type_via_pynput(self, text):
        """Last-resort character-by-character typing via pynput.

        Only used when no clipboard tool is available. Logs how many characters
        failed instead of swallowing every error silently, so dropped dictation
        is visible in the journal rather than vanishing without a trace.
        """
        failures = 0
        for element in text:
            try:
                self.kb.type(element)
                time.sleep(0.0025)
            except Exception:
                failures += 1
        if failures:
            print('[pynput] %d/%d characters failed to type' % (failures, len(text)))

    def add_to_history(self, text):
        """Put text in CopyQ history only (no clipboard change, no paste)."""
        try:
            return subprocess.run(['copyq', 'add', '--', text], timeout=3, check=False,
                                  stderr=subprocess.DEVNULL).returncode
        except Exception as exc:
            return type(exc).__name__

    def type_text(self, text, echo=True):
        """Inject text into the active window, clipboard-first for every app.

        echo=False keeps the text out of stdout/the journal (rewrite mode).
        """
        if not text:
            return
        if echo:
            print(text)
        focus_ok = self._restore_target_window()
        wm_class = self._get_active_window_class()
        terminal = wm_class in self.TERMINAL_CLASSES
        result = self._set_clipboard(text)
        result['paste'] = 'not_sent'
        result['focus_ok'] = focus_ok
        if not focus_ok:
            result['ok'] = False
        if result['ok']:
            try:
                self._send_paste(terminal)
                result['paste'] = 'shortcut_sent_not_acknowledged'
            except Exception as exc:
                result.update(ok=False, paste='failed', paste_error=str(exc))
        # No unverified character fallback: retained text goes to stop recovery.
        return result

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

        text = sanitize_transcript(text)

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


class MultiTapKeyListener():
    """Detect double-tap and single-tap on a modifier key, with momentary
    Left-Shift language selection.

    Ctrl_R:
    - Single tap   → deactivate_callback (stop recording, instant)
    - Double tap   → activate_callback(language) (start recording, instant)

    Left-Shift (momentary, no state kept):
    - Held while the double-tap fires → activate in `shift_language` (English)
    - Not held                        → activate in `default_language`
    The language is chosen at start time only; nothing is remembered between
    sessions. Right-Shift is deliberately ignored (name 'shift_r').

    Left-Alt (momentary, no state kept): held while the double-tap fires →
    rewrite mode for that session; Left-Shift then selects the language the
    rewrite is written in (the app decides what that means for recognition).
    Right-Alt (AltGr, 'alt_gr'/'alt_r') is ignored. With Shift already held, X
    reports the Left-Alt key as Meta_L (its second keymap level), so Meta_L
    counts as Left-Alt for press and release alike.

    The double-tap calls activate_callback(language, rewrite, keys); keys is
    metadata about the modifiers at that moment (held flags, how long held or
    how long ago released), for the journal.

    toggle_language_callback / language_key remain for backward compatibility
    (old stateful AltGr toggle) but are unused by default.
    """

    TAP_WINDOW = 0.3   # max seconds between taps for double-tap
    META_L_VK = 0xffe7  # keysym Meta_L: Left-Alt pressed or released while Shift is held

    def __init__(self, activate_callback, deactivate_callback,
                 toggle_language_callback=None, key=keyboard.Key.cmd_r,
                 language_key=None, default_language=None, shift_language='en'):
        self.activate_callback = activate_callback
        self.deactivate_callback = deactivate_callback
        self.toggle_language_callback = toggle_language_callback
        self.key = key
        self.language_key = language_key
        self.default_language = default_language
        self.shift_language = shift_language
        self.shift_held = False
        self.alt_held = False
        self.alt_as_meta = False
        # Last press / release time per modifier, for the journal only.
        self.mod_times = {'shift': [None, None], 'alt': [None, None]}
        self.last_press_time = 0
        self.tap_count = 0
        self.lang_last_press_time = 0
        self.lang_tap_count = 0

    @staticmethod
    def _get_vk(k):
        """Get virtual key code from any pynput key representation."""
        if hasattr(k, 'vk') and k.vk is not None:
            return k.vk
        if hasattr(k, 'value') and hasattr(k.value, 'vk'):
            return k.value.vk
        return None

    def _key_matches(self, key, target):
        """Check if pressed key matches target using vk codes for precision."""
        # Always compare by vk code to distinguish left/right modifiers
        a, b = self._get_vk(key), self._get_vk(target)
        if a is not None and b is not None:
            return a == b
        return key == target

    def _is_left_alt(self, key, name):
        return name in ('alt', 'alt_l') or (name is None and self._get_vk(key) == self.META_L_VK)

    def _mod_event(self, mod, pressed):
        self.mod_times[mod][0 if pressed else 1] = time.time()

    def keys_info(self, now=None):
        """Modifier state at the second tap: metadata only, no key text."""
        now = time.time() if now is None else now
        info = {}
        for mod, held in (('shift', self.shift_held), ('alt', self.alt_held)):
            down, up = self.mod_times[mod]
            info[mod] = held
            if held and down is not None:
                info[mod + '_held_s'] = round(now - down, 3)
            elif up is not None:
                info[mod + '_released_s_ago'] = round(now - up, 3)
        if self.alt_held and self.alt_as_meta:
            info['alt_as_meta'] = True
        return info

    def on_press(self, key):
        name = getattr(key, 'name', None)
        # Track Left-Shift held state for momentary language selection. Holding
        # Left-Shift while double-tapping the main key dictates in the shift
        # language (English) for that session only. Left-Shift reports name
        # 'shift' (vk 65505) on X11; Right-Shift ('shift_r') is ignored.
        if name in ('shift', 'shift_l'):
            if not self.shift_held:
                self._mod_event('shift', True)
            self.shift_held = True
            return
        # Left-Alt arrives as the generic name 'alt' (Alt_L), so it must be
        # tracked before the generic-modifier skip below; after Shift it
        # arrives as Meta_L (no name).
        if self._is_left_alt(key, name):
            if not self.alt_held:
                self._mod_event('alt', True)
                self.alt_as_meta = name is None
            self.alt_held = True
            return
        # Skip generic modifier events (e.g. Key.ctrl vk=65507) that pynput
        # sends alongside the specific Key.ctrl_r/Key.ctrl_l events.
        # Without this, pressing Ctrl_R triggers both Key.ctrl_r AND Key.ctrl,
        # and Key.ctrl shares vk=65507 with Key.ctrl_l causing false matches.
        if name in ('ctrl', 'alt'):
            return

        # Main key: Ctrl_R — double-tap start, single-tap stop
        if self._key_matches(key, self.key):
            current_time = time.time()
            if current_time - self.last_press_time < self.TAP_WINDOW:
                self.tap_count += 1
            else:
                self.tap_count = 1
            self.last_press_time = current_time

            if self.tap_count == 2:
                language = self.shift_language if self.shift_held else self.default_language
                self.activate_callback(language, self.alt_held, self.keys_info(current_time))
            elif self.tap_count == 1:
                self.deactivate_callback()

        # Language key: Ctrl_L — double-tap toggles language
        elif self.language_key and self._key_matches(key, self.language_key):
            current_time = time.time()
            if current_time - self.lang_last_press_time < self.TAP_WINDOW:
                self.lang_tap_count += 1
            else:
                self.lang_tap_count = 1
            self.lang_last_press_time = current_time

            if self.lang_tap_count == 2:
                self.lang_tap_count = 0
                if self.toggle_language_callback:
                    self.toggle_language_callback()

    def on_release(self, key):
        name = getattr(key, 'name', None)
        if name in ('shift', 'shift_l'):
            if self.shift_held:
                self._mod_event('shift', False)
            self.shift_held = False
        elif self._is_left_alt(key, name):
            if self.alt_held:
                self._mod_event('alt', False)
            self.alt_held = False

    def run(self):
        with keyboard.Listener(
                on_press=self.on_press,
                on_release=self.on_release) as listener:
            listener.join()


class MouseToggleListener():
    """Toggle dictation from the mouse side buttons.

    The keyboard trigger needs two hands: one on the mouse, one reaching for
    Ctrl_R. The thumb buttons are always under the hand that is already
    holding the mouse, so they carry the same start/stop action.

    Every configured button does the same thing -- there is no forward/back
    distinction here. A press starts a session when idle and stops it when
    recording, exactly like a double/single tap on Ctrl_R.

    Both thumb buttons pressed together (or one button bouncing) would
    otherwise read as start-then-immediate-stop, so presses inside
    COALESCE_WINDOW after an accepted press are swallowed.

    The listener does not suppress the button, so a browser still sees its
    normal back/forward navigation.
    """

    COALESCE_WINDOW = 0.4   # seconds; a second press inside this is the same intent

    def __init__(self, toggle_callback, button_names):
        self.toggle_callback = toggle_callback
        self.buttons = set()
        for raw in button_names:
            name = raw.strip()
            if not name:
                continue
            button = getattr(mouse.Button, name, None)
            if button is None:
                print('[mouse] Unknown button %r, ignored' % name)
                continue
            self.buttons.add(button)
        self.last_fire_time = 0.0

    def on_click(self, x, y, button, pressed):
        if not pressed or button not in self.buttons:
            return
        now = time.time()
        if now - self.last_fire_time < self.COALESCE_WINDOW:
            return
        self.last_fire_time = now
        self.toggle_callback()

    def start(self):
        """Run the listener on its own daemon thread (non-blocking)."""
        if not self.buttons:
            return None
        listener = mouse.Listener(on_click=self.on_click)
        listener.daemon = True
        listener.start()
        print('Mouse toggle: %s starts/stops dictation.'
              % ', '.join(sorted(b.name for b in self.buttons)))
        return listener


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

    parser.add_argument('--shift-language', type=str, default='en',
                        help='''\
Language used for a session when Left-Shift is held while double-tapping the
start key (momentary, no state kept). The unshifted double-tap uses --language.
Default: en (English).''')

    parser.add_argument('--silence-ms', type=int, default=1000,
                        help='''\
Silence duration in milliseconds before splitting a speech chunk (streaming mode).
Lower values give faster feedback but may split mid-sentence.
Default: 1000 (1 second).''')

    parser.add_argument('--min-chunk-s', type=float, default=2.5,
                        help='''\
Eager-flush floor in seconds (streaming mode). Once you have spoken this long
since the last cut, the audio is split on the next micro-pause and transcription
starts immediately instead of waiting for a full >=silence-ms pause. Lower gives
faster feedback but shorter (less context) chunks; higher waits for more speech.
Set to 0 to disable eager flushing. Default: 2.5.''')

    parser.add_argument('--max-chunk-s', type=float, default=10,
                        help='''\
Eager-flush ceiling in seconds (streaming mode). Upper bound on an eager chunk:
the cut lands on the latest micro-pause still within this window, so Whisper gets
as much context as allowed. Higher = more context/quality, more latency on long
turns. Default: 10.''')

    parser.add_argument('--eager-prob', type=float, default=0.35,
                        help='''\
Speech-probability threshold for an early eager cut (streaming mode). Within the
allowed window the quietest moment is found; if its Silero speech probability is
at or below this value it counts as a real pause and is cut early. Lower = only
cut early on clearer silences (longer chunks); higher = cut early more eagerly.
At the ceiling a cut always happens at the quietest point. Default: 0.35.''')

    parser.add_argument('--paste-min-s', type=float, default=12.0,
                        help='''\
Hold-back paste buffer (streaming mode). Instead of transcribing and typing
every short eager chunk on its own, accumulate audio chunks until they total
this many seconds, then transcribe them merged in one pass and paste the result
in one go. More context per pass = better accuracy; the trade-off is text lands
in fewer, larger blocks with a bit more delay. A real >=silence-ms pause (natural
sentence end) always flushes early, as does the end of the turn. Set to 0 to
transcribe each chunk immediately (old behaviour). Default: 12.''')

    parser.add_argument('--context-chars', type=int, default=500,
                        help='''\
Carry a rolling tail of previously transcribed text (this session) into each next
chunk as Whisper's initial_prompt (streaming mode). Restores cross-chunk context
lost by eager-flush splitting, improving accuracy on fluent dictation. The tail
is capped at this many characters (well under Whisper's 224-token prompt window)
and resets on session start and language toggle. Set to 0 to disable. Default: 500.''')

    parser.add_argument('--auto-stop-silence', type=int, default=10,
                        help='''\
Automatically stop recording after this many seconds of silence (streaming mode).
Set to 0 to disable auto-stop. Default: 10 seconds.''')

    parser.add_argument('--batch-mode', action='store_true',
                        help='''\
Use original batch mode: record all audio first, then transcribe, then type.
By default, streaming mode with VAD is used for real-time feedback.''')

    parser.add_argument('--no-transcribe-vad', action='store_true',
                        help='''\
Deprecated no-op: the VAD gate is off by default. Kept so existing service
files and scripts keep working.''')

    parser.add_argument('--transcribe-vad', action='store_true',
                        help='''\
Gate the transcribe call with VAD, removing silence before decoding. This stops
subtitle artefacts at the source, but it also shifts segment boundaries and can
change how unclear passages are recognised, so it is OFF by default: the phrase
filter removes the same artefacts afterwards without touching recognition.
Enable only if a new artefact slips through that the filter does not catch.''')

    parser.add_argument('--no-hallucination-filter', action='store_true',
                        help='''\
Disable the phrase filter that drops a chunk consisting entirely of a known
silence artefact (see HALLUCINATION_PHRASES). Text inside a real sentence is
never touched; each drop is logged as [filter].''')

    parser.add_argument('--no-space-rule-runs', action='store_true',
                        help='''\
Disable spacing of character runs. By default '***' is typed as '* * *' and
'---' as '- - -', so dictated punctuation can never render as a horizontal
rule, table separator or heading underline in Markdown.''')

    parser.add_argument('--ellipsis-style', choices=['space', 'single', 'keep'], default='space',
                        help='''\
How a dictated pause ('...') is typed. 'space' writes '. . .' so no editor or
chat client can auto-format it into a dash; 'single' writes one ellipsis
character; 'keep' leaves it as typed. Default: space.''')
    parser.add_argument('--mouse-buttons', type=str, default='button8,button9,button10,button11',
                        help='''\
Comma-separated pynput mouse buttons that toggle dictation, so a session can be
started without letting go of the mouse. Every listed button does the same
thing: press once to start, press again to stop. Presses within 0.4s of each
other count as one, so hitting both thumb buttons together is harmless.
button8/button9 are the thumb buttons as the kernel reports them; button10/11
are what they become once X remaps them away from browser back/forward (see
the xinput set-button-map line in install.sh), so both pairs are listed and
the trigger survives with or without that remap.
Set to an empty string to disable. Default: button8,button9,button10,button11.''')

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
        self.transcriber = SpeechTranscriber(m.finish_transcribing, args.model_name, args.device,
                                             args.compute_type, args.language,
                                             vad_filter=args.transcribe_vad)
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

    def start(self, language=_UNSET, rewrite_mode=False, keys=None):
        # Rewrite mode exists only in streaming mode; batch mode ignores it.
        if self.m.is_READY():
            if language is not _UNSET:
                self.transcriber.language = language
            # Start recording BEFORE beep so no audio is missed
            if self.args.max_time:
                self.timer = threading.Timer(self.args.max_time, self.timer_stop)
                self.timer.start()
            self.m.start_recording()
            self.replayer.capture_target_window()
            self.beep("start_recording", wait=False)
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

    def toggle_language(self):
        """Toggle transcription language between nl and en."""
        current = self.transcriber.language
        new_lang = 'en' if current == 'nl' else 'nl'
        self.transcriber.language = new_lang
        lang_name = 'English' if new_lang == 'en' else 'Nederlands'
        print('[Language] Switched to %s (%s)' % (lang_name, new_lang))
        # Audio feedback: two quick beeps (distinct from single start/stop beep)
        self.beep("start_recording", wait=True)
        self.beep("start_recording", wait=False)

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
            keylistener = MultiTapKeyListener(
                self.start, self.stop,
                key=normalize_key_names(key, parse=True),
                default_language=self.args.language,
                shift_language=self.args.shift_language,
            )
            self.m.on_enter_READY(lambda *_: print("Double tap %s to start/stop (language: %s). Hold Left-Shift + double tap to dictate in %s."
                                                   % (key, self.args.language or 'auto-detect', self.args.shift_language)))
        else:
            key = self.args.key_combo or '<win>+z'
            keylistener= KeyListener(self.toggle, normalize_key_names(key))
            self.m.on_enter_READY(lambda *_: print("Press ", key, " to start/stop recording."))
        self.mouse_listener = MouseToggleListener(
            self.toggle, (self.args.mouse_buttons or '').split(',')).start()
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
                   -> typing thread runs the stop check and sets session.done

    Sessions: stop() releases `active` as soon as the recorder is flushed, so
    a new start is never refused. The new session records at once, but its
    transcription waits for the previous session's done event: decoding,
    stop-check corrections and pastes stay strictly in session order.
    """

    class Session:
        """Per-session state, so a new start never shares queues, ledger,
        recorder or paste target with a previous session still finishing."""
        PREVIOUS_WAIT_S = 300

        def __init__(self, transcription_queue, typing_queue, ledger=None, recorder=None,
                     previous=None, language=_UNSET, target_wid=None, rewrite=False,
                     output_language=_UNSET, detect=None):
            self.transcription_queue = transcription_queue
            self.typing_queue = typing_queue
            self.ledger = ledger
            self.recorder = recorder
            self.previous = previous  # threading.Event of the previous session
            self.language = language
            self.target_wid = target_wid
            # Rewrite mode: nothing is pasted during the session; at stop the
            # full text is rewritten once and pasted into the window that is
            # active at stop (paste_wid). context lives only in this object
            # and is cleared when the session ends.
            self.rewrite = rewrite
            # Language the rewrite is written in; by default the recognition
            # language. detect: (fallback, alternative) for the recogniser.
            self.output_language = language if output_language is _UNSET else output_language
            self.detect = detect
            self.context = None
            self.paste_wid = None
            self.held = []
            self.done = threading.Event()

    def __init__(self, args):
        self.args = args
        # active = recording and accepting stop; a stopped session may still be
        # transcribing, running its stop check or pasting corrections.
        self.active = False
        self.session = None

        # Queues connecting the pipeline stages
        self.transcription_queue = queue.Queue()
        self.typing_queue = queue.Queue()

        # Workers
        self.transcription_worker = TranscriptionWorker(
            args.model_name, args.device, args.compute_type, args.language,
            context_chars=args.context_chars,
            vad_filter=args.transcribe_vad,
            filter_hallucinations=not args.no_hallucination_filter,
            space_rule_runs=not args.no_space_rule_runs,
            ellipsis_style=args.ellipsis_style,
        )
        self.replayer = KeyboardReplayer()
        self.recorder = None  # created fresh each session
        self.timer = None

        # Sound effects
        self.SOUND_EFFECTS = {
            "start_recording": loadwav("assets/granted-04.wav"),
            "finish_recording": loadwav("assets/beepbeep.wav")
        }
        self.SOUND_EFFECTS.update(rewrite_sounds(self.SOUND_EFFECTS["start_recording"]))

        # Rewrite mode (Left-Alt + double tap). Optional; without configuration
        # the hotkey falls back to ordinary dictation with a journal line.
        self.rewrite_config, self.rewrite_reason = rewrite.load_config()
        self._announce_rewrite()

    def _announce_rewrite(self):
        if not self.rewrite_config:
            print('Rewrite mode: not available (%s); Left-Alt sessions dictate normally.'
                  % self.rewrite_reason)
            return
        cfg = self.rewrite_config
        limit, problem = cfg.context_max_chars, cfg.context_max_chars_invalid
        if problem:
            # The reason only, never the configured value.
            diagnostic('rewrite_setting_invalid', setting=rewrite.CONTEXT_MAX_CHARS_ENV,
                       reason=problem, default=limit)
        print('Rewrite mode: model %s, time-out %gs, reply context up to %d characters%s.'
              % (cfg.model, cfg.timeout_s, limit,
                 ' (%s invalid: %s, default used)' % (rewrite.CONTEXT_MAX_CHARS_ENV, problem)
                 if problem else ''))

    def beep(self, k, wait=True):
        playsound(self.SOUND_EFFECTS[k], wait=wait)

    def start(self, language=_UNSET, rewrite_mode=False, keys=None):
        """Begin a streaming dictation session.

        language: transcription language for THIS session, chosen at start
        time from the hotkey modifier (Left-Shift held -> English, otherwise
        the configured default). Set every session so nothing is remembered.
        None means auto-detect. _UNSET (combo-mode toggle path) keeps the
        currently configured language untouched.

        rewrite_mode: Left-Alt was held. Without a usable configuration the
        session runs as ordinary dictation after an error tone. In rewrite
        mode `language` is the language to write in; when that differs from
        the configured recognition language (Left-Shift: English), the spoken
        language is detected on the first chunk, restricted to those two, so
        Dutch speech is recognised as Dutch and then written in English.

        keys: modifier metadata from the hotkey listener, for the journal.
        """
        if self.active:
            return None
        start_sound = 'start_recording'
        if rewrite_mode:
            unavailable = rewrite.availability(getattr(self, 'rewrite_config', None),
                                               getattr(self, 'rewrite_reason', None))
            if unavailable:
                diagnostic('rewrite_unavailable', reason=unavailable)
                rewrite_mode = False
                start_sound = 'rewrite_unavailable'

        self.active = True
        previous = self.session.done if self.session is not None else None
        if previous is not None and not previous.is_set():
            # Previous session still finishing: record now, decode and paste after it.
            diagnostic('start_while_finishing')

        # Fresh queues per session; a finishing session keeps its own.
        self.transcription_queue = queue.Queue()
        self.typing_queue = queue.Queue()

        self.ledger = CoverageLedger()
        # Publish the new session before anything slow (recorder start, xdotool):
        # a previous session that ends meanwhile must see it is no longer the
        # current one, or it would clear `active` while this one records.
        spoken, detect = language, None
        default = getattr(getattr(self, 'args', None), 'language', None)
        if rewrite_mode and language not in (_UNSET, None) and default and language != default:
            spoken, detect = None, (default, language)
        session = self.Session(self.transcription_queue, self.typing_queue, self.ledger,
                               None, previous, spoken, None, rewrite=rewrite_mode,
                               output_language=language, detect=detect)
        self.session = session
        start_meta = dict(language='detect' if detect else ('unchanged' if spoken is _UNSET else spoken or 'auto'))
        if rewrite_mode:
            start_meta['output_language'] = None if language is _UNSET else language
        if keys:
            start_meta['keys'] = keys
        diagnostic('session_start', max_seconds=MAX_SECONDS, pcm_bytes_per_minute=RATE*2*60,
                   rewrite=rewrite_mode, **start_meta)
        # Create fresh recorder and start BEFORE beep so no audio is missed
        self.recorder = StreamingRecorder(
            self.transcription_queue,
            silence_ms=self.args.silence_ms,
            auto_stop_silence_s=self.args.auto_stop_silence or None,
            on_auto_stop=lambda: self._auto_stop(session),
            min_chunk_s=self.args.min_chunk_s,
            max_chunk_s=self.args.max_chunk_s,
            eager_prob=self.args.eager_prob,
            ledger=self.ledger,
        )
        session.recorder = self.recorder
        self.recorder.start()
        session.target_wid = self.replayer.window_id()
        if rewrite_mode:
            # Read once, now: after the recorder (no audio lost) and before
            # anything of this session pastes. Never read again.
            context = context_capture.capture(max_chars=self.rewrite_config.context_max_chars)
            session.context = context.text
            diagnostic('rewrite_context', **context.journal())
            start_sound = 'rewrite_start_context' if context.text else 'rewrite_start'

        self.beep(start_sound, wait=False)
        print('Recording (streaming mode) ...')

        # Start transcription and typing threads
        self._transcription_thread = threading.Thread(
            target=self._transcription_loop, args=(session,), daemon=True
        )
        self._typing_thread = threading.Thread(
            target=self._typing_loop, args=(session,), daemon=True
        )
        self._transcription_thread.start()
        self._typing_thread.start()

        # Hard wall-clock cap as well as the sample-count cap in the recorder.
        duration = min(self.args.max_time or MAX_SECONDS, MAX_SECONDS)
        self.timer = threading.Timer(duration, self.timer_stop)
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
        self._fix_paste_window(getattr(self, 'session', None))

        # Stop recorder — this flushes remaining audio and sends sentinel
        self.recorder.stop()
        # A new start may begin now; this session finishes on its own threads.
        self.active = False
        return True

    def _auto_stop(self, session=None):
        """Auto-stop: double beep to distinguish from manual stop."""
        if session is not None and session is not getattr(self, 'session', None):
            # Silence timeout of an older session: stop only its own recorder,
            # never the current session's.
            if session.recorder is not None and getattr(session.recorder, 'recording', False):
                diagnostic('stale_auto_stop')
                session.recorder.stop()
            return None
        if not self.active:
            return None

        print('Auto-stopping (silence timeout) ...')
        self.beep("finish_recording", wait=True)
        self.beep("finish_recording", wait=False)
        if self.timer is not None:
            self.timer.cancel()
            self.timer = None
        self._fix_paste_window(getattr(self, 'session', None))

        self.recorder.stop()
        self.active = False
        return True

    def timer_stop(self):
        print('Timer stop')
        self.ledger.incident('timer_stop')
        self._auto_stop()

    def _fix_paste_window(self, session):
        """Rewrite mode pastes where the focus is at stop, fixed from then on:
        the reply field may be in another window than the selected text."""
        if session is not None and getattr(session, 'rewrite', False) and session.paste_wid is None:
            session.paste_wid = self.replayer.window_id()

    def toggle_language(self):
        """Toggle transcription language between nl and en."""
        current = self.transcription_worker.language
        new_lang = 'en' if current == 'nl' else 'nl'
        self.transcription_worker.language = new_lang
        # Old-language context would steer Whisper the wrong way after a switch
        self.transcription_worker.reset_context()
        lang_name = 'English' if new_lang == 'en' else 'Nederlands'
        print('[Language] Switched to %s (%s)' % (lang_name, new_lang))
        # Audio feedback: two quick beeps (distinct from single start/stop beep)
        self.beep("start_recording", wait=True)
        self.beep("start_recording", wait=False)

    def toggle(self):
        return self.start() or self.stop()

    def _session_from_attributes(self):
        return self.Session(self.transcription_queue, self.typing_queue,
                            getattr(self, 'ledger', None), getattr(self, 'recorder', None))

    def _await_previous(self, session):
        """Block until the previous session has pasted its corrections; then
        take over the shared worker and paste target. Audio keeps queueing."""
        if session.previous is not None and not session.previous.is_set():
            t0 = time.time()
            if not session.previous.wait(session.PREVIOUS_WAIT_S):
                diagnostic('previous_session_timeout', seconds=session.PREVIOUS_WAIT_S)
            diagnostic('waited_for_previous', seconds=time.time() - t0)
        if session.language is not _UNSET:
            self.transcription_worker.language = session.language
        self.transcription_worker.detect = getattr(session, 'detect', None)
        self.transcription_worker.reset_context()
        if session.target_wid is not None:
            self.replayer.target_wid = session.target_wid

    def _transcription_loop(self, session=None):
        """Consumer: takes audio chunks from transcription_queue, transcribes, puts text on typing_queue.

        Chunks arrive as (audio, is_boundary) tuples. Rather than transcribing
        every short eager chunk on its own, they are held in a paste buffer and
        merged into one longer audio segment before a single transcribe+type,
        so Whisper sees more context per pass (better accuracy) and text lands
        in fewer, larger pastes. The buffer flushes when it reaches
        paste_min_s of audio, when a boundary (real >=silence_ms pause) arrives,
        or at end of stream. Nothing already typed is ever revised — the merge
        happens *before* the paste. paste_min_s=0 falls back to per-chunk.
        """
        if session is None:
            session = self._session_from_attributes()  # direct call, no start()
        else:
            self._await_previous(session)
        paste_min_samples = int(self.args.paste_min_s * StreamingRecorder.SAMPLE_RATE) if self.args.paste_min_s else 0
        pending = []
        pending_samples = 0
        range_start = 0
        ledger = session.ledger
        transcription_queue, typing_queue = session.transcription_queue, session.typing_queue

        def flush():
            nonlocal pending, pending_samples
            if not pending:
                return
            merged = np.concatenate(pending) if len(pending) > 1 else pending[0]
            n_parts = len(pending)
            pending = []
            pending_samples = 0
            entry = None
            entry_start = range_start
            if ledger:
                covered = ledger.entries[-1]['end'] if ledger.entries else 0
                if entry_start > covered:
                    ledger.outcome(covered, entry_start, error='queue_gap')
                elif entry_start < covered:
                    ledger.incident('queue_overlap', start=entry_start, covered=covered)
                    merged = merged[covered-entry_start:]
                    entry_start = covered
                    if not len(merged):
                        return
            try:
                t0 = time.time()
                text = self.transcription_worker.transcribe_chunk(merged)
                raw = getattr(self.transcription_worker, 'last_raw', text)
                audio = audio_stats(merged)
                # Phrase filtering is opt-in for ordinary output, as before;
                # recovery always rejects captions (also in clean_correction).
                discard = transcript_discard_reason(
                    text, audio, drop_phrases=getattr(
                        self.transcription_worker, 'filter_hallucinations', True))
                if discard:
                    # No 150 ms of speech energy in the whole range, and either
                    # the text is already suspect (too few words for the audio)
                    # or not one sample came near speech level: whatever Whisper
                    # made of it is not Mark's voice. Never paste it, and keep it
                    # out of the next decode's prompt. Quiet but plausible short
                    # text (peak above the floor) is still pasted.
                    print('[filter] Dropped text on silent audio: %r (%s)' % (text.strip(), audio))
                    raw, text = raw or text, ''
                    reset = getattr(self.transcription_worker, 'reset_context', None)
                    if reset:
                        reset()
                if ledger:
                    entry = ledger.outcome(entry_start, entry_start + len(merged), text, raw,
                                           audio=audio, discard_reason=discard)
                elapsed = time.time() - t0
                audio_dur = len(merged) / StreamingRecorder.SAMPLE_RATE
                print('[transcribe] %.1fs audio (%d part%s) -> %.1fs processing: "%s"'
                      % (audio_dur, n_parts, '' if n_parts == 1 else 's', elapsed, text.strip()))
                if text.strip():
                    typing_queue.put((text, entry) if ledger else text)
            except Exception as e:
                print('[transcribe] Error: %s' % e)
                if ledger and entry is None:
                    ledger.outcome(entry_start, entry_start + len(merged), error=str(e))

        while True:
            item = transcription_queue.get()
            if item is None:
                # Sentinel: end of stream — paste whatever is still buffered.
                flush()
                typing_queue.put(None)
                break

            chunk, is_boundary = item[:2]
            if pending and len(item) > 2 and item[2] != range_start + pending_samples:
                flush()  # Never hide an internal queue gap inside one merged range.
            if not pending:
                range_start = item[2] if len(item) > 2 else 0
            pending.append(chunk)
            pending_samples += len(chunk)

            if paste_min_samples == 0 or is_boundary or pending_samples >= paste_min_samples:
                flush()

    def _typing_loop(self, session=None):
        """Consumer: takes text from typing_queue and types it."""
        if session is None:
            session = self._session_from_attributes()
        try:
            if getattr(session, 'rewrite', False):
                self._rewrite_session(session)
            else:
                self._typing_session(session)
        finally:
            # Context is reply material for this one session only.
            session.context = None
            # Release the next session even if this one failed.
            session.done.set()

    def _rewrite_session(self, session):
        """Rewrite mode: hold every range, run the ordinary stop check with
        recovered text in place, keep the raw text in CopyQ, rewrite once,
        paste once. Any rewrite failure pastes the raw text."""
        ledger = session.ledger
        status = []
        while True:
            item = session.typing_queue.get()
            if item is None:
                break
            entry = None
            if isinstance(item, tuple):
                item, entry = item
            session.held.append(item)
            if entry is not None:
                ledger.delivered(entry, dict(ok=True, path=HELD_FOR_REWRITE))
        try:
            if ledger:
                def deliver(line):
                    status.append(line)  # only the check line arrives here (inline)
                    return dict(ok=True, path=HELD_FOR_REWRITE)
                try:
                    ledger.finish(
                        lambda audio: self.transcription_worker.transcribe_chunk(audio, recovery=True),
                        deliver, is_artifact=is_hallucination,
                        discard_recovery=transcript_discard_reason, inline=True)
                except Exception as exc:
                    ledger.incident('stop_check_failed', error=str(exc))
                finally:
                    ledger.frames.clear()
                    if session.recorder is not None:
                        session.recorder._raw_frames.clear()
                raw = ledger.ordered_text()
            else:
                raw = re.sub(r'\s+', ' ', ' '.join(t.strip() for t in session.held)).strip()
            self._deliver_rewrite(session, raw, status)
        finally:
            session.held = []
            if getattr(self, 'session', None) in (None, session):
                self.active = False
            print('Session ended.')

    def _deliver_rewrite(self, session, raw, status):
        if not raw:
            diagnostic('rewrite_skipped', reason='no_text')
            return
        # Raw text first into CopyQ history, so it survives any later failure.
        history = self.replayer.add_to_history(raw)
        diagnostic('rewrite_raw_saved', copyq_add=history)
        output = getattr(session, 'output_language', session.language)
        spoken = getattr(self.transcription_worker, 'language', None)
        result = rewrite.rewrite(raw, None if output is _UNSET else output,
                                 session.context, getattr(self, 'rewrite_config', None),
                                 getattr(self, 'rewrite_reason', None),
                                 spoken=spoken if isinstance(spoken, str) else None)
        text = result.text
        if status:
            text = text + '\n\n' + '\n'.join(status)
        wid = session.paste_wid or session.target_wid
        if wid is not None:
            self.replayer.target_wid = wid
        try:
            delivery = self.replayer.type_text(text, echo=False)
        except Exception as exc:
            delivery = dict(ok=False, path='exception', error=str(exc))
        diagnostic('rewrite_delivery', rewritten=result.ok,
                   paste_window='stop' if session.paste_wid else 'start',
                   **{k: v for k, v in (delivery or {}).items() if k != 'text'})
        self.beep('rewrite_done' if result.ok else 'rewrite_failed', wait=False)

    def _typing_session(self, session):
        first_chunk = True
        ledger = session.ledger
        while True:
            text = session.typing_queue.get()
            if text is None:
                # Sentinel: end of stream (beep already played in stop())
                if ledger:
                    try:
                        # Corrections take the ordinary chunk path: clipboard plus
                        # paste into the start window, after the text pasted so far.
                        def deliver(correction):
                            nonlocal first_chunk
                            prefix = '' if first_chunk else ' '
                            first_chunk = False
                            return self.replayer.type_text(prefix + correction)
                        ledger.finish(
                            lambda audio: self.transcription_worker.transcribe_chunk(audio, recovery=True),
                            deliver, is_artifact=is_hallucination,
                            discard_recovery=transcript_discard_reason)
                    except Exception as exc:
                        ledger.incident('stop_check_failed', error=str(exc))
                    finally:
                        ledger.frames.clear()
                        if session.recorder is not None:
                            session.recorder._raw_frames.clear()
                # Only a session that ended without stop() (e.g. capture error)
                # still holds `active`; never clear a newer session's flag.
                if getattr(self, 'session', None) in (None, session):
                    self.active = False
                print('Session ended.')
                break

            entry = None
            if isinstance(text, tuple):
                text, entry = text
            # Strip leading space from first chunk only
            if first_chunk:
                if text.startswith(' '):
                    text = text[1:]
                first_chunk = False
            else:
                # Add space between chunks if the text doesn't start with punctuation
                if text and text[0] not in ' .,;:!?\'")-]}>':
                    text = ' ' + text

            try:
                result = self.replayer.type_text(text)
            except Exception as exc:
                result = dict(ok=False, path='exception', error=str(exc))
            if entry is not None:
                ledger.delivered(entry, result)

    def run(self):
        def normalize_key_names(keyseqs, parse=False):
            k = keyseqs.replace('<win>', '<cmd>').replace('<win_r>', '<cmd_r>').replace('<win_l>', '<cmd_l>').replace('<super>', '<cmd>').replace('<super_r>', '<cmd_r>').replace('<super_l>', '<cmd_l>')
            if parse:
                k = keyboard.HotKey.parse(k)[0]
            print('Using key:', k)
            return k

        auto_stop_info = ', auto-stop after %ds silence' % self.args.auto_stop_silence if self.args.auto_stop_silence else ', no auto-stop'
        lang_info = self.args.language or 'auto-detect'
        eager_info = (', eager flush %g-%gs at quietest point (p<=%.2f)' % (self.args.min_chunk_s, self.args.max_chunk_s, self.args.eager_prob)
                      if self.args.min_chunk_s else ', no eager flush')
        context_info = (', rolling context %d chars' % self.args.context_chars
                        if self.args.context_chars else ', no rolling context')
        paste_info = (', merge-paste >=%gs' % self.args.paste_min_s
                      if self.args.paste_min_s else ', paste per chunk')
        print('Streaming dictation mode (chunk silence: %dms%s%s%s%s, language: %s)' % (self.args.silence_ms, eager_info, paste_info, context_info, auto_stop_info, lang_info))

        if (platform.system() != 'Windows' and not self.args.key_combo) or self.args.double_key:
            key = self.args.double_key or (platform.system() == 'Linux' and '<ctrl_r>') or '<cmd_r>'
            keylistener = MultiTapKeyListener(
                self.start, self.stop,
                key=normalize_key_names(key, parse=True),
                default_language=self.args.language,
                shift_language=self.args.shift_language,
            )
            print("Double tap %s to start/stop (language: %s). Hold Left-Shift + double tap to dictate in %s."
                  " Add Left-Alt for rewrite mode." % (key, self.args.language or 'auto-detect', self.args.shift_language))
        else:
            key = self.args.key_combo or '<win>+z'
            keylistener = KeyListener(self.toggle, normalize_key_names(key))
            print("Press ", key, " to start/stop recording.")

        self.mouse_listener = MouseToggleListener(
            self.toggle, (self.args.mouse_buttons or '').split(',')).start()

        keylistener.run()


def warn_if_wayland():
    """Say plainly when global triggers cannot work in this session.

    pynput listens through X. Under a Wayland session it only sees input that
    goes to Xwayland windows, so Ctrl_R and the thumb buttons stay silent in
    every native Wayland app while the service still looks active.
    """
    if os.environ.get('XDG_SESSION_TYPE') == 'wayland':
        print('WARNING: Wayland session detected. The hotkey and mouse buttons '
              'only work in X11 windows. Log in with "Ubuntu on Xorg" '
              '(or set WaylandEnable=false in /etc/gdm3/custom.conf).')


if __name__ == "__main__":
    warn_if_wayland()
    args = parse_args()
    if args.batch_mode:
        print('Using batch mode (original behaviour)')
        BatchApp(args).run()
    else:
        App(args).run()
