"""Content-free capture diagnostics. Never change capture or VAD decisions."""
import json
import os
import subprocess

from coverage_ledger import RATE, audio_stats, diagnostic


def emit(kind, **fields):
    """Diagnostic failures must never interrupt recording."""
    try:
        diagnostic(kind, **fields)
    except Exception:
        pass


def pipewire_capture(pid, objects):
    """Only report links belonging to this process's capture stream, not defaults.

    A missing/directly unlinked source is unknown, never guessed from the default.
    Multiple streams in this process are explicitly ambiguous.
    """
    nodes = {o['id']: o.get('info', {}).get('props', {}) for o in objects
             if o.get('type') == 'PipeWire:Interface:Node'}
    devices = {str(o['id']): o.get('info', {}).get('props', {}) for o in objects
               if o.get('type') == 'PipeWire:Interface:Device'}
    streams = [i for i, p in nodes.items()
               if str(p.get('application.process.id')) == str(pid)
               and p.get('media.class') == 'Stream/Input/Audio']
    links = [o.get('info', {}) for o in objects
             if o.get('type') == 'PipeWire:Interface:Link']
    def node(i):
        p = nodes.get(i, {})
        device = devices.get(str(p.get('device.id')), {})
        return dict(id=i, name=p.get('node.name'), description=p.get('node.description'),
                    device_id=p.get('device.id'), device_name=device.get('device.name'),
                    device_description=device.get('device.description'),
                    media_class=p.get('media.class'), serial=p.get('object.serial'))
    routes = [dict(stream=node(i), sources=[node(l['output-node-id']) for l in links
                                           if l.get('input-node-id') == i
                                           and 'output-node-id' in l]) for i in streams]
    return dict(status='matched' if len(streams) == 1 and routes[0]['sources'] else 'unknown_or_ambiguous',
                routes=routes, physical_route_proven=False)


def log_pipewire_capture():
    try:
        result = subprocess.run(['pw-dump'], capture_output=True, text=True,
                                timeout=1, check=True)
        emit('capture_pipewire', **pipewire_capture(os.getpid(), json.loads(result.stdout)))
    except Exception as exc:
        emit('capture_pipewire', status='unavailable', error_type=type(exc).__name__)


def log_capture_device(pa, rate):
    try:
        device = pa.get_default_input_device_info()
        host = pa.get_host_api_info_by_index(int(device['hostApi']))
        emit('capture_device', index=device.get('index'), name=device.get('name'),
             host_api=host.get('name'), default_samplerate=device.get('defaultSampleRate'),
             requested_rate=rate, requested_channels=1, selection='portaudio_default',
             physical_route_proven=False)
    except Exception as exc:
        emit('capture_device', status='unavailable', error_type=type(exc).__name__)


class InputObservation:
    """Sample-clock cadence survives trims. All state is per recording.

    -45 dBFS is an energy heuristic, NOT a VAD threshold or proof of speech.
    Warn once after >=3s of observed low energy; normal short pauses do not warn.
    voiced_s in audio_stats is likewise energy-based, not Silero speech duration.
    """
    def __init__(self):
        self.last_log_end = None
        self.last_end = None
        self.weak_start = None
        self.warned = False

    def observe(self, audio, offset, processed, timestamps, trailing, threshold, timeout, stop):
        end = offset + len(audio)
        window = audio[-RATE:]
        start = end - len(window)
        stats = audio_stats(window)
        if self.last_end is None or end > self.last_end:
            if stats['rms_dbfs'] < -45 and stats['voiced_s'] == 0:
                if self.weak_start is None or (self.last_end is not None and start > self.last_end):
                    self.weak_start = start
                if end - self.weak_start >= 3 * RATE and not self.warned:
                    emit('input_level_warning', reason='sustained_low_input',
                         message='Microfoonsignaal langdurig erg zwak; controleer microfoon en bron.',
                         start=self.weak_start, end=end, weak_seconds=(end-self.weak_start)/RATE,
                         rms_threshold_dbfs=-45, audio=stats)
                    self.warned = True
            else:
                self.weak_start = None
            self.last_end = end
        if stop or self.last_log_end is None or end - self.last_log_end >= RATE:
            emit('vad_observation', buffer_start=offset, buffer_end=end,
                 processed_end=offset + processed,
                 last_speech_end=offset + timestamps[-1]['end'] if timestamps else None,
                 vad_speech_seconds=sum(t['end']-t['start'] for t in timestamps)/RATE,
                 trailing_silence_seconds=trailing, threshold=threshold,
                 auto_stop_silence_s=timeout, stop=bool(stop),
                 window_start=start, window_end=end, audio=stats)
            self.last_log_end = end
