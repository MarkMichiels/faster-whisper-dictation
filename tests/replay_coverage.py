"""CPU/int8 replay of real App loops; all desktop writes are receipts-only fakes.
Usage: venv/bin/python tests/replay_coverage.py /tmp/wp4_session.wav
Audio/transcripts never enter evidence; output is metrics only.
"""
import contextlib
import io
import json
import queue
import sys
import time
import types
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from coverage_ledger import CoverageLedger, RATE
from test_recovery import load_logic, worker

model = WhisperModel('large-v3', device='cpu', compute_type='int8', cpu_threads=6,
                     local_files_only=True)
audio = decode_audio(sys.argv[1], sampling_rate=RATE)
ledger = CoverageLedger()
ledger.capture((audio * 32767).astype(np.int16).tobytes())
logic = load_logic()
app = logic['App'].__new__(logic['App'])
app.args = types.SimpleNamespace(paste_min_s=0)
app.ledger = ledger
app.transcription_worker = worker([])
app.transcription_worker.model = model
original_decode = app.transcription_worker.transcribe_chunk
calls = [0]
def decode(samples, **kw):
    calls[0] += 1
    if calls[0] == 1:
        raise RuntimeError('simulated decoder loss')
    return original_decode(samples, **kw)
app.transcription_worker.transcribe_chunk = decode
app.transcription_queue, app.typing_queue = queue.Queue(), queue.Queue()
for start in range(0, len(audio), 10*RATE):
    app.transcription_queue.put((audio[start:start+10*RATE], True, start))
app.transcription_queue.put(None)
originals, corrections = [], []
def paste(text):
    (corrections if text.lstrip().startswith('[Correctie: ') else originals).append(text)
    return dict(ok=True, path='copyq')
app.replayer = types.SimpleNamespace(type_text=paste)
app.recorder = types.SimpleNamespace(_raw_frames=[])
app.active = True
with contextlib.redirect_stdout(io.StringIO()):
    start = time.perf_counter()
    app._transcription_loop()
    normal_seconds = time.perf_counter() - start
    start = time.perf_counter()
    app._typing_loop()
    extra_seconds = time.perf_counter() - start
assert ledger.frames == [] and not app.active
first_recovered = ledger.entries[0]['recovery'].get('text', '')
assert first_recovered and not logic['is_hallucination'](first_recovered), 'simulated loss not recovered as words'
assert corrections and corrections[0].lstrip().startswith('[Correctie: ')
print(json.dumps(dict(model='large-v3', device='cpu', compute_type='int8', threads=6,
    audio_seconds=len(audio)/RATE, ranges=len(ledger.entries), injected_decoder_errors=1,
    ordinary_seconds=normal_seconds, stop_check_extra_seconds=extra_seconds,
    originals=len(originals), correction_items=len(corrections),
    recovered_first_range_words=len(ledger.entries[0]['recovery']['text'].split()),
    checked_ranges=sum(e['recovery'] is not None for e in ledger.entries),
    ram_audio_released=not ledger.frames,
    gpu_seconds=None, desktop_writes=False), indent=2))
