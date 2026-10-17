"""Read-only CPU replay. Input audio stays outside the repository in /tmp.
Usage: venv/bin/python tests/replay_recovery.py /tmp/dct_wp3_clip.wav
"""
import contextlib
import io
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio
from test_recovery import load_logic, worker

old_path = Path('/tmp/dct_wp3_old.py')
old_path.write_bytes(subprocess.check_output(['git', 'show', 'd5654a7:dictation.py']))
old = load_logic(old_path)
new = load_logic()
model = WhisperModel('large-v3', device='cpu', compute_type='int8', cpu_threads=6,
                     local_files_only=True)
audio = decode_audio(sys.argv[1], sampling_rate=16000)
# An actual model, identical options/audio; only the recovery code differs.
for label, logic in [('old', old), ('new', new)]:
    obj = worker([])
    obj.__class__ = logic['TranscriptionWorker']
    obj.model = model
    obj._context = 'Dit is een onafgemaakte zin ' + '. ' * 18
    for step in range(2):
        t0 = time.monotonic()
        with contextlib.redirect_stdout(io.StringIO()) as messages:
            text = obj.transcribe_chunk(audio)
        print(json.dumps(dict(version=label, step=step, empty=not bool(text.strip()),
                              recovery_attempts=messages.getvalue().count('[recovery] Suspect'),
                              words=len(re.findall(r'\w+', text)),
                              pause_in_context=bool(re.search(r'(?:\.\s*){3,}', obj._context)),
                              elapsed=round(time.monotonic()-t0, 2)),
                         ensure_ascii=False), flush=True)
