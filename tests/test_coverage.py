"""No microphone, desktop writes or model; pipeline uses actual app loop methods."""
import ast
import queue
import subprocess
import sys
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import coverage_ledger as coverage
from test_recovery import load_logic, worker, logic as shared_logic
import test_recovery


def replayer_class():
    tree = ast.parse((Path(__file__).resolve().parents[1] / 'dictation.py').read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'KeyboardReplayer')
    ns = dict(subprocess=subprocess, time=types.SimpleNamespace(sleep=lambda _: None),
              diagnostic=coverage.diagnostic)
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'replayer', 'exec'), ns)
    return ns['KeyboardReplayer']


class CoverageTests(unittest.TestCase):
    def ledger(self, seconds=30):
        obj = coverage.CoverageLedger()
        obj.capture(np.arange(seconds * coverage.RATE, dtype=np.int16).tobytes())
        return obj

    def test_partition_and_gap_recovery(self):
        obj = self.ledger(2)
        pos = 0
        for length in (1, 1023, 1000, 16000):
            obj.outcome(pos, pos + length, 'tekst')
            obj.delivered(obj.entries[-1], dict(ok=True, path='copyq'))
            pos += length
        with self.assertRaises(ValueError):
            obj.outcome(pos-1, pos+1)
        obj.finish(lambda a: 'hersteld', lambda t: dict(ok=True, path='copyq'))
        samples = np.zeros(obj.samples, dtype=np.int8)
        for entry in obj.entries:
            samples[entry['start']:entry['end']] += 1
        self.assertTrue(np.all(samples == 1))
        self.assertEqual(obj.entries[-1]['error'], 'queue_gap')
        self.assertEqual(obj.frames, [])

    def test_error_artifact_and_copyq_failure_all_checked(self):
        obj = self.ledger()
        obj.outcome(0, 10*coverage.RATE, error='CUDA OOM')
        obj.outcome(10*coverage.RATE, 20*coverage.RATE, raw='Dank u wel.')
        e = obj.outcome(20*coverage.RATE, 30*coverage.RATE, text='afgeleverd via fallback')
        obj.delivered(e, dict(ok=True, path='xclip', copyq_add=1))
        self.assertEqual(len(obj.suspects()), 3)
        delivered = []
        output = obj.finish(lambda audio: 'teruggevonden woorden',
                            lambda text: delivered.append(text) or dict(ok=True, path='copyq'))
        self.assertEqual(len(output), 3)
        self.assertTrue(all(t.startswith('[mogelijk gemist ') for t in output))
        self.assertEqual(len(delivered), 1)
        self.assertEqual(obj.frames, [])

    def test_pipeline_originals_then_separate_correction(self):
        logic = load_logic()
        ledger = self.ledger(20)
        app = logic['App'].__new__(logic['App'])
        app.ledger = ledger
        app.args = types.SimpleNamespace(paste_min_s=0)
        app.transcription_worker = worker([RuntimeError('OOM'), 'Dit zijn de normale reeds afgeleverde woorden.',
                                          'Dit zijn teruggevonden woorden uit het eerste stuk.'])
        app.transcription_queue, app.typing_queue = queue.Queue(), queue.Queue()
        for start in (0, 10*coverage.RATE):
            app.transcription_queue.put((np.zeros(10*coverage.RATE, dtype=np.float32), True, start))
        app.transcription_queue.put(None)
        originals, corrections = [], []
        app.replayer = types.SimpleNamespace(
            type_text=lambda t: originals.append(t) or dict(ok=True, path='copyq'),
            _set_clipboard=lambda t: corrections.append(t) or dict(ok=True, path='copyq'))
        app.recorder = types.SimpleNamespace(_raw_frames=[b'audio'])
        app.active = True
        app._transcription_loop()
        app._typing_loop()
        self.assertEqual(originals, ['Dit zijn de normale reeds afgeleverde woorden.'])
        self.assertEqual(len(corrections), 1)
        self.assertIn('[mogelijk gemist 0.00–10.00s:', corrections[0])
        self.assertFalse(app.active)
        self.assertEqual(ledger.frames, [])
        self.assertEqual(app.recorder._raw_frames, [])

    def test_release_even_when_recovery_or_delivery_raises(self):
        for fail_decode in (True, False):
            obj = self.ledger(1)
            obj.outcome(0, coverage.RATE, error='OOM')
            def fail(*args):
                raise RuntimeError('failed')
            try:
                obj.finish(fail if fail_decode else lambda a: 'woorden', fail)
            except RuntimeError:
                pass
            self.assertEqual(obj.frames, [])

    def test_duration_limit_never_overwrites_head(self):
        with patch.object(coverage, 'MAX_SECONDS', 1):
            obj = coverage.CoverageLedger()
            head = b'\x01\x00' * coverage.RATE
            obj.capture(head)
            self.assertEqual(obj.capture(b'\x02\x00'*100), b'')
            self.assertEqual(obj.samples, coverage.RATE)
            self.assertEqual(b''.join(obj.frames), head)

    def test_recorder_absolute_ranges_survive_trim_and_stop(self):
        raw = np.arange(16384, dtype=np.int16)
        ledger = coverage.CoverageLedger()
        ledger.capture(raw.tobytes())
        recorder = test_recovery.AudioCoverageTests().recorder([p.tobytes() for p in raw.reshape(-1, 1024)])
        recorder.ledger, recorder._buffer_offset = ledger, 0
        recorder.vad_options = None
        shared_logic['get_speech_timestamps'] = lambda *a, **kw: [{'start': 4000, 'end': 8000}]
        recorder._check_and_send_chunks()
        self.assertGreater(recorder._buffer_offset, 0)
        recorder.stop()
        cursor = 0
        for item in iter(recorder.transcription_queue.get, None):
            audio, boundary, start = item
            self.assertEqual(start, cursor)
            ledger.outcome(start, start+len(audio), 'woorden')
            cursor += len(audio)
        self.assertEqual(cursor, ledger.samples)
        self.assertEqual([(e['start'], e['end']) for e in ledger.entries], [(0, 8000), (8000, 16384)])

    def test_internal_queue_gap_not_hidden_by_merging_or_decoder_error(self):
        ledger = self.ledger(3)
        app = load_logic()['App'].__new__(load_logic()['App'])
        app.ledger = ledger
        app.args = types.SimpleNamespace(paste_min_s=10)
        app.transcription_worker = worker(['Eerste woorden.', RuntimeError('OOM')])
        app.transcription_queue, app.typing_queue = queue.Queue(), queue.Queue()
        for start in (0, 2*coverage.RATE):
            app.transcription_queue.put((np.zeros(coverage.RATE), False, start))
        app.transcription_queue.put(None)
        app._transcription_loop()
        self.assertEqual([(e['start'], e['end']) for e in ledger.entries],
                         [(0, coverage.RATE), (coverage.RATE, 2*coverage.RATE), (2*coverage.RATE, 3*coverage.RATE)])
        self.assertEqual(ledger.entries[1]['error'], 'queue_gap')
        self.assertEqual(ledger.entries[2]['outcome'], 'decoder_error')

    def test_focus_or_paste_failure_returns_failed_delivery(self):
        cls = replayer_class()
        obj = cls.__new__(cls)
        obj._get_active_window_class = lambda: 'editor'
        obj._set_clipboard = lambda t: dict(ok=True, path='copyq')
        obj._restore_target_window = lambda: False
        obj._send_paste = lambda terminal: self.fail('must not paste without focus')
        self.assertFalse(obj.type_text('woorden')['ok'])
        obj._restore_target_window = lambda: True
        def fail(terminal):
            raise RuntimeError('keyboard error')
        obj._send_paste = fail
        result = obj.type_text('woorden')
        self.assertFalse(result['ok'])
        self.assertEqual(result['paste'], 'failed')

    def test_copyq_and_xclip_returncodes(self):
        cls = replayer_class()
        obj = cls.__new__(cls)
        for add, copy, clip, expected in ((1, 0, 0, True), (0, 2, 7, False), (0, 0, 0, True)):
            calls = []
            def run(cmd, **kwargs):
                calls.append(cmd[:2])
                code = clip if cmd[0] == 'xclip' else (add if cmd[1] == 'add' else copy)
                return types.SimpleNamespace(returncode=code)
            with patch.object(subprocess, 'run', run):
                result = obj._set_clipboard('test')
            self.assertEqual(result['ok'], expected)
            self.assertEqual(result['copyq_add'], add)
            if add or copy:
                self.assertEqual(result['path'], 'xclip')
                self.assertEqual(result['xclip'], clip)
            else:
                self.assertEqual(result['path'], 'copyq')

    def test_recovery_retains_literal_filtered_words_without_prompt(self):
        obj = worker(['Dank u wel.'])
        obj._context = 'Oude context'
        self.assertEqual(obj.transcribe_chunk(np.zeros(160000), recovery=True), 'Dank u wel.')
        self.assertEqual(len(obj.calls), 1)
        self.assertIsNone(obj.calls[0]['initial_prompt'])


if __name__ == '__main__':
    unittest.main()
