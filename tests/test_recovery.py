"""Offline tests: AST-load the pure logic, never import audio/desktop backends."""
import ast
from pathlib import Path
import re
import types
import threading
import queue
import time
import unittest
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import coverage_ledger


def load_logic(path=None):
    path = path or Path(__file__).resolve().parents[1] / 'dictation.py'
    tree = ast.parse(Path(path).read_text())
    # Top-level constants/functions through sanitize_transcript, plus worker.
    nodes = []
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.FunctionDef)):
            if getattr(node, 'lineno', 0) < next(n.lineno for n in tree.body if isinstance(n, ast.ClassDef)):
                nodes.append(node)
        elif isinstance(node, ast.ClassDef) and node.name in ('TranscriptionWorker', 'StreamingRecorder', 'App'):
            nodes.append(node)
    ns = {'re': re, 'np': np, 'threading': threading, 'queue': queue, 'time': time,
          'platform': types.SimpleNamespace(system=lambda: 'Linux'),
          'audio_stats': coverage_ledger.audio_stats, 'silent_audio': coverage_ledger.silent_audio}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), ns)
    return ns


logic = load_logic()


def worker(outputs):
    obj = logic['TranscriptionWorker'].__new__(logic['TranscriptionWorker'])
    obj.language = 'nl'
    obj.context_chars = 500
    obj._context = ''
    obj.vad_filter = False
    obj.filter_hallucinations = True
    obj.space_rule_runs = True
    obj.ellipsis_style = 'space'
    obj.calls = []
    outputs = iter(outputs)
    def transcribe(audio, **kw):
        obj.calls.append(kw)
        text = next(outputs)
        if isinstance(text, Exception):
            raise text
        return iter([types.SimpleNamespace(text=text)]), None
    obj.model = types.SimpleNamespace(transcribe=transcribe)
    return obj


class RecoveryTests(unittest.TestCase):
    def test_context_hygiene(self):
        for bad in ['. . . .', '...', '***', 'Dank u wel.', 'TV Gelderland 2021',
                    'ja ja ja ja ja']:
            self.assertEqual(logic['prompt_text'](bad), '')
        self.assertEqual(logic['prompt_text']('Een zin . . . .'), 'Een zin')
        self.assertEqual(logic['prompt_text']('Dit is goed.'), 'Dit is goed.')

    def test_prompt_scrubbed_before_decode(self):
        obj = worker(['Een normale zin.'])
        obj._context = 'Vorige zin . . . .'
        obj.transcribe_chunk(np.zeros(16000))
        self.assertEqual(obj.calls[0]['initial_prompt'], 'Vorige zin')

    def test_one_retry_recovers_and_breaks_loop(self):
        for bad in ['. . . .', 'Een begin . . . .', 'Dank u wel.', '', 'ja ja ja ja ja']:
            obj = worker([bad, 'Dit is de volledig herstelde tekst.'])
            obj._context = 'Oude context.'
            self.assertEqual(obj.transcribe_chunk(np.zeros(160000)),
                             'Dit is de volledig herstelde tekst.')
            self.assertEqual(len(obj.calls), 2)
            self.assertIsNone(obj.calls[1]['initial_prompt'])
            self.assertNotIn('Oude context', obj._context)

    def test_short_for_audio_retried_once(self):
        obj = worker(['Weinig.', 'Dit is nog steeds weinig.'])
        obj.transcribe_chunk(np.zeros(320000))
        self.assertEqual(len(obj.calls), 2)
        self.assertEqual(obj._context, '')

    def test_silence_stays_filtered(self):
        for bad in ['***', '. . . .', 'TV Gelderland 2021', 'Dank u wel.']:
            obj = worker([bad, bad])
            self.assertEqual(obj.transcribe_chunk(np.zeros(16000)), '')
            self.assertEqual(obj._context, '')
            self.assertEqual(len(obj.calls), 2)

    def test_retry_cannot_erase_first_pass_words(self):
        for retry in ['', 'TV Gelderland 2021', RuntimeError('model failure')]:
            obj = worker(['Een zin . . . .', retry])
            self.assertTrue(obj.transcribe_chunk(np.zeros(160000)).startswith('Een zin'))
            self.assertEqual(obj._context, '')

    def test_normal_no_retry_and_filter_regressions(self):
        obj = worker(['Een normale zin met genoeg woorden.'])
        self.assertEqual(obj.transcribe_chunk(np.zeros(16000)), 'Een normale zin met genoeg woorden.')
        self.assertEqual(len(obj.calls), 1)
        clean = logic['sanitize_transcript']
        self.assertEqual(clean('Dat is muziek en dank u wel.'), 'Dat is muziek en dank u wel.')
        self.assertEqual(clean('Een *** pauze...'), 'Een * * * pauze. . .')


class AudioCoverageTests(unittest.TestCase):
    def recorder(self, frames):
        obj = logic['StreamingRecorder'].__new__(logic['StreamingRecorder'])
        obj._buffer_lock = threading.Lock()
        obj._stop_lock = threading.Lock()
        obj._raw_frames = frames
        obj._vad_processed_samples = 0
        obj.transcription_queue = queue.Queue()
        obj.silence_ms = 100
        obj.auto_stop_silence_s = 0
        obj.min_chunk_samples = 0
        obj.recording = True
        obj._rec_thread = obj._vad_thread = types.SimpleNamespace(join=lambda *a, **kw: None)
        return obj

    def test_vad_gaps_and_stop_tail_cover_every_sample(self):
        raw = np.arange(16384, dtype=np.int16)
        obj = self.recorder([part.tobytes() for part in raw.reshape(-1, 1024)])
        logic['get_speech_timestamps'] = lambda *a, **kw: [{'start': 4000, 'end': 8000}]
        obj.vad_options = None
        obj._check_and_send_chunks()
        obj.stop()
        chunks = []
        while (item := obj.transcription_queue.get()) is not None:
            chunks.append(item[0])
        np.testing.assert_array_equal(np.concatenate(chunks), raw.astype(np.float32)/32768)

    def test_sub_100ms_tail_and_double_stop(self):
        obj = self.recorder([np.ones(1000, dtype=np.int16).tobytes()])
        obj.stop()
        obj.stop()
        self.assertEqual(len(obj.transcription_queue.get()[0]), 1000)
        self.assertIsNone(obj.transcription_queue.get())
        self.assertTrue(obj.transcription_queue.empty())

    def test_slow_consumer_drains_to_sentinel_without_queue_loss(self):
        obj = logic['App'].__new__(logic['App'])
        obj.args = types.SimpleNamespace(paste_min_s=0)
        obj.transcription_queue = queue.Queue()
        obj.typing_queue = queue.Queue()
        seen = []
        def transcribe(audio):
            seen.append(int(audio[0]))
            return 'chunk %d' % audio[0]
        obj.transcription_worker = types.SimpleNamespace(transcribe_chunk=transcribe)
        for i in range(40):
            obj.transcription_queue.put((np.array([i]), True))
        obj.transcription_queue.put(None)
        obj._transcription_loop()
        self.assertEqual(seen, list(range(40)))
        self.assertEqual(obj.typing_queue.qsize(), 41)
        self.assertTrue(obj.transcription_queue.empty())


if __name__ == '__main__':
    unittest.main()
