"""No microphone, desktop writes or model; pipeline uses actual app loop methods."""
import ast
import contextlib
import io
import json
import queue
import re
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
    ns = dict(re=re, subprocess=subprocess, time=types.SimpleNamespace(sleep=lambda _: None),
              diagnostic=coverage.diagnostic)
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'replayer', 'exec'), ns)
    return ns['KeyboardReplayer']


class CoverageTests(unittest.TestCase):
    def setUp(self):
        # Delivery-count tests predate the end-of-session status line.
        patcher = patch.object(coverage, 'STOP_STATUS', False)
        patcher.start()
        self.addCleanup(patcher.stop)

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
        e = obj.outcome(20*coverage.RATE, 30*coverage.RATE, text='niet aangekomen tekst')
        obj.delivered(e, dict(ok=False, path='none', copyq_add=1, xclip=1))
        self.assertEqual(len(obj.suspects()), 3)
        delivered = []
        output = obj.finish(lambda audio: 'teruggevonden woorden',
                            lambda text: delivered.append(text) or dict(ok=True, path='copyq'))
        # Three adjacent suspect ranges form one gap: one pasted correction.
        self.assertEqual(output, delivered)
        self.assertEqual(output, ['[Correctie: in deze opname is waarschijnlijk dit gemist: '
                                  '"teruggevonden woorden teruggevonden woorden niet aangekomen tekst"]'])
        self.assertEqual(obj.frames, [])

    def test_xclip_fallback_that_pasted_is_not_pasted_again(self):
        obj = self.ledger(10)
        e = obj.outcome(0, 10*coverage.RATE, text='deze zin werd al geplakt via xclip zonder CopyQ historie')
        obj.delivered(e, dict(ok=True, path='xclip', copyq_add=1, paste='shortcut_sent_not_acknowledged'))
        self.assertEqual(obj.suspects(), [])
        self.assertEqual(obj.finish(lambda a: self.fail('no decode'), lambda t: self.fail('no paste')), [])

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
            # Audible audio: an all-zero range is now (rightly) treated as silence.
            app.transcription_queue.put((np.full(10*coverage.RATE, 0.1, dtype=np.float32), True, start))
        app.transcription_queue.put(None)
        pasted = []
        app.replayer = types.SimpleNamespace(
            type_text=lambda t: pasted.append(t) or dict(ok=True, path='copyq',
                                                         paste='shortcut_sent_not_acknowledged'),
            _set_clipboard=lambda t: self.fail('corrections must use the ordinary paste path'))
        app.recorder = types.SimpleNamespace(_raw_frames=[b'audio'])
        app.active = True
        app._transcription_loop()
        app._typing_loop()
        # Original first, then the correction pasted through the same path, space-separated.
        self.assertEqual(pasted, [
            'Dit zijn de normale reeds afgeleverde woorden.',
            ' [Correctie: vóór "Dit zijn de normale…" is waarschijnlijk dit gemist: '
            '"Dit zijn teruggevonden woorden uit het eerste stuk."]'])
        self.assertEqual(ledger.entries[0]['recovery']['delivery']['paste'],
                         'shortcut_sent_not_acknowledged')
        self.assertFalse(app.active)
        self.assertEqual(ledger.frames, [])
        self.assertEqual(app.recorder._raw_frames, [])

    def test_correction_wording_without_seconds(self):
        f = coverage.correction_text
        self.assertEqual(f('Ik denk dat het eigenlijk een', 'hetgeen dat Jonas gisteren zei.', 'gemiste tekst'),
                         '[Correctie: tussen "…dat het eigenlijk een" en "hetgeen dat Jonas gisteren…" '
                         'is waarschijnlijk dit gemist: "gemiste tekst"]')
        self.assertEqual(f('kort stuk', '', 'staart'),
                         '[Correctie: na "kort stuk" is waarschijnlijk dit gemist: "staart"]')
        self.assertEqual(f('', 'Begin van de rest van de tekst', 'kop'),
                         '[Correctie: vóór "Begin van de rest…" is waarschijnlijk dit gemist: "kop"]')
        for text in (f('a b', 'c d', 'x'), f('', '', 'x')):
            self.assertNotRegex(text, r'\d|seconde|s:')

    def test_anchors_from_delivered_text_one_correction_per_gap(self):
        obj = self.ledger(50)
        R = coverage.RATE
        def ok(start, end, text):
            obj.delivered(obj.outcome(start, end, text), dict(ok=True, path='copyq'))
        ok(0, 10*R, 'Het eerste stuk eindigt dat het eigenlijk een')
        obj.outcome(10*R, 20*R, error='OOM')
        ok(20*R, 30*R, 'hetgeen dat Jonas zei over de pomp')
        ok(30*R, 40*R, 'en dan het slot van de zin')
        obj.outcome(40*R, 50*R, error='OOM')
        decoded = iter(['eerste gat', 'tweede gat'])
        pasted = []
        out = obj.finish(lambda a: next(decoded), lambda t: pasted.append(t) or dict(ok=True, path='copyq'))
        self.assertEqual(out, pasted)
        self.assertEqual(out, [
            '[Correctie: tussen "…dat het eigenlijk een" en "hetgeen dat Jonas zei…" '
            'is waarschijnlijk dit gemist: "eerste gat"]',
            '[Correctie: na "…slot van de zin" is waarschijnlijk dit gemist: "tweede gat"]'])

    def test_failed_delivery_pasted_as_correction_without_decode(self):
        obj = self.ledger(30)
        R = coverage.RATE
        obj.delivered(obj.outcome(0, 10*R, 'vooraf goed aangekomen'), dict(ok=True, path='copyq'))
        lost = obj.outcome(10*R, 20*R, 'herkend maar niet geplakt')
        obj.delivered(lost, dict(ok=False, path='copyq', paste='failed'))
        obj.delivered(obj.outcome(20*R, 30*R, 'daarna weer goed'), dict(ok=True, path='copyq'))
        pasted = []
        obj.finish(lambda a: self.fail('delivery failure needs no decode'),
                   lambda t: pasted.append(t) or dict(ok=True, path='copyq'))
        self.assertEqual(pasted, ['[Correctie: tussen "vooraf goed aangekomen" en "daarna weer goed" '
                                  'is waarschijnlijk dit gemist: "herkend maar niet geplakt"]'])

    def test_failed_correction_paste_stays_in_journal(self):
        obj = self.ledger(10)
        obj.outcome(0, 10*coverage.RATE, error='OOM')
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            obj.finish(lambda a: 'woorden', lambda t: dict(ok=False, path='copyq', paste='failed'))
        line = next(l for l in out.getvalue().splitlines() if '"correction_delivery"' in l)
        record = json.loads(line.removeprefix('DCT-COVERAGE '))
        self.assertFalse(record['ok'])
        self.assertIn('"woorden"', record['correction'])
        self.assertFalse(obj.entries[0]['recovery']['delivery']['ok'])

    def test_no_desktop_notifications(self):
        source = (Path(__file__).resolve().parents[1] / 'dictation.py').read_text()
        self.assertNotIn('notify-send', source)
        self.assertNotIn('notify-send', Path(coverage.__file__).read_text())
        calls = []
        with patch.object(subprocess, 'run', lambda *a, **kw: calls.append(a)), \
                contextlib.redirect_stdout(io.StringIO()):
            obj = self.ledger(10)
            obj.outcome(0, 10*coverage.RATE, error='OOM')
            obj.incident('capture_error', error='x')
            obj.finish(lambda a: 'woorden', lambda t: dict(ok=True, path='copyq'))
        self.assertEqual(calls, [])
        self.assertFalse(hasattr(load_logic()['App'], '_coverage_notice'))

    def test_restart_during_stop_check_is_accepted_and_ordered(self):
        """Start during A's stop check: B records at once, decodes/pastes after A."""
        R = coverage.RATE
        logic = load_logic()
        release, a_checking = threading.Event(), threading.Event()
        events, recorders = [], []

        class FakeRecorder:
            SAMPLE_RATE = R
            def __init__(self, transcription_queue, ledger=None, **kw):
                self.queue, self.ledger, self._raw_frames = transcription_queue, ledger, []
                self.name = 'AB'[len(recorders)]
                recorders.append(self)
            def start(self):
                events.append('record ' + self.name)  # mic open: audio is captured from here
                self.ledger.capture(np.full(2*R, 8000, dtype=np.int16).tobytes())  # audible
            def stop(self):
                value = 0.25 if self.name == 'A' else 1.0
                for start in (0, R):
                    self.queue.put((np.full(R, value, dtype=np.float32), True, start))
                self.queue.put(None)

        class FakeWorker:
            language = 'nl'
            def reset_context(self):
                events.append('reset ' + str(self.language))
            def transcribe_chunk(self, audio, recovery=False):
                name = 'A' if audio[0] < 0.5 else 'B'
                if recovery:
                    a_checking.set()
                    release.wait(5)
                    return 'teruggevonden woorden'
                events.append('decode ' + name)
                if name == 'A' and events.count('decode A') == 1:
                    raise RuntimeError('OOM')
                return 'tekst %s' % name

        class FakeReplayer:
            target_wid = None
            def __init__(self):
                self.windows = iter(['win-A', 'win-B'])
            def window_id(self):
                return next(self.windows)
            def type_text(self, text):
                events.append('paste %s @%s' % (text.strip(), self.target_wid))
                return dict(ok=True, path='copyq', paste='shortcut_sent_not_acknowledged')

        logic.update(StreamingRecorder=FakeRecorder, CoverageLedger=coverage.CoverageLedger,
                     diagnostic=coverage.diagnostic, MAX_SECONDS=coverage.MAX_SECONDS, RATE=R)
        app = logic['App'].__new__(logic['App'])
        app.args = types.SimpleNamespace(paste_min_s=0, silence_ms=1000, auto_stop_silence=0,
                                         min_chunk_s=1, max_chunk_s=10, eager_prob=0.5, max_time=60)
        app.active, app.session, app.timer = False, None, None
        app.beep = lambda *a, **kw: None
        app.transcription_worker, app.replayer = FakeWorker(), FakeReplayer()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertTrue(app.start('nl'))
            session_a = app.session
            self.assertTrue(app.stop())
            self.assertTrue(a_checking.wait(5))  # A is now in its stop check
            self.assertTrue(app.start('en'))  # not refused
            session_b = app.session
            self.assertTrue(app.stop())
            self.assertFalse(session_a.done.is_set())
            self.assertNotIn('decode B', events)  # B waits, its audio is queued
            release.set()
            self.assertTrue(session_b.done.wait(5))
        self.assertIn('"start_while_finishing"', out.getvalue())
        self.assertNotIn('start_refused_busy', out.getvalue())
        self.assertLess(events.index('record B'), events.index('paste [Correctie: vóór "tekst A" is '
                                                              'waarschijnlijk dit gemist: "teruggevonden woorden"] @win-A'))
        pastes = [e for e in events if e.startswith('paste')]
        self.assertEqual(pastes, [
            'paste tekst A @win-A',
            'paste [Correctie: vóór "tekst A" is waarschijnlijk dit gemist: "teruggevonden woorden"] @win-A',
            'paste tekst B @win-B', 'paste tekst B @win-B'])
        # Every captured sample of B was decoded and delivered; nothing lost.
        ledger_b = session_b.ledger
        self.assertEqual([(e['start'], e['end'], e['outcome']) for e in ledger_b.entries],
                         [(0, R, 'text'), (R, 2*R, 'text')])
        self.assertTrue(all(e['delivery']['ok'] for e in ledger_b.entries))
        self.assertEqual(app.transcription_worker.language, 'en')
        self.assertFalse(app.active)

    def test_previous_session_ending_during_start_keeps_new_session_active(self):
        """Regression 18/10: A ended while B's start was still opening (xdotool),
        cleared `active`, so the next press started C instead of stopping B and
        sessions chained forever. A stale auto-stop of A must not stop B either."""
        R = coverage.RATE
        logic = load_logic()
        release, a_checking = threading.Event(), threading.Event()
        recorders, holder = [], {}

        class FakeRecorder:
            SAMPLE_RATE = R
            def __init__(self, transcription_queue, ledger=None, on_auto_stop=None, **kw):
                self.queue, self.ledger, self._raw_frames = transcription_queue, ledger, []
                self.on_auto_stop, self.recording, self.stops = on_auto_stop, False, 0
                recorders.append(self)
            def start(self):
                self.recording = True
                self.ledger.capture(np.full(R, 8000, dtype=np.int16).tobytes())
            def stop(self):
                self.stops += 1
                if not self.recording:
                    return
                self.recording = False
                self.queue.put((np.full(R, 0.5, dtype=np.float32), True, 0))
                self.queue.put(None)

        class FakeWorker:
            language = 'nl'
            def reset_context(self):
                pass
            def transcribe_chunk(self, audio, recovery=False):
                return 'tekst'

        class FakeReplayer:
            target_wid = None
            calls = 0
            def window_id(self):
                FakeReplayer.calls += 1
                if FakeReplayer.calls == 2:  # B is starting: let A finish right now
                    release.set()
                    assert holder['a'].done.wait(5)
                return 'win'
            def type_text(self, text):
                if not release.is_set():  # A is still pasting when B is pressed
                    a_checking.set()
                    release.wait(5)
                return dict(ok=True, path='copyq', paste='shortcut_sent_not_acknowledged')

        logic.update(StreamingRecorder=FakeRecorder, CoverageLedger=coverage.CoverageLedger,
                     diagnostic=coverage.diagnostic, MAX_SECONDS=coverage.MAX_SECONDS, RATE=R)
        app = logic['App'].__new__(logic['App'])
        app.args = types.SimpleNamespace(paste_min_s=0, silence_ms=1000, auto_stop_silence=0,
                                         min_chunk_s=1, max_chunk_s=10, eager_prob=0.5, max_time=60)
        app.active, app.session, app.timer = False, None, None
        app.beep = lambda *a, **kw: None
        app.transcription_worker, app.replayer = FakeWorker(), FakeReplayer()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertTrue(app.start('nl'))
            holder['a'] = app.session
            self.assertTrue(app.stop())
            self.assertTrue(a_checking.wait(5))
            self.assertTrue(app.start('nl'))  # A ends inside this call
            session_b = app.session
            self.assertTrue(holder['a'].done.is_set())
            self.assertTrue(app.active)  # B still records
            recorders[0].on_auto_stop()  # stale silence timeout of A
            self.assertTrue(recorders[1].recording)
            self.assertTrue(app.active)
            app.toggle()  # the next press stops B, it does not start C
            self.assertEqual(len(recorders), 2)
            self.assertFalse(recorders[1].recording)
            self.assertTrue(session_b.done.wait(5))
        self.assertFalse(app.active)

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


class StopStatusTests(unittest.TestCase):
    def test_status_texts(self):
        self.assertEqual(coverage.stop_status(0, 0, 0, 0.04), '[Controle: niets gemist, 0,0 s]')
        self.assertEqual(coverage.stop_status(2, 1, 1, 2.14),
                         '[Controle: 2 stukken nagekeken, 1 correctie, 1 niet terug te vinden, 2,1 s]')

    def test_no_status_line_when_nothing_deviated(self):
        obj = coverage.CoverageLedger()
        obj.capture(np.zeros(2 * coverage.RATE, dtype=np.int16).tobytes())
        obj.outcome(0, 2 * coverage.RATE, raw='hallo', text='hallo')
        obj.entries[-1]['delivery'] = dict(ok=True, path='copyq')
        pasted = []
        with contextlib.redirect_stdout(io.StringIO()):
            obj.finish(lambda audio: '', lambda text: pasted.append(text) or dict(ok=True))
        self.assertEqual(pasted, [])  # Mark 18/10: clean session pastes nothing extra.
        empty = coverage.CoverageLedger()
        with contextlib.redirect_stdout(io.StringIO()):
            empty.finish(lambda audio: '', lambda text: pasted.append(text) or dict(ok=True))
        self.assertEqual(pasted, [])
