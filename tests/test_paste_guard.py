"""Paste guard, offline: after stop, a click or typed key holds text in CopyQ.

No audio, desktop, clipboard or network. The guard journals metadata only;
held text goes to the replayer with a hold reason instead of being pasted.
"""
import ast
import contextlib
import io
import queue
import re
import sys
import time
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))
import coverage_ledger as coverage  # noqa: E402
import rewrite  # noqa: E402
import selection_release  # noqa: E402
from selection_release import TraceSnapshot, input_since  # noqa: E402
from test_rewrite import R, FakeReplayer, app_logic, config, llm_answer, worker  # noqa: E402

SPOKEN = 'Kwartelkoffie bij de Zeppelinstraat'


class FakeTrace:
    def __init__(self, **kw):
        kw.setdefault('keys_known', True)
        kw.setdefault('mouse_known', True)
        self.data = TraceSnapshot(**kw)

    def snapshot(self):
        return self.data


class InputSinceTests(unittest.TestCase):
    def test_nothing_after(self):
        self.assertIsNone(input_since(TraceSnapshot(), 10.0))
        self.assertIsNone(input_since(TraceSnapshot(last_mouse_press_at=9.0, last_other_key_at=9.5,
                                                    select_all_at=8.0), 10.0))

    def test_mouse_key_and_select_all(self):
        self.assertEqual(input_since(TraceSnapshot(last_mouse_press_at=11.0), 10.0), 'mouse')
        self.assertEqual(input_since(TraceSnapshot(last_other_key_at=11.0), 10.0), 'key')
        self.assertEqual(input_since(TraceSnapshot(select_all_at=11.0), 10.0), 'key')
        self.assertEqual(input_since(TraceSnapshot(last_mouse_press_at=11.0, last_other_key_at=12.0), 10.0),
                         'mouse+key')

    def test_gesture_press_is_skipped(self):
        snap = TraceSnapshot(last_mouse_press_at=10.3)
        self.assertIsNone(input_since(snap, 10.0, mouse_after=10.5))
        self.assertEqual(input_since(TraceSnapshot(last_mouse_press_at=10.6), 10.0, mouse_after=10.5), 'mouse')
        # mouse_after never lowers the floor below since.
        self.assertIsNone(input_since(TraceSnapshot(last_mouse_press_at=9.0), 10.0, mouse_after=5.0))

    def test_own_injected_keys_do_not_count(self):
        clock = [100.0]
        trace = selection_release.InputTrace(focus=lambda: None, clock=lambda: clock[0], spawn=lambda fn: fn())
        trace.injecting()
        clock[0] = 100.1
        trace.other_key()
        trace.mouse_press()
        self.assertIsNone(input_since(trace.snapshot(), 99.0))
        clock[0] = 101.0
        trace.other_key()
        self.assertEqual(input_since(trace.snapshot(), 99.0), 'key')


class GuardTestCase(unittest.TestCase):
    def setUp(self):
        self.logic = app_logic()
        self.journal = []
        self.logic['diagnostic'] = lambda kind, **meta: self.journal.append((kind, meta))
        self.app = self.logic['App'].__new__(self.logic['App'])
        self.now = time.monotonic()

    def session(self, stopped_ago=None, stop_mouse=False, mode='dictate'):
        return types.SimpleNamespace(
            mode=mode, started_at=self.now - 30, last_paste_at=None, held_texts=[],
            stopped_at=None if stopped_ago is None else self.now - stopped_ago, stop_mouse=stop_mouse)

    def guard(self, session, **trace):
        self.app.input_trace = FakeTrace(**trace)
        return self.app._paste_guard(session)


class PasteGuardTests(GuardTestCase):
    def test_unknown_without_key_trace(self):
        self.app.input_trace = None
        hold, meta = self.app._paste_guard(self.session(stopped_ago=1))
        self.assertIsNone(hold)
        self.assertEqual(meta['guard'], 'unknown')
        hold, meta = self.guard(self.session(stopped_ago=1), keys_known=False, last_mouse_press_at=self.now)
        self.assertEqual((hold, meta['guard']), (None, 'unknown'))

    def test_recording_never_holds(self):
        hold, meta = self.guard(self.session(), last_mouse_press_at=self.now - 1, last_other_key_at=self.now - 2)
        self.assertIsNone(hold)
        self.assertEqual(meta['guard'], 'recording')
        self.assertEqual(meta['input_since_paste'], 'mouse+key')
        self.assertNotIn('input_after_stop', meta)

    def test_clear_after_stop_without_input(self):
        hold, meta = self.guard(self.session(stopped_ago=2), last_mouse_press_at=self.now - 10)
        self.assertIsNone(hold)
        self.assertEqual(meta['guard'], 'clear')
        self.assertIsNone(meta['input_after_stop'])
        self.assertAlmostEqual(meta['since_stop_s'], 2.0, delta=0.2)
        self.assertAlmostEqual(meta['mouse_ago_s'], 10.0, delta=0.2)

    def test_click_or_key_after_stop_holds(self):
        for trace, after in [(dict(last_mouse_press_at=self.now - 1), 'mouse'),
                             (dict(last_other_key_at=self.now - 1), 'key'),
                             (dict(select_all_at=self.now - 1), 'key')]:
            with self.subTest(after=after):
                hold, meta = self.guard(self.session(stopped_ago=2), **trace)
                self.assertEqual(hold, 'input_after_stop')
                self.assertEqual((meta['guard'], meta['input_after_stop']), ('held', after))

    def test_setting_off_only_journals(self):
        self.logic['HOLD_AFTER_INPUT'] = False
        hold, meta = self.guard(self.session(stopped_ago=2), last_mouse_press_at=self.now - 1)
        self.assertIsNone(hold)
        self.assertEqual((meta['guard'], meta['input_after_stop']), ('off', 'mouse'))

    def test_thumb_button_stop_press_is_grace(self):
        # The thumb press that stopped the session lands right after stopped_at.
        session = self.session(stopped_ago=2, stop_mouse=True)
        hold, meta = self.guard(session, last_mouse_press_at=session.stopped_at + 0.1)
        self.assertEqual((hold, meta['guard']), (None, 'clear'))
        hold, meta = self.guard(session, last_mouse_press_at=session.stopped_at + 1.0)
        self.assertEqual((hold, meta['guard']), ('input_after_stop', 'held'))
        # A key stop has no grace for the mouse.
        session = self.session(stopped_ago=2)
        hold, _ = self.guard(session, last_mouse_press_at=session.stopped_at + 0.1)
        self.assertEqual(hold, 'input_after_stop')

    def test_gesture_grace_from_toggle(self):
        session = self.session(stopped_ago=2)
        self.app._gesture_at = session.stopped_at
        hold, meta = self.guard(session, last_mouse_press_at=session.stopped_at + 0.2)
        self.assertEqual((hold, meta['guard']), (None, 'clear'))

    def test_never_raises(self):
        self.app.input_trace = types.SimpleNamespace(snapshot=Mock(side_effect=RuntimeError('boom')))
        hold, meta = self.app._paste_guard(self.session(stopped_ago=1))
        self.assertIsNone(hold)
        self.assertEqual((meta['guard'], meta['error']), ('error', 'RuntimeError'))


class PasteTests(GuardTestCase):
    def setUp(self):
        super().setUp()
        self.calls = []

        def type_text(text, **kw):
            self.calls.append((text, kw))
            return dict(ok=True, held=True, paste='held_' + kw['hold']) if kw.get('hold') else dict(ok=True)
        self.app.replayer = types.SimpleNamespace(type_text=type_text)

    def test_plain_paste_has_no_hold_argument(self):
        session = self.session()
        self.app.input_trace = FakeTrace()
        self.app._paste(session, 'een', terminal_mode=True)
        self.assertEqual(self.calls, [('een', dict(terminal_mode=True))])
        self.assertIsNotNone(session.last_paste_at)
        kind, meta = self.journal[-1]
        self.assertEqual((kind, meta['mode'], meta['guard']), ('paste_guard', 'dictate', 'recording'))

    def test_held_text_accumulates(self):
        session = self.session(stopped_ago=3)
        self.app.input_trace = FakeTrace(last_mouse_press_at=self.now - 1)
        self.app._paste(session, SPOKEN)
        self.app._paste(session, ' en verder')
        self.assertEqual([c[0] for c in self.calls], [SPOKEN, SPOKEN + ' en verder'])
        self.assertTrue(all(c[1] == dict(hold='input_after_stop') for c in self.calls))

    def test_guard_journal_is_metadata_only(self):
        session = self.session(stopped_ago=3)
        self.app.input_trace = FakeTrace(last_mouse_press_at=self.now - 1)
        self.app._paste(session, SPOKEN)
        self.assertNotIn('Kwartel', repr(self.journal))
        allowed = {'mode', 'since_stop_s', 'input_after_stop', 'input_since_paste', 'mouse_ago_s', 'key_ago_s',
                   'guard', 'error'}
        for kind, meta in self.journal:
            self.assertEqual(kind, 'paste_guard')
            self.assertLessEqual(set(meta), allowed)

    def test_held_counts_as_not_pasted(self):
        session = types.SimpleNamespace(pasted=0, paste_failed=0)
        self.app._count_paste(session, dict(ok=True, held=True))
        self.app._count_paste(session, dict(ok=True))
        self.assertEqual((session.pasted, session.paste_failed), (1, 1))

    def test_mark_stopped_once(self):
        session = self.session()
        self.app._mark_stopped(session, 'mouse', 'button9')
        first = session.stopped_at
        self.app._mark_stopped(session, 'key')
        self.assertEqual(session.stopped_at, first)
        self.assertTrue(session.stop_mouse)
        self.assertEqual(self.journal, [('session_stop', dict(trigger='mouse', button='button9'))])
        self.app._mark_stopped(None, 'key')  # never raises


class HoldReplayer(FakeReplayer):
    def type_text(self, text, echo=True, rewrite_mode=False, html=None, terminal_mode=False, hold=None):
        if not hold:
            return super().type_text(text, echo, rewrite_mode, html, terminal_mode)
        self.events.append(('held', text))
        return dict(ok=True, path='copyq', paste='held_' + hold, held=True)


class FakeBox:
    def __init__(self):
        self.shown = []

    def show(self, phase, *rest):
        self.shown.append(phase)


class RewriteHeldTests(unittest.TestCase):
    def test_rewrite_held_is_not_reported_pasted(self):
        logic = app_logic()
        events = []
        app = logic['App'].__new__(logic['App'])
        app.args = types.SimpleNamespace(paste_min_s=0)
        app.transcription_worker = worker(['Eerste deel over de pomp.'])
        app.replayer = HoldReplayer(events)
        app.beep = lambda k, wait=True: events.append(('beep', k))
        app.rewrite_config, app.rewrite_reason, app.active = config(), None, False
        app.status = FakeBox()
        logged = []
        app.llm_log = types.SimpleNamespace(record=lambda **row: logged.append(row))
        ledger = coverage.CoverageLedger()
        ledger.capture(np.full(10 * R, 8000, dtype=np.int16).tobytes())
        session = app.Session(queue.Queue(), queue.Queue(), ledger, types.SimpleNamespace(_raw_frames=[]),
                              None, 'nl', 'win-start', rewrite=True)
        session.context, session.paste_wid = None, 'win-stop'
        session.stopped_at = time.monotonic() - 1
        app.session = session
        app.input_trace = FakeTrace(last_mouse_press_at=time.monotonic() - 0.5)
        session.transcription_queue.put((np.full(10 * R, 0.1, dtype=np.float32), True, 0))
        session.transcription_queue.put(None)
        with patch.object(rewrite, 'http_call', lambda cfg, body, timeout: llm_answer('Herschreven.')), \
                contextlib.redirect_stdout(io.StringIO()):
            app._transcription_loop(session)
            app._typing_loop(session)
        self.assertIn(('held', 'Herschreven.'), events)
        self.assertNotIn('paste', [e[0] for e in events])
        self.assertIn(('beep', 'rewrite_failed'), events)
        self.assertEqual(app.status.shown[-1], 'paste_refused')
        # LLM log: held text is in CopyQ and on the clipboard, not pasted.
        self.assertEqual(logged[0]['delivered'], 'clipboard')


class ReplayerHoldTests(unittest.TestCase):
    def setUp(self):
        source = ROOT / 'dictation.py'
        node = next(n for n in ast.parse(source.read_text()).body
                    if isinstance(n, ast.ClassDef) and n.name == 'KeyboardReplayer')
        self.events = []
        ns = dict(re=re, diagnostic=lambda event, **meta: self.events.append((event, meta)),
                  keyboard=types.SimpleNamespace(Key=types.SimpleNamespace(ctrl='CTRL', shift='SHIFT')))
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), ns)
        cls = ns['KeyboardReplayer']
        self.obj = cls.__new__(cls)
        self.obj.kb = Mock()
        self.obj.target_wid = 7
        self.obj.window_id = Mock(return_value=9)
        self.obj._restore_target_window = Mock(return_value=True)
        self.obj._get_active_window_class = Mock(return_value='editor')
        self.obj._set_clipboard = Mock(return_value=dict(ok=True, path='fake'))
        self.obj._clipboard_owned = Mock(return_value=True)
        self.obj._delivery_alert = Mock()

    def test_hold_sends_no_keys_and_no_activation(self):
        with contextlib.redirect_stdout(io.StringIO()):
            result = self.obj.type_text(SPOKEN, hold='input_after_stop')
        self.obj.kb.press.assert_not_called()
        self.obj._restore_target_window.assert_not_called()
        self.obj._set_clipboard.assert_called_once_with(SPOKEN)
        self.obj._delivery_alert.assert_called_once()
        self.assertEqual((result['paste'], result['held'], result['app']),
                         ('held_input_after_stop', True, 'editor'))
        kind, meta = self.events[-1]
        self.assertEqual(kind, 'paste_delivery')
        self.assertNotIn('Kwartel', repr(self.events))

    def test_paste_calls_inject_hook_and_reports_refocus(self):
        order = []
        self.obj.inject_hook = lambda: order.append('hook')
        self.obj.kb.press.side_effect = lambda k: order.append(('press', k))
        with contextlib.redirect_stdout(io.StringIO()):
            result = self.obj.type_text('een')
        self.assertEqual(order[:2], ['hook', ('press', 'CTRL')])
        self.assertTrue(result['refocused'])
        self.assertFalse(result.get('held', False))

    def test_no_hook_attribute_is_fine(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.obj.type_text('een')
        self.obj.kb.press.assert_any_call('CTRL')


if __name__ == '__main__':
    unittest.main()
