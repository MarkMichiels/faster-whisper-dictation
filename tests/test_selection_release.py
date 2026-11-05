"""Selection release and context badge, offline: fake keys, fake focus, fake clock.

Never touches a real display, keyboard, clipboard or selection. The badge
process is never started; launch() gets a fake Popen.
"""
import ast
import contextlib
import io
import json
import subprocess
import sys
import threading
import time
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))
import context_badge  # noqa: E402
import context_capture  # noqa: E402
import selection_release as sr  # noqa: E402
from test_rewrite import AppStart, config, key, CTRL_R, CTRL, ALT, SHIFT, llm_answer  # noqa: E402
import rewrite  # noqa: E402

WIN = 0x2600007
OTHER = 0x3a00004


def trace(**kw):
    base = dict(select_all_at=100.0, select_all_window=WIN, keys_known=True, mouse_known=True)
    base.update(kw)
    return sr.TraceSnapshot(**base)


def start(**kw):
    base = dict(now=102.0, context_source='primary', primary_age_s=2.03, focus_window=WIN,
                focus_is_terminal=False, owner_same_client=True, probe_done=102.1)
    base.update(kw)
    return sr.StartState(**base)


class PlanTests(unittest.TestCase):
    """Every refusal reason of plan_release, one by one."""

    def test_ok(self):
        self.assertEqual(sr.plan_release(trace(), start()), (True, 'ok'))

    def test_each_refusal(self):
        cases = [
            ('context_not_primary', trace(), start(context_source='clipboard')),
            ('context_not_primary', trace(), start(context_source='none')),
            ('terminal', trace(), start(focus_is_terminal=True)),
            ('no_focus', trace(), start(focus_window=None)),
            ('keys_unknown', trace(keys_known=False), start()),
            ('mouse_unknown', trace(mouse_known=False), start()),
            ('no_select_all', trace(select_all_at=None), start()),
            ('select_all_too_old', trace(select_all_at=81.9), start()),
            ('select_all_too_old', trace(select_all_at=102.5), start()),
            ('focus_moved', trace(select_all_window=OTHER), start()),
            ('focus_moved', trace(select_all_window=None), start()),  # focus read not finished
            ('mouse_after_select', trace(last_mouse_press_at=100.5), start()),
            ('mouse_after_select', trace(last_mouse_press_at=100.0), start()),
            ('key_after_select', trace(last_other_key_at=101.0), start()),
            ('owner_elsewhere', trace(), start(owner_same_client=False)),
            ('owner_elsewhere', trace(), start(owner_same_client=None)),
            ('selection_not_from_select_all', trace(), start(primary_age_s=None)),
            ('selection_not_from_select_all', trace(), start(primary_age_s=1.3)),   # newer selection
            ('selection_not_from_select_all', trace(), start(primary_age_s=2.8)),   # older selection
            ('previous_session_busy', trace(), start(previous_busy=True)),
        ]
        for reason, tr, st in cases:
            with self.subTest(reason=reason):
                self.assertEqual(sr.plan_release(tr, st), (False, reason))

    def test_input_before_select_all_does_not_count(self):
        tr = trace(last_mouse_press_at=99.0, last_other_key_at=99.5)
        self.assertEqual(sr.plan_release(tr, start()), (True, 'ok'))

    def test_timestamp_window_spans_the_probe(self):
        # The probe ran between now (2.0 s after Ctrl+A) and probe_done (2.1 s after).
        for age in (1.41, 2.0, 2.05, 2.69):
            self.assertTrue(sr.plan_release(trace(), start(primary_age_s=age))[0], age)
        for age in (1.39, 2.71):
            self.assertFalse(sr.plan_release(trace(), start(primary_age_s=age))[0], age)


class FireTests(unittest.TestCase):
    def fire(self, tr=None, now=100.5, held=False, focus=WIN, active=True):
        return sr.may_fire(tr or trace(select_all_at=90.0), 100.0, now, held, focus, WIN, active)

    def test_reasons(self):
        self.assertEqual(self.fire(), (True, 'ok'))
        self.assertEqual(self.fire(held=True), (None, 'waiting_for_release'))
        self.assertEqual(self.fire(held=True, now=102.01), (False, 'modifiers_held_too_long'))
        self.assertEqual(self.fire(tr=trace(last_other_key_at=100.2)), (False, 'typed_after_start'))
        self.assertEqual(self.fire(tr=trace(last_mouse_press_at=100.2)), (False, 'mouse_after_start'))
        self.assertEqual(self.fire(focus=OTHER), (False, 'focus_moved_after_start'))
        self.assertEqual(self.fire(focus=None), (False, 'focus_moved_after_start'))
        self.assertEqual(self.fire(active=False), (False, 'session_ended'))


class Clock:
    def __init__(self, t=100.0):
        self.t = t

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


class WaitTests(unittest.TestCase):
    def run_wait(self, release_at=None, focus=lambda: WIN, events=None, active=lambda: True):
        clock = Clock()
        tr = sr.InputTrace(focus=lambda: WIN, clock=clock, spawn=lambda fn: fn())
        tr.set_known(keys=True, mouse=True)
        sent = []
        held = (lambda: True) if release_at is None else (lambda: clock.t < release_at)
        result = sr.wait_and_release(tr, 100.0, WIN, held, lambda: sent.append(clock.t),
                                     session_active=active, focus=focus, clock=clock, sleep=clock.sleep)
        return result, sent

    def test_fires_once_after_release(self):
        result, sent = self.run_wait(release_at=100.63)
        self.assertEqual(len(sent), 1)
        self.assertGreaterEqual(sent[0], 100.63)
        self.assertEqual((result['fired'], result['reason']), (True, 'ok'))
        self.assertAlmostEqual(result['wait_s'], 0.65, places=2)

    def test_held_too_long_never_fires(self):
        result, sent = self.run_wait(release_at=None)
        self.assertEqual(sent, [])
        self.assertEqual((result['fired'], result['reason']), (False, 'modifiers_held_too_long'))

    def test_focus_error_is_a_refusal(self):
        def broken():
            raise OSError('no display')
        result, sent = self.run_wait(release_at=100.1, focus=broken)
        self.assertEqual((sent, result['reason']), ([], 'focus_moved_after_start'))

    def test_send_error_never_raises(self):
        tr = sr.InputTrace(focus=lambda: WIN, spawn=lambda fn: fn())

        def boom():
            raise RuntimeError('xtest')
        result = sr.wait_and_release(tr, time.monotonic(), WIN, lambda: False, boom, focus=lambda: WIN)
        self.assertEqual((result['fired'], result['reason'], result['error']), (False, 'error', 'RuntimeError'))

    def test_own_injection_is_not_user_input(self):
        clock = Clock()
        tr = sr.InputTrace(focus=lambda: WIN, clock=clock, spawn=lambda fn: fn())
        tr.injecting()
        tr.other_key()
        tr.mouse_press()
        tr.select_all()
        snap = tr.snapshot()
        self.assertEqual((snap.last_other_key_at, snap.last_mouse_press_at, snap.select_all_at),
                         (None, None, None))
        clock.t += sr.INJECT_QUIET_S + 0.01
        tr.other_key()
        self.assertEqual(tr.snapshot().last_other_key_at, clock.t)


class TraceTests(unittest.TestCase):
    def test_focus_read_off_the_callback_and_latest_wins(self):
        pending = []
        tr = sr.InputTrace(focus=iter([WIN, OTHER]).__next__, spawn=pending.append)
        tr.select_all()
        tr.select_all()
        self.assertIsNone(tr.snapshot().select_all_window)  # callback did not block on X
        pending[1]()   # newest press resolves first
        pending[0]()   # stale result of the first press is dropped
        self.assertEqual(tr.snapshot().select_all_window, WIN)

    def test_focus_error_leaves_window_unknown(self):
        def broken():
            raise OSError
        tr = sr.InputTrace(focus=broken, spawn=lambda fn: fn())
        tr.select_all()
        self.assertIsNone(tr.snapshot().select_all_window)

    def test_snapshot_is_a_copy(self):
        tr = sr.InputTrace(focus=lambda: WIN, spawn=lambda fn: fn())
        snap = tr.snapshot()
        snap.select_all_at = 1.0
        self.assertIsNone(tr.snapshot().select_all_at)


def char(c, vk=None):
    return types.SimpleNamespace(char=c, vk=vk if vk is not None else ord(c))


class ClassifyTests(unittest.TestCase):
    def test_classes(self):
        c = sr.classify_key
        self.assertEqual(c(char('a'), True, False), 'select_all')
        self.assertEqual(c(char('A'), True, False), 'select_all')
        self.assertEqual(c(char('\x01', 97), True, False), 'select_all')
        self.assertEqual(c(char('a'), False, False), 'other')
        self.assertEqual(c(char('c'), True, False), 'neutral')
        self.assertEqual(c(key('insert', 0xff63), True, False), 'neutral')
        self.assertEqual(c(key('insert', 0xff63), False, False), 'other')
        self.assertEqual(c(key('home', 0xff50), True, False), 'other')
        self.assertEqual(c(key('enter', 0xff0d), False, False), 'other')
        for mod in (CTRL, ALT, SHIFT, key('alt_gr', 0xfe03), key(None, 0xffe7), key('shift_r', 0xffe2)):
            self.assertEqual(c(mod, False, False), 'modifier')
        self.assertEqual(c(CTRL_R, False, True), 'start')


def listener_with_trace(tr):
    tree = ast.parse((ROOT / 'dictation.py').read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MultiTapKeyListener')
    ns = dict(keyboard=types.SimpleNamespace(Key=types.SimpleNamespace(cmd_r=key('cmd_r', 65516))),
              time=time, selection_release=sr)
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'listener', 'exec'), ns)
    calls = []
    listener = ns['MultiTapKeyListener'](lambda *a: calls.append(a), lambda: None, key=CTRL_R,
                                         default_language='nl')
    listener.trace = tr
    return listener, calls


class ListenerTests(unittest.TestCase):
    def setUp(self):
        self.tr = sr.InputTrace(focus=lambda: WIN, spawn=lambda fn: fn())

    def test_ctrl_a_with_either_ctrl(self):
        listener, _ = listener_with_trace(self.tr)
        for ctrl in (CTRL, CTRL_R):
            self.tr._data = sr.TraceSnapshot()
            listener.on_press(ctrl)
            listener.on_press(char('a'))
            listener.on_release(char('a'))
            listener.on_release(ctrl)
            snap = self.tr.snapshot()
            self.assertIsNotNone(snap.select_all_at)
            self.assertEqual(snap.select_all_window, WIN)
            self.assertIsNone(snap.last_other_key_at)

    def test_start_chord_and_copy_are_not_other_keys(self):
        listener, calls = listener_with_trace(self.tr)
        listener.on_press(CTRL)
        listener.on_press(char('c'))
        listener.on_release(char('c'))
        listener.on_release(CTRL)
        listener.on_press(ALT)
        for _ in range(2):
            listener.on_press(CTRL_R)
            listener.on_release(CTRL_R)
        self.assertTrue(calls and calls[-1][1])  # rewrite started
        self.assertTrue(listener.modifiers_held())
        listener.on_release(ALT)
        self.assertFalse(listener.modifiers_held())
        self.assertIsNone(self.tr.snapshot().last_other_key_at)

    def test_plain_a_and_arrows_are_other_keys(self):
        listener, _ = listener_with_trace(self.tr)
        listener.on_press(char('a'))
        self.assertIsNotNone(self.tr.snapshot().last_other_key_at)
        self.assertIsNone(self.tr.snapshot().select_all_at)

    def test_without_trace_listener_is_unchanged(self):
        listener, calls = listener_with_trace(None)
        listener.on_press(char('a'))
        for _ in range(2):
            listener.on_press(CTRL_R)
            listener.on_release(CTRL_R)
        self.assertEqual(calls[-1][:2], ('nl', False))


class SendTests(unittest.TestCase):
    def test_exactly_ctrl_home(self):
        kb = Mock()
        k = types.SimpleNamespace(ctrl='CTRL', home='HOME')
        sr.send_ctrl_home(kb, k)
        self.assertEqual([tuple(c) for c in kb.mock_calls],
                         [('press', ('CTRL',), {}), ('press', ('HOME',), {}),
                          ('release', ('HOME',), {}), ('release', ('CTRL',), {})])

    def test_ctrl_released_even_when_home_fails(self):
        kb = Mock()
        kb.press.side_effect = [None, RuntimeError('xtest')]
        with self.assertRaises(RuntimeError):
            sr.send_ctrl_home(kb, types.SimpleNamespace(ctrl='CTRL', home='HOME'))
        kb.release.assert_called_with('CTRL')

    def test_source_sends_no_other_key(self):
        """No Enter, Tab, Escape or clearmodifiers anywhere in the release path."""
        source = (ROOT / 'selection_release.py').read_text()
        tree = ast.parse(source)
        attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        self.assertFalse(attrs & {'enter', 'tab', 'esc', 'return_', 'left', 'right', 'up', 'down', 'end'})
        docs = {id(n.body[0].value) for n in ast.walk(tree)
                if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef)) and ast.get_docstring(n)}
        strings = [n.value for n in ast.walk(tree)
                   if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docs]
        self.assertFalse([v for v in strings if 'clearmodifiers' in v or 'xdotool' in v])


class CaptureOriginTests(unittest.TestCase):
    def test_probe_origin_reaches_context_without_text_in_journal(self):
        mail = 'Beste, kunnen jullie de levering voor vrijdag bevestigen? Groeten.'
        meta = {'focus': WIN, 'primary': dict(owner=True, targets=['TIMESTAMP'], age_s=2.0, same_client=True),
                'clipboard': dict(owner=False)}

        def run(argv, **kw):
            if '--probe' in argv:
                return types.SimpleNamespace(returncode=0, stdout=json.dumps(meta).encode())
            return types.SimpleNamespace(returncode=0, stdout=mail.encode())
        ctx = context_capture.capture(run)
        self.assertEqual((ctx.source, ctx.focus_window, ctx.owner_same_client), ('primary', WIN, True))
        self.assertTrue(ctx.journal()['owner_same_client'])
        self.assertNotIn('levering', json.dumps(ctx.journal()))
        meta['primary']['same_client'] = 'yes'  # malformed: unknown, never True
        meta['focus'] = 'x'
        ctx = context_capture.capture(run)
        self.assertEqual((ctx.focus_window, ctx.owner_same_client), (None, None))


class FakeKeyListener:
    def __init__(self, held=lambda: False):
        self.modifiers_held = held


class AppReleaseTests(AppStart, unittest.TestCase):
    MAIL = 'Beste, kunnen jullie de levering voor vrijdag bevestigen? Groeten.'

    def run_start(self, ctx, trace_setup=None, release=True, badge=True, terminal_mode=False,
                  wm_class='google-chrome'):
        def capture(**kw):
            return ctx
        app, events = self.start_app(config(), None, capture=capture)
        logic_globals = app.start.__globals__
        logic_globals['RELEASE_SELECTION'] = release
        logic_globals['CONTEXT_BADGE'] = badge
        badge_mod = logic_globals['context_badge']
        badge_mod.calls.clear()
        sent = threading.Event()
        app.replayer.send_ctrl_home = lambda: (events.append(('ctrl_home',)), sent.set())
        app.replayer._get_active_window_class = lambda: wm_class
        app.replayer.TERMINAL_CLASSES = {'kitty'}
        app.input_trace = sr.InputTrace(focus=lambda: WIN, spawn=lambda fn: fn())
        app.input_trace.set_known(keys=True, mouse=True)
        app.key_listener = FakeKeyListener()
        if trace_setup:
            trace_setup(app.input_trace)
        out = io.StringIO()
        keys = {'ctrl': True} if terminal_mode else {'alt': True}
        with patch.object(rewrite, 'http_call', lambda *a: llm_answer('Een korte herschreven bevestiging.')), \
                patch.object(sr, 'focus_window', lambda: WIN), contextlib.redirect_stdout(out):
            app.start('nl', True, keys)
            session = app.session
            sent.wait(1)
            deadline = time.monotonic() + 2
            while '"selection_release"' not in out.getvalue() and time.monotonic() < deadline:
                time.sleep(0.01)
            app.stop()
            self.assertTrue(session.done.wait(5))
        journal = out.getvalue()
        record = json.loads(next(l for l in journal.splitlines() if '"selection_release"' in l)
                            .removeprefix('DCT-COVERAGE '))
        return events, record, journal, badge_mod.calls

    def ctx(self, source='primary', age=None, text=None, **kw):
        text = text if text is not None else self.MAIL
        return context_capture.Context(text, source, len(text), age, focus_window=WIN,
                                       owner_same_client=True, **kw)

    def select_all_now(self, offset=0.0):
        def setup(tr):
            tr.select_all()
            tr._data.select_all_at -= offset
        return setup

    def test_ctrl_a_selection_is_released(self):
        events, record, journal, badge = self.run_start(self.ctx(age=0.0), self.select_all_now())
        self.assertEqual(record['fired'], True)
        self.assertEqual(record['reason'], 'ok')
        self.assertEqual(events.count(('ctrl_home',)), 1)
        self.assertLess(events.index(('ctrl_home',)), [i for i, e in enumerate(events) if e[0] == 'paste'][0])
        self.assertEqual(badge, [len(self.MAIL)])
        self.assertNotIn('levering', journal.split('Een korte')[0])

    def test_refusals_are_journaled_and_send_nothing(self):
        cases = [
            ('disabled', dict(release=False)),
            ('no_select_all', dict()),
            ('terminal', dict(wm_class='kitty', trace_setup=self.select_all_now())),
            ('terminal', dict(terminal_mode=True, trace_setup=self.select_all_now())),
            ('context_not_primary', dict(ctx=self.ctx(source='clipboard', age=0.0),
                                         trace_setup=self.select_all_now())),
            ('selection_not_from_select_all', dict(ctx=self.ctx(age=5.0), trace_setup=self.select_all_now())),
        ]
        for reason, kw in cases:
            with self.subTest(reason=reason):
                ctx = kw.pop('ctx', self.ctx(age=0.0))
                events, record, journal, _ = self.run_start(ctx, **kw)
                self.assertEqual((record['fired'], record['reason']), (False, reason))
                self.assertNotIn(('ctrl_home',), events)

    def test_typing_after_start_cancels(self):
        def setup(tr):
            tr.select_all()
        app_listener_held = {'v': True}

        def capture(**kw):
            return self.ctx(age=0.0)
        app, events = self.start_app(config(), None, capture=capture)
        app.start.__globals__['RELEASE_SELECTION'] = True
        app.replayer.send_ctrl_home = lambda: events.append(('ctrl_home',))
        app.replayer._get_active_window_class = lambda: ''
        app.replayer.TERMINAL_CLASSES = set()
        app.input_trace = sr.InputTrace(focus=lambda: WIN, spawn=lambda fn: fn())
        app.input_trace.set_known(keys=True, mouse=True)
        app.key_listener = FakeKeyListener(lambda: app_listener_held['v'])
        setup(app.input_trace)
        out = io.StringIO()
        with patch.object(rewrite, 'http_call', lambda *a: llm_answer('Een korte herschreven bevestiging.')), \
                patch.object(sr, 'focus_window', lambda: WIN), contextlib.redirect_stdout(out):
            app.start('nl', True, {'alt': True})
            session = app.session
            time.sleep(0.12)
            app.input_trace.other_key()   # the user types while Alt is still down
            app_listener_held['v'] = False
            deadline = time.monotonic() + 2
            while '"selection_release"' not in out.getvalue() and time.monotonic() < deadline:
                time.sleep(0.01)
            app.stop()
            self.assertTrue(session.done.wait(5))
        record = json.loads(next(l for l in out.getvalue().splitlines() if '"selection_release"' in l)
                            .removeprefix('DCT-COVERAGE '))
        self.assertEqual((record['fired'], record['reason']), (False, 'typed_after_start'))
        self.assertNotIn(('ctrl_home',), events)

    def test_badge_only_with_context_and_when_enabled(self):
        _, _, _, badge = self.run_start(context_capture.Context(None, reasons={'primary': 'empty'}))
        self.assertEqual(badge, [])
        _, _, _, badge = self.run_start(self.ctx(source='clipboard', age=3.0), badge=False)
        self.assertEqual(badge, [])
        _, _, _, badge = self.run_start(self.ctx(source='clipboard', age=3.0))
        self.assertEqual(badge, [len(self.MAIL)])


class BadgeTests(unittest.TestCase):
    def test_label_shows_only_the_length(self):
        self.assertEqual(context_badge.label(4312), 'Context opgepakt: 4.312 tekens')
        self.assertEqual(context_badge.label(100000), 'Context opgepakt: 100.000 tekens')
        self.assertEqual(context_badge.label(-3), 'Context opgepakt: 0 tekens')

    def test_monitor_choice_and_position(self):
        mons = [(0, 0, 1920, 1080), (1920, 0, 2560, 1440)]
        self.assertEqual(context_badge.choose_monitor(mons, (2500, 300), (4480, 1440)), mons[1])
        self.assertEqual(context_badge.choose_monitor(mons, (100, 100), (4480, 1440)), mons[0])
        self.assertEqual(context_badge.choose_monitor(mons, None, (4480, 1440)), mons[0])
        self.assertEqual(context_badge.choose_monitor([], (5, 5), (1600, 1000)), (0, 0, 1600, 1000))
        self.assertEqual(context_badge.badge_position(mons[1], 300), (1920 + 2560 - 300 - 16, 16))

    def test_launch_passes_only_the_number(self):
        seen = {}

        class Proc:
            def wait(self, timeout=None):
                return 0

        def popen(argv, **kw):
            seen['argv'], seen['kw'] = argv, kw
            return Proc()
        logs = []
        self.assertTrue(context_badge.launch(4312, popen=popen, log=lambda *a, **k: logs.append((a, k)),
                                             spawn=lambda fn: fn()))
        self.assertEqual(seen['argv'][-2:], ['--chars', '4312'])
        self.assertEqual(seen['kw']['stdin'], subprocess.DEVNULL)
        self.assertEqual(logs, [])

    def test_failed_start_is_silent_but_journaled(self):
        logs = []

        def popen(argv, **kw):
            raise FileNotFoundError('python')
        self.assertFalse(context_badge.launch(10, popen=popen, log=lambda kind, **k: logs.append((kind, k))))
        self.assertEqual(logs, [('context_badge', {'started': False, 'error': 'FileNotFoundError'})])

    def test_stuck_badge_is_killed(self):
        proc = Mock()
        proc.wait.side_effect = [subprocess.TimeoutExpired('badge', 5), 0]
        logs = []
        context_badge.launch(10, popen=lambda *a, **k: proc, log=lambda kind, **k: logs.append((kind, k)),
                             spawn=lambda fn: fn())
        proc.kill.assert_called_once()
        self.assertEqual(proc.wait.call_args_list[0].kwargs, {'timeout': context_badge.KILL_AFTER_S})
        self.assertEqual(logs, [('context_badge', {'started': True, 'killed': True})])


class SourceTests(unittest.TestCase):
    def test_no_personal_paths_and_bounded_imports(self):
        for name in ('selection_release.py', 'context_badge.py', 'tests/test_selection_release.py'):
            source = (ROOT / name).read_text()
            self.assertNotIn('/' + 'home/', source)
            self.assertNotIn('notify-' + 'send', source)
        for name in ('selection_release.py', 'context_badge.py'):
            tree = ast.parse((ROOT / name).read_text())
            modules = {a.name.split('.')[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
            modules |= {n.module.split('.')[0] for n in ast.walk(tree)
                        if isinstance(n, ast.ImportFrom) and n.module}
            self.assertLessEqual(modules, {'__future__', 'argparse', 'dataclasses', 'os', 'select',
                                           'subprocess', 'sys', 'threading', 'time', 'typing', 'Xlib'})

    def test_badge_never_takes_focus(self):
        source = (ROOT / 'context_badge.py').read_text()
        self.assertIn('override_redirect=True', source)
        self.assertIn('input=0', source)
        self.assertIn('shape.SK.Input', source)
        self.assertNotIn('set_input_focus', source)


if __name__ == '__main__':
    unittest.main()
