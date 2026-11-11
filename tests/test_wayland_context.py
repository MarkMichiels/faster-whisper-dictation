"""Wayland reply context: selection age/focus from the shell, release after Ctrl+A, badge.

Offline: fake gdbus answers, fake probe/xclip, synthetic evdev events, fake
clock. Never touches a real display, input device, clipboard or selection.
"""
import contextlib
import io
import json
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
import context_capture as cc  # noqa: E402
import selection_release as sr  # noqa: E402
import wayland_io as w  # noqa: E402
import rewrite  # noqa: E402
from test_rewrite import AppStart, config, llm_answer  # noqa: E402
from test_wayland import listeners  # noqa: E402

MAIL = 'Beste, kunnen jullie de levering voor vrijdag bevestigen? Groeten, Jan.'
COPIED = 'Gekopieerde alinea uit een ander programma met genoeg tekens erin.'
GMAIL, POPOUT, VIEWER = '42', '43', '44'


class FakeShell:
    """Answers like ShellClient.selection_info; never holds any text."""

    def __init__(self, **selections):
        self.selections = selections
        self.calls = []

    def selection_info(self, name):
        self.calls.append(name)
        value = self.selections.get(name, dict(state='unknown', age_s=None, change_window=None,
                                               focus_window=GMAIL, same_app=False))
        if isinstance(value, Exception):
            raise value
        return value


def info(age=2.0, change=GMAIL, focus=GMAIL, same_app=True, state='set'):
    return dict(state=state, age_s=age, change_window=change, focus_window=focus, same_app=same_app)


def x_probe(primary_targets=('UTF8_STRING', 'TARGETS'), clipboard=True, clipboard_targets=('UTF8_STRING',)):
    """What the Xwayland mirror answers: owner and TARGETS, never a TIMESTAMP (D4)."""
    meta = {'focus': 2097155,
            'primary': dict(owner=True, targets=list(primary_targets), age_s=None, same_client=True),
            'clipboard': dict(owner=clipboard, targets=list(clipboard_targets), age_s=None,
                              same_client=False) if clipboard else dict(owner=False)}
    return meta


def runner(meta, texts=None):
    texts = texts or {'primary': MAIL, 'clipboard': COPIED}
    reads = []

    def run(argv, **kw):
        if '--probe' in argv:
            return types.SimpleNamespace(returncode=0, stdout=json.dumps(meta).encode())
        selection = argv[argv.index('-selection') + 1]
        reads.append(selection)
        return types.SimpleNamespace(returncode=0, stdout=texts[selection].encode())
    run.reads = reads
    return run


class ShellClientTests(unittest.TestCase):
    def test_gdbus_text_parsing_keeps_strings_intact(self):
        parse = w.parse_gvariant_text
        self.assertEqual(parse("('set', 2.5, '42', '42', true)"), ('set', 2.5, '42', '42', True))
        self.assertEqual(parse("('unknown', -1.0, '', '42', false)"), ('unknown', -1.0, '', '42', False))
        self.assertEqual(parse('("it\'s true", \'x\')'), ("it's true", 'x'))
        self.assertEqual(parse("('a\\'false', true)"), ("a'false", True))
        self.assertEqual(parse('(true,)'), (True,))
        self.assertEqual(parse('()'), ())

    def test_selection_info_allowlist_and_shape(self):
        call = Mock(return_value=('set', 2.4567, '42', '42', True))
        client = w.ShellClient(call)
        self.assertEqual(client.selection_info('primary'), info(age=2.457))
        call.assert_called_once_with('SelectionInfo', 'primary')
        call.return_value = ('unknown', -1.0, '', '', False)
        self.assertEqual(client.selection_info('clipboard'),
                         dict(state='unknown', age_s=None, change_window=None, focus_window=None, same_app=False))
        for bad in ('PRIMARY', 'dnd', ''):
            with self.assertRaises(ValueError):
                client.selection_info(bad)
        call.return_value = ('weird', 1.0, '', '', False)
        with self.assertRaises(ValueError):
            client.selection_info('primary')
        call.reset_mock()
        client.badge(4312.7)
        call.assert_called_once_with('ShowBadge', '4312')
        client.badge(-5)
        self.assertEqual(call.call_args.args, ('ShowBadge', '0'))


class WaylandCaptureTests(unittest.TestCase):
    """Age and focus rule for PRIMARY under Wayland; CLIPBOARD as before plus the shell age."""

    def capture(self, shell, meta=None, texts=None):
        run = runner(meta or x_probe(), texts)
        return cc.capture(run, shell=shell), run.reads

    def test_live_failure_without_shell_is_no_timestamp(self):
        # The X11 rule alone, on the mutter mirror: what the live journal showed.
        meta = x_probe(clipboard_targets=('UTF8_STRING', cc.COPYQ_OWNER_TARGET))
        ctx = cc.capture(runner(meta))
        self.assertEqual((ctx.source, ctx.reasons),
                         ('none', {'primary': 'no_timestamp', 'clipboard': 'copyq_owner'}))
        self.assertNotIn('backend', ctx.journal())

    def test_fresh_selection_in_focused_program_is_context(self):
        ctx, reads = self.capture(FakeShell(primary=info(age=3.2)))
        self.assertEqual((ctx.text, ctx.source, ctx.age_s), (MAIL, 'primary', 3.2))
        self.assertEqual((ctx.focus_window, ctx.owner_same_client), (GMAIL, True))
        self.assertEqual(reads, ['primary'])
        journal = json.dumps(ctx.journal())
        self.assertEqual(ctx.journal()['backend'], 'wayland')
        self.assertNotIn('levering', journal)

    def test_other_window_of_same_program_is_context_but_not_same_window(self):
        ctx, _ = self.capture(FakeShell(primary=info(change=GMAIL, focus=POPOUT, same_app=True)))
        self.assertEqual(ctx.source, 'primary')
        self.assertEqual((ctx.focus_window, ctx.owner_same_client), (POPOUT, False))

    def test_refusals(self):
        copyq = x_probe(primary_targets=('UTF8_STRING', cc.COPYQ_OWNER_TARGET), clipboard=False)
        secret = x_probe(primary_targets=('UTF8_STRING', 'x-kde-passwordManagerHint'), clipboard=False)
        cases = [
            ('too_old', info(age=cc.PRIMARY_MAX_AGE_S + 0.5), None),
            ('focus_elsewhere', info(change=VIEWER, same_app=False), None),
            ('focus_elsewhere', info(change='', same_app=False), None),  # no focus at the change
            ('no_change_seen', info(state='unknown', age=None, change=None, same_app=False), None),
            ('empty', info(state='cleared', same_app=False), None),
            ('shell_unavailable', OSError('extension missing'), None),
            ('shell_unavailable', ValueError('old extension signature'), None),
            ('copyq_owner', info(), copyq),
            ('secret_hint', info(), secret),
        ]
        for reason, answer, meta in cases:
            with self.subTest(reason=reason):
                meta = meta or x_probe(clipboard=False)
                ctx, reads = self.capture(FakeShell(primary=answer), meta)
                self.assertEqual(ctx.source, 'none')
                self.assertEqual(ctx.reasons['primary'], reason)
                self.assertNotIn('primary', reads)  # refused on metadata, text never read

    def test_old_selection_is_refused_even_in_same_window(self):
        ctx, reads = self.capture(FakeShell(primary=info(age=35.0)), x_probe(clipboard=False))
        self.assertEqual((ctx.source, ctx.reasons['primary']), ('none', 'too_old'))
        self.assertEqual(reads, [])

    def test_clipboard_fallback_uses_shell_age_without_focus_rule(self):
        shell = FakeShell(primary=info(change=VIEWER, same_app=False),
                          clipboard=info(age=40.0, change=VIEWER, same_app=False))
        ctx, reads = self.capture(shell)
        self.assertEqual((ctx.source, ctx.text, ctx.age_s), ('clipboard', COPIED, 40.0))
        self.assertEqual(ctx.reasons, {'primary': 'focus_elsewhere'})
        self.assertEqual(reads, ['clipboard'])
        shell.selections['clipboard'] = info(age=cc.CLIPBOARD_MAX_AGE_S + 1)
        ctx, _ = self.capture(shell)
        self.assertEqual(ctx.reasons, {'primary': 'focus_elsewhere', 'clipboard': 'too_old'})
        copyq = x_probe(clipboard_targets=('UTF8_STRING', cc.COPYQ_OWNER_TARGET))
        shell.selections['clipboard'] = info(age=1.0)
        ctx, _ = self.capture(shell, copyq)
        self.assertEqual(ctx.reasons['clipboard'], 'copyq_owner')

    def test_text_checks_still_apply(self):
        ctx, _ = self.capture(FakeShell(primary=info()), x_probe(clipboard=False),
                              texts={'primary': 'kort', 'clipboard': ''})
        self.assertEqual(ctx.reasons['primary'], 'too_short')

    def test_configured_ages_are_honoured(self):
        run = runner(x_probe(clipboard=False))
        ctx = cc.capture(run, max_ages=(5, 60), shell=FakeShell(primary=info(age=6.0)))
        self.assertEqual(ctx.reasons['primary'], 'too_old')


class WaylandTraceTests(unittest.TestCase):
    """The Ctrl+A trace from evdev (be/us keymap) into plan_release, with shell window ids."""

    def setUp(self):
        self.now = [100.0]
        ns = listeners()
        self.tap = ns['MultiTapKeyListener'](Mock(), Mock(), key=w.Key.ctrl_r,
                                             default_language='nl', shift_language='en')
        self.focus = [GMAIL]
        self.trace = sr.InputTrace(focus=lambda: self.focus[0], clock=lambda: self.now[0],
                                   spawn=lambda fn: fn())
        self.trace.set_known(keys=True, mouse=True)
        self.tap.trace = self.trace
        self.hub = w.InputHub(provider=lambda: {}, selector=Mock(), layout='be')
        self.hub.key_listener = self.tap
        self.hub.mouse_callbacks.append(lambda x, y, b, pressed: pressed and self.trace.mouse_press())

    def ev(self, code, value=1, device='kbd'):
        self.hub.event(device, types.SimpleNamespace(type=1, code=code, value=value))

    def tap_key(self, code, device='kbd'):
        self.ev(code, 1, device)
        self.ev(code, 0, device)

    def ctrl_a(self, a_code=16):
        self.ev(29)
        self.tap_key(a_code)
        self.ev(29, 0)

    def rewrite_chord(self):
        self.ev(56)            # Left-Alt held
        self.tap_key(97)
        self.tap_key(97)       # double tap Right-Ctrl

    def start(self, age, **kw):
        base = dict(now=self.now[0], context_source='primary', primary_age_s=age, focus_window=GMAIL,
                    focus_is_terminal=False, owner_same_client=True, probe_done=self.now[0] + 0.1)
        base.update(kw)
        return sr.StartState(**base)

    def test_ctrl_a_then_start_plans_release(self):
        self.ctrl_a()
        self.now[0] = 102.0
        self.rewrite_chord()
        snap = self.trace.snapshot()
        self.assertEqual((snap.select_all_at, snap.select_all_window, snap.last_other_key_at),
                         (100.0, GMAIL, None))
        self.assertEqual(sr.plan_release(snap, self.start(age=1.98)), (True, 'ok'))

    def test_us_layout_uses_key_a(self):
        self.hub.layout = 'us'
        self.ctrl_a(a_code=16)       # physical Q on us is q: not select-all
        self.assertIsNone(self.trace.snapshot().select_all_at)
        self.ctrl_a(a_code=30)
        self.assertEqual(self.trace.snapshot().select_all_at, 100.0)

    def test_mouse_selection_and_touchpad_are_left_alone(self):
        for code in (272, 330):  # BTN_LEFT, BTN_TOUCH (tap-to-click)
            with self.subTest(code=code):
                self.setUp()
                self.ctrl_a()
                self.now[0] = 101.0
                self.tap_key(code, device='mouse')
                self.now[0] = 102.0
                self.assertEqual(sr.plan_release(self.trace.snapshot(), self.start(age=2.0)),
                                 (False, 'mouse_after_select'))

    def test_mouse_selection_without_ctrl_a(self):
        self.tap_key(272, device='mouse')
        self.now[0] = 102.0
        self.assertEqual(sr.plan_release(self.trace.snapshot(), self.start(age=1.0)),
                         (False, 'no_select_all'))

    def test_old_or_foreign_selection_is_refused(self):
        self.ctrl_a()
        self.now[0] = 102.0
        snap = self.trace.snapshot()
        # PRIMARY changed long before the Ctrl+A (old selection kept, Ctrl+A did nothing).
        self.assertEqual(sr.plan_release(snap, self.start(age=30.0)), (False, 'selection_not_from_select_all'))
        # PRIMARY changed after the Ctrl+A (something else selected since).
        self.assertEqual(sr.plan_release(snap, self.start(age=0.5)), (False, 'selection_not_from_select_all'))
        # PRIMARY changed while another window had focus.
        self.assertEqual(sr.plan_release(snap, self.start(age=2.0, owner_same_client=False)),
                         (False, 'owner_elsewhere'))
        self.assertEqual(sr.plan_release(snap, self.start(age=2.0, owner_same_client=None)),
                         (False, 'owner_elsewhere'))
        # Focus is another window now.
        self.assertEqual(sr.plan_release(snap, self.start(age=2.0, focus_window=POPOUT)),
                         (False, 'focus_moved'))
        # Ctrl+A too long ago.
        self.now[0] = 100.0 + sr.SELECT_ALL_MAX_AGE_S + 1
        self.assertEqual(sr.plan_release(self.trace.snapshot(), self.start(age=21.0)),
                         (False, 'select_all_too_old'))

    def test_typing_after_ctrl_a_and_terminal(self):
        self.ctrl_a()
        self.tap_key(107)  # End
        self.now[0] = 102.0
        self.assertEqual(sr.plan_release(self.trace.snapshot(), self.start(age=2.0)),
                         (False, 'key_after_select'))
        self.setUp()
        self.ctrl_a()
        self.now[0] = 102.0
        self.assertEqual(sr.plan_release(self.trace.snapshot(), self.start(age=2.0, focus_is_terminal=True)),
                         (False, 'terminal'))

    def test_unknown_focus_at_ctrl_a_fails_closed(self):
        def broken():
            raise OSError('extension missing')
        self.trace._focus = broken
        self.ctrl_a()
        self.now[0] = 102.0
        self.assertEqual(sr.plan_release(self.trace.snapshot(), self.start(age=2.0)), (False, 'focus_moved'))

    def test_wait_uses_shell_focus(self):
        self.ctrl_a()
        sent = []
        result = sr.wait_and_release(self.trace, 102.0, GMAIL, lambda: False, lambda: sent.append(1),
                                     focus=lambda: POPOUT, clock=lambda: 102.1, sleep=lambda s: None)
        self.assertEqual((result['fired'], result['reason'], sent), (False, 'focus_moved_after_start', []))
        result = sr.wait_and_release(self.trace, 102.0, GMAIL, lambda: False, lambda: sent.append(1),
                                     focus=lambda: GMAIL, clock=lambda: 102.1, sleep=lambda s: None)
        self.assertEqual((result['fired'], sent), (True, [1]))


class FakeKeyListener:
    def __init__(self, held=lambda: False):
        self.modifiers_held = held


class WaylandAppTests(AppStart, unittest.TestCase):
    """App.start under Wayland: shell-based capture, release via ctrl_home, shell badge."""

    def run_start(self, ctx, setup=None, wm_class='google-chrome', badge_error=None, release=True):
        seen = {}

        def capture(**kw):
            seen.update(kw)
            return ctx
        app, events = self.start_app(config(), None, capture=capture)
        logic = app.start.__globals__
        logic.update(WAYLAND=True, RELEASE_SELECTION=release, CONTEXT_BADGE=True)
        logic['context_badge'].calls.clear()
        shell = object()
        app.replayer.shell = shell
        sent = threading.Event()
        app.replayer.send_ctrl_home = lambda: (events.append(('ctrl_home',)), sent.set())
        app.replayer._get_active_window_class = lambda: wm_class
        app.replayer.TERMINAL_CLASSES = {'kitty', 'org.gnome.terminal'}
        app.replayer.focus_id = lambda: GMAIL

        def show_badge(chars):
            if badge_error:
                raise badge_error
            events.append(('shell_badge', chars))
        app.replayer.show_badge = show_badge
        app.input_trace = sr.InputTrace(focus=app._focus_reader(), spawn=lambda fn: fn())
        app.input_trace.set_known(keys=True, mouse=True)
        app.key_listener = FakeKeyListener()
        if setup:
            setup(app.input_trace)
        out = io.StringIO()
        with patch.object(rewrite, 'http_call', lambda *a: llm_answer('Een korte herschreven bevestiging.')), \
                patch.object(sr, 'focus_window', Mock(side_effect=AssertionError('X focus used'))), \
                contextlib.redirect_stdout(out):
            app.start('nl', True, {'alt': True})
            session = app.session
            sent.wait(1)
            deadline = time.monotonic() + 2
            while ('"selection_release"' not in out.getvalue() or
                   (ctx.text and '"context_badge"' not in out.getvalue())) and time.monotonic() < deadline:
                time.sleep(0.01)
            app.stop()
            self.assertTrue(session.done.wait(5))
        lines = [json.loads(l.removeprefix('DCT-COVERAGE ')) for l in out.getvalue().splitlines()
                 if l.startswith('DCT-COVERAGE ')]
        record = next(l for l in lines if l['kind'] == 'selection_release')
        badges = [l for l in lines if l['kind'] == 'context_badge']
        return events, record, badges, seen, logic['context_badge'].calls, out.getvalue()

    def ctx(self, age=0.0, same=True, source='primary'):
        return cc.Context(MAIL, source, len(MAIL), age, focus_window=GMAIL, owner_same_client=same,
                          backend='wayland')

    def test_ctrl_a_selection_released_through_shell_and_badge_shown(self):
        events, record, badges, seen, x11_badge, out = self.run_start(self.ctx(), lambda tr: tr.select_all())
        self.assertIn('shell', seen)
        self.assertEqual((record['fired'], record['reason']), (True, 'ok'))
        self.assertEqual(events.count(('ctrl_home',)), 1)
        paste = [i for i, e in enumerate(events) if e[0] == 'paste'][0]
        self.assertLess(events.index(('ctrl_home',)), paste)
        self.assertIn(('shell_badge', len(MAIL)), events)
        self.assertEqual(x11_badge, [])
        self.assertEqual(badges, [dict(badges[0], started=True, backend='shell')])
        self.assertNotIn('levering', out.split('Een korte')[0])

    def test_refusals_send_nothing(self):
        def mouse_after(tr):
            tr.select_all()
            tr.mouse_press()
        cases = [
            ('mouse_after_select', dict(setup=mouse_after)),
            ('no_select_all', dict()),
            ('terminal', dict(setup=lambda tr: tr.select_all(), wm_class='org.gnome.terminal')),
            ('owner_elsewhere', dict(ctx=self.ctx(same=False), setup=lambda tr: tr.select_all())),
            ('selection_not_from_select_all', dict(ctx=self.ctx(age=25.0), setup=lambda tr: tr.select_all())),
            ('context_not_primary', dict(ctx=self.ctx(source='clipboard'), setup=lambda tr: tr.select_all())),
            ('disabled', dict(release=False, setup=lambda tr: tr.select_all())),
        ]
        for reason, kw in cases:
            with self.subTest(reason=reason):
                ctx = kw.pop('ctx', self.ctx())
                events, record, _, _, _, _ = self.run_start(ctx, **kw)
                self.assertEqual((record['fired'], record['reason']), (False, reason))
                self.assertNotIn(('ctrl_home',), events)

    def test_badge_failure_is_journaled_only(self):
        events, record, badges, _, _, _ = self.run_start(self.ctx(), badge_error=OSError('no extension'))
        self.assertEqual(badges[0]['started'], False)
        self.assertEqual(badges[0]['error'], 'OSError')
        self.assertTrue(any(e[0] == 'paste' for e in events))

    def test_no_context_no_badge(self):
        events, _, badges, _, _, _ = self.run_start(cc.Context(None, reasons={'primary': 'too_old'},
                                                               backend='wayland'))
        self.assertEqual(badges, [])
        self.assertFalse([e for e in events if e[0] == 'shell_badge'])


class WaylandMouseTraceTests(unittest.TestCase):
    def test_always_on_mouse_trace_registers_on_the_hub(self):
        ns = listeners()
        _, _, mouse_cls = w.bind(ns['KeyboardReplayer'], ns['MultiTapKeyListener'], ns['MouseToggleListener'])
        trace = sr.InputTrace(focus=lambda: GMAIL, spawn=lambda fn: fn())
        callbacks = []
        mouse_cls.add_press_callback(callbacks.append)
        self.assertEqual(len(callbacks), 0)  # registered on the hub, not called
        # Through the real method of App (AST-loaded), with the bound class.
        import ast
        tree = ast.parse((ROOT / 'dictation.py').read_text())
        app = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'App')
        method = next(n for n in app.body if isinstance(n, ast.FunctionDef)
                      and n.name == '_start_input_mouse_listener')
        hub_callbacks = []
        fake_cls = types.SimpleNamespace(add_press_callback=hub_callbacks.append)
        g = dict(WAYLAND=True, MouseToggleListener=fake_cls, diagnostic=lambda *a, **k: None,
                 mouse=None)
        exec(compile(ast.Module(body=[method], type_ignores=[]), 'dictation.py', 'exec'), g)
        holder = types.SimpleNamespace(input_trace=trace)
        g['_start_input_mouse_listener'](holder)
        self.assertTrue(trace.snapshot().mouse_known)
        hub_callbacks[0](0, 0, None, True)
        self.assertIsNotNone(trace.snapshot().last_mouse_press_at)


class SourceTests(unittest.TestCase):
    def test_extension_returns_no_selection_content(self):
        source = (ROOT / 'gnome-extension/dictation-helper@faster-whisper-dictation/extension.js').read_text()
        for forbidden in ('get_text', 'get_content', 'transfer_async', 'read_bytes', 'get_mimetypes',
                          'get_title'):
            self.assertNotIn(forbidden, source)
        self.assertNotIn('/' + 'home/', (ROOT / 'tests/test_wayland_context.py').read_text())


if __name__ == '__main__':
    unittest.main()
