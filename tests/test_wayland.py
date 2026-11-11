"""Offline kernel-path proof: fake evdev streams and D-Bus, no desktop IO."""
import ast
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import types
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import wayland_io as w
import selection_release


def listeners():
    tree = ast.parse((ROOT / 'dictation.py').read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and
             n.name in ('MultiTapKeyListener', 'MouseToggleListener', 'KeyboardReplayer')]
    import time
    import re
    ns = dict(keyboard=w.keyboard, mouse=w.mouse, time=time, re=re,
              diagnostic=lambda *args, **kwargs: None,
              selection_release=selection_release)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'dictation.py', 'exec'), ns)
    return ns


class FakeDevice:
    def __init__(self):
        self.closed = False
    def capabilities(self):
        return {1: [29, 97]}
    def close(self):
        self.closed = True


class WaylandTests(unittest.TestCase):
    def setUp(self):
        self.ns = listeners()
        self.start, self.stop = Mock(), Mock()
        self.tap = self.ns['MultiTapKeyListener'](self.start, self.stop,
            key=w.Key.ctrl_r, default_language='nl', shift_language='en')
        self.hub = w.InputHub(provider=lambda: {}, selector=Mock(), layout='be')
        self.hub.key_listener = self.tap

    def key(self, code, value=1, device='keyboard'):
        self.hub.event(device, types.SimpleNamespace(type=1, code=code, value=value))

    def test_double_tap_modifiers_and_repeat(self):
        for code in (42, 56, 29):
            self.key(code)
        self.key(97)
        self.key(97, 2)
        self.key(97, 0)
        self.key(97)
        self.stop.assert_called_once()
        language, rewrite, info = self.start.call_args.args
        self.assertEqual((language, rewrite), ('en', True))
        self.assertTrue(all(info[k] for k in ('shift', 'alt', 'ctrl')))
        self.assertNotIn('text', info)
        for code in (42, 56, 29, 97):
            self.key(code, 0)
        self.assertFalse(self.tap.modifiers_held())

    def test_right_modifiers_do_not_select_mode(self):
        self.key(54)
        self.key(100)
        self.key(97)
        self.key(97, 0)
        self.key(97)
        self.assertEqual(self.start.call_args.args[:2], ('nl', False))

    def test_ctrl_a_be_and_us(self):
        trace = Mock()
        self.tap.trace = trace
        self.key(29)
        self.key(16)  # physical Q -> a on Belgian AZERTY
        trace.select_all.assert_called_once()
        self.key(30)  # physical A -> q, must not match
        trace.select_all.assert_called_once()
        us = w.InputHub(provider=lambda: {}, selector=Mock(), layout='us')
        self.assertEqual(us.key(30).char, 'a')
        self.assertIsNone(us.key(16).char)
        with self.assertRaises(ValueError):
            w.InputHub(layout='unknown')

    def test_mouse_buttons_coalesce(self):
        toggle = Mock()
        listener = self.ns['MouseToggleListener'](toggle, ['x1', 'x2'])
        self.hub.mouse_callbacks.append(listener.on_click)
        self.key(275, device='mouse')
        self.key(276, device='mouse')
        toggle.assert_called_once()
        self.key(275, 0, 'mouse')
        toggle.assert_called_once()

    def test_multi_device_modifiers_and_unplug(self):
        self.key(29, device='one')
        self.key(29, device='two')
        self.hub.release('one')
        self.assertTrue(self.tap.ctrl_held)
        self.hub.release('two')
        self.assertFalse(self.tap.ctrl_held)
        self.key(97)
        self.hub.release('keyboard')
        self.assertFalse(self.tap.main_down)
        self.key(97)
        self.start.assert_not_called()

    def test_hotplug_and_cleanup(self):
        first, second = FakeDevice(), FakeDevice()
        provider = Mock(return_value={'one': lambda: first})
        hub = w.InputHub(provider=provider, selector=Mock())
        hub.rescan()
        self.assertIn('one', hub.devices)
        provider.return_value = {}
        hub.rescan()  # no keyboard temporarily is not fatal after startup
        self.assertTrue(first.closed)
        provider.return_value = {'two': lambda: second}
        hub.rescan()
        self.assertIn('two', hub.devices)
        hub.running = False
        hub.run()
        self.assertTrue(second.closed)
        hub.selector.close.assert_called_once()

    def test_no_readable_device_fails_clearly(self):
        with self.assertRaisesRegex(RuntimeError, 'input group'):
            self.hub.rescan()

    def test_syn_dropped_never_creates_tap(self):
        self.key(97)
        self.hub.event('keyboard', types.SimpleNamespace(type=0, code=3, value=0))
        self.key(97)
        self.hub.event('keyboard', types.SimpleNamespace(type=0, code=0, value=0))
        self.key(97)
        self.start.assert_not_called()

    def test_dbus_allowlist_and_focus(self):
        call = Mock(side_effect=[('42', 'Kitty'), (True,), ()])
        client = w.ShellClient(call)
        self.assertEqual(client.focus(), ('42', 'kitty'))
        self.assertTrue(client.activate('42'))
        client.shortcut('paste_terminal')
        with self.assertRaises(ValueError):
            client.shortcut('Return')
        self.assertEqual(call.call_count, 3)
        with patch.object(w.subprocess, 'check_output', return_value=b'(true,)'):
            self.assertTrue(w.ShellClient().activate('42'))
        with patch.object(w.subprocess, 'check_output', return_value=b"('42', 'editor')"):
            self.assertEqual(w.ShellClient().focus(), ('42', 'editor'))

    def test_wayland_replayer_never_constructs_x_controller(self):
        cls, _, _ = w.bind(self.ns['KeyboardReplayer'],
            self.ns['MultiTapKeyListener'], self.ns['MouseToggleListener'])
        with patch.object(w, 'ShellClient') as shell:
            replay = cls()
            replay.target_wid = '42'
            shell.return_value.focus.return_value = ('43', 'editor')
            with self.assertRaises(RuntimeError):
                replay._send_paste(False)
            shell.return_value.shortcut.assert_not_called()
            shell.return_value.focus.return_value = ('42', 'kitty')
            replay._send_paste(True)
            shell.return_value.shortcut.assert_called_once_with('paste_terminal')

    def test_delivery_clipboard_policy_and_missing_shell(self):
        cls, _, _ = w.bind(self.ns['KeyboardReplayer'],
            self.ns['MultiTapKeyListener'], self.ns['MouseToggleListener'])
        with patch.object(w, 'ShellClient') as shell:
            replay = cls()
            replay.target_wid = '42'
            replay._set_clipboard = Mock(return_value=dict(ok=True, path='copyq'))
            replay._clipboard_owned = Mock(return_value=True)
            replay._delivery_alert = Mock()
            shell.return_value.focus.return_value = ('42', 'kitty')
            shell.return_value.activate.return_value = True
            result = replay.type_text('first\nsecond', echo=False)
            replay._set_clipboard.assert_called_once_with('first second')
            shell.return_value.shortcut.assert_called_once_with('paste_terminal')
            self.assertTrue(result['ok'])
            shell.return_value.reset_mock()
            shell.return_value.focus.side_effect = OSError('not running')
            shell.return_value.activate.side_effect = OSError('not running')
            replay._set_clipboard.return_value = dict(ok=True, path='copyq')
            result = replay.type_text('recoverable', echo=False)
            self.assertFalse(result['ok'])
            self.assertFalse(result['focus_ok'])
            replay._set_clipboard.assert_called_with('recoverable')
            shell.return_value.shortcut.assert_not_called()

    def test_xclip_fallback_target_per_backend(self):
        """Wayland apps paste UTF8_STRING through Xwayland; a bare text/plain stays X11-only."""
        base = self.ns['KeyboardReplayer']
        cls, _, _ = w.bind(base, self.ns['MultiTapKeyListener'], self.ns['MouseToggleListener'])
        calls = []

        def run(argv, **kw):
            calls.append(argv)
            return types.SimpleNamespace(returncode=1 if argv[0] == 'copyq' else 0)
        fake = types.SimpleNamespace(run=run, DEVNULL=subprocess.DEVNULL,
                                     check_output=lambda argv, **kw: calls.append(argv) or b'tekst')
        self.ns.update(subprocess=fake)
        with patch.object(w, 'ShellClient'):
            for replayer, target in ((base.__new__(base), 'text/plain'), (cls(), 'UTF8_STRING')):
                calls.clear()
                result = replayer._set_clipboard('tekst')
                self.assertEqual((result['ok'], result['path']), (True, 'xclip'))
                self.assertEqual(calls[-1], ['xclip', '-selection', 'clipboard', '-t', target])
                self.assertTrue(replayer._clipboard_owned('tekst', None, 'xclip', None))
                self.assertEqual(calls[-1], ['xclip', '-selection', 'clipboard', '-o', '-t', target])

    def test_x11_import_route_and_wayland_no_pynput(self):
        tree = ast.parse((ROOT / 'dictation.py').read_text())
        nodes = [n for n in tree.body if isinstance(n, ast.AnnAssign)]
        route = next(n for n in tree.body if isinstance(n, ast.If) and
                     isinstance(n.test, ast.Name) and n.test.id == 'WAYLAND')
        code = compile(ast.Module(body=nodes + [route], type_ignores=[]), 'route', 'exec')
        fake = types.ModuleType('pynput')
        fake.keyboard, fake.mouse = object(), object()
        with patch.dict(sys.modules, pynput=fake):
            for session in ('x11', 'wayland'):
                ns = {'os': types.SimpleNamespace(environ={'XDG_SESSION_TYPE': session})}
                exec(code, ns)
                self.assertIs(ns['keyboard'], w.keyboard if session == 'wayland' else fake.keyboard)

    def test_installer_wayland_group_gate_is_read_only(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            commands = root / 'commands'
            commands.mkdir()
            identity = commands / 'id'
            identity.write_text('#!/bin/sh\necho users\n')
            identity.chmod(0o755)
            env = dict(os.environ, HOME=str(root / 'home'),
                       XDG_DATA_HOME=str(root / 'data'), XDG_SESSION_TYPE='x11',
                       PATH=str(commands) + ':/usr/bin:/bin')
            result = subprocess.run(['bash', str(ROOT / 'install.sh'), '--check', '--wayland'],
                                    env=env, capture_output=True, text=True, timeout=5)
            self.assertEqual(result.returncode, 1)
            self.assertIn('input group', result.stderr)
            self.assertFalse((root / 'home').exists())
            self.assertFalse((root / 'data').exists())
            identity.write_text('#!/bin/sh\necho users input\n')
            result = subprocess.run(['bash', str(ROOT / 'install.sh'), '--check', '--wayland'],
                                    env=env, capture_output=True, text=True, timeout=5)
            self.assertEqual(result.returncode, 1)
            self.assertIn('Helper is missing', result.stderr)
            self.assertFalse((root / 'home').exists())
            self.assertFalse((root / 'data').exists())

    def test_wayland_unit_has_no_xinput_and_x11_bytes_unchanged(self):
        source = (ROOT / 'install.sh').read_text()
        funcs = source[source.index('detect_ld_path() {'):source.index('unit_is_current() {')]
        env = dict(os.environ, SCRIPT_DIR='/sandbox', VENV_DIR='/sandbox/venv',
                   LD_PATH='', REVISION='test')
        def render(wayland, function):
            return subprocess.check_output(['bash', '-c', 'set -eu\n' + funcs + '\n' + function],
                                           env=dict(env, WAYLAND=wayland))
        self.assertEqual(render('false', 'render_service'), render('false', 'render_service_base'))
        unit = render('true', 'render_service').decode()
        self.assertNotIn('ExecStartPre=', unit)
        self.assertNotIn('ExecStartPost=', unit)
        self.assertIn('Environment=XDG_SESSION_TYPE=wayland', unit)


if __name__ == '__main__':
    unittest.main()
