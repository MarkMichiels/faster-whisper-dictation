"""Offline paste-boundary tests: no audio, desktop, clipboard or API access."""
import ast
import contextlib
import io
import re
import types
import unittest
from pathlib import Path
from unittest.mock import Mock

SOURCE = Path(__file__).resolve().parents[1] / 'dictation.py'
SEPARATORS = '\r\n\v\f\x1c\x1d\x1e\x85\u2028\u2029'


class NoEnterTests(unittest.TestCase):
    def setUp(self):
        self.tree = ast.parse(SOURCE.read_text())
        node = next(n for n in self.tree.body
                    if isinstance(n, ast.ClassDef) and n.name == 'KeyboardReplayer')
        self.events = []
        self.ns = dict(re=re, diagnostic=lambda event, **meta: self.events.append((event, meta)),
                       keyboard=types.SimpleNamespace(Key=types.SimpleNamespace(
                           ctrl='CTRL', shift='SHIFT')))
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), 'exec'), self.ns)
        self.cls = self.ns['KeyboardReplayer']

    def replayer(self, window='editor', clipboard_ok=True, focus_ok=True):
        obj = self.cls.__new__(self.cls)
        obj.kb = Mock()
        obj._restore_target_window = Mock(return_value=focus_ok)
        obj._get_active_window_class = Mock(return_value=window)
        obj._set_clipboard = Mock(return_value=dict(ok=clipboard_ok, path='fake'))
        return obj

    def paste(self, obj, text, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return obj.type_text(text, **kwargs)

    def test_ordinary_all_separators_every_window(self):
        for window in ['editor', '', *sorted(self.cls.TERMINAL_CLASSES)]:
            for separator in list(SEPARATORS) + ['\r\n', '\n\n', ' \r\n \t\u2028 ']:
                with self.subTest(window=window, separator=repr(separator)):
                    obj = self.replayer(window)
                    self.paste(obj, 'first' + separator + 'second')
                    obj._set_clipboard.assert_called_once_with('first second')

    def test_rewrite_terminal_is_single_line(self):
        for window in self.cls.TERMINAL_CLASSES:
            obj = self.replayer(window)
            self.paste(obj, 'first' + SEPARATORS + 'second', rewrite_mode=True, echo=False)
            obj._set_clipboard.assert_called_once_with('first second')

    def test_rewrite_gui_preserves_paragraphs_and_normalizes_cr(self):
        obj = self.replayer()
        self.paste(obj, 'Dear reader,\r\n\r\nBody\rEnd\n\nCheck',
                   rewrite_mode=True, echo=False)
        obj._set_clipboard.assert_called_once_with('Dear reader,\n\nBody\nEnd\n\nCheck')

    def test_default_is_safe_for_correction_and_check_lines(self):
        for text in [' correction\ncontinued', '\n\n[check]\r\nchecked']:
            obj = self.replayer()
            self.paste(obj, text)
            pasted = obj._set_clipboard.call_args.args[0]
            self.assertFalse(any(c in SEPARATORS for c in pasted))

    def test_new_journal_event_is_metadata_only(self):
        obj = self.replayer('kitty')
        self.paste(obj, 'private\r\n\nwords', rewrite_mode=True, echo=False)
        self.assertEqual(self.events, [('paste_line_safety', dict(
            flattened_line_breaks=3, terminal=True))])
        obj = self.replayer()
        self.paste(obj, 'private\rwords', rewrite_mode=True, echo=False)
        self.assertEqual(self.events[-1], ('paste_line_safety', dict(
            flattened_line_breaks=0, terminal=False)))

    def test_only_paste_shortcut_keys_even_on_failure(self):
        for terminal in (False, True):
            for rewrite_mode in (False, True):
                for clipboard_ok, focus_ok in [(True, True), (False, True), (True, False)]:
                    obj = self.replayer('kitty' if terminal else 'editor', clipboard_ok, focus_ok)
                    self.paste(obj, 'one\n\rtwo', rewrite_mode=rewrite_mode)
                    expected = [('press', 'CTRL')]
                    if terminal:
                        expected.append(('press', 'SHIFT'))
                    expected += [('press', 'v'), ('release', 'v')]
                    if terminal:
                        expected.append(('release', 'SHIFT'))
                    expected.append(('release', 'CTRL'))
                    actual = [(call[0], call.args[0]) for call in obj.kb.mock_calls]
                    self.assertEqual(actual, expected if clipboard_ok and focus_ok else [])
        obj = self.replayer()
        obj.kb.press.side_effect = RuntimeError('injection failed')
        self.assertFalse(self.paste(obj, 'one\ntwo')['ok'])
        obj.kb.type.assert_not_called()

    def test_no_other_synthetic_key_paths_in_source(self):
        # A new typing backend or dynamic key must explicitly revisit this invariant.
        calls = [n for n in ast.walk(self.tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute)
                 and isinstance(n.func.value, ast.Attribute) and n.func.value.attr == 'kb']
        self.assertTrue(calls)
        for call in calls:
            self.assertIn(call.func.attr, ('press', 'release'))
            key = call.args[0]
            if isinstance(key, ast.Constant):
                self.assertEqual(key.value, 'v')
            else:
                self.assertIsInstance(key, ast.Attribute)
                self.assertIn(key.attr, ('ctrl', 'shift'))
        self.assertFalse(hasattr(self.cls, '_type_via_pynput'))


if __name__ == '__main__':
    unittest.main()
