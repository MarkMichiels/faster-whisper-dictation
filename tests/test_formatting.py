"""Formatting choices, renderer and clipboard race tests: entirely offline.

Never import the desktop application. Every subprocess and synthetic key is
replaced before a method can execute. No paid model calls.
"""
import contextlib
import io
import itertools
import json
import subprocess
import types
import unittest
from html.parser import HTMLParser
from unittest.mock import Mock, patch

import rewrite
import rich_text
import test_no_enter as no_enter
import test_rewrite as rewrite_tests
from test_no_enter import SEPARATORS
from test_rewrite import (listener_class, key, CTRL_R, CTRL, ALT, SHIFT, META_L,
                          config, llm_answer, app_logic)


class ChoiceTests(unittest.TestCase):
    def test_complete_table_and_no_sticky_state(self):
        for ctrl, alt, shift in itertools.product((False, True), repeat=3):
            with self.subTest(ctrl=ctrl, alt=alt, shift=shift):
                starts, stops = [], []
                listener = listener_class()(lambda *a: starts.append(a), lambda: stops.append(True),
                                            key=CTRL_R, default_language='nl')
                held = ([CTRL] if ctrl else []) + ([SHIFT] if shift else []) + ([META_L] if alt and shift else [ALT] if alt else [])
                rewrite_tests.KeyTests().tap_twice(listener, held)
                language, ai, keys = starts[-1]
                self.assertEqual(language, 'en' if shift else 'nl')
                self.assertEqual(ai, ctrl or alt)
                self.assertEqual(keys['ctrl'], ctrl)
                self.assertEqual(keys['alt'], alt)
                self.assertEqual(len(stops), 1)
                rewrite_tests.KeyTests().tap_twice(listener)
                self.assertEqual(starts[-1][:2], ('nl', False))
                self.assertFalse(starts[-1][2]['ctrl'])

    def test_left_ctrl_does_not_count_and_repeat_is_not_a_tap(self):
        starts, stops = [], []
        listener = listener_class()(lambda *a: starts.append(a), lambda: stops.append(True),
                                    key=CTRL_R, default_language='nl')
        listener.on_press(CTRL)
        listener.on_press(key('ctrl_l', 65507))
        self.assertEqual(listener.tap_count, 0)
        listener.on_press(CTRL_R)
        listener.on_press(CTRL_R)
        self.assertFalse(starts)
        listener.on_release(CTRL_R)
        listener.on_release(CTRL)
        listener.on_press(CTRL_R)
        self.assertEqual(starts[-1][:2], ('nl', False))


    def test_stop_after_double_tap_is_immediate(self):
        stops = []
        listener = listener_class()(lambda *a: None, lambda: stops.append(True), key=CTRL_R)
        rewrite_tests.KeyTests().tap_twice(listener, (CTRL,))
        listener.on_press(CTRL_R)
        self.assertEqual(len(stops), 2)


class DeliveryTests(unittest.TestCase):
    def test_session_modes_and_language_snapshot(self):
        helper = rewrite_tests.EnglishRewriteTests()
        for ctrl, alt, shift in itertools.product((False, True), repeat=3):
            with self.subTest(ctrl=ctrl, alt=alt, shift=shift):
                language = 'en' if shift else 'nl'
                app, session, records, bodies = helper.run_start(
                    language, ctrl or alt, {'ctrl': ctrl, 'alt': alt, 'shift': shift})
                self.assertEqual(session.mode, 'terminal' if ctrl else 'rich' if alt else 'plain')
                self.assertEqual(session.terminal_mode, ctrl)
                self.assertEqual(session.output_language, language)
                self.assertEqual(session.language, 'nl' if ctrl or alt else language)
                if bodies:
                    self.assertEqual('limited Markdown' in bodies[0]['messages'][0]['content'], not ctrl)

    def test_single_delivery_raw_history_fallback_and_status(self):
        logic = app_logic()
        for terminal, success in itertools.product((False, True), repeat=2):
            app = logic['App'].__new__(logic['App'])
            app.replayer = Mock()
            app.replayer.add_to_history.return_value = 0
            app.replayer.type_text.return_value = dict(ok=True)
            app.beep = Mock()
            app.transcription_worker = types.SimpleNamespace(language='nl')
            session = types.SimpleNamespace(language='nl', output_language='en', context=None,
                                            target_wid='start', paste_wid='stop', terminal_mode=terminal)
            result = rewrite.RewriteResult('Final text' if success else 'Raw\ntext', success,
                                           html='<p>Final text</p>' if success and not terminal else None)
            with patch.object(rewrite, 'rewrite', return_value=result) as call, contextlib.redirect_stdout(io.StringIO()):
                app._deliver_rewrite(session, 'Raw\ntext', ['Check <literal>'])
            app.replayer.add_to_history.assert_called_once_with('Raw\ntext')
            app.replayer.type_text.assert_called_once()
            args, kwargs = app.replayer.type_text.call_args
            self.assertEqual(kwargs['terminal_mode'], terminal)
            self.assertEqual(call.call_args.kwargs['rich'], not terminal)
            self.assertFalse(kwargs['echo'])
            if success and not terminal:
                self.assertIn('<p>Check &lt;literal&gt;</p>', kwargs['html'])
            else:
                self.assertIsNone(kwargs['html'])
            self.assertEqual(app.replayer.target_wid, 'stop')


class Tags(HTMLParser):
    def handle_starttag(self, tag, attrs):
        assert tag in {'p', 'ul', 'ol', 'li', 'strong', 'br'}, tag
        assert not attrs, attrs


class RenderingTests(unittest.TestCase):
    def test_paragraphs_lists_nested_bold_and_same_visible_text(self):
        source = 'First **bold** & text.\n\n- Alpha\n  - Nested\n- Beta\n\n1. One\n2. Two'
        plain, html = rich_text.render(source)
        Tags().feed(html)
        self.assertIn('<strong>bold</strong>', html)
        self.assertIn('<ul>', html)
        self.assertIn('<ol>', html)
        self.assertIn('Nested', plain)
        self.assertNotIn('**', plain)
        self.assertNotIn('\n', rich_text.single_line(plain))

    def test_hostile_html_links_images_and_escapes(self):
        for source in ['<script>alert("x")</script>', '[label](https://example.invalid/a)',
                       '![image](https://example.invalid/tracker)', '[bad](javascript:alert(1))',
                       '<img src=x onerror=alert(1)>', '**<b>quoted & unsafe</b>**']:
            with self.subTest(source=source):
                plain, html = rich_text.render(source)
                Tags().feed(html)
                self.assertNotIn('<script', html)
                self.assertNotIn('<img', html)
                self.assertNotIn('href=', html)
                self.assertTrue(plain)

    def test_unsupported_blocks_fail_closed(self):
        for source in ['# heading', '> quote', '```html\n<script>x</script>\n```', '---', '']:
            with self.subTest(source=source), self.assertRaises(ValueError):
                rich_text.render(source)

    def test_model_modes_email_gate_and_literal_mail_parts(self):
        answer = dict(mail=True, aanhef='Dag <reader>,', tekst='One **bold** point.\n\n- Alpha\n- Beta', afsluiting='Groeten,')
        data = llm_answer('unused')
        data['choices'][0]['message']['content'] = json.dumps(answer)
        raw = 'Dit is een mail met one bold point en alpha en beta en nog woorden.'
        for rich in (True, False):
            calls = []
            def call(cfg, body, timeout):
                calls.append(body)
                return data
            out = rewrite.rewrite(raw, 'nl', None, config(signature='Local Name'), call=call, rich=rich)
            self.assertTrue(out.ok, out.reason)
            if rich:
                self.assertNotIn('**', out.text)
                self.assertFalse(any(c in SEPARATORS for c in out.text))
                self.assertTrue(out.html.startswith('<p>Dag &lt;reader&gt;,</p>'))
                self.assertIn('<p>Groeten,<br>Local Name</p>', out.html)
                self.assertIn('<strong>bold</strong>', out.html)
                Tags().feed(out.html)
                self.assertIn('limited Markdown', calls[0]['messages'][0]['content'])
            else:
                self.assertIsNone(out.html)
                self.assertIn('geen Markdown', calls[0]['messages'][0]['content'])
        out = rewrite.rewrite('one bold point alpha beta en nog wat woorden', 'nl', None,
                              config(), call=lambda *a: data, rich=True)
        self.assertTrue(out.ok)
        self.assertNotIn('Dag', out.text)
        self.assertNotIn('Groeten', out.html)

    def test_render_failure_and_visible_word_limit(self):
        out = rewrite.rewrite('enkele woorden over iets hier', 'nl', None, config(),
                              call=lambda *a: llm_answer('# Heading'), rich=True)
        self.assertFalse(out.ok)
        self.assertEqual(out.reason, 'render_failed')
        self.assertIsNone(out.html)
        out = rewrite.rewrite('Alpha beta gamma', 'nl', None, config(),
                              call=lambda *a: llm_answer('- **Alpha**\n- beta\n- gamma'), rich=True)
        self.assertTrue(out.ok, out.reason)


class ClipboardTests(unittest.TestCase):
    """Use AST-isolated output class; mock subprocess even on error paths."""
    replayer = no_enter.NoEnterTests.replayer
    paste = no_enter.NoEnterTests.paste

    def setUp(self):
        no_enter.NoEnterTests.setUp(self)
        self.run = Mock(return_value=types.SimpleNamespace(returncode=0))
        self.read = Mock()
        fake = types.SimpleNamespace(run=self.run, check_output=self.read, DEVNULL=subprocess.DEVNULL)
        self.ns.update(subprocess=fake, time=types.SimpleNamespace(sleep=lambda *a: None))

    def real_replayer(self):
        obj = self.replayer()
        obj._set_clipboard = types.MethodType(self.cls._set_clipboard, obj)
        obj._clipboard_owned = types.MethodType(self.cls._clipboard_owned, obj)
        return obj

    def install_payload_reader(self):
        def read(cmd, **kwargs):
            write = next(c.args[0] for c in self.run.call_args_list if c.args[0][:2] == ['copyq', 'copy'])
            payload = dict(zip(write[2::2], write[3::2]))
            mime = cmd[-1]
            return ('\n'.join(payload) if mime == '?' else payload.get(mime, '')).encode()
        self.read.side_effect = read

    def test_double_mime_history_and_ownership(self):
        obj = self.real_replayer()
        self.install_payload_reader()
        result = self.paste(obj, 'First\nsecond', html='<p>First</p><p>second</p>', rewrite_mode=True)
        self.assertTrue(result['ok'])
        self.assertTrue(result['html_offered'])
        self.assertEqual(result['ownership'], 'matched')
        self.assertEqual(result['shortcut'], 'Ctrl+V')
        self.assertEqual(self.run.call_args_list[0].args[0], ['copyq', 'add', '--', 'First second'])
        self.assertEqual(self.read.call_count, 3)
        obj._delivery_alert.assert_not_called()
        self.assertNotIn('First', repr(self.events))
        self.assertNotIn('<p>', repr(self.events))

    def test_explicit_terminal_overrules_html_but_wm_class_does_not(self):
        for explicit in (False, True):
            obj = self.real_replayer()
            obj._get_active_window_class.return_value = 'kitty'
            self.run.reset_mock()
            self.install_payload_reader()
            result = self.paste(obj, 'First\nsecond', html='<p>First</p>', rewrite_mode=True, terminal_mode=explicit)
            self.assertTrue(result['ok'])
            self.assertEqual(result['html_offered'], not explicit)
            self.assertEqual(result['shortcut'], 'Ctrl+Shift+V')
        obj = self.replayer('unrecognized-terminal')
        result = self.paste(obj, 'One\ntwo', terminal_mode=True)
        self.assertEqual(result['shortcut'], 'Ctrl+Shift+V')

    def test_replacement_read_error_and_changed_html_refuse_without_rewrite(self):
        for failure in ('owner', 'plain', 'html', 'error'):
            obj = self.real_replayer()
            self.run.reset_mock()
            self.install_payload_reader()
            original = self.read.side_effect
            def read(cmd, **kw):
                if failure == 'error':
                    raise subprocess.TimeoutExpired(cmd, 2)
                if cmd[-1] == {'owner': 'application/x-dictation-owner', 'plain': 'text/plain', 'html': 'text/html'}[failure]:
                    return b'OTHER'
                return original(cmd, **kw)
            self.read.side_effect = read
            result = self.paste(obj, 'private text', html='<p>private text</p>', rewrite_mode=True)
            self.assertFalse(result['ok'])
            self.assertEqual(result['paste'], 'refused_clipboard_changed')
            obj.kb.press.assert_not_called()
            obj._delivery_alert.assert_called_once()
            self.assertEqual(self.run.call_count, 2)  # history and own write only; no restore
            self.assertNotIn('private text', repr(self.events))

    def test_copyq_failure_plain_xclip_and_audible_degradation(self):
        for rc in (0, 1):
            obj = self.real_replayer()
            self.run.side_effect = lambda cmd, **kw: types.SimpleNamespace(returncode=rc if cmd[0] == 'xclip' else 1)
            self.read.side_effect = lambda *a, **kw: b'First second'
            result = self.paste(obj, 'First\nsecond', html='<p>First</p>', rewrite_mode=True)
            self.assertEqual(result['ok'], rc == 0)
            self.assertFalse(result['html_offered'])
            self.assertEqual(result['path'], 'xclip')
            obj._delivery_alert.assert_called_once()
            self.assertEqual(self.run.call_args.kwargs['input'], b'First second')
            self.assertNotIn('text/html', self.run.call_args.args[0])

    def test_xclip_replacement_refuses_and_never_restores(self):
        obj = self.real_replayer()
        self.run.side_effect = lambda cmd, **kw: types.SimpleNamespace(returncode=0 if cmd[0] == 'xclip' else 1)
        self.read.return_value = b'other writer'
        result = self.paste(obj, 'mine', echo=False)
        self.assertFalse(result['ok'])
        obj.kb.press.assert_not_called()
        obj._delivery_alert.assert_called_once()
        self.assertEqual(self.run.call_count, 2)
        self.assertEqual(result['ownership'], 'mismatch_or_unreadable')
        self.assertNotIn('other writer', repr(self.events))

    def test_timeout_metadata_cannot_contain_subprocess_payload(self):
        obj = self.real_replayer()
        def run(cmd, **kw):
            if cmd[0] == 'copyq':
                raise subprocess.TimeoutExpired(cmd, 3)
            return types.SimpleNamespace(returncode=0)
        self.run.side_effect = run
        self.read.return_value = b'PRIVATE PAYLOAD'
        result = self.paste(obj, 'PRIVATE PAYLOAD', echo=False, rewrite_mode=True, html='<p>PRIVATE PAYLOAD</p>')
        self.assertTrue(result['ok'])
        self.assertEqual(result['copyq_error'], 'TimeoutExpired')
        self.assertNotIn('PRIVATE PAYLOAD', repr(self.events))
        obj._delivery_alert.assert_called_once()

    def test_plain_ownership_rejects_unexpected_html(self):
        obj = self.real_replayer()
        self.install_payload_reader()
        original = self.read.side_effect
        self.read.side_effect = lambda cmd, **kw: original(cmd, **kw) + (b'\ntext/html' if cmd[-1] == '?' else b'')
        result = self.paste(obj, 'plain', echo=False)
        self.assertFalse(result['ok'])
        obj.kb.press.assert_not_called()
        obj._delivery_alert.assert_called_once()

    def test_plain_write_removes_old_html_and_checks_absence(self):
        obj = self.real_replayer()
        self.install_payload_reader()
        result = self.paste(obj, 'plain', html='<p>not allowed</p>')
        self.assertTrue(result['ok'])
        self.assertFalse(result['html_offered'])
        write = self.run.call_args.args[0]
        self.assertNotIn('text/html', write)
        self.assertEqual(self.read.call_args.args[0][-1], '?')


if __name__ == '__main__':
    unittest.main()
