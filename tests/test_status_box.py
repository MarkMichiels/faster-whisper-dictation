"""Status box, offline: real App loops, real StatusBox and ShellClient, fake D-Bus call.

Never touches a display, the session bus, the clipboard or the network. The
central check: no dictation text and no selection content ever reaches the
D-Bus call; only allowlisted codes and counts do.
"""
import contextlib
import io
import os
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))
import context_capture  # noqa: E402
import rewrite  # noqa: E402
import status_box  # noqa: E402
import wayland_io  # noqa: E402
from test_rewrite import AppStart, config, llm_answer  # noqa: E402

# Distinctive synthetic text: must never appear in anything sent to the shell.
SPOKEN = 'Zeppelinkoffie bij Kwartelstraat negen'
SELECTED = 'Geachte heer Blauwvingerhoed, uw factuur 77351 voor de pompen'
REWRITTEN = 'Beste Blauwvingerhoed, de factuur klopt.'
SECRETS = ('Zeppelin', 'Kwartel', 'Blauwvinger', '77351', 'factuur', 'pomp', 'Beste', 'Geachte')


class FakeShell:
    """The D-Bus call of ShellClient, recorded. Optionally fails like an old helper."""

    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def __call__(self, method, *args):
        self.calls.append((method, args))
        if self.fail and method in ('SetStatus', 'ClearStatus'):
            raise RuntimeError('UnknownMethod')
        return ()

    def phases(self):
        return [args[0].strip("'") for method, args in self.calls if method == 'SetStatus']

    def statuses(self):
        return [tuple(a.strip("'") for a in args) for method, args in self.calls if method == 'SetStatus']


def sync_box(shell, log=None):
    client = wayland_io.ShellClient(call=shell)
    return status_box.StatusBox(client.status, client.clear_status, badge=client.badge,
                                log=log, spawn=lambda fn: fn())


class StatusBoxUnitTests(unittest.TestCase):
    def test_codes_and_count_only_quoted(self):
        shell = FakeShell()
        box = sync_box(shell)
        box.show('recording', 'rewrite', 'en', 4312)
        box.clear()
        self.assertEqual(shell.calls, [('SetStatus', ("'recording'", "'rewrite'", "'en'", '4312')),
                                       ('ClearStatus', ())])

    def test_client_refuses_anything_but_codes(self):
        client = wayland_io.ShellClient(call=FakeShell())
        for args in [(SPOKEN, 'dictate'), ('recording', SELECTED), ('recording', 'dictate', 'Nederlands'),
                     ('recording', 'dictate', "e'n"), ('recording', 'dictate', 'EN'),
                     ("recording'", 'dictate')]:
            with self.assertRaises(ValueError):
                client.status(*args)

    def test_language_code_and_mode(self):
        self.assertEqual([status_box.language_code(v) for v in ('en', 'EN', None, 'auto', '', 'e n', 'nl')],
                         ['en', 'en', '', '', '', '', 'nl'])
        self.assertEqual(status_box.mode_of(True, True, False), 'rewrite_line')
        self.assertEqual(status_box.mode_of(True, False, False), 'rewrite')
        self.assertEqual(status_box.mode_of(False, True, True), 'dictate_fallback')
        self.assertEqual(status_box.mode_of(False, False, False), 'dictate')

    def test_newest_state_wins_behind_a_slow_call(self):
        sent, gate, spawned = [], threading.Event(), []

        def send(*state):
            gate.wait(2)
            sent.append(state[0])
        box = status_box.StatusBox(send, lambda: sent.append('clear'),
                                   spawn=lambda fn: spawned.append(threading.Thread(target=fn)) or spawned[-1].start())
        box.show('recording', 'dictate')
        time.sleep(0.05)
        for phase in ('finishing', 'rewriting', 'pasted'):
            box.show(phase, 'dictate')
        gate.set()
        spawned[0].join(2)
        self.assertEqual(sent, ['recording', 'pasted'])
        self.assertEqual(len(spawned), 1)   # one sender at a time

    def test_failure_is_quiet_logged_once_and_falls_back_to_the_badge(self):
        shell, log = FakeShell(fail=True), []
        box = sync_box(shell, log=lambda kind, **d: log.append((kind, d)))
        box.show('recording', 'rewrite', '', 120)    # context: old helper -> badge
        box.show('finishing', 'rewrite', '', 120)
        box.show('recording', 'dictate', '', 0)      # no context: no badge
        self.assertEqual([c for c in shell.calls if c[0] == 'ShowBadge'], [('ShowBadge', ('120',))])
        self.assertEqual(log[0], ('status_box', dict(available=False, error='RuntimeError')))
        self.assertEqual([k for k, _ in log].count('status_box'), 1)
        self.assertFalse(box.available)
        shell.fail = False
        box.show('recording', 'dictate', '', 0)
        self.assertEqual(log[-1], ('status_box', dict(available=True)))

    def test_spawn_failure_never_raises(self):
        log = []
        box = status_box.StatusBox(lambda *a: None, lambda: None, log=lambda k, **d: log.append(d),
                                   spawn=lambda fn: (_ for _ in ()).throw(RuntimeError('no thread')))
        box.show('recording', 'dictate')
        self.assertEqual(log, [dict(sent=False, error='RuntimeError')])
        box.show('recording', 'dictate')   # not stuck as busy
        self.assertEqual(len(log), 2)


class AppStatusTests(AppStart, unittest.TestCase):
    """The phases of whole sessions through the real App code."""

    def run_session(self, rewrite_mode=False, keys=None, language='nl', context=None, http=None,
                    paste_ok=True, spoken=SPOKEN, cfg='default', silent=False):
        def capture(**kw):
            return context or context_capture.Context(None)
        app, events = self.start_app(config() if cfg == 'default' else cfg, None if cfg else 'not_configured',
                                     capture=capture)
        app.args.language = 'nl'
        shell = FakeShell()
        app.status = sync_box(shell)
        logic = app.start.__globals__
        logic['context_badge'].calls.clear()
        app.transcription_worker.transcribe_chunk = lambda audio, recovery=False: spoken
        if not paste_ok:
            original = app.replayer.type_text

            def refuse(text, **kw):
                original(text, **kw)
                return dict(ok=False, path='copyq', paste='refused_clipboard_changed')
            app.replayer.type_text = refuse
        answer = http or (lambda *a: llm_answer(REWRITTEN))
        out = io.StringIO()
        with patch.object(rewrite, 'http_call', answer), contextlib.redirect_stdout(out):
            app.start(language, rewrite_mode, keys)
            session = app.session
            if silent:
                app._mic_silent_alert(session)
            app.stop()
            self.assertTrue(session.done.wait(5))
        self.everything = repr(shell.calls)
        for secret in SECRETS:
            self.assertNotIn(secret, self.everything)
        for method, args in shell.calls:
            if method == 'SetStatus':
                phase, mode, lang, chars = (a.strip("'") for a in args)
                self.assertIn(phase, wayland_io.STATUS_PHASES)
                self.assertIn(mode, wayland_io.STATUS_MODES)
                self.assertRegex(lang, r'^([a-z]{2,3})?$')
                self.assertRegex(chars, r'^\d+$')
            else:
                self.assertIn(method, ('ClearStatus', 'ShowBadge'))
        return shell, events, logic['context_badge'].calls, session

    def test_plain_dictation(self):
        shell, events, badge, _ = self.run_session()
        self.assertEqual(shell.statuses(), [('recording', 'dictate', '', '0'),
                                            ('finishing', 'dictate', '', '0'),
                                            ('pasted', 'dictate', '', '0')])
        self.assertTrue(any(e[0] == 'paste' and SPOKEN in e[1] for e in events))

    def test_shift_english(self):
        shell, _, _, _ = self.run_session(language='en')
        self.assertEqual(shell.statuses()[0], ('recording', 'dictate', 'en', '0'))

    def test_rewrite_with_context_replaces_the_badge(self):
        ctx = context_capture.Context(SELECTED, 'primary', len(SELECTED), 0.5)
        shell, events, badge, _ = self.run_session(True, {'alt': True}, context=ctx)
        n = str(len(SELECTED))
        self.assertEqual(shell.statuses(), [('recording', 'rewrite', '', n), ('finishing', 'rewrite', '', n),
                                            ('rewriting', 'rewrite', '', n), ('pasted', 'rewrite', '', n)])
        self.assertEqual(badge, [])   # no separate 1.5 s badge
        self.assertNotIn('ShowBadge', [m for m, _ in shell.calls])
        self.assertTrue(any(e[0] == 'paste' and 'Blauwvinger' in e[1] for e in events))

    def test_rewrite_one_line_english(self):
        shell, _, _, _ = self.run_session(True, {'ctrl': True, 'shift': True}, language='en')
        self.assertEqual(shell.statuses()[0], ('recording', 'rewrite_line', 'en', '0'))

    def test_rewrite_failure_pastes_raw(self):
        def broken(*a):
            raise OSError('network down')
        shell, events, _, _ = self.run_session(True, {'alt': True}, http=broken)
        self.assertEqual(shell.phases()[-2:], ['rewriting', 'raw_pasted'])

    def test_paste_refused(self):
        shell, _, _, _ = self.run_session(paste_ok=False)
        self.assertEqual(shell.phases()[-1], 'paste_refused')
        shell, _, _, _ = self.run_session(True, {'alt': True}, paste_ok=False)
        self.assertEqual(shell.phases()[-1], 'paste_refused')

    def test_rewrite_unavailable_falls_back(self):
        shell, _, _, _ = self.run_session(True, {'alt': True}, cfg=None)
        self.assertEqual(shell.phases()[0], 'recording')
        self.assertEqual(shell.statuses()[0][1], 'dictate_fallback')

    def test_nothing_recognised_and_silent_microphone(self):
        shell, _, _, _ = self.run_session(spoken='')
        self.assertEqual(shell.phases()[-1], 'nothing')
        shell, _, _, _ = self.run_session(spoken='', silent=True)
        self.assertEqual(shell.phases(), ['recording', 'recording_silent', 'finishing', 'mic_silent'])
        shell, _, _, _ = self.run_session(True, {'alt': True}, spoken='', silent=True)
        self.assertEqual(shell.phases()[-1], 'mic_silent')

    def test_only_the_newest_session_shows(self):
        app, _ = self.start_app(config(), None)
        shell = FakeShell()
        app.status = sync_box(shell)
        old = app.Session(None, None)
        app.session = app.Session(None, None)
        app._status(old, 'pasted', final=True)
        self.assertEqual(shell.calls, [])
        app._status(app.session, 'finishing')
        self.assertEqual(shell.phases(), ['finishing'])

    def test_no_status_box_keeps_the_old_badge(self):
        ctx = context_capture.Context(SELECTED, 'primary', len(SELECTED), 0.5)
        app, _ = self.start_app(config(), None, capture=lambda **kw: ctx)
        logic = app.start.__globals__
        logic['context_badge'].calls.clear()
        with patch.object(rewrite, 'http_call', lambda *a: llm_answer(REWRITTEN)), \
                contextlib.redirect_stdout(io.StringIO()):
            app.start('nl', True, {'alt': True})
            session = app.session
            app.stop()
            self.assertTrue(session.done.wait(5))
        self.assertEqual(logic['context_badge'].calls, [len(SELECTED)])

    def test_error_inside_the_session_clears_the_box(self):
        app, _ = self.start_app(config(), None)
        shell = FakeShell()
        app.status = sync_box(shell)
        session = app.Session(None, None)
        app.session = session
        app._typing_session = lambda s: (_ for _ in ()).throw(RuntimeError('boom'))
        with self.assertRaises(RuntimeError):
            app._typing_loop(session)
        self.assertEqual(shell.calls[-1], ('ClearStatus', ()))


class MakeStatusBoxTests(unittest.TestCase):
    def test_only_on_wayland_and_when_enabled(self):
        from test_rewrite import app_logic
        logic = app_logic()
        app = logic['App'].__new__(logic['App'])
        shell = FakeShell()

        class Replayer:
            show_status = wayland_io.ShellClient(call=shell).status
            clear_status = wayland_io.ShellClient(call=shell).clear_status
            show_badge = wayland_io.ShellClient(call=shell).badge
        app.replayer = Replayer()
        logic.update(WAYLAND=False, STATUS_BOX=True)
        self.assertIsNone(app._make_status_box())
        logic.update(WAYLAND=True, STATUS_BOX=False)
        self.assertIsNone(app._make_status_box())
        logic.update(WAYLAND=True, STATUS_BOX=True)
        with contextlib.redirect_stdout(io.StringIO()):
            box = app._make_status_box()
        self.assertIsInstance(box, status_box.StatusBox)
        deadline = time.monotonic() + 2
        while box.available is None and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(shell.calls, [('ClearStatus', ())])   # leftover box of a dead instance


if __name__ == '__main__':
    unittest.main()
