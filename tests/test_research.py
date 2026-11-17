"""Research mode, offline: keys, settings, the runner with a fake Pi, delivery, the guard.

Never starts a real Pi, never touches the clipboard, CopyQ, a display or the
network. The fake Pi is a small Python script that writes Pi-shaped JSONL.
"""
import io
import json
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))
import coverage_ledger as coverage  # noqa: E402
import research  # noqa: E402
import settings  # noqa: E402
import status_box  # noqa: E402
import wayland_io  # noqa: E402
from test_recovery import worker  # noqa: E402
from test_rewrite import ALT, ALT_GR, CTRL_R, SHIFT, FakeReplayer, app_logic, key, listener_class  # noqa: E402

R = coverage.RATE
NODE = shutil.which('node') or next(
    (str(p) for p in sorted(Path.home().glob('.nvm/versions/node/*/bin/node'))), None)
DICTATION = 'Zoek uit wanneer de Zeppelinpomp van Kwartelstraat geleverd wordt'
SELECTION = 'Geachte heer Blauwvingerhoed, factuur 77351'


def tap_twice(listener, held=()):
    listener.last_press_time = 0
    for k in held:
        listener.on_press(k)
    for _ in range(2):
        listener.on_press(CTRL_R)
        listener.on_release(CTRL_R)
    for k in held:
        listener.on_release(k)


class KeyTests(unittest.TestCase):
    def listener(self, calls, **kw):
        cls = listener_class()
        return cls(lambda *a: calls.append(a), lambda: calls.append('stop'),
                   key=kw.pop('main', CTRL_R), default_language='nl', shift_language='en', **kw)

    def test_altgr_x11_selects_research_not_rewrite(self):
        calls = []
        listener = self.listener(calls)
        tap_twice(listener)
        tap_twice(listener, (ALT_GR,))
        tap_twice(listener, (SHIFT, ALT_GR))
        tap_twice(listener, (ALT, ALT_GR))     # AltGr wins over Left-Alt
        tap_twice(listener, (ALT,))            # afterwards rewrite is unchanged
        tap_twice(listener)
        starts = [c for c in calls if c != 'stop']
        self.assertEqual([c[:2] for c in starts],
                         [('nl', False), ('nl', False), ('en', False), ('nl', False), ('nl', True), ('nl', False)])
        self.assertEqual([bool(c[2].get('research')) for c in starts], [False, True, True, True, False, False])
        self.assertNotIn('altgr', starts[0][2])          # ordinary starts keep their journal shape
        self.assertTrue(starts[1][2]['altgr'])
        self.assertIn('altgr_held_s', starts[1][2])
        self.assertFalse(starts[4][2]['altgr'])
        self.assertIn('altgr_released_s_ago', starts[4][2])
        self.assertFalse(listener.modifiers_held())

    def test_altgr_as_vk_only_and_modifiers_held(self):
        calls = []
        listener = self.listener(calls)
        listener.on_press(key(None, 0xfe03))     # ISO_Level3_Shift without a pynput name
        self.assertTrue(listener.modifiers_held())   # selection_release must wait for it
        listener.on_release(key(None, 0xfe03))
        self.assertFalse(listener.altgr_held)

    def test_evdev_route_altgr_and_escape(self):
        calls, escapes = [], []
        listener = self.listener(calls, main=wayland_io.Key.ctrl_r)
        listener.escape_callback = lambda: escapes.append(1)
        hub = wayland_io.InputHub(provider=lambda: {}, selector=object(), layout='be', trace=False)
        hub.key_listener = listener

        def ev(code, value):
            hub.event('/dev/input/event3', types.SimpleNamespace(type=1, code=code, value=value))
        ev(100, 1)                      # AltGr down
        for _ in range(2):
            listener.last_press_time = listener.last_press_time or 0
            ev(97, 1)
            ev(97, 0)
        ev(100, 0)
        self.assertEqual(calls[-1][:2], ('nl', False))
        self.assertTrue(calls[-1][2]['research'])
        ev(1, 1)
        ev(1, 0)
        self.assertEqual(escapes, [1])
        ev(30, 1)                       # another key: no callback
        ev(30, 0)
        self.assertEqual(escapes, [1])

    def test_escape_without_callback_and_failing_callback(self):
        calls = []
        listener = self.listener(calls)
        listener.on_press(key('esc', 0xff1b))           # no callback: ignored
        listener.escape_callback = lambda: 1 / 0
        listener.on_press(key('esc', 0xff1b))           # a failing callback never stops the listener
        tap_twice(listener)
        self.assertEqual(calls[-1][:2], ('nl', False))


class SettingsTests(unittest.TestCase):
    def load(self, text):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'config.yaml')
            Path(path).write_text(textwrap.dedent(text))
            return settings.load(environ={}, explicit_path=path)

    def test_defaults(self):
        loaded = settings.load(environ={}, explicit_path='/nonexistent/config.yaml')
        self.assertFalse(loaded.get('research.enabled'))
        self.assertEqual(loaded.get('research.timeout_s'), 600)
        self.assertTrue(loaded.get('research.escape_cancels'))
        self.assertEqual(research.load_config(loaded), (None, 'disabled'))

    def test_provider_allowlist(self):
        for model, ok in (('anthropic/claude-sonnet-5-5', True), ('openai-codex/gpt-5.5', True),
                          ('openrouter/deepseek/deepseek-v4.1-flash-1017', True), ('google/gemini-3-pro', False),
                          ('anthropic', False), ('anthropic/x; rm -rf /', False), ('ANTHROPIC/x', False)):
            with self.subTest(model=model):
                loaded = self.load('research:\n  model: %s\n' % json.dumps(model))
                if ok:
                    self.assertEqual(loaded.get('research.model'), model)
                else:
                    self.assertEqual(loaded.get('research.model'), 'anthropic/claude-sonnet-5-5')
                    events = [d for k, d in loaded.events if d.get('key') == 'research.model']
                    self.assertTrue(events, loaded.events)
                    if '/' in model and model.split('/')[0] == 'google':
                        self.assertEqual(events[0].get('reason'), 'provider_not_allowed')

    def test_load_config_reasons(self):
        with tempfile.TemporaryDirectory() as tmp:
            pi = os.path.join(tmp, 'pi')
            Path(pi).write_text('#!/bin/sh\n')
            node = os.path.join(tmp, 'node')
            Path(node).write_text('#!/bin/sh\n')
            os.chmod(node, 0o755)
            base = 'research:\n  enabled: true\n'
            self.assertEqual(research.load_config(self.load(base))[1], 'pi_bin_missing')
            text = base + '  pi_bin: %s\n  node_bin: %s\n' % (pi, node)
            self.assertEqual(research.load_config(self.load(text))[1], 'pi_bin_not_executable')
            os.chmod(pi, 0o755)
            self.assertEqual(research.load_config(self.load(text))[1], 'cwd_missing')
            self.assertEqual(research.load_config(self.load(text + '  cwd: %s/nope\n' % tmp))[1],
                             'cwd_not_a_directory')
            self.assertEqual(research.load_config(self.load(text + '  cwd: %s\n  notes_dir: ../x\n' % tmp))[1],
                             'notes_dir_outside_cwd')
            cfg, reason = research.load_config(self.load(text + '  cwd: %s\n' % tmp))
            self.assertEqual(reason, 'ok')
            self.assertEqual(cfg.notes_dir, os.path.join(os.path.realpath(tmp), 'research_notes'))
            ok, lines = research.check(self.load(text + '  cwd: %s\n' % tmp))
            self.assertTrue(ok, lines)
            ok, lines = research.check(self.load(text + '  cwd: %s\n  model: google/x\n' % tmp))
            self.assertFalse(ok)
            self.assertIn('provider_not_allowed', lines[0])


FAKE_PI = r'''#!/usr/bin/env python3
import json, os, signal, subprocess, sys, time
mode = os.environ.get('FAKE_PI_MODE', 'result')
prompt = sys.stdin.read()
argv = ' '.join(sys.argv)
with open(os.environ['FAKE_PI_LOG'], 'w') as f:
    json.dump(dict(argv=sys.argv, prompt=prompt, cwd=os.getcwd(), pgid=os.getpgid(0), pid=os.getpid(),
                   env={k: v for k, v in os.environ.items() if k.startswith(('DICTATION_RESEARCH', 'PI_DASH'))}), f)
sid = sys.argv[sys.argv.index('--session-id') + 1]
def out(record):
    sys.stdout.write(json.dumps(record) + '\n'); sys.stdout.flush()
out({'type': 'session', 'version': 3, 'id': sid, 'cwd': os.getcwd()})
out({'type': 'agent_start'})
out({'type': 'message_update', 'assistantMessageEvent': {'type': 'thinking_start'}})
out({'type': 'tool_execution_start', 'toolCallId': 't1', 'toolName': 'bash', 'args': {'command': 'rg -n pomp docs'}})
out({'type': 'tool_execution_end', 'toolCallId': 't1', 'toolName': 'bash', 'result': {'content': [{'type': 'text', 'text': 'x'}]}, 'isError': False})
if mode == 'blocked' or mode == 'result':
    out({'type': 'tool_execution_start', 'toolCallId': 't2', 'toolName': 'bash', 'args': {'command': 'git push'}})
    out({'type': 'tool_execution_end', 'toolCallId': 't2', 'toolName': 'bash', 'isError': True,
         'result': {'content': [{'type': 'text', 'text': 'Blocked by the dictation research guard [git_remote]: command not allowed. Do not try.'}]}})
out({'type': 'message_end', 'message': {'role': 'assistant', 'usage': {'cost': {'total': 0.0123}}, 'stopReason': 'toolUse'}})
if mode in ('sleep', 'grandchild'):
    if mode == 'grandchild':
        child = subprocess.Popen(['sleep', '60'])
        with open(os.environ['FAKE_PI_LOG'] + '.child', 'w') as f:
            f.write(str(child.pid))
    time.sleep(60)
if mode == 'exit3':
    sys.stderr.write('Error: something broke\n'); sys.exit(3)
if mode == 'auth':
    sys.stderr.write('No API key found for anthropic. Please login\n'); sys.exit(1)
if mode in ('result', 'blocked', 'question', 'linger', 'orphan'):
    status = 'question' if mode == 'question' else 'done'
    text = 'Levering: vrijdag. ' + prompt.split('<opdracht>')[1].split('</opdracht>')[0].strip()[:12]
    out({'type': 'tool_execution_start', 'toolCallId': 't3', 'toolName': 'deliver_result',
         'args': {'text': text, 'summary': 'note written', 'status': status}})
    out({'type': 'tool_execution_end', 'toolCallId': 't3', 'toolName': 'deliver_result', 'isError': False,
         'result': {'content': [{'type': 'text', 'text': 'Result delivered'}]}})
    if mode == 'linger':
        time.sleep(60)
out({'type': 'agent_end'})
if mode == 'orphan':
    child = subprocess.Popen(['sleep', '60'])     # inherits stdout: keeps the pipe open after we exit
    with open(os.environ['FAKE_PI_LOG'] + '.child', 'w') as f:
        f.write(str(child.pid))
'''


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.pi = os.path.join(self.tmp, 'pi')
        Path(self.pi).write_text(FAKE_PI.replace('#!/usr/bin/env python3', '#!' + sys.executable, 1))
        os.chmod(self.pi, 0o755)
        self.log = os.path.join(self.tmp, 'pi.log')
        self.journal = []

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def cfg(self, timeout=20.0):
        return research.ResearchConfig(pi_bin=self.pi, model='anthropic/fake', thinking='low', cwd=self.tmp,
                                       notes_dir=os.path.join(self.tmp, 'notes'), timeout_s=timeout)

    def run_fake(self, mode, timeout=20.0, cancel_after=None):
        progress, finished = [], []
        env = dict(os.environ, FAKE_PI_MODE=mode, FAKE_PI_LOG=self.log)
        with patch.dict(os.environ, env):
            runner = research.Runner(self.cfg(timeout), log=lambda kind, **d: self.journal.append((kind, d)))
            run = runner.start(research.build_prompt(DICTATION, SELECTION, 'nl'),
                               lambda step, steps: progress.append((step, steps)), finished.append)
            self.assertIsNotNone(run)
            self.assertTrue(runner.busy)
            self.assertIsNone(runner.start('second'))      # one session at a time
            if cancel_after is not None:
                deadline = time.monotonic() + 10
                while not os.path.exists(self.log + ('.child' if mode == 'grandchild' else '')) \
                        and time.monotonic() < deadline:
                    time.sleep(0.05)
                time.sleep(cancel_after)
                self.assertTrue(runner.cancel('escape'))
            self.assertTrue(run.wait(30))
        deadline = time.monotonic() + 5
        while not finished and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertFalse(runner.busy)
        return finished[0], progress, json.loads(Path(self.log).read_text()), run

    def test_result_prompt_on_stdin_only(self):
        outcome, progress, seen, run = self.run_fake('result')
        self.assertEqual(outcome.status, 'done')
        self.assertTrue(outcome.text.startswith('Levering: vrijdag. Zoek uit wan'))
        self.assertEqual(outcome.summary, 'note written')
        self.assertEqual(outcome.blocked, ['bash'])
        self.assertEqual(outcome.cost, 0.0123)
        self.assertEqual(outcome.session_id, run.session_id)
        # Dictation and selection travel on stdin, never in argv (ps shows argv).
        self.assertIn(DICTATION, seen['prompt'])
        self.assertIn(SELECTION, seen['prompt'])
        argv = ' '.join(seen['argv'])
        self.assertNotIn('Zeppelin', argv)
        self.assertNotIn('Blauwvinger', argv)
        for flag in ('--mode json', '-p', '--no-mcp', '--exclude-tools', '--append-system-prompt'):
            self.assertIn(flag, argv)
        self.assertIn(research.EXTENSION, seen['argv'])
        self.assertEqual(seen['cwd'], self.tmp)
        self.assertEqual(seen['pgid'], seen['pid'])      # own process group
        self.assertEqual(seen['env']['DICTATION_RESEARCH_NOTES'], os.path.join(self.tmp, 'notes'))
        self.assertEqual(seen['env']['PI_DASHBOARD_VISIBLE'], '1')
        steps = [s for s, _ in progress]
        self.assertEqual(steps[:3], ['starting', 'thinking', 'bash:rg'])
        self.assertIn('blocked', steps)
        self.assertIn('deliver_result', steps)
        # The journal: codes and tool names only.
        text = json.dumps(self.journal)
        for secret in ('Zeppelin', 'Kwartel', 'Blauwvinger', '77351', 'vrijdag', 'pomp'):
            self.assertNotIn(secret, text)
        self.assertIn(('research_tool_blocked', {'tool': 'bash', 'rule': 'git_remote'}), self.journal)

    def test_question(self):
        outcome = self.run_fake('question')[0]
        self.assertEqual(outcome.status, 'question')

    def test_timeout(self):
        start = time.monotonic()
        outcome = self.run_fake('sleep', timeout=1.0)[0]
        self.assertEqual((outcome.status, outcome.reason), ('timeout', 'timeout'))
        self.assertLess(time.monotonic() - start, 10)
        self.assertIsNone(outcome.text)

    def test_cancel_stops_the_whole_group(self):
        outcome = self.run_fake('grandchild', cancel_after=0.2)[0]
        self.assertEqual((outcome.status, outcome.reason), ('cancelled', 'escape'))
        child = int(Path(self.log + '.child').read_text())
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                state = Path('/proc/%d/stat' % child).read_text().split(') ')[1][0]
            except FileNotFoundError:
                break
            if state == 'Z':
                break
            time.sleep(0.05)
        else:
            self.fail('grandchild still running')

    def test_no_deliver_result(self):
        outcome = self.run_fake('none')[0]
        self.assertEqual((outcome.status, outcome.reason), ('no_result', 'no_deliver_result'))

    def test_exit_nonzero(self):
        outcome = self.run_fake('exit3')[0]
        self.assertEqual((outcome.status, outcome.reason, outcome.returncode), ('failed', 'pi_exit_3', 3))
        outcome = self.run_fake('auth')[0]
        self.assertEqual((outcome.status, outcome.reason), ('failed', 'auth'))

    def test_session_that_lingers_after_result_is_stopped(self):
        with patch.object(research, 'AFTER_RESULT_S', 0.3):
            outcome = self.run_fake('linger')[0]
        self.assertEqual(outcome.status, 'done')

    def assert_gone(self, pid):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                state = Path('/proc/%d/stat' % pid).read_text().split(') ')[1][0]
            except FileNotFoundError:
                return
            if state == 'Z':
                return
            time.sleep(0.05)
        self.fail('process %d still running' % pid)

    def test_left_behind_process_does_not_hang_the_run(self):
        with patch.object(research, 'READER_GRACE_S', 0.5):
            start = time.monotonic()
            outcome = self.run_fake('orphan')[0]
        self.assertEqual(outcome.status, 'done')
        self.assertLess(time.monotonic() - start, 8)
        self.assert_gone(int(Path(self.log + '.child').read_text()))
        self.assertTrue(any(k == 'research_output_open_after_exit' for k, _ in self.journal))

    def test_step_code_never_carries_arguments(self):
        self.assertEqual(research.step_code('bash', {'command': 'FOO=1 /usr/bin/python3.12 -c "print(42)"'}),
                         'bash:python')
        self.assertEqual(research.step_code('bash', {'command': 'Zeppelin --x'}), 'bash')
        self.assertEqual(research.step_code('Read'), 'read')
        self.assertEqual(research.step_code('weird tool name!'), 'tool')
        self.assertEqual(research.step_code(None), 'tool')


class StatusBoxTests(unittest.TestCase):
    def test_research_state_and_allowlist(self):
        calls = []
        box = status_box.StatusBox(lambda *a: calls.append(('show',) + a), lambda: calls.append(('clear',)),
                                   research=lambda *a: calls.append(('research',) + a), spawn=lambda fn: fn())
        box.research('researching', 'bash:rg', 12.7, 3)
        self.assertEqual(calls, [('research', 'researching', 'bash:rg', 12, 3)])
        sent = []
        shell = wayland_io.ShellClient(call=lambda *a: sent.append(a))
        shell.research('failed', 'timeout', 5, 1)
        self.assertEqual(sent, [('SetResearch', "'failed'", "'timeout'", '5', '1')])
        for bad in (('bezig', ''), ('researching', 'Zoek de pomp'), ('done', "x'; y")):
            with self.assertRaises(ValueError):
                shell.research(*bad)
        with self.assertRaises(RuntimeError):
            status_box.StatusBox(lambda *a: None, lambda: None).research('done')


class FakeRunner:
    def __init__(self, busy=False):
        self.started, self.cancelled, self._busy = [], [], busy

    @property
    def busy(self):
        return self._busy

    def start(self, prompt, on_progress=None, on_finish=None):
        if self._busy:
            return None
        self.started.append((prompt, on_progress, on_finish))
        self._busy = True
        return types.SimpleNamespace(session_id='sid')

    def cancel(self, reason='cancelled'):
        self.cancelled.append(reason)
        was, self._busy = self._busy, False
        return was


class Replayer(FakeReplayer):
    def __init__(self, events, focus='win-stop', paste_ok=True):
        super().__init__(events)
        self.focus, self.paste_ok = focus, paste_ok

    def window_id(self):
        return self.focus

    def type_text(self, text, echo=True, rewrite_mode=False, html=None, terminal_mode=False):
        assert not echo and rewrite_mode
        self.events.append(('paste', text, self.target_wid, html))
        if self.paste_ok:
            return dict(ok=True, path='copyq', copyq_copy=0, paste='shortcut_sent_not_acknowledged')
        return dict(ok=False, path='copyq', copyq_copy=0, paste='not_sent', focus_ok=False)

    def _set_clipboard(self, text, html=None):
        self.events.append(('clipboard', text, html))
        return dict(ok=True, path='copyq')


class AppTests(unittest.TestCase):
    def app(self, runner=None, focus='win-stop', paste_ok=True):
        logic = app_logic()
        app = logic['App'].__new__(logic['App'])
        events = []
        app.args = types.SimpleNamespace(paste_min_s=0, language='nl')
        app.transcription_worker = worker([DICTATION])
        app.replayer = Replayer(events, focus=focus, paste_ok=paste_ok)
        app.beep = lambda k, wait=True: events.append(('beep', k))
        app.active = False
        app.research_runner = runner if runner is not None else FakeRunner()
        app.research_config = types.SimpleNamespace(model='anthropic/fake', escape_cancels=True)
        app._paste_lock = threading.Lock()
        app.status = None
        logged = []
        app.llm_log = types.SimpleNamespace(record=lambda **row: logged.append(row))
        return logic, app, events, logged

    def run_session(self, app, logic):
        ledger = coverage.CoverageLedger()
        ledger.capture(np.full(10 * R, 8000, dtype=np.int16).tobytes())
        session = app.Session(queue.Queue(), queue.Queue(), ledger, types.SimpleNamespace(_raw_frames=[]),
                              None, 'nl', 'win-start', rewrite=True, research=True)
        session.context, session.paste_wid = SELECTION, 'win-stop'
        app.session = session
        session.transcription_queue.put((np.full(10 * R, 0.1, dtype=np.float32), True, 0))
        session.transcription_queue.put(None)
        out = io.StringIO()
        with patch('sys.stdout', out):
            app._transcription_loop(session)
            app._typing_loop(session)
        return session

    def test_dictation_is_released_at_once(self):
        logic, app, events, _ = self.app()
        session = self.run_session(app, logic)
        self.assertTrue(session.done.is_set())               # the next dictation never waits
        self.assertIsNone(session.context)                    # wiped as for rewrite
        prompt, _, _ = app.research_runner.started[0]
        self.assertIn(DICTATION, prompt)
        self.assertIn(SELECTION, prompt)                      # copied before the wipe
        self.assertEqual(events[0], ('history', DICTATION))  # raw text kept in CopyQ first
        self.assertFalse(any(e[0] == 'paste' for e in events))
        self.assertEqual(app._research_job.paste_wid, 'win-stop')

    def finish(self, app, status='done', text='**Vrijdag** geleverd.', reason=None):
        job = app._research_job
        outcome = research.Outcome(status=status, reason=reason, text=text, session_id='sid-1',
                                   session_file='/x/sid-1.jsonl', seconds=42.0, steps=5, blocked=['bash'])
        with patch.object(logic_time := __import__('time'), 'sleep', lambda s: None):
            app._research_finished(job, outcome)
        return job

    def test_paste_when_focus_unchanged(self):
        logic, app, events, logged = self.app(focus='win-stop')
        self.run_session(app, logic)
        app.replayer.target_wid = 'other-session-window'
        job = self.finish(app)
        paste = [e for e in events if e[0] == 'paste']
        self.assertEqual(len(paste), 1)
        self.assertEqual(paste[0][1], 'Vrijdag geleverd.')
        self.assertEqual(paste[0][2], 'win-stop')
        self.assertIn('<strong>Vrijdag</strong>', paste[0][3])
        self.assertEqual(app.replayer.target_wid, 'other-session-window')   # restored
        self.assertEqual((job.phase, job.step), ('done', ''))
        self.assertIn(('beep', 'rewrite_done'), events)
        row = logged[0]
        self.assertEqual((row['kind'], row['mode'], row['status'], row['delivered']),
                         ('research', 'research', 'ok', 'pasted'))
        self.assertEqual((row['pi_session_id'], row['pi_session_file']), ('sid-1', '/x/sid-1.jsonl'))
        self.assertNotIn('call', row)                     # llm_content only, no llm_call row
        self.assertEqual(row['input'], DICTATION)
        self.assertEqual(row['context'], SELECTION)
        self.assertIsNone(job.context)

    def test_journal_has_no_text(self):
        logic, app, events, logged = self.app(focus='win-stop')
        out = io.StringIO()
        with patch('sys.stdout', out), patch('sys.stderr', out):
            self.run_session(app, logic)
            self.finish(app, text='Vrijdag geleverd door Blauwvingerhoed.')
        journal = out.getvalue()
        self.assertIn('research_finished', journal)
        for secret in ('Zeppelin', 'Kwartel', 'Blauwvinger', '77351', 'Vrijdag', 'factuur'):
            self.assertNotIn(secret, journal)

    def test_clipboard_when_focus_moved(self):
        logic, app, events, logged = self.app(focus='win-elsewhere')
        self.run_session(app, logic)
        job = self.finish(app, text='Regel een.\n\nRegel twee.')
        self.assertFalse(any(e[0] == 'paste' for e in events))
        clip = [e for e in events if e[0] == 'clipboard']
        self.assertEqual(clip[0][1], 'Regel een. Regel twee.')   # plain is single-line
        self.assertEqual((job.phase, job.step), ('clipboard', 'focus_changed'))
        self.assertEqual(logged[0]['delivered'], 'clipboard')

    def test_question_always_to_clipboard(self):
        logic, app, events, logged = self.app(focus='win-stop')
        self.run_session(app, logic)
        job = self.finish(app, status='question', text='Pomp A of pomp B?')
        self.assertFalse(any(e[0] == 'paste' for e in events))
        self.assertEqual((job.phase, job.step), ('clipboard', 'question'))

    def test_refused_paste_leaves_clipboard(self):
        logic, app, events, logged = self.app(focus='win-stop', paste_ok=False)
        self.run_session(app, logic)
        job = self.finish(app)
        self.assertEqual((job.phase, job.step), ('clipboard', 'paste_refused'))
        self.assertFalse(any(e[0] == 'clipboard' for e in events))   # type_text already set it

    def test_failures(self):
        for status, reason, step in (('timeout', 'timeout', 'timeout'), ('cancelled', 'escape', 'escape'),
                                     ('failed', 'pi_exit_3', 'pi_exit_3'), ('no_result', None, 'no_result'),
                                     ('no_result', 'no_deliver_result', 'no_deliver_result')):
            with self.subTest(status=status, reason=reason):
                logic, app, events, logged = self.app()
                self.run_session(app, logic)
                job = self.finish(app, status=status, reason=reason,
                                  text='Niets gevonden.' if status == 'no_result' else None)
                self.assertEqual((job.phase, job.step), ('failed', step))
                self.assertFalse(any(e[0] in ('paste', 'clipboard') for e in events))
                self.assertIn(('beep', 'rewrite_failed'), events)

    def test_waits_for_running_dictation_then_clipboard(self):
        logic, app, events, logged = self.app(focus='win-stop')
        self.run_session(app, logic)
        app.active = True            # Mark is dictating again and never stops
        app.RESEARCH_SETTLE_S = 0.2
        job = app._research_job
        outcome = research.Outcome(status='done', text='Klaar.', session_id='s')
        app._research_finished(job, outcome)
        self.assertFalse(any(e[0] == 'paste' for e in events))
        self.assertEqual((job.phase, job.step), ('clipboard', 'dictation_busy'))

    def test_busy_runner_refuses_and_hotkey_cancels(self):
        logic, app, events, _ = self.app(runner=FakeRunner(busy=True))
        session = self.run_session(app, logic)
        self.assertTrue(session.done.is_set())
        self.assertEqual((app._research_job.phase, app._research_job.step), ('failed', 'busy'))
        self.assertIn(('beep', 'rewrite_failed'), events)
        # AltGr + double tap while a session runs: cancel, nothing recorded.
        self.assertIsNone(app.start('nl', False, {'research': True}))
        self.assertEqual(app.research_runner.cancelled, ['hotkey'])

    def test_escape_cancels_only_while_busy(self):
        logic, app, events, _ = self.app()
        app._research_escape()
        self.assertEqual(app.research_runner.cancelled, [])
        app.research_runner._busy = True
        app._research_escape()
        self.assertEqual(app.research_runner.cancelled, ['escape'])

    def test_box_waits_for_the_dictation(self):
        logic, app, events, _ = self.app()
        sent = []
        app.status = types.SimpleNamespace(research=lambda *a: sent.append(a))
        app._box_writes = 0
        app._dictation_box_until = 0.0
        job = types.SimpleNamespace(phase='researching', step='bash:rg', steps=2, started=time.monotonic(),
                                    shown=None, writes=None)
        app.active = True
        self.assertFalse(app._research_show(job))
        app.active = False
        app._dictation_box_until = time.monotonic() + 60
        self.assertFalse(app._research_show(job))
        app._dictation_box_until = 0.0
        self.assertTrue(app._research_show(job))
        self.assertTrue(app._research_show(job))
        self.assertEqual(len(sent), 1)                 # unchanged: not sent again
        app._box_writes += 1                           # the dictation drew over it
        app._research_show(job)
        job.step = 'web_search'
        app._research_show(job)
        self.assertEqual([s[1] for s in sent], ['bash:rg', 'bash:rg', 'web_search'])


@unittest.skipIf(NODE is None, 'node not available')
class NodeTests(unittest.TestCase):
    def test_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.path.realpath(tmp)
            os.makedirs(os.path.join(cwd, 'notes'))
            os.makedirs(os.path.join(cwd, 'dossier', 'issues'))
            os.makedirs(os.path.join(cwd, 'credentials'))
            Path(cwd, 'dossier', 'PLAN.md').write_text('## Beslissen\n- a\n')
            Path(cwd, 'notes', 'old.md').write_text('old')
            env = dict(DICTATION_RESEARCH_CWD=cwd, DICTATION_RESEARCH_NOTES=os.path.join(cwd, 'notes'),
                       DICTATION_RESEARCH_PROPOSALS='1')
            cases = [
                ('bash', {'command': 'git push origin main'}, 'git_remote'),
                ('bash', {'command': 'git -C x commit -m y'}, 'git_write'),
                ('bash', {'command': 'git log --oneline -3'}, None),
                ('bash', {'command': 'rg -n service docs'}, None),
                ('bash', {'command': 'cat credentials/token.json'}, 'secret_path'),
                ('bash', {'command': 'ls; rm -rf notes'}, 'delete'),
                ('bash', {'command': 'systemctl --user restart x'}, 'service'),
                ('bash', {'command': 'kill 1234'}, 'signal'),
                ('bash', {'command': 'curl -X POST https://x.invalid'}, 'http_write'),
                ('bash', {'command': 'curl -s https://example.org'}, None),
                ('bash', {'command': 'echo x > notes/new.md'}, None),
                ('bash', {'command': 'echo x > dossier/PLAN.md'}, 'redirect_outside'),
                ('bash', {'command': 'python3 -c "import subprocess"'}, 'script_write'),
                ('read', {'path': 'credentials/token.json'}, 'secret_path'),
                ('read', {'path': '/home/u/.pi/agent/auth.json'}, 'secret_path'),
                ('read', {'path': 'dossier/PLAN.md'}, None),
                ('write', {'path': 'notes/2026-11-17_pomp.md', 'content': 'x'}, None),
                ('edit', {'path': 'notes/2026-11-17_pomp.md', 'edits': [{'oldText': 'x', 'newText': 'y'}]}, None),
                ('write', {'path': 'notes/old.md', 'content': 'x'}, 'overwrite'),
                ('write', {'path': '/tmp/elsewhere.md', 'content': 'x'}, 'write_outside'),
                ('write', {'path': 'dossier/code.py', 'content': 'x'}, 'write_outside'),
                ('write', {'path': 'dossier/issues/new_issue.md', 'content': 'x'}, None),
                ('edit', {'path': 'dossier/PLAN.md', 'edits': [{'oldText': '- a\n', 'newText': '- a\n- b\n'}]}, None),
                ('edit', {'path': 'dossier/PLAN.md', 'edits': [{'oldText': '- a', 'newText': '- z'}]},
                 'edit_not_insert'),
                ('update_roles', {}, 'tool_denied'),
                ('gmail_draft', {}, 'tool_denied'),
                ('deliver_result', {'text': 'x', 'summary': 'y', 'status': 'done'}, None),
            ]
            data = ''.join(json.dumps(dict(tool=t, input=i, env=env)) + '\n' for t, i, _ in cases)
            out = subprocess.run([NODE, str(ROOT / 'research_ext' / 'check_policy.mjs')], input=data.encode(),
                                 capture_output=True, timeout=30, check=True).stdout.decode().splitlines()
            for (tool, inp, rule), line in zip(cases, out):
                with self.subTest(tool=tool, input=inp):
                    decision = json.loads(line)
                    self.assertEqual(decision.get('rule') if decision['block'] else None, rule)
            self.assertEqual(len(out), len(cases))
            # Without the environment every write is refused (fail closed).
            data = json.dumps(dict(tool='write', input={'path': os.path.join(cwd, 'notes', 'n.md')}, env={}))
            out = subprocess.run([NODE, str(ROOT / 'research_ext' / 'check_policy.mjs')], input=data.encode(),
                                 capture_output=True, timeout=30, check=True).stdout.decode()
            self.assertTrue(json.loads(out)['block'])

    def test_extension_status_box_js(self):
        subprocess.run([NODE, str(ROOT / 'tests' / 'test_wayland_extension.mjs')], capture_output=True,
                       timeout=60, check=True)


if __name__ == '__main__':
    unittest.main()
