"""Research mode: one dictation starts one autonomous Pi session in the background.

AltGr + double tap records like rewrite mode (selection as context). At stop,
instead of one rewrite call, this module starts `pi --mode json -p` as a
subprocess of its own and returns at once: ordinary dictation never waits for
it. The session ends with the tool `deliver_result` (research_ext/research.ts);
only that text is ever pasted, never free model prose.

Safety, in layers (docs/research.md):
- a fixed session brief (research_ext/session_brief.md), not changeable by the
  dictation;
- the extension's tool_call guard (research_ext/policy.mjs): writes only in the
  notes directory (plus, optionally, new issue files and insert-only PLAN.md
  edits), no credential reads, no external writes, no deletion, no service or
  process control;
- `--no-mcp` and `--exclude-tools` for tools that act outside the machine or
  change accounts.

Process control: the runner keeps the Popen object it started (own process
group). Time-out and cancel signal that group only while the process is still
unreaped, so the PID cannot have been reused. Never a kill by pattern.

What reaches the journal: codes, counts, durations, tool names. Never the
dictation, the selection, the result or Pi's stderr.
"""
from __future__ import annotations

import glob
import json
import os
import re
import shutil
import signal
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
EXTENSION = os.path.join(HERE, 'research_ext', 'research.ts')
BRIEF = os.path.join(HERE, 'research_ext', 'session_brief.md')
RESULT_TOOL = 'deliver_result'
# Tools that must not even be offered: account/role changes, the user's
# browser, questions that cannot be answered headless, cockpit writes.
EXCLUDE_TOOLS = ('update_roles', 'browser_*', 'ask_user', 'cockpit_claim', 'canvas', 'mcp__*', 'gmail_*',
                 'craft_*')
# Marker the guard puts at the start of every refusal (policy.mjs, BLOCK_PREFIX).
BLOCK_PREFIX = 'Blocked by the dictation research guard'
TERM_GRACE_S = 5.0
AFTER_RESULT_S = 10.0       # a session that delivered but does not exit is stopped
                            # (research.ts already exits Pi 3 s after its shutdown)
STDERR_TAIL = 4096
READER_GRACE_S = 3.0        # output still open this long after Pi exited: signal the group
STEP_RE = re.compile(r'[a-z][a-z0-9_.:-]{0,39}')
FINAL_STATUSES = ('done', 'no_result', 'question')
# Programs the status box may name for a bash step. Anything else shows as
# 'bash': the first word of a command could be dictated text.
KNOWN_PROGRAMS = frozenset((
    'awk', 'base64', 'basename', 'bc', 'cat', 'cd', 'column', 'curl', 'cut', 'date', 'df', 'diff', 'dirname',
    'du', 'echo', 'env', 'fd', 'fdfind', 'file', 'find', 'git', 'grep', 'head', 'jq', 'less', 'ls', 'make',
    'md5sum', 'node', 'od', 'pandoc', 'pdfinfo', 'pdftotext', 'printf', 'pwd', 'python', 'readlink',
    'realpath', 'rg', 'sed', 'sha256sum', 'sort', 'sqlite', 'stat', 'tail', 'test', 'tr', 'tree', 'true',
    'uniq', 'wc', 'wget', 'which', 'xxd', 'yq'))


@dataclass(frozen=True)
class ResearchConfig:
    pi_bin: str
    model: str
    thinking: str
    cwd: str
    notes_dir: str              # absolute
    timeout_s: float
    node_bin: Optional[str] = None
    visible: bool = True
    proposals: bool = True
    escape_cancels: bool = True
    extension: str = EXTENSION
    brief: str = BRIEF


def _executable(path: Optional[str]) -> bool:
    return bool(path) and os.path.isabs(path) and os.path.isfile(path) and os.access(path, os.X_OK)


def load_config(loaded: Any) -> tuple[Optional[ResearchConfig], str]:
    """(config, 'ok') or (None, reason code). Never raises; reasons carry no values."""
    try:
        get = loaded.get
        if not get('research.enabled'):
            return None, 'disabled'
        pi_bin = get('research.pi_bin')
        pi_bin = os.path.expanduser(pi_bin) if pi_bin else None
        if not pi_bin:
            return None, 'pi_bin_missing'
        if not _executable(pi_bin):
            return None, 'pi_bin_not_executable'
        node_bin = get('research.node_bin')
        node_bin = os.path.expanduser(node_bin) if node_bin else None
        if node_bin and not _executable(node_bin):
            return None, 'node_bin_not_executable'
        if not node_bin and not shutil.which('node'):
            return None, 'node_missing'
        cwd = get('research.cwd')
        cwd = os.path.expanduser(cwd) if cwd else None
        if not cwd:
            return None, 'cwd_missing'
        if not (os.path.isabs(cwd) and os.path.isdir(cwd)):
            return None, 'cwd_not_a_directory'
        notes = get('research.notes_dir')
        if os.path.isabs(notes) or '..' in notes.replace('\\', '/').split('/'):
            return None, 'notes_dir_outside_cwd'
        if not (os.path.isfile(EXTENSION) and os.path.isfile(BRIEF)):
            return None, 'extension_missing'
        return ResearchConfig(
            pi_bin=pi_bin, node_bin=node_bin, model=get('research.model'), thinking=get('research.thinking'),
            cwd=os.path.realpath(cwd), notes_dir=os.path.join(os.path.realpath(cwd), notes),
            timeout_s=float(get('research.timeout_s')), visible=bool(get('research.visible')),
            proposals=bool(get('research.proposals')),
            escape_cancels=bool(get('research.escape_cancels'))), 'ok'
    except Exception as exc:  # noqa: BLE001 - never stop dictation for this mode
        return None, 'error_' + type(exc).__name__


def build_prompt(dictation: str, context: Optional[str], language: Optional[str]) -> str:
    """The task as sent on stdin (never on the command line, where ps shows it)."""
    parts = ['<opdracht>\n%s\n</opdracht>' % dictation.strip()]
    if context:
        parts.append('<selectie>\n%s\n</selectie>' % context.strip())
    if language:
        parts.append('Taal van het dictaat: %s' % language)
    return '\n\n'.join(parts) + '\n'


ARGV_MESSAGE = ('Carry out the task in <opdracht> above (with <selectie> as context, if present), following '
                'the dictation research brief. Finish with deliver_result.')


def command(cfg: ResearchConfig, session_id: str, name: str) -> list[str]:
    argv = [cfg.pi_bin, '--mode', 'json', '-p', '--session-id', session_id, '--name', name,
            '--model', cfg.model, '--thinking', cfg.thinking, '--no-mcp',
            '-e', cfg.extension, '--exclude-tools', ','.join(EXCLUDE_TOOLS),
            '--append-system-prompt', cfg.brief,
            '--append-system-prompt',
            'Your notes directory for this session: %s' % os.path.relpath(cfg.notes_dir, cfg.cwd),
            '--', ARGV_MESSAGE]
    return argv


def environment(cfg: ResearchConfig, base: Optional[dict] = None) -> dict:
    env = dict(os.environ if base is None else base)
    path = env.get('PATH', '/usr/bin:/bin')
    if cfg.node_bin:
        # A stable alias (e.g. dictation-node -> nvm/.../bin/node) may
        # have a different basename. Pi's /usr/bin/env node shebang needs
        # the target directory, not the alias directory.
        path = os.path.dirname(os.path.realpath(cfg.node_bin)) + os.pathsep + path
    env['PATH'] = path
    env['DICTATION_RESEARCH_CWD'] = cfg.cwd
    env['DICTATION_RESEARCH_NOTES'] = cfg.notes_dir
    env['DICTATION_RESEARCH_PROPOSALS'] = '1' if cfg.proposals else '0'
    if cfg.visible:
        env['PI_DASHBOARD_VISIBLE'] = '1'
    else:
        env.pop('PI_DASHBOARD_VISIBLE', None)
    return env


def step_code(tool: Any, args: Any = None) -> str:
    """Status-box step from a tool name: a code, never arguments (bash: the program name only)."""
    name = tool.lower() if isinstance(tool, str) else ''
    if not STEP_RE.fullmatch(name):
        return 'tool'
    if name == 'bash' and isinstance(args, dict) and isinstance(args.get('command'), str):
        words = args['command'].strip().split()
        while words and re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*=\S*', words[0]):
            words.pop(0)  # VAR=value prefixes
        program = os.path.basename(words[0]) if words else ''
        program = re.sub(r'\d+(\.\d+)*$', '', program.lower())  # python3.12 -> python
        if program in KNOWN_PROGRAMS:
            return 'bash:' + program
    return name


@dataclass
class Outcome:
    status: str                      # done | no_result | question | timeout | cancelled | failed
    reason: Optional[str] = None     # code only
    text: Optional[str] = None       # deliver_result.text
    summary: Optional[str] = None
    session_id: Optional[str] = None
    session_file: Optional[str] = None
    seconds: float = 0.0
    steps: int = 0
    blocked: list = field(default_factory=list)   # tool names only
    cost: Optional[float] = None
    returncode: Optional[int] = None

    @property
    def has_text(self) -> bool:
        return self.status == 'done' and bool(self.text and self.text.strip())


class Run:
    """One Pi session. start() returns at once; on_finish(outcome) runs on a thread of the run."""

    def __init__(self, cfg: ResearchConfig, prompt: str,
                 on_progress: Optional[Callable[[str, int], None]] = None,
                 on_finish: Optional[Callable[[Outcome], None]] = None,
                 log: Optional[Callable[..., None]] = None,
                 popen: Callable[..., Any] = subprocess.Popen,
                 clock: Callable[[], float] = time.monotonic,
                 sessions_dir: Optional[str] = None) -> None:
        self.cfg, self.prompt = cfg, prompt
        self.on_progress = on_progress or (lambda step, steps: None)
        self.on_finish = on_finish or (lambda outcome: None)
        self.log = log or (lambda kind, **details: None)
        self.popen, self.clock = popen, clock
        self.sessions_dir = sessions_dir
        self.session_id = str(uuid.uuid4())
        self.proc = None
        self.started = None
        self.step, self.steps = 'starting', 0
        self.blocked: list = []
        self.cost = 0.0
        self.cost_known = False
        self.result: Optional[dict] = None
        self._pending: dict = {}
        self._ended: Optional[str] = None     # timeout | cancelled (set before signalling)
        self._end_reason: Optional[str] = None
        self._lock = threading.Lock()
        self._done = threading.Event()
        self._stderr = bytearray()
        self.provider_error = False

    # --- control ---------------------------------------------------------
    @property
    def pid(self) -> Optional[int]:
        return getattr(self.proc, 'pid', None)

    @property
    def running(self) -> bool:
        return self.proc is not None and not self._done.is_set()

    def start(self) -> 'Run':
        name = 'dictatie-onderzoek %s' % time.strftime('%H:%M')
        self.started = self.clock()
        self.proc = self.popen(
            command(self.cfg, self.session_id, name), cwd=self.cfg.cwd, env=environment(self.cfg),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=True, close_fds=True)
        self.log('research_started', pid=self.pid, session_id=self.session_id, model=self.cfg.model,
                 thinking=self.cfg.thinking, timeout_s=self.cfg.timeout_s)
        threading.Thread(target=self._feed, daemon=True, name='research-stdin').start()
        self._stderr_reader = threading.Thread(target=self._read_stderr, daemon=True, name='research-stderr')
        self._reader = threading.Thread(target=self._read, daemon=True, name='research-reader')
        self._stderr_reader.start()
        self._reader.start()
        threading.Thread(target=self._watchdog, daemon=True, name='research-watchdog').start()
        threading.Thread(target=self._finish, daemon=True, name='research-finish').start()
        self._progress('starting')
        return self

    def cancel(self, reason: str = 'cancelled') -> bool:
        """Stop the session (cancel/time-out). True when a running session was signalled."""
        with self._lock:
            if self._done.is_set() or self.proc is None:
                return False
            if self._ended is None:
                self._ended = 'timeout' if reason == 'timeout' else 'cancelled'
                self._end_reason = reason
        self._terminate()
        return True

    def wait(self, timeout: Optional[float] = None) -> bool:
        return self._done.wait(timeout)

    def _signal(self, sig: int) -> bool:
        """Signal our own process group, only while our child is still unreaped."""
        proc = self.proc
        if proc is None or proc.poll() is not None:
            return False
        try:
            pgid = os.getpgid(proc.pid)
            if pgid == proc.pid:      # start_new_session: the group is ours
                os.killpg(pgid, sig)
            else:
                proc.send_signal(sig)
            return True
        except (ProcessLookupError, PermissionError, OSError):
            return False

    def _terminate(self) -> None:
        if not self._signal(signal.SIGTERM):
            return
        try:
            self.proc.wait(TERM_GRACE_S)
        except subprocess.TimeoutExpired:
            self._signal(signal.SIGKILL)
            self.log('research_killed', pid=self.pid)

    def _watchdog(self) -> None:
        if not self._done.wait(self.cfg.timeout_s):
            self.cancel('timeout')

    # --- I/O ---------------------------------------------------------------
    def _feed(self) -> None:
        try:
            self.proc.stdin.write(self.prompt.encode('utf-8'))
            self.proc.stdin.close()
        except (OSError, ValueError, AttributeError):
            pass

    def _read_stderr(self) -> None:
        stream = getattr(self.proc, 'stderr', None)
        if stream is None:
            return
        try:
            for chunk in iter(lambda: stream.read(1024), b''):
                self._stderr += chunk
                del self._stderr[:-STDERR_TAIL]
        except (OSError, ValueError):
            pass

    def _progress(self, step: str) -> None:
        self.step = step
        try:
            self.on_progress(step, self.steps)
        except Exception as exc:  # noqa: BLE001
            self.log('research_progress_error', error=type(exc).__name__)

    def handle(self, record: dict) -> None:
        """One JSONL record from Pi. Public for tests."""
        kind = record.get('type')
        if kind == 'session':
            sid = record.get('id')
            if isinstance(sid, str) and sid and sid != self.session_id:
                self.session_id = sid
        elif kind == 'tool_execution_start':
            name = record.get('toolName')
            self.steps += 1
            if name == RESULT_TOOL:
                self._pending[record.get('toolCallId')] = record.get('args')
            self._progress(step_code(name, record.get('args')))
        elif kind == 'tool_execution_end':
            name = record.get('toolName')
            if record.get('isError'):
                text = _result_text(record.get('result'))
                if text.startswith(BLOCK_PREFIX):
                    tool = name if isinstance(name, str) and STEP_RE.fullmatch(name.lower() or '-') else 'tool'
                    self.blocked.append(tool)
                    self.log('research_tool_blocked', tool=tool, rule=_rule_code(text))
                    self._progress('blocked')
            elif name == RESULT_TOOL:
                args = self._pending.pop(record.get('toolCallId'), None)
                if isinstance(args, dict):
                    self.result = args
                    threading.Thread(target=self._stop_after_result, daemon=True).start()
        elif kind == 'message_update':
            event = record.get('assistantMessageEvent') or {}
            sub = event.get('type')
            if sub == 'thinking_start' and self.step != 'thinking':
                self._progress('thinking')
            elif sub == 'text_start' and self.step != 'writing':
                self._progress('writing')
        elif kind == 'message_end':
            message = record.get('message') or {}
            if message.get('role') == 'assistant':
                usage = message.get('usage') or {}
                cost = (usage.get('cost') or {}).get('total') if isinstance(usage.get('cost'), dict) else None
                if isinstance(cost, (int, float)) and cost >= 0:
                    self.cost += cost
                    self.cost_known = True
                if message.get('stopReason') == 'error':
                    self.provider_error = True

    def _stop_after_result(self) -> None:
        if not self._done.wait(AFTER_RESULT_S):
            self.log('research_no_exit_after_result', pid=self.pid)
            with self._lock:
                if self._ended is None:
                    self._end_reason = 'no_exit_after_result'
            self._terminate()

    def _read(self) -> None:
        """Parse Pi's JSONL until EOF. The outcome comes from _finish, not from here."""
        try:
            for raw in iter(self.proc.stdout.readline, b''):
                line = raw.rstrip(b'\r\n')
                if not line:
                    continue
                try:
                    record = json.loads(line.decode('utf-8'))
                except (ValueError, UnicodeDecodeError):
                    continue
                if isinstance(record, dict):
                    try:
                        self.handle(record)
                    except Exception as exc:  # noqa: BLE001
                        self.log('research_record_error', error=type(exc).__name__)
        except (OSError, ValueError):
            pass

    def _finish(self) -> None:
        """Wait for Pi to exit, then for its output; report the outcome once.

        A process Pi left behind in our group can keep stdout open after Pi
        exits. The group is then signalled once (it cannot have been reused
        while it has members) and the outcome is reported regardless."""
        try:
            returncode = self.proc.wait()
        except Exception:  # noqa: BLE001
            returncode = None
        self._reader.join(READER_GRACE_S)
        if self._reader.is_alive():
            self.log('research_output_open_after_exit', pid=self.pid)
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError, OSError):
                pass
            self._reader.join(READER_GRACE_S)
        self._stderr_reader.join(READER_GRACE_S)
        if not self._reader.is_alive():
            for stream in (self.proc.stdout, getattr(self.proc, 'stderr', None)):
                try:
                    if stream is not None:
                        stream.close()
                except Exception:  # noqa: BLE001
                    pass
        outcome = self._outcome(returncode)
        self._done.set()
        try:
            self.on_finish(outcome)
        except Exception as exc:  # noqa: BLE001
            self.log('research_finish_error', error=type(exc).__name__)

    def _outcome(self, returncode: Optional[int]) -> Outcome:
        seconds = round(self.clock() - (self.started or self.clock()), 1)
        out = Outcome(status='failed', session_id=self.session_id, seconds=seconds, steps=self.steps,
                      blocked=list(self.blocked), cost=round(self.cost, 6) if self.cost_known else None,
                      returncode=returncode, session_file=self._session_file())
        result = self.result
        if isinstance(result, dict):
            status = result.get('status')
            out.status = status if status in FINAL_STATUSES else 'done'
            out.text = result.get('text') if isinstance(result.get('text'), str) else None
            out.summary = result.get('summary') if isinstance(result.get('summary'), str) else None
            if out.status == 'done' and not (out.text and out.text.strip()):
                out.status, out.reason = 'no_result', 'empty_text'
        elif self._ended:
            out.status, out.reason = self._ended, self._end_reason
        elif returncode not in (0, None):
            out.reason = 'provider_error' if self.provider_error else _stderr_reason(bytes(self._stderr), returncode)
        else:
            out.status, out.reason = 'no_result', 'provider_error' if self.provider_error else 'no_deliver_result'
        return out

    def _session_file(self) -> Optional[str]:
        base = self.sessions_dir or os.path.join(
            os.environ.get('PI_CODING_AGENT_DIR') or os.path.expanduser('~/.pi/agent'), 'sessions')
        try:
            found = glob.glob(os.path.join(glob.escape(base), '*', '*_%s.jsonl' % glob.escape(self.session_id)))
            return found[0] if found else None
        except Exception:  # noqa: BLE001
            return None


def _result_text(result: Any) -> str:
    if isinstance(result, dict):
        for part in result.get('content') or ():
            if isinstance(part, dict) and part.get('type') == 'text' and isinstance(part.get('text'), str):
                return part['text']
    return ''


def _rule_code(text: str) -> str:
    """The guard writes 'Blocked by ... [rule]: explanation'; keep the rule code only."""
    match = re.search(r'\[([a-z_]{1,40})\]', text)
    return match.group(1) if match else 'unknown'


def _stderr_reason(tail: bytes, returncode: int) -> str:
    """A reason code from Pi's stderr; the text itself is never journalled."""
    text = tail.decode('utf-8', 'replace').lower()
    if ("syntaxerror: the requested module 'node:module' does not provide an export named 'enablecompilecache'" in text
            or "syntaxerror: the requested module 'node:util' does not provide an export named 'styletext'" in text):
        return 'node_runtime_incompatible'
    if 'no api key' in text or 'not logged in' in text or 'unauthorized' in text or 'login' in text:
        return 'auth'
    if 'model' in text and ('not found' in text or 'unknown' in text):
        return 'model_unknown'
    return 'pi_exit_%d' % returncode if returncode >= 0 else 'pi_signal_%d' % -returncode


class Runner:
    """At most one research session; the dictation keeps its own threads.

    start() refuses while one runs (the caller decides: AltGr + double tap then
    cancels instead). Callbacks run on the session's reader thread.
    """

    def __init__(self, cfg: ResearchConfig, log: Optional[Callable[..., None]] = None,
                 run_factory: Callable[..., Run] = Run) -> None:
        self.cfg = cfg
        self.log = log or (lambda kind, **details: None)
        self.run_factory = run_factory
        self.current: Optional[Run] = None
        self._lock = threading.Lock()

    @property
    def busy(self) -> bool:
        run = self.current
        return run is not None and run.running

    def start(self, prompt: str, on_progress=None, on_finish=None) -> Optional[Run]:
        with self._lock:
            if self.busy:
                self.log('research_refused', reason='busy')
                return None
            run = self.run_factory(self.cfg, prompt, on_progress=on_progress, on_finish=on_finish, log=self.log)
            self.current = run
        try:
            return run.start()
        except Exception as exc:  # noqa: BLE001 - e.g. pi vanished since the check
            with self._lock:
                if self.current is run:
                    self.current = None
            self.log('research_start_failed', error=type(exc).__name__)
            raise

    def cancel(self, reason: str = 'cancelled') -> bool:
        run = self.current
        if run is None or not run.running:
            return False
        threading.Thread(target=run.cancel, args=(reason,), daemon=True).start()
        return True


# --- command line (install.sh --check) ----------------------------------------

EXTENSION_FILES = ('research.ts', 'policy.mjs', 'session_brief.md')


def check(loaded: Any) -> tuple[bool, list[str]]:
    """(ok, lines): off is fine; on needs pi, node and the extension files."""
    if not loaded.get('research.enabled'):
        return True, ['Research mode: off (research.enabled is false)']
    lines, ok = [], True
    for kind, details in getattr(loaded, 'events', ()):
        if str(details.get('key', '')).startswith('research.'):
            ok = False
            lines.append('Research mode: %s refused (%s); the default applies'
                         % (details.get('key'), details.get('reason', kind)))
    for name in EXTENSION_FILES:
        if not os.path.isfile(os.path.join(HERE, 'research_ext', name)):
            ok = False
            lines.append('Research mode: MISSING research_ext/%s' % name)
    cfg, reason = load_config(loaded)
    if cfg is None:
        ok = False
        lines.append('Research mode: NOT USABLE (%s); AltGr sessions dictate normally' % reason)
        return ok, lines
    node = cfg.node_bin or shutil.which('node')
    try:
        version = subprocess.run([node, '--version'], capture_output=True, timeout=10,
                                 check=False).stdout.decode().strip()
    except Exception as exc:  # noqa: BLE001
        version, ok = '', False
        lines.append('Research mode: node does not run (%s)' % type(exc).__name__)
    if ok:
        lines.append('Research mode: on; pi %s, node %s %s, model %s, cwd %s, notes %s'
                     % (cfg.pi_bin, node, version, cfg.model, cfg.cwd, os.path.relpath(cfg.notes_dir, cfg.cwd)))
    return ok, lines


def main(argv: Optional[list] = None) -> int:
    import argparse
    import sys
    import settings
    parser = argparse.ArgumentParser(description='Research mode helper')
    parser.add_argument('--check', action='store_true', help='check pi, node and the extension')
    parser.add_argument('--config', help='config.yaml path (default: see docs/configuration.md)')
    args = parser.parse_args(argv)
    if not args.check:
        parser.print_help()
        return 2
    ok, lines = check(settings.load(explicit_path=args.config))
    print('\n'.join(lines), file=sys.stdout if ok else sys.stderr)
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
