"""Searchable local log of every LLM call (rewrite now; research sessions later).

One SQLite database in WAL mode, outside the repository, with two tables:
llm_call holds one content-free record per direct call in the shared
llm-call/1 form (model, provider, times, tokens, cost, status, sha256 of
request and answer); llm_content holds the text with an FTS5 index, keyed by
the same request_id. File: $XDG_DATA_HOME/faster-whisper-dictation/llm_log.db
(directory 0700, file 0600). Reference: docs/llm_log.md.

Never in the paste path: the dictation thread only calls LlmLog.record(),
which is a put_nowait on a bounded queue. One daemon writer thread filters,
writes and runs the retention job. A full queue, a slow disk, a locked
database or any write error drops the row with a journal line that holds
numbers and reasons only, never text.

Privacy gates that no setting can switch off:
- secret filter: API keys, tokens, private-key blocks, password assignments
  and long hex/base64 runs become [geheim] before anything is written;
- no row at all when the target window is a password manager or the context
  came from a source marked as secret;
- no audio, no request headers, no endpoint or key.

Retention (log.text_days, log.meta_days): the text columns are wiped after
text_days (the llm_call record and its sha256 stay), both rows are deleted
after meta_days. secure_delete, an FTS
'optimize' and a WAL checkpoint keep wiped text out of free pages and the WAL.
The job runs when the writer starts (service start) and then once a day inside
the service; no separate timer.

Command line:
  python3 llm_log.py --check [--config PATH]   directory and file permissions
  python3 llm_log.py --purge [--config PATH]   run the retention job once now
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import queue
import re
import sqlite3
import stat
import sys
import threading
import time
import uuid
from typing import Any, Callable, Optional

APP_DIR = 'faster-whisper-dictation'
DB_NAME = 'llm_log.db'
STATUS_NAME = 'llm_log_status.json'
QUEUE_SIZE = 1000
BUSY_TIMEOUT_MS = 2000
PURGE_INTERVAL_S = 24 * 3600
SECRET_MARK = '[geheim]'
TEXT_COLUMNS = ('context', 'input', 'output')
SCHEMA_VERSION = 1

# Window classes (lowercased substrings) of password managers and secret prompts.
PASSWORD_WINDOW_HINTS = ('keepass', 'bitwarden', '1password', 'lastpass', 'enpass', 'keeweb',
                         'protonpass', 'proton-pass', 'seahorse', 'secrets', 'kwallet',
                         'pinentry', 'gcr-prompter', 'polkit')

# Order matters: whole blocks first, then prefixed tokens, then generic runs.
_SECRET_PATTERNS = (
    re.compile(r'-----BEGIN [A-Z0-9 ]*(?:PRIVATE KEY|CERTIFICATE|PGP [A-Z ]+)-----.*?'
               r'(?:-----END [A-Z0-9 ]*-----|\Z)', re.S),
    re.compile(r'(?<![\w-])(?:sk-|sk_live_|sk_test_|rk_live_|ghp_|gho_|ghu_|ghs_|ghr_|github_pat_|glpat-|'
               r'xox[abprs]-|AKIA|ASIA|AIza|ya29\.|hf_|npm_)[A-Za-z0-9_\-.]{8,}'),
    re.compile(r'\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}'),       # JWT
    re.compile(r'(?i)\b(bearer|basic)\s+[A-Za-z0-9_\-.=+/]{16,}'),
    re.compile(r'(?i)\b(password|passwd|wachtwoord|passphrase|secret|api[_-]?key|token)\b(\s*[:=]\s*)\S+'),
    re.compile(r'(?<![A-Za-z0-9])[A-Fa-f0-9]{32,}(?![A-Za-z0-9])'),                      # long hex
    re.compile(r'(?<![A-Za-z0-9+/=_-])(?=[A-Za-z0-9+/_-]*\d)(?=[A-Za-z0-9+/_-]*[A-Za-z])'
               r'[A-Za-z0-9+/_-]{40,}={0,2}'),                                          # long base64
)


def filter_secrets(text: Optional[str]) -> tuple[Optional[str], int]:
    """(text with every secret-looking run replaced by [geheim], replacements)."""
    if not text:
        return text, 0
    total = 0
    for pattern in _SECRET_PATTERNS:
        if pattern.groups >= 2:   # keep the label ("password: "), hide the value
            text, n = pattern.subn(lambda m: m.group(1) + m.group(2) + SECRET_MARK
                                   if m.lastindex and m.lastindex >= 2 else SECRET_MARK, text)
        elif pattern.groups == 1:  # "Bearer <token>"
            text, n = pattern.subn(lambda m: m.group(1) + ' ' + SECRET_MARK, text)
        else:
            text, n = pattern.subn(SECRET_MARK, text)
        total += n
    return text, total


def is_password_window(window_class: Optional[str]) -> bool:
    """True when the target window class looks like a password manager or secret prompt."""
    name = (window_class or '').lower()
    return bool(name) and any(hint in name for hint in PASSWORD_WINDOW_HINTS)


def prompt_version(system_prompt: str) -> str:
    """Short stable hash of the system prompt that was sent (prompt changes become visible)."""
    return hashlib.sha256((system_prompt or '').encode('utf-8')).hexdigest()[:12]


def default_path(environ=None) -> str:
    env = os.environ if environ is None else environ
    base = env.get('XDG_DATA_HOME') or os.path.join(os.path.expanduser('~'), '.local', 'share')
    return os.path.join(base, APP_DIR, DB_NAME)


def resolve_path(configured: Optional[str], environ=None) -> str:
    return os.path.expanduser(configured) if configured else default_path(environ)


# Content-free call record, exactly the shared llm-call/1 contract (flat SQLite
# form). Nothing of the text ever goes in here; usage tellers are as the
# provider reported them, a missing value stays NULL (unknown), never 0.
CALL_COLUMNS = (
    'schema', 'ts', 'request_id', 'response_id', 'provider_request_id', 'producer', 'caller', 'step',
    'traffic', 'session_id', 'parent_session_id', 'correlation_id', 'assignment_project',
    'assignment_session_id', 'provider', 'model', 'transport', 'route', 'execution_status',
    'http_status', 'error', 'duration_s', 'prompt_tokens', 'completion_tokens', 'cached_tokens',
    'cache_write_tokens', 'reasoning_tokens', 'usage_complete', 'cost_usd', 'price_basis',
    'content_store', 'content_key', 'input_sha256', 'input_bytes', 'output_sha256', 'output_bytes',
)
CALL_SCHEMA = 'llm-call/1'
CALL_SQL = """
CREATE TABLE IF NOT EXISTS llm_call (
  request_id TEXT PRIMARY KEY,
  schema TEXT NOT NULL CHECK (schema = 'llm-call/1'),
  ts TEXT NOT NULL,
  producer TEXT NOT NULL,
  response_id TEXT,
  provider_request_id TEXT,
  caller TEXT,
  step TEXT,
  traffic TEXT NOT NULL,
  session_id TEXT,
  parent_session_id TEXT,
  correlation_id TEXT,
  assignment_project TEXT,
  assignment_session_id TEXT,
  provider TEXT NOT NULL,
  model TEXT NOT NULL,
  transport TEXT,
  route TEXT,
  execution_status TEXT NOT NULL,
  http_status INTEGER,
  error TEXT,
  duration_s REAL,
  prompt_tokens INTEGER,
  completion_tokens INTEGER,
  cached_tokens INTEGER,
  cache_write_tokens INTEGER,
  reasoning_tokens INTEGER,
  usage_complete INTEGER NOT NULL CHECK (usage_complete IN (0, 1)),
  cost_usd REAL,
  price_basis TEXT NOT NULL,
  content_store TEXT,
  content_key TEXT,
  input_sha256 TEXT,
  input_bytes INTEGER,
  output_sha256 TEXT,
  output_bytes INTEGER
);
CREATE UNIQUE INDEX IF NOT EXISTS llm_call_response_id ON llm_call(response_id) WHERE response_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS llm_call_ts ON llm_call(ts);
"""

# This application's own table: the text, its search index and retention,
# plus what only this application knows (mode, delivery, prompt hash). Keyed
# by request_id; a rewrite row has a matching llm_call row. A research row
# (a Pi session, which gets no llm-call record) has only this row.
CONTENT_COLUMNS = ('request_id', 'epoch', 'ts', 'kind', 'mode', 'language', 'prompt_version', 'status',
                   'reason', 'delivered', 'context', 'input', 'output', 'secrets_replaced',
                   'pi_session_id', 'pi_session_file')
CONTENT_SQL = """
CREATE TABLE IF NOT EXISTS llm_content(
  request_id TEXT PRIMARY KEY,   -- = llm_call.request_id for a direct call
  epoch REAL NOT NULL,           -- seconds since 1970, for retention
  ts TEXT NOT NULL,              -- ISO 8601, local time with offset
  kind TEXT NOT NULL,            -- rewrite | research
  mode TEXT,                     -- opmaak | regel | research
  language TEXT,                 -- written language (nl, en, ...)
  prompt_version TEXT,           -- hash of the system prompt sent
  status TEXT,                   -- ok | failed | timeout | cancelled | no_result | running
  reason TEXT,                   -- failure reason, no text
  delivered TEXT,                -- pasted | clipboard | none
  context TEXT,                  -- selection / field context as sent
  input TEXT,                    -- dictation
  output TEXT,                   -- text the model returned (also when rejected)
  secrets_replaced INTEGER NOT NULL DEFAULT 0,
  pi_session_id TEXT, pi_session_file TEXT,   -- research sessions only
  text_purged_at TEXT            -- set when retention wiped the text
);
CREATE INDEX IF NOT EXISTS llm_content_epoch ON llm_content(epoch);
CREATE INDEX IF NOT EXISTS llm_content_pi_session ON llm_content(pi_session_id);
CREATE VIRTUAL TABLE IF NOT EXISTS llm_content_fts USING fts5(context, input, output,
  content=llm_content, content_rowid=rowid, tokenize='unicode61 remove_diacritics 2');
CREATE TRIGGER IF NOT EXISTS llm_content_ai AFTER INSERT ON llm_content BEGIN
  INSERT INTO llm_content_fts(rowid, context, input, output) VALUES (new.rowid, new.context, new.input, new.output);
END;
CREATE TRIGGER IF NOT EXISTS llm_content_ad AFTER DELETE ON llm_content BEGIN
  INSERT INTO llm_content_fts(llm_content_fts, rowid, context, input, output)
    VALUES ('delete', old.rowid, old.context, old.input, old.output);
END;
CREATE TRIGGER IF NOT EXISTS llm_content_au AFTER UPDATE OF context, input, output ON llm_content BEGIN
  INSERT INTO llm_content_fts(llm_content_fts, rowid, context, input, output)
    VALUES ('delete', old.rowid, old.context, old.input, old.output);
  INSERT INTO llm_content_fts(rowid, context, input, output) VALUES (new.rowid, new.context, new.input, new.output);
END;
"""
SCHEMA_SQL = CALL_SQL + CONTENT_SQL


def _sha(text: Optional[str]) -> tuple[Optional[str], Optional[int]]:
    if text is None:
        return None, None
    data = text.encode('utf-8')
    return hashlib.sha256(data).hexdigest(), len(data)


def _count(value: Any) -> Optional[int]:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def call_row(request_id: str, row: dict[str, Any], producer: str, traffic: str) -> Optional[tuple]:
    """Flat llm-call/1 values for a direct call, or None (no attempt data, e.g. a research row).

    Missing stays unknown: an absent counter or cost is NULL, never 0."""
    attempt = row.get('call')
    if not isinstance(attempt, dict):
        return None
    usage = attempt.get('usage') if isinstance(attempt.get('usage'), dict) else {}
    prompt_details = usage.get('prompt_tokens_details') if isinstance(usage.get('prompt_tokens_details'), dict) else {}
    completion_details = (usage.get('completion_tokens_details')
                          if isinstance(usage.get('completion_tokens_details'), dict) else {})
    prompt_tokens, completion_tokens = _count(usage.get('prompt_tokens')), _count(usage.get('completion_tokens'))
    cost = usage.get('cost')
    cost = float(cost) if isinstance(cost, (int, float)) and not isinstance(cost, bool) and cost >= 0 else None
    reason = row.get('reason')
    sent, received = bool(attempt.get('sent')), bool(attempt.get('received'))
    http_status = attempt.get('http_status')
    http_status = http_status if isinstance(http_status, int) and 100 <= http_status <= 599 else None
    if not sent:
        execution = 'error_before_send'
    elif received and reason not in ('provider_error', 'invalid_response'):
        execution = 'ok'      # the endpoint answered; a rejection by this application is no call error
    else:
        execution = 'error_after_send'
    if cost is not None:
        basis = 'provider'
    elif not sent:
        basis = 'not_sent'
    elif reason == 'provider_error' or (http_status is not None and 400 <= http_status < 500):
        basis = 'rejected'
    else:
        basis = 'unknown'
    started = attempt.get('started')
    started = float(started) if isinstance(started, (int, float)) else float(row['epoch'])
    in_sha, in_bytes = _sha(attempt.get('request'))
    out_sha, out_bytes = _sha(attempt.get('response'))
    seconds = row.get('seconds')
    def short(value: Any) -> Optional[str]:
        return value[:200] if isinstance(value, str) and value and '\n' not in value else None
    values = dict(
        schema=CALL_SCHEMA, ts=_utc_iso(started), request_id=request_id,
        response_id=short(attempt.get('response_id')), producer=producer, caller=row.get('caller') or 'dictation',
        step=row.get('kind') or 'rewrite', traffic=row.get('traffic') or traffic,
        session_id=short(row.get('pi_session_id')),
        provider=short(row.get('provider')) or 'unknown',
        model=short(attempt.get('model')) or short(row.get('model')) or 'unknown',
        transport=short(attempt.get('transport')), execution_status=execution, http_status=http_status,
        error=None if execution == 'ok' else short(reason) or 'unknown',
        duration_s=float(seconds) if isinstance(seconds, (int, float)) and seconds >= 0 else None,
        prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
        cached_tokens=_count(prompt_details.get('cached_tokens')),
        cache_write_tokens=_count(prompt_details.get('cache_write_tokens')),
        reasoning_tokens=_count(completion_details.get('reasoning_tokens')),
        usage_complete=int(prompt_tokens is not None and completion_tokens is not None),
        cost_usd=cost, price_basis=basis, content_store='sqlite', content_key=request_id,
        input_sha256=in_sha, input_bytes=in_bytes, output_sha256=out_sha, output_bytes=out_bytes)
    return tuple(values.get(c) for c in CALL_COLUMNS)


def _ensure_private_dir(path: str) -> None:
    """Create the directory 0700. An existing directory is never re-moded (log.path may
    point into a shared one); install.sh --check reports it, the files stay 0600."""
    if not os.path.isdir(path):
        os.makedirs(path, mode=0o700, exist_ok=True)
        os.chmod(path, 0o700)


def _ensure_private_file(path: str) -> None:
    """Create the file 0600 before SQLite opens it; tighten an existing one."""
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    os.close(fd)
    if stat.S_IMODE(os.stat(path).st_mode) != 0o600:
        os.chmod(path, 0o600)


def connect(path: str, busy_timeout_ms: int = BUSY_TIMEOUT_MS) -> sqlite3.Connection:
    """Open (and create) the log with private permissions, WAL and the schema."""
    _ensure_private_dir(os.path.dirname(path) or '.')
    _ensure_private_file(path)
    conn = sqlite3.connect(path, timeout=busy_timeout_ms / 1000, isolation_level=None,
                           check_same_thread=False)
    conn.execute('PRAGMA busy_timeout=%d' % int(busy_timeout_ms))
    conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('PRAGMA synchronous=NORMAL')
    conn.execute('PRAGMA secure_delete=ON')   # wiped text does not linger in free pages
    conn.executescript(SCHEMA_SQL)
    conn.execute('PRAGMA user_version=%d' % SCHEMA_VERSION)
    for suffix in ('-wal', '-shm'):            # SQLite copies the main file's mode; make sure
        side = path + suffix
        if os.path.exists(side) and stat.S_IMODE(os.stat(side).st_mode) != 0o600:
            os.chmod(side, 0o600)
    return conn


def purge(conn: sqlite3.Connection, text_days: int, meta_days: int,
          now: Optional[float] = None) -> dict[str, int]:
    """Wipe text older than text_days, delete rows older than meta_days (0 = keep)."""
    now = time.time() if now is None else now
    stamp = _iso(now)
    wiped = deleted = calls_deleted = 0
    conn.execute('BEGIN IMMEDIATE')
    try:
        if meta_days > 0:
            cutoff = now - meta_days * 86400
            deleted = conn.execute('DELETE FROM llm_content WHERE epoch < ?', (cutoff,)).rowcount
            # Same UTC text form as written, so string order is time order.
            calls_deleted = conn.execute('DELETE FROM llm_call WHERE ts < ?', (_utc_iso(cutoff),)).rowcount
        if text_days > 0:
            # The llm_call row and its sha256 stay: proof that the call existed.
            wiped = conn.execute(
                'UPDATE llm_content SET context=NULL, input=NULL, output=NULL, text_purged_at=? '
                'WHERE epoch < ? AND text_purged_at IS NULL', (stamp, now - text_days * 86400)).rowcount
        conn.execute('COMMIT')
    except BaseException:
        conn.execute('ROLLBACK')
        raise
    if wiped or deleted or calls_deleted:
        conn.execute("INSERT INTO llm_content_fts(llm_content_fts) VALUES ('optimize')")
        conn.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    return dict(text_wiped=wiped, rows_deleted=deleted, calls_deleted=calls_deleted)


def _iso(epoch: float) -> str:
    return datetime.datetime.fromtimestamp(epoch).astimezone().isoformat(timespec='seconds')


def _utc_iso(epoch: float) -> str:
    """UTC with an explicit zone, one fixed form (llm-call/1 ts)."""
    return datetime.datetime.fromtimestamp(epoch, datetime.timezone.utc).isoformat(
        timespec='milliseconds').replace('+00:00', 'Z')


def _shorten(text: str, max_chars: int) -> str:
    if not max_chars or len(text) <= max_chars:
        return text
    from context_capture import shorten   # same head + [ingekort] + tail rule as the context itself
    return shorten(text, max_chars)


class LlmLog:
    """Bounded queue plus one writer thread. record() never blocks and never raises."""

    def __init__(self, path: str, text_days: int = 90, meta_days: int = 730, context_max_chars: int = 0,
                 diagnostic: Optional[Callable[..., None]] = None, queue_size: int = QUEUE_SIZE,
                 purge_interval_s: float = PURGE_INTERVAL_S, busy_timeout_ms: int = BUSY_TIMEOUT_MS,
                 clock: Callable[[], float] = time.time, producer: str = 'DCT', traffic: str = 'real'):
        self.path = path
        self.producer = producer
        self.traffic = traffic
        self.text_days = text_days
        self.meta_days = meta_days
        self.context_max_chars = context_max_chars
        self.diagnostic = diagnostic or (lambda kind, **details: None)
        self.purge_interval_s = purge_interval_s
        self.busy_timeout_ms = busy_timeout_ms
        self.clock = clock
        self.queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self.status_path = os.path.join(os.path.dirname(path) or '.', STATUS_NAME)
        self.counts = dict(written=0, dropped=0, skipped_password=0, secrets_replaced=0)
        self.last_write: Optional[str] = None
        self.last_purge: Optional[dict[str, Any]] = None
        self.last_error: Optional[str] = None
        self._conn: Optional[sqlite3.Connection] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = object()
        self._next_purge = 0.0   # due at once: the first job runs when the writer starts

    # --- dictation side -------------------------------------------------
    def start(self) -> 'LlmLog':
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name='llm-log-writer', daemon=True)
            self._thread.start()
        return self

    def record(self, **row: Any) -> bool:
        """Queue one call. False (and a journal line) when the row is skipped or dropped."""
        try:
            if row.pop('password_field', False) or is_password_window(row.pop('window_class', None)):
                self.counts['skipped_password'] += 1
                self.diagnostic('llm_log_skipped', reason='password_field',
                                skipped=self.counts['skipped_password'])
                return False
            row.setdefault('epoch', self.clock())
            self.queue.put_nowait(row)
            return True
        except queue.Full:
            self._dropped('queue_full')
        except Exception as exc:  # noqa: BLE001 - logging must never hurt dictation
            self._dropped('record_error', error=type(exc).__name__)
        return False

    def close(self, timeout: float = 2.0) -> None:
        if self._thread is not None:
            try:
                self.queue.put(self._stop, timeout=timeout)
            except queue.Full:
                return
            self._thread.join(timeout)

    # --- writer thread --------------------------------------------------
    def _dropped(self, reason: str, **details: Any) -> None:
        self.counts['dropped'] += 1
        self.last_error = reason
        self.diagnostic('llm_log_dropped', reason=reason, dropped=self.counts['dropped'], **details)

    def _connection(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = connect(self.path, self.busy_timeout_ms)
        return self._conn

    def _run(self) -> None:
        while True:
            self._maybe_purge()
            wait = max(1.0, min(3600.0, self._next_purge - time.monotonic()))
            try:
                row = self.queue.get(timeout=wait)
            except queue.Empty:
                continue
            if row is self._stop:
                break
            self._write(row)
        if self._conn is not None:
            try:
                self._conn.close()
            except sqlite3.Error:
                pass

    def _prepare(self, row: dict[str, Any]) -> tuple[Optional[tuple], tuple]:
        epoch = float(row.get('epoch') or self.clock())
        values = dict(row, epoch=epoch, ts=row.get('ts') or _iso(epoch))
        values['request_id'] = values.get('request_id') or uuid.uuid4().hex
        # Hashes and counters from the raw attempt, before any filtering.
        call = call_row(values['request_id'], values, self.producer, self.traffic)
        system = values.pop('system_prompt', None)
        if system and not values.get('prompt_version'):
            values['prompt_version'] = prompt_version(system)
        replaced = 0
        if values.get('context') and self.context_max_chars:
            values['context'] = _shorten(values['context'], self.context_max_chars)
        for column in TEXT_COLUMNS:
            values[column], n = filter_secrets(values.get(column))
            replaced += n
        values['secrets_replaced'] = replaced
        self.counts['secrets_replaced'] += replaced
        return call, tuple(values.get(column) for column in CONTENT_COLUMNS)

    def _write(self, row: dict[str, Any]) -> None:
        try:
            call, content = self._prepare(row)
            conn = self._connection()
            conn.execute('BEGIN IMMEDIATE')
            try:
                if call is not None:
                    conn.execute('INSERT INTO llm_call(%s) VALUES (%s)'
                                 % (', '.join(CALL_COLUMNS), ', '.join('?' * len(CALL_COLUMNS))), call)
                conn.execute('INSERT INTO llm_content(%s) VALUES (%s)'
                             % (', '.join(CONTENT_COLUMNS), ', '.join('?' * len(CONTENT_COLUMNS))), content)
                conn.execute('COMMIT')
            except BaseException:
                if conn.in_transaction:
                    conn.execute('ROLLBACK')
                raise
        except sqlite3.OperationalError as exc:
            self._dropped('locked' if 'locked' in str(exc) or 'busy' in str(exc) else 'write_error',
                          error=type(exc).__name__)
            self._reset_connection()
        except Exception as exc:  # noqa: BLE001
            self._dropped('write_error', error=type(exc).__name__)
            self._reset_connection()
        else:
            self.counts['written'] += 1
            self.last_write = _iso(self.clock())
            self.last_error = None
        self._write_status()

    def _reset_connection(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except sqlite3.Error:
                pass
            self._conn = None

    def _maybe_purge(self) -> None:
        if time.monotonic() < self._next_purge:
            return
        self._next_purge = time.monotonic() + self.purge_interval_s
        if not os.path.exists(self.path):
            return   # nothing to purge; the database is created at the first write
        try:
            result = purge(self._connection(), self.text_days, self.meta_days, self.clock())
        except Exception as exc:  # noqa: BLE001 - retried at the next interval
            self.last_error = 'purge_failed'
            self.diagnostic('llm_log_purge_failed', error=type(exc).__name__)
            self._reset_connection()
        else:
            self.last_purge = dict(result, at=_iso(self.clock()))
            self.diagnostic('llm_log_purged', **result)
        self._write_status()

    def _write_status(self) -> None:
        """Small health file for Dream or the dashboard: counts and times, never text."""
        status = dict(path=self.path, last_write=self.last_write, last_purge=self.last_purge,
                      last_error=self.last_error, text_days=self.text_days, meta_days=self.meta_days,
                      **self.counts)
        try:
            _ensure_private_dir(os.path.dirname(self.status_path) or '.')
            tmp = self.status_path + '.tmp'
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, 'w', encoding='utf-8') as fh:
                json.dump(status, fh)
            os.replace(tmp, self.status_path)
        except OSError:
            pass


def from_settings(loaded, diagnostic: Optional[Callable[..., None]] = None, environ=None) -> Optional[LlmLog]:
    """A started LlmLog from the `log:` settings, or None when the log is off."""
    if not loaded.get('log.enabled'):
        diagnostic and diagnostic('llm_log_off')
        return None
    log = LlmLog(resolve_path(loaded.get('log.path'), environ), text_days=loaded.get('log.text_days'),
                 meta_days=loaded.get('log.meta_days'), context_max_chars=loaded.get('log.context_max_chars'),
                 diagnostic=diagnostic, producer=loaded.get('log.producer'))
    if diagnostic:
        diagnostic('llm_log_on', text_days=log.text_days, meta_days=log.meta_days,
                   context_max_chars=log.context_max_chars)
    return log.start()


# --- command line -------------------------------------------------------------

def check(path: str, enabled: bool) -> tuple[bool, list[str]]:
    """(ok, lines) for install.sh --check: directory 0700, files 0600, owned by this user."""
    if not enabled:
        return True, ['LLM log: off (log.enabled is false)']
    lines, ok = [], True
    directory = os.path.dirname(path) or '.'
    if not os.path.isdir(directory):
        return True, ['LLM log: on; %s is created at the first call' % directory]
    for target, wanted in [(directory, 0o700)] + [(path + s, 0o600) for s in ('', '-wal', '-shm')] + \
            [(os.path.join(directory, STATUS_NAME), 0o600)]:
        if not os.path.exists(target):
            continue
        info = os.stat(target)
        mode = stat.S_IMODE(info.st_mode)
        if mode != wanted or info.st_uid != os.getuid():
            ok = False
            lines.append('LLM log: WRONG permissions %s on %s (want %s, owner this user)'
                         % (oct(mode), target, oct(wanted)))
    if ok:
        lines.append('LLM log: on; %s and its files are private' % directory)
    return ok, lines


def main(argv=None) -> int:
    import settings
    parser = argparse.ArgumentParser(description='LLM log helper')
    parser.add_argument('--check', action='store_true', help='check directory and file permissions')
    parser.add_argument('--purge', action='store_true', help='run the retention job once now')
    parser.add_argument('--config', help='config.yaml path (default: see docs/configuration.md)')
    args = parser.parse_args(argv)
    loaded = settings.load(explicit_path=args.config)
    path = resolve_path(loaded.get('log.path'))
    if args.check:
        ok, lines = check(path, loaded.get('log.enabled'))
        print('\n'.join(lines), file=sys.stdout if ok else sys.stderr)
        return 0 if ok else 1
    if args.purge:
        if not os.path.exists(path):
            print('No log at %s: nothing to purge.' % path)
            return 0
        conn = connect(path)
        print(json.dumps(purge(conn, loaded.get('log.text_days'), loaded.get('log.meta_days'))))
        conn.close()
        return 0
    parser.print_help()
    return 2


if __name__ == '__main__':
    sys.exit(main())
