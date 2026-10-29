"""Rewrite mode: capture the text the user is replying to, once, at session start.

Source order: the X11 PRIMARY selection (what is highlighted) when it is at most
PRIMARY_MAX_AGE_S old, else CLIPBOARD when at most CLIPBOARD_MAX_AGE_S old and
not written by CopyQ (after an ordinary dictation CLIPBOARD holds our own
previous text). Anything else: no context.

Age comes from the selection owner's TIMESTAMP target compared with the X
server clock; an owner that gives no TIMESTAMP counts as too old (fail closed).
Metadata (owner, TARGETS, TIMESTAMP) is read by a short-lived helper process
(this file with --probe, python-xlib) so a hanging owner can be killed by a hard
time-out; the text itself is read with xclip, also with a time-out.

Privacy: the captured text stays in memory only. Nothing here prints or logs
it, not even a hash; Context.journal() holds metadata only.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

PRIMARY_MAX_AGE_S = 300
CLIPBOARD_MAX_AGE_S = 120
MAX_CHARS = 8000
MIN_CHARS = 20
PROBE_TIMEOUT_S = 1.5
READ_TIMEOUT_S = 0.5
TRUNCATED_MARK = '[ingekort]'

# Set by password managers (KeePassXC and others) and by CopyQ for hidden items.
SECRET_HINT_TARGETS = ('x-kde-passwordManagerHint', 'application/x-copyq-hidden')
COPYQ_OWNER_TARGET = 'application/x-copyq-owner'

_SECRET_RE = re.compile(r'-----BEGIN|(?<![\w-])(?:sk-|sk_live_|ghp_|gho_|github_pat_|glpat-|xox[abprs]-|AKIA|AIza)')


@dataclass
class Context:
    text: Optional[str]
    source: str = 'none'
    chars: int = 0
    age_s: Optional[float] = None
    truncated: bool = False
    reasons: dict[str, str] = field(default_factory=dict)

    def journal(self) -> dict[str, Any]:
        """Metadata only: never the text."""
        return dict(source=self.source, chars=self.chars, age_s=self.age_s,
                    truncated=self.truncated, reasons=dict(self.reasons))


def secret_reason(text: str) -> Optional[str]:
    """'secret_pattern' or 'too_short' for text that is no reply context."""
    stripped = text.strip()
    if _SECRET_RE.search(stripped):
        return 'secret_pattern'
    if (len(stripped) >= 12 and not re.search(r'\s', stripped)
            and re.search(r'\d', stripped) and re.search(r'[^\d]', stripped)):
        return 'secret_pattern'  # one long token with digits: key, password, token
    if len(stripped) < MIN_CHARS:
        return 'too_short'
    return None


def judge(selection: str, meta: Optional[dict[str, Any]], max_age: float) -> Optional[str]:
    """Reason to refuse a selection on metadata alone, or None."""
    if not meta or not meta.get('owner'):
        return 'empty'
    targets = meta.get('targets') or []
    if any(t in targets for t in SECRET_HINT_TARGETS):
        return 'secret_hint'
    if selection == 'clipboard' and COPYQ_OWNER_TARGET in targets:
        return 'copyq_owner'
    age = meta.get('age_s')
    if age is None:
        return 'no_timestamp'
    if age > max_age:
        return 'too_old'
    return None


def probe(run: Callable = subprocess.run, timeout: float = PROBE_TIMEOUT_S) -> Optional[dict[str, Any]]:
    """Owner, TARGETS and age of PRIMARY and CLIPBOARD from the helper process."""
    try:
        result = run([sys.executable, os.path.abspath(__file__), '--probe'],
                     capture_output=True, timeout=timeout, check=False)
        if result.returncode != 0:
            return None
        return json.loads(result.stdout)
    except Exception:
        return None


def read_text(selection: str, run: Callable = subprocess.run,
              timeout: float = READ_TIMEOUT_S) -> Optional[str]:
    try:
        result = run(['xclip', '-selection', selection, '-t', 'UTF8_STRING', '-o'],
                     capture_output=True, timeout=timeout, check=False)
    except Exception:
        return None
    if result.returncode != 0:
        return None
    data = result.stdout
    return data.decode('utf-8', errors='replace') if isinstance(data, bytes) else str(data)


def capture(run: Callable = subprocess.run,
            max_ages: tuple[float, float] = (PRIMARY_MAX_AGE_S, CLIPBOARD_MAX_AGE_S),
            max_chars: int = MAX_CHARS) -> Context:
    """Read the reply context once. Never raises."""
    reasons: dict[str, str] = {}
    meta = probe(run)
    if meta is None:
        return Context(None, reasons={'primary': 'probe_failed', 'clipboard': 'probe_failed'})
    for selection, max_age in zip(('primary', 'clipboard'), max_ages):
        reason = judge(selection, meta.get(selection), max_age)
        if reason:
            reasons[selection] = reason
            continue
        text = read_text(selection, run)
        if text is None:
            reasons[selection] = 'read_failed'
            continue
        reason = secret_reason(text)
        if reason:
            reasons[selection] = reason
            continue
        text = text.strip()
        truncated = len(text) > max_chars
        if truncated:
            # Newest message sits on top in a mail client: keep the beginning.
            text = text[:max_chars].rstrip() + '\n' + TRUNCATED_MARK
        return Context(text, selection, len(text), meta[selection].get('age_s'), truncated, reasons)
    return Context(None, reasons=reasons)


# ---------------------------------------------------------------------------
# Helper process: python3 context_capture.py --probe
# ---------------------------------------------------------------------------

def _probe_x(names: tuple = ('PRIMARY', 'CLIPBOARD'), deadline_s: float = 1.0) -> dict[str, Any]:
    import select
    from Xlib import X, Xatom, display

    d = display.Display()
    try:
        atom = d.intern_atom
        win = d.screen().root.create_window(0, 0, 1, 1, 0, X.CopyFromParent,
                                            event_mask=X.PropertyChangeMask)
        clock = atom('DCT_PROBE_CLOCK')
        # Our own property change returns the current server time.
        win.change_property(clock, Xatom.STRING, 8, b'1')
        result: dict[str, Any] = {}
        pending: dict[tuple[int, int], tuple[str, str, int]] = {}
        for name in names:
            selection = atom(name)
            owner = d.get_selection_owner(selection)
            if getattr(owner, 'id', owner) in (0, None):
                result[name.lower()] = dict(owner=False)
                continue
            result[name.lower()] = dict(owner=True, targets=None, timestamp=None)
            for target in ('TARGETS', 'TIMESTAMP'):
                prop = atom('DCT_PROBE_%s_%s' % (name, target))
                win.convert_selection(selection, atom(target), prop, X.CurrentTime)
                pending[(selection, atom(target))] = (name.lower(), target, prop)
        d.flush()
        now = None
        end = time.monotonic() + deadline_s
        while (pending or now is None) and time.monotonic() < end:
            if not d.pending_events():
                ready, _, _ = select.select([d.fileno()], [], [], max(0.0, end - time.monotonic()))
                if not ready:
                    break
            event = d.next_event()
            if event.type == X.PropertyNotify and event.atom == clock:
                now = event.time
            elif event.type == X.SelectionNotify:
                key = (event.selection, event.target)
                if key not in pending:
                    continue
                name, target, prop = pending.pop(key)
                if event.property == X.NONE:
                    continue
                reply = win.get_full_property(prop, X.AnyPropertyType)
                values = list(reply.value) if reply is not None else []
                if target == 'TARGETS':
                    result[name]['targets'] = [d.get_atom_name(v) for v in values]
                elif values and int(values[0]):
                    result[name]['timestamp'] = int(values[0])
        for meta in result.values():
            stamp = meta.pop('timestamp', None)
            meta['age_s'] = (round(((now - stamp) & 0xFFFFFFFF) / 1000.0, 3)
                             if stamp and now is not None else None)
        return result
    finally:
        d.close()


if __name__ == '__main__':
    # An explicit selection list is for checking the probe against a private
    # test selection; the dictation always probes PRIMARY and CLIPBOARD.
    if sys.argv[1:2] == ['--probe'] and len(sys.argv) <= 3:
        names = tuple(sys.argv[2].split(',')) if len(sys.argv) == 3 else ('PRIMARY', 'CLIPBOARD')
        print(json.dumps(_probe_x(names)))
        sys.exit(0)
    print('usage: context_capture.py --probe [SELECTION,...]', file=sys.stderr)
    sys.exit(2)
