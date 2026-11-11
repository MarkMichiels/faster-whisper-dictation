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

GNOME Wayland (D4 in docs/wayland.md): mutter mirrors PRIMARY and CLIPBOARD to
Xwayland, so TARGETS and the text are still read the X way, but the mirror has
no TIMESTAMP and is always owned by mutter. Age and origin then come from the
Dictation Helper extension (capture(shell=...)): the time of the last owner
change of that selection and the window focused at that moment. PRIMARY counts
only when that window belongs to the program focused now (same process), so a
selection made in another program is refused ('focus_elsewhere'); copy it with
Ctrl+C to use it through CLIPBOARD. A PRIMARY owned by CopyQ (clipboard-to-
selection sync: our own previous output) is refused as well. The extension
never returns selection text.

Long text: up to max_chars characters (default DEFAULT_MAX_CHARS; the caller
passes the configured limit) are kept. Longer text keeps its beginning and its end, with
TRUNCATED_MARK in between: a mail thread shows the oldest message on top, a
quoted reply the newest, so either end can hold the message being answered.

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

# A selection made for a reply is seconds old when the rewrite starts (measured:
# 3-10 s); one left over from reading something else is older (34 s, 50 s).
PRIMARY_MAX_AGE_S = 20
CLIPBOARD_MAX_AGE_S = 120
DEFAULT_MAX_CHARS = 100_000
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
    original_chars: Optional[int] = None
    # Origin of PRIMARY, for selection_release: the input focus window at the
    # probe and whether the PRIMARY owner is the same X client. No text.
    # Wayland: a Meta.Window id string; owner_same_client then means "PRIMARY
    # last changed while this same window had focus".
    focus_window: Optional[Any] = None
    owner_same_client: Optional[bool] = None
    backend: str = 'x11'

    def journal(self) -> dict[str, Any]:
        """Metadata only: never the text."""
        meta = dict(source=self.source, chars=self.chars, age_s=self.age_s,
                    truncated=self.truncated, original_chars=self.original_chars,
                    reasons=dict(self.reasons), owner_same_client=self.owner_same_client)
        if self.backend != 'x11':
            meta['backend'] = self.backend
        return meta


def shorten(text: str, max_chars: int) -> str:
    """Keep the beginning and the end of text, at most max_chars characters plus the mark."""
    if len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head
    return '%s\n%s\n%s' % (text[:head].rstrip(), TRUNCATED_MARK, text[len(text) - tail:].lstrip())


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


def shell_meta(selection: str, x_meta: Optional[dict[str, Any]], shell: Any
               ) -> tuple[Optional[dict[str, Any]], Optional[str], dict[str, Any]]:
    """Wayland: X metadata (owner, TARGETS) with age and origin from the shell.

    Returns (meta for judge, reason or None, shell info). Fail closed: an
    unreachable or outdated extension, no change seen, a cleared selection or
    a PRIMARY owned by CopyQ give a reason. The focus rule follows in capture().
    """
    try:
        info = shell.selection_info(selection)
    except Exception:
        return x_meta, 'shell_unavailable', {}
    if not x_meta or not x_meta.get('owner') or info.get('state') == 'cleared':
        return x_meta, 'empty', info
    if info.get('state') != 'set' or info.get('age_s') is None:
        return x_meta, 'no_change_seen', info
    meta = dict(x_meta, age_s=info['age_s'])
    if selection == 'primary' and COPYQ_OWNER_TARGET in (meta.get('targets') or []):
        return meta, 'copyq_owner', info
    return meta, None, info


def capture(run: Callable = subprocess.run,
            max_ages: tuple[float, float] = (PRIMARY_MAX_AGE_S, CLIPBOARD_MAX_AGE_S),
            max_chars: int = DEFAULT_MAX_CHARS, shell: Any = None) -> Context:
    """Read the reply context once. Never raises.

    shell: the Wayland Dictation Helper client (wayland_io.ShellClient); None on X11.
    """
    reasons: dict[str, str] = {}
    backend = 'x11' if shell is None else 'wayland'
    meta = probe(run)
    if meta is None:
        return Context(None, reasons={'primary': 'probe_failed', 'clipboard': 'probe_failed'},
                       backend=backend)
    if shell is None:
        focus = meta.get('focus') if isinstance(meta.get('focus'), int) else None
        primary = meta.get('primary') or {}
        same = primary.get('same_client') if isinstance(primary.get('same_client'), bool) else None
    else:
        focus, same = None, None
    origin = dict(focus_window=focus, owner_same_client=same, backend=backend)
    for selection, max_age in zip(('primary', 'clipboard'), max_ages):
        if shell is not None:
            sel_meta, reason, info = shell_meta(selection, meta.get(selection), shell)
            meta[selection] = sel_meta
            if selection == 'primary':
                focus = info.get('focus_window')
                same = (bool(focus) and info.get('change_window') == focus
                        and info.get('state') == 'set') if info else None
                origin.update(focus_window=focus, owner_same_client=same)
            if reason:
                reasons[selection] = reason
                continue
        reason = judge(selection, meta.get(selection), max_age)
        if not reason and shell is not None and selection == 'primary' and not info.get('same_app'):
            reason = 'focus_elsewhere'
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
        original = len(text)
        text = shorten(text, max_chars)
        return Context(text, selection, len(text), meta[selection].get('age_s'),
                       original > max_chars, reasons, original, **origin)
    return Context(None, reasons=reasons, **origin)


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
        # Input focus and X client (resource-id base) of each owner: lets the
        # caller check that PRIMARY belongs to the focused program. Ids only.
        mask = d.display.info.resource_id_mask
        focus = d.get_input_focus().focus
        focus_id = getattr(focus, 'id', focus) or 0
        focus_id = focus_id if focus_id > 1 else 0  # 0 = None, 1 = PointerRoot
        # Our own property change returns the current server time.
        win.change_property(clock, Xatom.STRING, 8, b'1')
        result: dict[str, Any] = {'focus': focus_id or None}
        pending: dict[tuple[int, int], tuple[str, str, int]] = {}
        for name in names:
            selection = atom(name)
            owner = d.get_selection_owner(selection)
            if getattr(owner, 'id', owner) in (0, None):
                result[name.lower()] = dict(owner=False)
                continue
            owner_id = getattr(owner, 'id', owner)
            result[name.lower()] = dict(owner=True, targets=None, timestamp=None,
                                        same_client=bool(focus_id) and (owner_id & ~mask) == (focus_id & ~mask))
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
        for meta in (v for v in result.values() if isinstance(v, dict)):
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
