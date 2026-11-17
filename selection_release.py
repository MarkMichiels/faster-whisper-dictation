"""Rewrite mode: release a Ctrl+A selection that was taken as reply context.

When you press Ctrl+A in a reply field and then start a rewrite, the whole
field is selected and read as context. Without help the selection stays: the
dictated text would replace everything at paste. This module decides whether
the app may send one Ctrl+Home to collapse that selection with the caret at
the top, and sends it only after the start modifiers are released.

X11 cannot tell *where* a selection lies: a selection in a read-only message
pane and the caret in a reply field of the same program share one X client.
The reliable ground is what the app itself heard: a Ctrl+A in the same window,
nothing typed or clicked since, and a PRIMARY timestamp that matches the
moment of that Ctrl+A. Anything else (mouse selections, Shift+arrows, the
clipboard, terminals) is left alone. Every refusal is journaled with its
reason; no text is ever logged.

Why Ctrl+Home and not Left: after Ctrl+A, Left leaves the caret one character
before the end in some editors; Ctrl+Home puts it at the top in every tested
field. Why wait for the release: with Left Alt still held the key becomes
Alt+key (a browser goes back in history), with Left Ctrl held it becomes a
word jump. Never --clearmodifiers: that sends a synthetic Alt release and
press, which can open menus.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

SELECT_ALL_MAX_AGE_S = 20.0      # same as context_capture.PRIMARY_MAX_AGE_S
TIMESTAMP_TOLERANCE_S = 0.6      # PRIMARY age vs time since the Ctrl+A press
RELEASE_WAIT_S = 2.0             # give up when the start chord is held longer
POLL_S = 0.05
INJECT_QUIET_S = 0.3             # listener ignores our own Ctrl+Home this long

# X keysyms of modifier keys (Shift_L ... Hyper_R) and AltGr (ISO_Level3_Shift).
MODIFIER_VKS = set(range(0xffe1, 0xffef)) | {0xfe03}
MODIFIER_NAMES = {'shift', 'shift_l', 'shift_r', 'ctrl', 'ctrl_l', 'ctrl_r', 'alt', 'alt_l',
                  'alt_r', 'alt_gr', 'cmd', 'cmd_l', 'cmd_r'}
LEFT_CTRL_VK, RIGHT_CTRL_VK = 65507, 65508
SELECT_ALL_CHARS = ('a', 'A', '\x01')
COPY_CHARS = ('c', 'C', '\x03')
INSERT_VK = 0xff63


@dataclass
class TraceSnapshot:
    """What the key and mouse listeners saw, in time.monotonic() seconds."""
    select_all_at: Optional[float] = None      # last Ctrl+A press
    select_all_window: Optional[int] = None    # focused window at that press
    last_mouse_press_at: Optional[float] = None
    last_other_key_at: Optional[float] = None  # any key but modifiers, Ctrl+A, copy keys, the start key
    mouse_known: bool = False                  # the always-on mouse listener runs
    keys_known: bool = False                   # the key listener feeds this trace


class InputTrace:
    """Thread-safe record fed by the key and mouse listener callbacks.

    Callbacks must stay cheap: the focused window at Ctrl+A is read on a short
    helper thread, never inside the callback.
    """

    def __init__(self, focus: Optional[Callable[[], Optional[int]]] = None,
                 clock: Callable[[], float] = time.monotonic, spawn: Optional[Callable] = None):
        self._lock = threading.Lock()
        self._data = TraceSnapshot()
        self._focus = focus or focus_window
        self._clock = clock
        self._spawn = spawn or (lambda fn: threading.Thread(target=fn, daemon=True).start())
        self._quiet_until = 0.0
        self._seq = 0

    def snapshot(self) -> TraceSnapshot:
        with self._lock:
            return TraceSnapshot(**vars(self._data))

    def set_known(self, keys: Optional[bool] = None, mouse: Optional[bool] = None) -> None:
        with self._lock:
            if keys is not None:
                self._data.keys_known = keys
            if mouse is not None:
                self._data.mouse_known = mouse

    def injecting(self) -> None:
        """Our own key events follow; do not count them as user input."""
        with self._lock:
            self._quiet_until = self._clock() + INJECT_QUIET_S

    def _quiet(self, now: float) -> bool:
        return now < self._quiet_until

    def select_all(self) -> None:
        now = self._clock()
        with self._lock:
            if self._quiet(now):
                return
            self._seq += 1
            seq = self._seq
            self._data.select_all_at = now
            self._data.select_all_window = None

        def read_focus():
            try:
                window = self._focus()
            except Exception:
                window = None
            with self._lock:
                if self._seq == seq:
                    self._data.select_all_window = window
        self._spawn(read_focus)

    def other_key(self) -> None:
        now = self._clock()
        with self._lock:
            if not self._quiet(now):
                self._data.last_other_key_at = now

    def mouse_press(self) -> None:
        now = self._clock()
        with self._lock:
            if not self._quiet(now):
                self._data.last_mouse_press_at = now


def classify_key(key, ctrl_down: bool, start_key_match: bool) -> str:
    """'modifier', 'start', 'select_all', 'neutral' or 'other' for one key press."""
    if start_key_match:
        return 'start'
    name = getattr(key, 'name', None)
    vk = getattr(key, 'vk', None)
    if vk is None and hasattr(key, 'value'):
        vk = getattr(key.value, 'vk', None)
    if name in MODIFIER_NAMES or vk in MODIFIER_VKS:
        return 'modifier'
    char = getattr(key, 'char', None)
    if ctrl_down and (char in SELECT_ALL_CHARS or vk in (97, 65)):
        return 'select_all'
    if ctrl_down and (char in COPY_CHARS or vk in (99, 67) or name == 'insert' or vk == INSERT_VK):
        return 'neutral'  # copying keeps the selection
    return 'other'


@dataclass
class StartState:
    now: float                     # monotonic time just before the context was read
    context_source: str            # 'primary', 'clipboard' or 'none'
    primary_age_s: Optional[float]
    focus_window: Optional[int]
    focus_is_terminal: bool        # WM_CLASS is a terminal, or a Left-Ctrl (terminal) session
    owner_same_client: Optional[bool]  # PRIMARY owner and focus window share the X client
    probe_done: Optional[float] = None  # monotonic time after the context was read
    previous_busy: bool = False    # an earlier session may still paste


def plan_release(trace: TraceSnapshot, start: StartState) -> tuple[bool, str]:
    """(True, 'ok') when a release may be scheduled, else (False, reason). Fail closed."""
    if start.context_source != 'primary':
        return False, 'context_not_primary'
    if start.focus_is_terminal:
        return False, 'terminal'
    if not start.focus_window:
        return False, 'no_focus'
    if not trace.keys_known:
        return False, 'keys_unknown'
    if not trace.mouse_known:
        return False, 'mouse_unknown'
    if trace.select_all_at is None:
        return False, 'no_select_all'
    since = start.now - trace.select_all_at
    if since < 0 or since > SELECT_ALL_MAX_AGE_S:
        return False, 'select_all_too_old'
    if trace.select_all_window != start.focus_window:
        return False, 'focus_moved'
    if trace.last_mouse_press_at is not None and trace.last_mouse_press_at >= trace.select_all_at:
        return False, 'mouse_after_select'
    if trace.last_other_key_at is not None and trace.last_other_key_at >= trace.select_all_at:
        return False, 'key_after_select'
    if start.owner_same_client is not True:
        return False, 'owner_elsewhere'
    # The probe ran somewhere between `now` and `probe_done`.
    latest = (start.probe_done if start.probe_done is not None else start.now) - trace.select_all_at
    age = start.primary_age_s
    if age is None or age < since - TIMESTAMP_TOLERANCE_S or age > latest + TIMESTAMP_TOLERANCE_S:
        return False, 'selection_not_from_select_all'
    if start.previous_busy:
        return False, 'previous_session_busy'
    return True, 'ok'


def may_fire(trace: TraceSnapshot, started_at: float, now: float, modifiers_held: bool,
             focus: Optional[int], planned_window: int, session_active: bool = True
             ) -> tuple[Optional[bool], str]:
    """Called repeatedly after start. None = keep waiting, True = send Ctrl+Home once, False = drop."""
    if not session_active:
        return False, 'session_ended'
    if trace.last_other_key_at is not None and trace.last_other_key_at >= started_at:
        return False, 'typed_after_start'
    if trace.last_mouse_press_at is not None and trace.last_mouse_press_at >= started_at:
        return False, 'mouse_after_start'
    if focus != planned_window:
        return False, 'focus_moved_after_start'
    if modifiers_held:
        if now - started_at > RELEASE_WAIT_S:
            return False, 'modifiers_held_too_long'
        return None, 'waiting_for_release'
    return True, 'ok'


def input_since(trace: TraceSnapshot, since: float, mouse_after: Optional[float] = None) -> Optional[str]:
    """'mouse', 'key', 'mouse+key' or None: user input the trace saw after `since`.

    Used by the paste guard: a click or typed key after stop may have moved
    the caret. `mouse_after` (>= since) skips presses up to that moment, the
    press that stopped or started dictation itself. Ctrl+A counts (a paste
    would replace the selection); modifiers, the start key and copy keys do
    not, and neither do our own injected keys.
    """
    floor = since if mouse_after is None else max(since, mouse_after)
    mouse = trace.last_mouse_press_at is not None and trace.last_mouse_press_at > floor
    key = any(at is not None and at > since for at in (trace.last_other_key_at, trace.select_all_at))
    return '+'.join(name for name, seen in (('mouse', mouse), ('key', key)) if seen) or None


def wait_and_release(trace: InputTrace, started_at: float, planned_window: int,
                     modifiers_held: Callable[[], bool], send: Callable[[], None],
                     session_active: Callable[[], bool] = lambda: True,
                     focus: Optional[Callable[[], Optional[int]]] = None,
                     clock: Callable[[], float] = time.monotonic,
                     sleep: Callable[[float], None] = time.sleep) -> dict:
    """Wait for the release of the start chord, then send one Ctrl+Home. Never raises."""
    focus = focus or focus_window
    try:
        while True:
            now = clock()
            try:
                current = focus()
            except Exception:
                current = None
            verdict, reason = may_fire(trace.snapshot(), started_at, now, modifiers_held(),
                                       current, planned_window, session_active())
            if verdict is None:
                sleep(POLL_S)
                continue
            if verdict:
                trace.injecting()
                send()
            return dict(fired=bool(verdict), reason=reason, wait_s=round(clock() - started_at, 3))
    except Exception as exc:
        return dict(fired=False, reason='error', error=type(exc).__name__,
                    wait_s=round(clock() - started_at, 3))


def send_ctrl_home(kb, key) -> None:
    """Exactly Ctrl+Home via XTest (a pynput Controller and its Key enum); no other key,
    no --clearmodifiers."""
    kb.press(key.ctrl)
    try:
        kb.press(key.home)
        kb.release(key.home)
    finally:
        kb.release(key.ctrl)


def focus_window() -> Optional[int]:
    """The X input focus window id, or None. Same source as the context probe."""
    from Xlib import display
    d = display.Display()
    try:
        focus = d.get_input_focus().focus
        wid = getattr(focus, 'id', focus) or 0
        return wid if wid > 1 else None  # 0 = None, 1 = PointerRoot
    finally:
        d.close()
