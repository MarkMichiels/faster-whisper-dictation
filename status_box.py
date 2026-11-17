"""Wayland status box: what the dictation is doing right now, top right.

The GNOME Dictation Helper draws the box (SetStatus/ClearStatus). This module
only decides *when* and sends it on a thread of its own, so a slow or missing
helper never holds up recording, recognition or a paste.

What crosses to the shell: a phase code, a mode code, a language code and the
number of context characters. Never dictation text, selection content, window
titles or paths; the words on the box live in the extension.

Phases (codes, see wayland_io.STATUS_PHASES):
  recording, recording_silent   while the microphone runs
  finishing                     after stop: last recognition and stop check
  rewriting                     the one LLM call of rewrite mode
  pasted                        done; the box hides after 1.5 s
  nothing, mic_silent,          ends without a (complete) paste; the box
  paste_refused, raw_pasted     stays a few seconds longer

Research mode (research.py) has a box state of its own, research(phase, step,
seconds, steps): phase researching/done/clipboard/failed, a step or reason
code, the elapsed seconds and the number of tool steps. The app sends it only
while the dictation does not need the box.

Only the newest state matters: when states queue up behind a slow call, the
sender skips to the latest one. A failed call is journalled once per change
of availability. When the helper is too old to know SetStatus and context was
taken at the start, the old context badge is shown instead (fallback).
"""
from __future__ import annotations

import threading
from typing import Any, Callable, Optional

FINAL_PHASES = frozenset(('pasted', 'nothing', 'mic_silent', 'paste_refused', 'raw_pasted'))


def language_code(value: Any) -> str:
    """A short lowercase language code, or '' (auto, unset or anything else)."""
    if isinstance(value, str) and 2 <= len(value) <= 3 and value.isascii() and value.isalpha():
        return value.lower()
    return ''


def mode_of(rewrite: bool, terminal: bool, fallback: bool) -> str:
    if rewrite:
        return 'rewrite_line' if terminal else 'rewrite'
    return 'dictate_fallback' if fallback else 'dictate'


class StatusBox:
    """Fire-and-forget sender for the status box. Never raises to the caller."""

    def __init__(self, send: Callable[..., None], clear: Callable[[], None],
                 badge: Optional[Callable[[int], None]] = None,
                 research: Optional[Callable[..., None]] = None,
                 log: Optional[Callable[..., None]] = None,
                 spawn: Optional[Callable[[Callable[[], None]], None]] = None) -> None:
        self._send, self._clear, self._badge = send, clear, badge
        self._research = research
        self._log = log or (lambda kind, **details: None)
        self._spawn = spawn or (lambda fn: threading.Thread(target=fn, daemon=True).start())
        self._lock = threading.Lock()
        self._pending: Optional[tuple] = None
        self._busy = False
        self.available: Optional[bool] = None   # None until the first call returns

    def show(self, phase: str, mode: str, language: str = '', chars: int = 0) -> None:
        self._post(('show', phase, mode, language_code(language), max(0, int(chars or 0))))

    def clear(self) -> None:
        self._post(('clear',))

    def research(self, phase: str, step: str = '', seconds: int = 0, steps: int = 0) -> None:
        """Research box (codes and counts only). Raises when no sender: the caller skips."""
        if self._research is None:
            raise RuntimeError('research box not available')
        self._post(('research', phase, step or '', max(0, int(seconds or 0)), max(0, int(steps or 0))))

    def _post(self, state: tuple) -> None:
        try:
            with self._lock:
                self._pending = state
                if self._busy:
                    return   # the running sender picks up the newest state
                self._busy = True
            self._spawn(self._drain)
        except Exception as exc:
            with self._lock:
                self._busy = False
            self._log('status_box', sent=False, error=type(exc).__name__)

    def _drain(self) -> None:
        while True:
            with self._lock:
                state, self._pending = self._pending, None
                if state is None:
                    self._busy = False
                    return
            self._deliver(state)

    def _deliver(self, state: tuple) -> None:
        try:
            if state[0] == 'clear':
                self._clear()
            elif state[0] == 'research':
                self._research(*state[1:])
            else:
                self._send(*state[1:])
            if self.available is not True:
                self.available = True
                self._log('status_box', available=True)
        except Exception as exc:
            if self.available is not False:
                self.available = False
                self._log('status_box', available=False, error=type(exc).__name__)
            if state[0] == 'show' and state[1] == 'recording' and state[4] and self._badge:
                try:
                    self._badge(state[4])
                    self._log('context_badge', started=True, backend='shell', fallback='status_box')
                except Exception as badge_exc:
                    self._log('context_badge', started=False, backend='shell',
                              error=type(badge_exc).__name__)
