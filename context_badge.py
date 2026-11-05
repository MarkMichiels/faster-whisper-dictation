"""Rewrite mode: a small corner badge that shows reply context was taken.

    python3 context_badge.py --chars 4312 [--seconds 1.5]

Shows "Context opgepakt: 4.312 tekens" in the top-right corner of the monitor
that holds the focused window, for 1.5 s. It shows only the length, never any
of the text.

It never takes focus or input:
- override-redirect window: the window manager does not manage it, so it gets
  no frame, no taskbar entry and no focus; SetInputFocus is never called;
- WM_HINTS input=False and _NET_WM_WINDOW_TYPE_NOTIFICATION for compositors;
- an empty SHAPE input region: clicks pass through to the window below.

The dictation starts it as its own short-lived process (launch()), fire and
forget, with a hard kill after KILL_AFTER_S, so a stuck badge never holds up
the dictation. A start failure is journaled and otherwise silent.
Only python-xlib, which context_capture already uses.
"""
from __future__ import annotations

import argparse
import os
import select
import subprocess
import sys
import threading
import time
from typing import Callable, Optional

SHOW_S = 1.5
KILL_AFTER_S = 5.0
MARGIN = 16
HEIGHT = 40
BACKGROUND, FOREGROUND = '#1f6f3f', '#ffffff'


def label(chars: int) -> str:
    """'Context opgepakt: 4.312 tekens' (dot as thousands separator)."""
    return 'Context opgepakt: %s tekens' % format(max(0, int(chars)), ',').replace(',', '.')


def choose_monitor(monitors: list[tuple[int, int, int, int]], point: Optional[tuple[int, int]],
                   root_size: tuple[int, int]) -> tuple[int, int, int, int]:
    """(x, y, width, height) of the monitor containing point; else the first; else the root."""
    if point is not None:
        px, py = point
        for x, y, w, h in monitors:
            if x <= px < x + w and y <= py < y + h:
                return x, y, w, h
    if monitors:
        return monitors[0]
    return 0, 0, root_size[0], root_size[1]


def badge_position(monitor: tuple[int, int, int, int], width: int) -> tuple[int, int]:
    """Top-right corner of the monitor, inside the margin."""
    x, y, w, _ = monitor
    return x + max(0, w - width - MARGIN), y + MARGIN


def launch(chars: int, popen: Callable = subprocess.Popen, log: Optional[Callable] = None,
           spawn: Optional[Callable] = None) -> bool:
    """Start the badge process, fire and forget. Never raises; a failure only logs.

    Only the number of characters crosses the process boundary.
    """
    try:
        proc = popen([sys.executable, os.path.abspath(__file__), '--chars', str(int(chars))],
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL, close_fds=True)
    except Exception as exc:
        if log:
            log('context_badge', started=False, error=type(exc).__name__)
        return False

    def reap():
        try:
            proc.wait(timeout=KILL_AFTER_S)
        except Exception:
            try:
                proc.kill()  # our own child, by its handle
                proc.wait(timeout=1)
            except Exception:
                pass
            if log:
                log('context_badge', started=True, killed=True)
    try:
        (spawn or (lambda fn: threading.Thread(target=fn, daemon=True).start()))(reap)
    except Exception:
        pass
    return True


# ---------------------------------------------------------------------------
# Badge process
# ---------------------------------------------------------------------------

def _monitors(d, root) -> list[tuple[int, int, int, int]]:
    try:
        if not d.has_extension('RANDR'):
            return []
        from Xlib.ext import randr  # noqa: F401  (registers the extension methods)
        reply = root.xrandr_get_monitors(is_active=True)
        return [(m.x, m.y, m.width_in_pixels, m.height_in_pixels) for m in reply.monitors]
    except Exception:
        return []


def _focus_center(d, root) -> Optional[tuple[int, int]]:
    try:
        focus = d.get_input_focus().focus
        if getattr(focus, 'id', focus) in (0, 1, None):
            return None
        geo = focus.get_geometry()
        pos = root.translate_coords(focus, 0, 0)
        return pos.x + geo.width // 2, pos.y + geo.height // 2
    except Exception:
        return None


def show(chars: int, seconds: float = SHOW_S) -> None:
    from Xlib import X, Xatom, display
    from Xlib.ext import shape

    d = display.Display()
    try:
        screen = d.screen()
        root = screen.root
        font = None
        for name in ('-*-dejavu sans-bold-r-normal--20-*-*-*-*-*-iso10646-1',
                     '-*-helvetica-bold-r-normal--20-*-*-*-*-*-iso8859-1', 'fixed'):
            try:
                font = d.open_font(name)
                if font is not None:
                    font.query()
                    break
            except Exception:
                font = None
        text = label(chars).encode('latin-1', 'replace')
        width = (font.query_text_extents(text).overall_width if font else 8 * len(text)) + 28
        monitor = choose_monitor(_monitors(d, root), _focus_center(d, root),
                                 (screen.width_in_pixels, screen.height_in_pixels))
        x, y = badge_position(monitor, width)
        cmap = screen.default_colormap
        bg = cmap.alloc_named_color(BACKGROUND).pixel
        fg = cmap.alloc_named_color(FOREGROUND).pixel
        win = root.create_window(x, y, width, HEIGHT, 0, screen.root_depth, X.InputOutput,
                                 X.CopyFromParent, background_pixel=bg,
                                 override_redirect=True, event_mask=X.ExposureMask)
        win.set_wm_name('dictation-context-badge')
        win.set_wm_class('dictation-badge', 'DictationBadge')
        win.set_wm_hints(flags=1, input=0)  # InputHint, input=False
        win.change_property(d.intern_atom('_NET_WM_WINDOW_TYPE'), Xatom.ATOM, 32,
                            [d.intern_atom('_NET_WM_WINDOW_TYPE_NOTIFICATION')])
        if d.has_extension('SHAPE'):
            win.shape_rectangles(shape.SO.Set, shape.SK.Input, X.Unsorted, 0, 0, [])
        gc = win.create_gc(foreground=fg, background=bg, **({'font': font} if font else {}))
        win.map()
        d.flush()
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            select.select([d.fileno()], [], [], max(0.0, end - time.monotonic()))
            while d.pending_events():
                event = d.next_event()
                if event.type == X.Expose and event.count == 0:
                    win.image_text(gc, 14, 27, text)
                    d.flush()
        win.destroy()
        d.flush()
    finally:
        d.close()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description='Show the reply-context badge once.')
    parser.add_argument('--chars', type=int, required=True)
    parser.add_argument('--seconds', type=float, default=SHOW_S)
    args = parser.parse_args(argv)
    show(max(0, args.chars), min(max(args.seconds, 0.2), KILL_AFTER_S - 1))


if __name__ == '__main__':
    main()
