"""GNOME Wayland kernel I/O. No XTest, uinput, grabs or selection text.

Only evdev key identities/state are retained, never an input transcript.
The explicit keymap contract is documented in docs/wayland.md.
"""
import enum
import os
import selectors
import subprocess
import time
from types import SimpleNamespace
from typing import Any, Callable

BUS = 'org.gnome.Shell.Extensions.DictationHelper'
PATH = '/org/gnome/Shell/Extensions/DictationHelper'
SHORTCUTS = frozenset(('paste', 'paste_terminal', 'ctrl_home'))
SELECTIONS = frozenset(('primary', 'clipboard'))
# Status box codes (the words live in the extension). Never text.
STATUS_PHASES = frozenset(('recording', 'recording_silent', 'finishing', 'rewriting', 'pasted',
                           'nothing', 'mic_silent', 'paste_refused', 'raw_pasted'))
STATUS_MODES = frozenset(('dictate', 'rewrite', 'rewrite_line', 'dictate_fallback'))


def parse_gvariant_text(text: str) -> Any:
    """Parse gdbus call output (a tuple of strings, numbers and booleans).

    gdbus prints booleans as true/false; they are translated outside quoted
    strings only, so a window class can never change the structure.
    """
    import ast
    out, quote, escape = [], None, False
    i = 0
    while i < len(text):
        ch = text[i]
        if quote:
            out.append(ch)
            if escape:
                escape = False
            elif ch == '\\':
                escape = True
            elif ch == quote:
                quote = None
            i += 1
            continue
        if ch in ('"', "'"):
            quote = ch
            out.append(ch)
            i += 1
            continue
        for word, value in (('true', 'True'), ('false', 'False')):
            if text.startswith(word, i) and not text[i + len(word):i + len(word) + 1].isalnum():
                out.append(value)
                i += len(word)
                break
        else:
            out.append(ch)
            i += 1
    return ast.literal_eval(''.join(out))


class ShellClient:
    def __init__(self, call: Callable[..., tuple[Any, ...]] | None = None) -> None:
        self._call = call or self._gdbus

    @staticmethod
    def _gdbus(method: str, *args: str) -> tuple[Any, ...]:
        # gdbus has a bounded timeout; output contains metadata only.
        result = subprocess.check_output([
            'gdbus', 'call', '--session', '--dest', BUS, '--object-path', PATH,
            '--method', BUS + '.' + method, *map(str, args)], timeout=3,
            stderr=subprocess.DEVNULL).decode().strip()
        return parse_gvariant_text(result)

    def focus(self) -> tuple[str, str]:
        wid, wm_class = self._call('GetFocus')
        return str(wid), wm_class.lower()

    def activate(self, wid: str) -> bool:
        return bool(self._call('Activate', str(int(wid)))[0])

    def shortcut(self, name: str) -> None:
        if name not in SHORTCUTS:
            raise ValueError('shortcut not allowed')
        self._call('SendShortcut', name)

    def selection_info(self, name: str) -> dict[str, Any]:
        """When the selection owner last changed and the focus then and now. No text.

        state: 'set', 'cleared' or 'unknown' (no change seen since the helper
        was enabled). same_app: the window focused at the change and the one
        focused now belong to one process (or are the same window).
        """
        if name not in SELECTIONS:
            raise ValueError('selection not allowed')
        state, age, change_window, focus_window, same_app = self._call('SelectionInfo', name)
        if state not in ('set', 'cleared', 'unknown'):
            raise ValueError('unexpected selection state')
        age = float(age)
        return dict(state=state, age_s=round(age, 3) if age >= 0 else None,
                    change_window=str(change_window) or None,
                    focus_window=str(focus_window) or None, same_app=bool(same_app))

    def badge(self, chars: int) -> None:
        """Show the context badge; only the number of characters is sent."""
        self._call('ShowBadge', str(max(0, int(chars))))

    def status(self, phase: str, mode: str, language: str = '', chars: int = 0) -> None:
        """Show the status box. Only allowlisted codes and one count are sent.

        Codes are quoted GVariant strings, so gdbus never has to guess the type
        (an empty language would otherwise not parse).
        """
        if phase not in STATUS_PHASES or mode not in STATUS_MODES:
            raise ValueError('status not allowed')
        if not (language == '' or (2 <= len(language) <= 3 and language.isascii()
                                   and language.isalpha() and language.islower())):
            raise ValueError('status not allowed')
        self._call('SetStatus', "'%s'" % phase, "'%s'" % mode, "'%s'" % language,
                   str(max(0, int(chars))))

    def clear_status(self) -> None:
        self._call('ClearStatus')


class Key(enum.Enum):
    ctrl = 65507
    ctrl_l = 65507
    ctrl_r = 65508
    shift = 65505
    shift_l = 65505
    shift_r = 65506
    alt = 65513
    alt_l = 65513
    alt_r = 65514
    cmd_r = 65516

    @property
    def vk(self) -> int:
        return self.value


class HotKey:
    @staticmethod
    def parse(value: str) -> list[Key]:
        if value != '<ctrl_r>':
            raise ValueError('Wayland supports only <ctrl_r> double-tap')
        return [Key.ctrl_r]


class Button(enum.Enum):
    left = 272
    right = 273
    middle = 274
    x1 = 275
    x2 = 276
    button8 = 275
    button9 = 276
    button10 = 277  # BTN_FORWARD, for devices exposing separate codes
    button11 = 278  # BTN_BACK


keyboard = SimpleNamespace(Key=Key, HotKey=HotKey)
mouse = SimpleNamespace(Button=Button)


def _diagnostic(kind: str, **details: Any) -> None:
    from coverage_ledger import diagnostic
    diagnostic(kind, **details)


def event_stamp(event: Any) -> float | None:
    """Kernel time of an evdev event (CLOCK_REALTIME, like time.time()), or None."""
    sec, usec = getattr(event, 'sec', None), getattr(event, 'usec', None)
    if isinstance(sec, (int, float)) and isinstance(usec, (int, float)):
        return sec + usec / 1e6
    return None


class InputHub:
    """One reader per device, periodic hotplug rescan, never exclusive-grab.

    Inject a device provider and selector in tests. Opening devices happens only
    in run(), never on import or in the constructor. Removal releases held keys.

    Optional key trace (DICTATION_KEY_TRACE=1, off by default) journals one
    'key_trace' line per event of the TRACE_CODES keys only: the modifiers and
    Right-Ctrl, with value, kernel time, read latency and what the hub did with
    it. Every other key is never journalled, not even as a count.
    """
    MODS = {29: Key.ctrl, 97: Key.ctrl_r, 42: Key.shift,
            54: Key.shift_r, 56: Key.alt, 100: Key.alt_r}
    TRACE_CODES = {29: 'ctrl_l', 97: 'ctrl_r', 42: 'shift_l', 54: 'shift_r',
                   56: 'alt_l', 100: 'alt_r', 125: 'meta_l', 126: 'meta_r'}

    def __init__(self, provider: Callable[[], dict[str, Callable[[], Any]]] | None = None,
                 selector: Any = None, layout: str | None = None,
                 trace: bool | None = None, log: Callable[..., None] | None = None) -> None:
        self.layout = layout or os.environ.get('DICTATION_EVDEV_LAYOUT', 'be')
        if self.layout not in ('be', 'us'):
            raise ValueError('DICTATION_EVDEV_LAYOUT must be be or us')
        self.provider = provider or self._devices
        self.selector = selector or selectors.DefaultSelector()
        self.devices = {}
        self.held = {}
        self.key_listener = None
        self.mouse_callbacks = []
        self.running = True
        self.started = False
        self.desynced = set()
        if trace is None:
            trace = os.environ.get('DICTATION_KEY_TRACE', '') == '1'
        self.trace = trace
        self.log = log or _diagnostic

    def _trace(self, path: str, event: Any, action: str) -> None:
        """One key_trace line for a traced key or a sync loss; never raises."""
        if not self.trace:
            return
        try:
            if event.type == 1:
                name = self.TRACE_CODES.get(event.code)
                if name is None:
                    return  # never any other key
                info = dict(key=name, value=event.value)
            elif event.type == 0 and event.code == 3:
                info = dict(key='syn_dropped')
            else:
                return
            stamp = event_stamp(event)
            if stamp is not None:
                info['t'] = round(stamp, 3)
                info['read_latency_ms'] = round((time.time() - stamp) * 1000, 1)
            self.log('key_trace', device=os.path.basename(path), action=action, **info)
        except Exception:
            pass

    @staticmethod
    def _devices() -> dict[str, Callable[[], Any]]:
        import evdev
        return {path: (lambda path=path: evdev.InputDevice(path))
                for path in evdev.list_devices()}

    def key(self, code: int) -> Key | SimpleNamespace:
        if code in self.MODS:
            return self.MODS[code]
        # Linux evdev codes, not X keycodes. be AZERTY: physical Q -> a.
        a_code, c_code = (16, 46) if self.layout == 'be' else (30, 46)
        char = 'a' if code == a_code else 'c' if code == c_code else None
        return SimpleNamespace(vk=ord(char) if char else -code, char=char)

    def event(self, path: str, event: Any) -> None:
        # SYN_DROPPED: discard state, not a new trigger.
        if event.type == 0 and event.code == 3:
            self._trace(path, event, 'released_all')
            self.release(path)
            self.desynced.add(path)
            return
        if path in self.desynced:
            # Ignore everything through SYN_REPORT after loss. No key presses
            # are reconstructed: a fresh physical release/press is required.
            if event.type == 0 and event.code == 0:
                self.desynced.discard(path)
            elif event.type == 1:
                self._trace(path, event, 'ignored_desynced')
            return
        if event.type != 1:
            return
        if event.value == 2:
            self._trace(path, event, 'ignored_repeat')
            return
        state = self.held.setdefault(path, set())
        code = event.code
        if event.value:
            if code in state:
                self._trace(path, event, 'ignored_already_down')
                return
            already = any(code in keys for keys in self.held.values())
            state.add(code)
            if already:
                self._trace(path, event, 'ignored_held_elsewhere')
                return
        else:
            if code not in state:
                self._trace(path, event, 'ignored_not_down')
                return
            state.remove(code)
            if any(code in keys for keys in self.held.values()):
                self._trace(path, event, 'ignored_held_elsewhere')
                return
        self._trace(path, event, 'passed')
        if code >= 272:
            # Every button code counts for the selection-release trace,
            # touchpad contacts (BTN_TOUCH, BTN_TOOL_*) included: a tap-to-click
            # never shows up as BTN_LEFT at evdev level.
            try:
                button = Button(code)
            except ValueError:
                button = None
            for callback in self.mouse_callbacks:
                callback(0, 0, button, bool(event.value))
        elif self.key_listener:
            callback = (self.key_listener.on_press if event.value
                        else self.key_listener.on_release)
            self.key_listener.event_time = event_stamp(event)
            try:
                callback(self.key(code))
            finally:
                self.key_listener.event_time = None

    def release(self, path: str) -> None:
        for code in list(self.held.get(path, ())):
            self.event(path, SimpleNamespace(type=1, code=code, value=0))
        self.held.pop(path, None)
        if self.key_listener:
            # A lost device must not complete an old half-double-tap.
            self.key_listener.tap_count = 0
            self.key_listener.last_press_time = 0

    def rescan(self) -> None:
        available = self.provider()
        for path in list(self.devices):
            if path not in available:
                self.remove(path)
        for path, opener in available.items():
            if path in self.devices:
                continue
            device = None
            try:
                device = opener()
                if 1 not in device.capabilities():
                    device.close()
                    continue
                self.selector.register(device, selectors.EVENT_READ, path)
                self.devices[path] = device
                if self.trace and self.started:
                    self.log('key_trace', device=os.path.basename(path), action='device_added')
            except (OSError, ValueError):
                if device:
                    device.close()
        if not self.devices and not self.started:
            raise RuntimeError('No readable evdev input devices; check input group and login')
        if self.trace and not self.started:
            self.log('key_trace', action='enabled',
                     devices=sorted(os.path.basename(p) for p in self.devices))
        self.started = True

    def remove(self, path: str) -> None:
        if self.trace:
            self.log('key_trace', device=os.path.basename(path), action='device_removed')
        self.release(path)
        device = self.devices.pop(path)
        self.selector.unregister(device)
        device.close()

    def run(self) -> None:
        try:
            while self.running:
                self.rescan()
                for ready, _ in self.selector.select(timeout=1):
                    path = ready.data
                    try:
                        for event in self.devices[path].read():
                            self.event(path, event)
                    except BlockingIOError:
                        pass
                    except OSError:
                        self.remove(path)
        finally:
            for path in list(self.devices):
                self.remove(path)
            self.selector.close()


def bind(replayer: type, tap_listener: type, mouse_listener: type) -> tuple[type, type, type]:
    """Keep the existing tap/modifier and clipboard/delivery policy unchanged."""
    hub = InputHub()

    class WaylandReplayer(replayer):
        # xclip fallback: GTK4/Chromium on Wayland paste UTF8_STRING via
        # Xwayland, not a bare text/plain (nested GNOME proof).
        XCLIP_TARGET = 'UTF8_STRING'

        def __init__(self, callback: Callable[..., Any] | None = None) -> None:
            self.callback = callback
            self.target_wid = None
            self.shell = ShellClient()

        def window_id(self) -> str | None:
            try:
                wid, _ = self.shell.focus()
                return wid or None
            except Exception:
                return None

        def _get_active_window_class(self) -> str:
            try:
                return self.shell.focus()[1]
            except Exception:
                return ''

        def _restore_target_window(self) -> bool:
            if not self.target_wid:
                return False
            import time
            try:
                if not self.shell.activate(self.target_wid):
                    return False
            except Exception:
                return False
            for _ in range(10):
                if self.window_id() == self.target_wid:
                    return True
                time.sleep(.05)
            return False

        def _send_paste(self, terminal: bool) -> None:
            # Never paste into a different window after a focus failure.
            if not self.target_wid or self.window_id() != self.target_wid:
                raise RuntimeError('Wayland target has no focus')
            self.shell.shortcut('paste_terminal' if terminal else 'paste')

        def send_ctrl_home(self) -> None:
            self.shell.shortcut('ctrl_home')

        def focus_id(self) -> str | None:
            """Focused Meta.Window id for selection_release; raises when unknown."""
            return self.shell.focus()[0] or None

        def show_badge(self, chars: int) -> None:
            self.shell.badge(chars)

        def show_status(self, phase: str, mode: str, language: str = '', chars: int = 0) -> None:
            self.shell.status(phase, mode, language, chars)

        def clear_status(self) -> None:
            self.shell.clear_status()

    class WaylandTap(tap_listener):
        def run(self) -> None:
            if self.key != Key.ctrl_r:
                raise ValueError('Wayland supports only Right-Ctrl double-tap')
            hub.key_listener = self
            hub.run()

    class WaylandMouse(mouse_listener):
        def start(self) -> 'WaylandMouse':
            if self.buttons:
                hub.mouse_callbacks.append(self.on_click)
            return self

        @staticmethod
        def add_press_callback(callback: Callable[..., Any]) -> None:
            """Always-on press record for selection_release (no positions)."""
            hub.mouse_callbacks.append(callback)

    # The shared hub, for synthetic-event proofs that must not open devices.
    WaylandTap.hub = WaylandMouse.hub = hub
    return WaylandReplayer, WaylandTap, WaylandMouse
