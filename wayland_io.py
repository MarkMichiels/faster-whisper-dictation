"""GNOME Wayland kernel I/O. No XTest, uinput, grabs or selection text.

Only evdev key identities/state are retained, never an input transcript.
The explicit keymap contract is documented in docs/wayland.md.
"""
import enum
import os
import selectors
import subprocess
from types import SimpleNamespace
from typing import Any, Callable

BUS = 'org.gnome.Shell.Extensions.DictationHelper'
PATH = '/org/gnome/Shell/Extensions/DictationHelper'
SHORTCUTS = frozenset(('paste', 'paste_terminal', 'ctrl_home'))


class ShellClient:
    def __init__(self, call: Callable[..., tuple[Any, ...]] | None = None) -> None:
        self._call = call or self._gdbus

    @staticmethod
    def _gdbus(method: str, *args: str) -> tuple[Any, ...]:
        # gdbus has a bounded timeout; output contains metadata only.
        import ast
        result = subprocess.check_output([
            'gdbus', 'call', '--session', '--dest', BUS, '--object-path', PATH,
            '--method', BUS + '.' + method, *map(str, args)], timeout=3,
            stderr=subprocess.DEVNULL).decode().strip()
        if result in ('(true,)', '(false,)'):
            return (result == '(true,)',)
        return ast.literal_eval(result)

    def focus(self) -> tuple[str, str]:
        wid, wm_class = self._call('GetFocus')
        return str(wid), wm_class.lower()

    def activate(self, wid: str) -> bool:
        return bool(self._call('Activate', str(int(wid)))[0])

    def shortcut(self, name: str) -> None:
        if name not in SHORTCUTS:
            raise ValueError('shortcut not allowed')
        self._call('SendShortcut', name)


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


class InputHub:
    """One reader per device, periodic hotplug rescan, never exclusive-grab.

    Inject a device provider and selector in tests. Opening devices happens only
    in run(), never on import or in the constructor. Removal releases held keys.
    """
    MODS = {29: Key.ctrl, 97: Key.ctrl_r, 42: Key.shift,
            54: Key.shift_r, 56: Key.alt, 100: Key.alt_r}

    def __init__(self, provider: Callable[[], dict[str, Callable[[], Any]]] | None = None,
                 selector: Any = None, layout: str | None = None) -> None:
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
        # SYN_DROPPED: discard state, not a new trigger. Selection release is
        # disabled for this kernel-only backend until the selection WP.
        if event.type == 0 and event.code == 3:
            self.release(path)
            self.desynced.add(path)
            return
        if path in self.desynced:
            # Ignore everything through SYN_REPORT after loss. No key presses
            # are reconstructed: a fresh physical release/press is required.
            if event.type == 0 and event.code == 0:
                self.desynced.discard(path)
            return
        if event.type != 1 or event.value == 2:
            return
        state = self.held.setdefault(path, set())
        code = event.code
        if event.value:
            if code in state:
                return
            already = any(code in keys for keys in self.held.values())
            state.add(code)
            if already:
                return
        else:
            if code not in state:
                return
            state.remove(code)
            if any(code in keys for keys in self.held.values()):
                return
        if code >= 272:
            try:
                button = Button(code)
            except ValueError:
                button = None
            for callback in self.mouse_callbacks:
                callback(0, 0, button, bool(event.value))
        elif self.key_listener:
            callback = (self.key_listener.on_press if event.value
                        else self.key_listener.on_release)
            callback(self.key(code))

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
            except (OSError, ValueError):
                if device:
                    device.close()
        if not self.devices and not self.started:
            raise RuntimeError('No readable evdev input devices; check input group and login')
        self.started = True

    def remove(self, path: str) -> None:
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

    return WaylandReplayer, WaylandTap, WaylandMouse
