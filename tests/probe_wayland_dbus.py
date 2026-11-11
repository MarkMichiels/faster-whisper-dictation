"""Real GJS/Gio D-Bus contract, fake compositor. No display/input/clipboard.

This proves marshalling, not extension loading in GNOME Shell. All wrapper
methods come from extension.js. Only Clutter/the Shell base class are fakes.
"""
from pathlib import Path
import os
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wayland_io import BUS, PATH, ShellClient

PREFIX = """
import Gio from 'gi://Gio';
import GLib from 'gi://GLib';
const Clutter = {
    KEY_Control_L: 1, KEY_Shift_L: 2, KEY_v: 3, KEY_Home: 4,
    KeyState: {PRESSED: 1, RELEASED: 0}, InputDeviceType: {KEYBOARD_DEVICE: 1},
    get_default_backend: () => ({get_default_seat: () => ({
        create_virtual_device: () => ({notify_keyval() {}})})}),
};
class Extension {}
const Meta = {SelectionType: {SELECTION_PRIMARY: 0, SELECTION_CLIPBOARD: 1}};
const St = {Label: class { constructor(p) { Object.assign(this, p); }
    get_preferred_width() { return [0, 100]; } set_position() {} destroy() {} }};
const Main = {layoutManager: {uiGroup: {add_child() {}}}};
let ownerChanged = null;
const global = {
    display: {focus_window: {get_id: () => 42, get_wm_class: () => "it's true", get_pid: () => 7,
                             get_monitor: () => 0},
              get_selection: () => ({connect: (_n, fn) => { ownerChanged = fn; return 1; }, disconnect() {}}),
              get_current_monitor: () => 0,
              get_monitor_geometry: () => ({x: 0, y: 0, width: 800, height: 600})},
    get_window_actors: () => [], get_current_time: () => 100,
};
"""
SUFFIX = """
const helper = new DictationHelper();
helper.enable();
ownerChanged(null, Meta.SelectionType.SELECTION_PRIMARY, {});
const loop = new GLib.MainLoop(null, false);
GLib.timeout_add(GLib.PRIORITY_DEFAULT, 20000, () => { loop.quit(); return GLib.SOURCE_REMOVE; });
loop.run();
helper.disable();
"""


def inside():
    import re
    source = (ROOT / 'gnome-extension/dictation-helper@faster-whisper-dictation/extension.js').read_text()
    source = re.sub(r'^import .*;\n', '', source, flags=re.M).replace('export default class', 'class')
    script = Path(os.environ['HOME']) / 'contract.mjs'
    script.write_text(PREFIX + source + SUFFIX)
    child = subprocess.Popen(['gjs', '-m', str(script)], stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
    try:
        subprocess.run(['gdbus', 'wait', '--session', '--timeout', '5', BUS],
                       check=True, timeout=7)
        client = ShellClient()
        assert client.focus() == ('42', "it's true")
        assert client.activate('999') is False
        primary = client.selection_info('primary')
        assert (primary['state'], primary['change_window'], primary['focus_window'],
                primary['same_app']) == ('set', '42', '42', True), primary
        assert 0 <= primary['age_s'] < 10, primary
        assert client.selection_info('clipboard') == dict(
            state='unknown', age_s=None, change_window=None, focus_window='42', same_app=False)
        result = subprocess.run(['gdbus', 'call', '--session', '--dest', BUS,
            '--object-path', PATH, '--method', BUS + '.SelectionInfo', 'dnd'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=3)
        assert result.returncode != 0
        client.badge(3)
        for name in ('paste', 'paste_terminal', 'ctrl_home'):
            client.shortcut(name)
        # Bypass the Python allowlist to prove the service refuses arbitrary input.
        result = subprocess.run(['gdbus', 'call', '--session', '--dest', BUS,
            '--object-path', PATH, '--method', BUS + '.SendShortcut', 'Return'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=3)
        assert result.returncode != 0
        print('PASS: actual GJS/Gio XML export and Python gdbus transport, selection metadata, badge, three shortcuts, server rejection')
    finally:
        if child.poll() is None:
            subprocess.run(['ps', '-o', 'pid,args', '-p', str(child.pid)], check=True)
            child.terminate()
            try:
                child.wait(timeout=3)
            except subprocess.TimeoutExpired:
                subprocess.run(['ps', '-o', 'pid,args', '-p', str(child.pid)], check=True)
                child.kill()
                child.wait(timeout=3)


def main():
    with tempfile.TemporaryDirectory(prefix='dbus-probe-', dir=ROOT / 'evidence') as tmp:
        env = dict(PATH='/usr/bin:/bin', HOME=tmp, LANG='C.UTF-8',
                   XDG_RUNTIME_DIR=tmp, TMPDIR=tmp)
        subprocess.run(['dbus-run-session', '--', sys.executable, __file__, '--inside'],
                       env=env, check=True, timeout=40)


if __name__ == '__main__':
    inside() if '--inside' in sys.argv else main()
