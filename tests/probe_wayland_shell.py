"""Isolated, bounded GNOME smoke probe. No host display, clipboard or input.

Run manually with Python. It uses a private HOME, runtime and D-Bus session;
only headless Mutter is started. A temporary extension install is destroyed on
exit. No shortcuts are sent to a real application. Logs are metadata only.
"""
from pathlib import Path
import os
import shutil
import signal
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
UUID = 'dictation-helper@faster-whisper-dictation'
BUS = 'org.gnome.Shell.Extensions.DictationHelper'
OBJECT = '/org/gnome/Shell/Extensions/DictationHelper'


def inside():
    subprocess.run(['gsettings', 'set', 'org.gnome.shell', 'enabled-extensions',
                    "['" + UUID + "']"], check=True, timeout=5)
    log = Path(os.environ['HOME']) / 'shell.log'
    with log.open('w') as output:
        shell = subprocess.Popen(['gnome-shell', '--headless', '--wayland', '--no-x11',
            '--virtual-monitor', '1024x768', '--wayland-display', 'dictation-test'],
            stdout=output, stderr=subprocess.STDOUT)
        try:
            subprocess.run(['gdbus', 'wait', '--session', '--timeout', '20', BUS],
                           check=True, timeout=22)
            def call(method, *args):
                return subprocess.check_output(['gdbus', 'call', '--session', '--dest', BUS,
                    '--object-path', OBJECT, '--method', BUS + '.' + method, *args],
                    stderr=subprocess.STDOUT, timeout=3).decode().strip()
            assert call('GetFocus') == "('', '')"
            assert call('Activate', '999999') == '(false,)'
            assert call('SelectionInfo') == '(false,)'
            assert call('ShowBadge', '1') == '()'
            for shortcut in ('paste', 'paste_terminal', 'ctrl_home'):
                assert call('SendShortcut', shortcut) == '()'
            try:
                call('SendShortcut', 'Return')
            except subprocess.CalledProcessError:
                pass
            else:
                raise AssertionError('arbitrary shortcut accepted')
            print('PASS: extension loaded; focus/activate/stubs, three shortcuts, rejection')
        finally:
            if shell.poll() is None:
                # Inspect the exact child PID before signalling it, never patterns.
                subprocess.run(['ps', '-o', 'pid,args', '-p', str(shell.pid)], check=True)
                shell.terminate()
                try:
                    shell.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    subprocess.run(['ps', '-o', 'pid,args', '-p', str(shell.pid)], check=True)
                    shell.kill()
                    shell.wait(timeout=3)
            print('SHELL LOG (isolated session):')
            print(log.read_text()[-10000:])


def main():
    with tempfile.TemporaryDirectory(prefix='wayland-probe-', dir=ROOT / 'evidence') as tmp:
        home = Path(tmp)
        runtime = home / 'runtime'
        runtime.mkdir(mode=0o700)
        target = home / '.local/share/gnome-shell/extensions' / UUID
        shutil.copytree(ROOT / 'gnome-extension' / UUID, target)
        env = dict(PATH='/usr/bin:/bin', HOME=tmp, XDG_RUNTIME_DIR='/proc/self/cwd/runtime',
                   XDG_CONFIG_HOME=str(home / '.config'), XDG_DATA_HOME=str(home / '.local/share'),
                   XDG_CACHE_HOME=str(home / '.cache'), XDG_SESSION_TYPE='wayland',
                   XDG_CURRENT_DESKTOP='GNOME', LIBGL_ALWAYS_SOFTWARE='1',
                   GALLIUM_DRIVER='llvmpipe', LANG='C.UTF-8', NO_AT_BRIDGE='1')
        # Hide host input, DRM, runtime buses and sockets, not just environment
        # variables. Fail closed if unprivileged bwrap is unavailable.
        subprocess.run(['bwrap', '--unshare-all', '--share-net', '--die-with-parent',
                       '--ro-bind', '/', '/', '--bind', tmp, tmp,
                       '--dev', '/dev', '--proc', '/proc', '--tmpfs', '/run',
                       '--tmpfs', '/tmp', '--chdir', tmp,
                       'dbus-run-session', '--', sys.executable, __file__, '--inside'],
                       env=env, check=True, timeout=90)


if __name__ == '__main__':
    inside() if '--inside' in sys.argv else main()
