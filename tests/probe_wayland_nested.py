"""Nested GNOME Shell proof of the Wayland reply-context path. Run by hand.

    python tests/probe_wayland_nested.py --out DIR [--explore]

Everything runs on a private display stack that this script starts and stops:
an Xvfb display (a free number from 140 up), a private D-Bus session bus, a
private stand-in for the system bus (so the shell cannot reach logind or the
display manager), a stub for the systemd user manager on the private session
bus (the shell starts an X11 services target before Xwayland is usable), and
`gnome-shell --nested --wayland` with a private HOME that has only the
Dictation Helper extension enabled. The shell runs under strace (skip with
PROBE_NO_STRACE=1); the result lists every /dev path it opened (no /dev/input
expected). --explore only checks the X probe and one paste.

The dictation side is the real fork code (App.start, context_capture,
selection_release, the Wayland replayer and ShellClient) with a fake recorder
and no model or LLM call. Key and mouse input reach the nested apps through
xdotool on the private Xvfb; the same keys are fed as synthetic evdev events
into the backend's InputHub, which never opens a device here. CopyQ is
shadowed by a stub that fails, so the xclip fallback writes the private
clipboard only. All text is synthetic.

Every started process is recorded by PID, checked with ps before it is
signalled, and stopped in finally blocks; each stage has a hard time-out.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UUID = 'dictation-helper@faster-whisper-dictation'
BUS = 'org.gnome.Shell.Extensions.DictationHelper'
WAYLAND_NAME = 'wl-dctn'


def stop(proc: subprocess.Popen | None, name: str, log=print) -> None:
    """Stop exactly this child after a ps check of its PID; never by pattern."""
    if proc is None or proc.poll() is not None:
        return
    check = subprocess.run(['ps', '-o', 'pid=,args=', '-p', str(proc.pid)],
                           capture_output=True, text=True, timeout=5)
    log('stop %s: %s' % (name, check.stdout.strip()[:120]))
    if not check.stdout.strip():
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def free_display() -> int:
    for number in range(140, 200):
        if not Path('/tmp/.X11-unix/X%d' % number).exists() and not Path('/tmp/.X%d-lock' % number).exists():
            return number
    raise RuntimeError('no free display number')


def wait_for(predicate, timeout: float, step: float = 0.1):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        value = predicate()
        if value:
            return value
        time.sleep(step)
    return None


# ---------------------------------------------------------------------------
# Outer: Xvfb, private buses, then the inner part in dbus-run-session.
# ---------------------------------------------------------------------------

def outer(out: Path, explore: bool) -> int:
    out.mkdir(parents=True, exist_ok=True)
    number = free_display()
    tmp = Path(tempfile.mkdtemp(prefix='dctn-', dir='/tmp'))
    xvfb = sysbus = None
    try:
        xvfb = subprocess.Popen(['Xvfb', ':%d' % number, '-screen', '0', '1600x1000x24', '-nolisten', 'tcp'],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if not wait_for(lambda: Path('/tmp/.X11-unix/X%d' % number).exists(), 10):
            raise RuntimeError('Xvfb did not start')
        sysbus = subprocess.Popen(['dbus-daemon', '--session', '--nofork', '--nopidfile',
                                   '--address=unix:path=%s/sysbus' % tmp],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if not wait_for(lambda: (tmp / 'sysbus').exists(), 10):
            raise RuntimeError('private system bus did not start')
        home = tmp / 'home'
        shim = tmp / 'shim'
        for path in (home, shim, tmp / 'rt'):
            path.mkdir(mode=0o700)
        stub = shim / 'copyq'
        stub.write_text('#!/bin/sh\nexit 1\n')  # never reach a real CopyQ
        stub.chmod(0o755)
        env = dict(PATH='%s:/usr/bin:/bin' % shim, HOME=str(home), LANG='C.UTF-8',
                   XDG_RUNTIME_DIR=str(tmp / 'rt'), XDG_CONFIG_HOME=str(home / '.config'),
                   XDG_DATA_HOME=str(home / '.local/share'), XDG_CACHE_HOME=str(home / '.cache'),
                   XDG_CURRENT_DESKTOP='GNOME', GSETTINGS_BACKEND='keyfile', NO_AT_BRIDGE='1',
                   LIBGL_ALWAYS_SOFTWARE='1', DISPLAY=':%d' % number, PROBE_XVFB=':%d' % number,
                   DBUS_SYSTEM_BUS_ADDRESS='unix:path=%s/sysbus' % tmp, PROBE_TMP=str(tmp),
                   PROBE_OUT=str(out), PROBE_PYTHON=sys.executable,
                   MUTTER_DEBUG_DUMMY_MODE_SPECS='1600x1000')
        args = [sys.executable, __file__, '--inside'] + (['--explore'] if explore else [])
        result = subprocess.run(['dbus-run-session', '--'] + args, env=env, timeout=600)
        return result.returncode
    finally:
        stop(sysbus, 'private system bus')
        stop(xvfb, 'Xvfb')
        for name in ('shell.log', 'strace.log'):
            if (tmp / name).exists():
                shutil.copy(tmp / name, out / ('nested_' + name))
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Inner: nested shell, synthetic editors, then the driver.
# ---------------------------------------------------------------------------

def inner(explore: bool) -> int:
    tmp, out = Path(os.environ['PROBE_TMP']), Path(os.environ['PROBE_OUT'])
    home = Path(os.environ['HOME'])
    target = home / '.local/share/gnome-shell/extensions' / UUID
    shutil.copytree(ROOT / 'gnome-extension' / UUID, target)
    keyfile = home / '.config/glib-2.0/settings/keyfile'
    keyfile.parent.mkdir(parents=True)
    keyfile.write_text("[org/gnome/shell]\nenabled-extensions=['%s']\ndisable-user-extensions=false\n"
                       "welcome-dialog-last-shown-version='999'\n" % UUID)
    procs: list[tuple[str, subprocess.Popen]] = []
    try:
        if '/run/user/' in os.environ.get('DBUS_SESSION_BUS_ADDRESS', '/run/user/'):
            raise RuntimeError('not on a private session bus')
        # The shell starts an X11 services target through systemd before Xwayland
        # is usable; a stub on the private bus completes that job at once.
        stub = subprocess.Popen(['/usr/bin/python3', str(ROOT / 'tests/nested_systemd_stub.py')])
        procs.append(('systemd stub', stub))
        subprocess.run(['gdbus', 'wait', '--session', '--timeout', '10', 'org.freedesktop.systemd1'],
                       check=True, timeout=15)
        trace = ([] if os.environ.get('PROBE_NO_STRACE') else
                 ['strace', '-f', '-qq', '-e', 'trace=open,openat', '-o', str(tmp / 'strace.log')])
        with open(tmp / 'shell.log', 'w') as log:
            shell = subprocess.Popen(trace + ['gnome-shell', '--nested', '--wayland',
                                              '--wayland-display', WAYLAND_NAME],
                                     stdout=log, stderr=subprocess.STDOUT)
        procs.append(('gnome-shell', shell))
        subprocess.run(['gdbus', 'wait', '--session', '--timeout', '40', BUS], check=True, timeout=45)
        nested_x = wait_for(lambda: nested_display(tmp / 'shell.log', shell.pid), 20)
        if not nested_x or nested_x == os.environ['PROBE_XVFB']:
            raise RuntimeError('nested Xwayland display not verified')
        app_env = dict(os.environ, WAYLAND_DISPLAY=WAYLAND_NAME, GDK_BACKEND='wayland')
        app_env.pop('DISPLAY')
        # Leave the startup overview: Escape through the private Xvfb only.
        time.sleep(2.0)
        subprocess.run(['xdotool', 'key', 'Escape'], env=dict(os.environ, DISPLAY=os.environ['PROBE_XVFB']),
                       timeout=10, check=False)
        time.sleep(1.5)
        windows = {}

        def launch(name, argv, env, ready=None):
            before = focus()
            proc = subprocess.Popen(argv, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            procs.append((name, proc))
            if ready and not wait_for(ready, 20):
                raise RuntimeError('%s did not start' % name)
            got = wait_for(lambda: focus()[0] not in ('', before[0]) and focus(), 20)
            if not got:
                raise RuntimeError('%s never got focus' % name)
            windows[name] = dict(id=got[0], wm_class=got[1], pid=proc.pid)
            time.sleep(0.8)

        editors = {}
        for name in ('A', 'B'):
            state = tmp / ('editor_%s.json' % name)
            editors[name] = str(state)
            launch('editor_' + name, ['/usr/bin/python3', str(ROOT / 'tests/nested_editor.py'),
                                      '--app-id', 'org.example.DictationProbe' + name,
                                      '--state', str(state), '--title', 'Editor ' + name],
                   app_env, ready=state.exists)
        launch('terminal', ['kitty', '--class', 'kitty', '--start-as', 'maximized',
                            '-o', 'linux_display_server=wayland',
                            '/bin/sh', '-c', 'printf "synthetische terminalregel voor de proef\\n"; '
                            'exec sleep 600'],
               dict(app_env, KITTY_CONFIG_DIRECTORY=str(tmp / 'kitty')))
        print('windows:', windows)
        driver_env = dict(os.environ, DISPLAY=nested_x, XDG_SESSION_TYPE='wayland',
                          PROBE_EDITORS=json.dumps(editors), PROBE_NESTED_X=nested_x,
                          PROBE_WINDOWS=json.dumps(windows),
                          DICTATION_EVDEV_LAYOUT='be')
        driver_env.pop('WAYLAND_DISPLAY', None)
        # Mutter's Xwayland requires its own cookie (in the private runtime dir).
        auth = sorted((tmp / 'rt').glob('.mutter-Xwaylandauth.*'))
        if not auth:
            raise RuntimeError('nested Xwayland cookie not found')
        driver_env['XAUTHORITY'] = str(auth[0])
        args = [os.environ['PROBE_PYTHON'], __file__, '--driver'] + (['--explore'] if explore else [])
        result = subprocess.run(args, env=driver_env, timeout=420, cwd=str(ROOT))
        return result.returncode
    finally:
        for name, proc in reversed(procs):
            stop(proc, name)
        opened = devices_opened(tmp / 'strace.log')
        (out / 'nested_devices.json').write_text(json.dumps(opened, indent=1))
        print('nested shell /dev opens:', opened)


def focus() -> tuple[str, str]:
    """GetFocus of the nested helper (private session bus)."""
    try:
        text = subprocess.check_output(['gdbus', 'call', '--session', '--dest', BUS, '--object-path',
                                        '/org/gnome/Shell/Extensions/DictationHelper', '--method',
                                        BUS + '.GetFocus'], timeout=5, stderr=subprocess.DEVNULL).decode()
    except Exception:
        return ('', '')
    import ast
    return tuple(ast.literal_eval(text.strip()))


def nested_display(log: Path, shell_pid: int) -> str | None:
    """The nested shell's public X11 display, verified by owner.

    Mutter takes the first free display number, which may be :0 when no
    other server holds it. It counts only when its lock file names a process
    whose parent chain reaches our own nested shell.
    """
    import re
    match = re.search(r'Using public X11 display (:\d+)', log.read_text(errors='replace'))
    if not match:
        return None
    display = match.group(1)
    try:
        pid = int(Path('/tmp/.X%s-lock' % display[1:]).read_text().strip())
    except (OSError, ValueError):
        return None
    for _ in range(10):
        if pid == shell_pid:
            return display
        try:
            stat = Path('/proc/%d/stat' % pid).read_text()
        except OSError:
            return None
        pid = int(stat.rsplit(')', 1)[1].split()[1])
        if pid <= 1:
            return None
    return None


def devices_opened(path: Path) -> dict[str, int]:
    import re
    counts: dict[str, int] = {}
    if not path.exists():
        return counts
    for line in path.read_text(errors='replace').splitlines():
        match = re.search(r'"(/dev/[^"]*)"', line)
        if match:
            counts[match.group(1)] = counts.get(match.group(1), 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Driver: the real fork code against the nested shell.
# ---------------------------------------------------------------------------

def driver(explore: bool) -> int:
    from nested_driver import run_cases
    return run_cases(explore)


if __name__ == '__main__':
    sys.path.insert(0, str(ROOT / 'tests'))
    if '--inside' in sys.argv:
        sys.exit(inner('--explore' in sys.argv))
    if '--driver' in sys.argv:
        sys.exit(driver('--explore' in sys.argv))
    if '--out' not in sys.argv:
        sys.exit('usage: probe_wayland_nested.py --out DIR [--explore]')
    if os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'):
        print('note: the desktop DISPLAY/WAYLAND_DISPLAY are not passed on')
    sys.exit(outer(Path(sys.argv[sys.argv.index('--out') + 1]).resolve(), '--explore' in sys.argv))
