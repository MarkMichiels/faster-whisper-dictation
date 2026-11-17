"""Nested GNOME proof: what a mouse thumb button does to the caret under Wayland.

    python tests/probe_thumb_nested.py --out DIR

Under X11 free_mouse_thumb_buttons.sh remaps the thumb buttons (8/9) to 10/11,
so apps ignore the press that starts or stops dictation. Under Wayland that
remap does not exist: the press reaches the focused app as BTN_SIDE/BTN_EXTRA.
This probe measures, on a private nested shell (same private stack as
probe_wayland_nested.py), what one such press does to the caret in

- Chrome (native Wayland, temporary profile) with a synthetic textarea page
  that has pushState history like a single-page app, and
- the synthetic GTK4 editor (nested_editor.py).

Per case: caret at the end, pointer parked over earlier text, one press of
X button 8 or 9 on the private Xvfb (mutter turns it into BTN_SIDE/EXTRA),
then one typed 'X'. The result is where the caret was before and after the
press and where the X landed. All text is synthetic; nothing reaches the
desktop session. Every process is stopped by its own PID after a ps check.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests'))
import probe_wayland_nested as base  # noqa: E402

PAGE = """<!doctype html><meta charset=utf-8><title>probe</title>
<body style="margin:0"><textarea id=t autocomplete=off style="width:100%;height:90vh;font-size:28px">
Eerste synthetische regel van de proef.
Tweede synthetische regel van de proef.
Derde synthetische regel van de proef.</textarea>
<script>
const t = document.getElementById('t');
function log(kind, extra) {
  console.log('PROBE ' + JSON.stringify(Object.assign({kind, caret: t.selectionStart,
    sel_end: t.selectionEnd, len: t.value.length, x_at: t.value.indexOf('X'),
    path: location.hash, focus: document.activeElement === t}, extra || {})));
}
history.pushState({}, '', '#een'); history.pushState({}, '', '#twee');
for (const k of ['mousedown', 'mouseup', 'auxclick'])
  addEventListener(k, e => log(k, {button: e.button}), true);
addEventListener('popstate', () => log('popstate'));
t.addEventListener('input', () => log('input'));
document.addEventListener('selectionchange', () => log('selectionchange'));
t.focus(); t.setSelectionRange(t.value.length, t.value.length); log('ready');
setInterval(() => log('tick'), 400);
</script>"""


def xvfb(*args: str) -> None:
    subprocess.run(['xdotool', *args], env=dict(os.environ, DISPLAY=os.environ['PROBE_XVFB']),
                   timeout=10, check=False, capture_output=True)


def shot(out: Path, name: str) -> None:
    subprocess.run(['import', '-display', os.environ['PROBE_XVFB'], '-window', 'root', str(out / name)],
                   timeout=10, check=False, capture_output=True)


def chrome_lines(log: Path) -> list[dict]:
    rows = []
    for line in log.read_text(errors='replace').splitlines():
        match = re.search(r'PROBE (\{.*?\})"?, source', line) or re.search(r'PROBE (\{.*\})', line)
        if match:
            try:
                rows.append(json.loads(match.group(1).replace('\\"', '"')))
            except ValueError:
                pass
    return rows


def inner() -> int:
    tmp, out = Path(os.environ['PROBE_TMP']), Path(os.environ['PROBE_OUT'])
    home = Path(os.environ['HOME'])
    target = home / '.local/share/gnome-shell/extensions' / base.UUID
    shutil.copytree(ROOT / 'gnome-extension' / base.UUID, target)
    keyfile = home / '.config/glib-2.0/settings/keyfile'
    keyfile.parent.mkdir(parents=True)
    keyfile.write_text("[org/gnome/shell]\nenabled-extensions=['%s']\ndisable-user-extensions=false\n"
                       "welcome-dialog-last-shown-version='999'\n" % base.UUID)
    procs: list[tuple[str, subprocess.Popen]] = []
    results: dict = {}
    try:
        if '/run/user/' in os.environ.get('DBUS_SESSION_BUS_ADDRESS', '/run/user/'):
            raise RuntimeError('not on a private session bus')
        stub = subprocess.Popen(['/usr/bin/python3', str(ROOT / 'tests/nested_systemd_stub.py')])
        procs.append(('systemd stub', stub))
        subprocess.run(['gdbus', 'wait', '--session', '--timeout', '10', 'org.freedesktop.systemd1'],
                       check=True, timeout=15)
        with open(tmp / 'shell.log', 'w') as log:
            shell = subprocess.Popen(['gnome-shell', '--nested', '--wayland',
                                      '--wayland-display', base.WAYLAND_NAME],
                                     stdout=log, stderr=subprocess.STDOUT)
        procs.append(('gnome-shell', shell))
        subprocess.run(['gdbus', 'wait', '--session', '--timeout', '40', base.BUS], check=True, timeout=45)
        nested_x = base.wait_for(lambda: base.nested_display(tmp / 'shell.log', shell.pid), 20)
        if not nested_x or nested_x == os.environ['PROBE_XVFB']:
            raise RuntimeError('nested Xwayland display not verified')
        app_env = dict(os.environ, WAYLAND_DISPLAY=base.WAYLAND_NAME, GDK_BACKEND='wayland')
        app_env.pop('DISPLAY')
        time.sleep(2.0)
        xvfb('key', 'Escape')
        time.sleep(1.5)

        def case(name, read, settle=0.7):
            """Pointer over earlier text, one thumb press, then one typed X."""
            row = {}
            for button in os.environ.get('PROBE_BUTTONS', '1,8,9').split(','):
                reset = read('reset')
                xvfb('mousemove', '300', '130')   # over the first line, caret stays at the end
                time.sleep(0.6)
                before = read('state')
                xvfb('click', button)
                time.sleep(settle)
                after_press = read('state')
                xvfb('type', 'X')
                time.sleep(0.6)
                after_x = read('state')
                shot(out, 'thumb_%s_button%s.png' % (name, button))
                row['button' + button] = dict(reset=reset, before=before, after_press=after_press,
                                              after_x=after_x)
            results[name] = row

        # GTK4 editor (native Wayland).
        if os.environ.get('PROBE_SKIP_GTK') != '1':
            state = tmp / 'editor.json'
            editor = subprocess.Popen(['/usr/bin/python3', str(ROOT / 'tests/nested_editor.py'),
                                       '--app-id', 'org.example.DictationProbeThumb', '--state', str(state),
                                       '--title', 'Editor'], env=app_env,
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            procs.append(('editor', editor))
            if not base.wait_for(state.exists, 20):
                raise RuntimeError('editor did not start')
            time.sleep(1.5)

            def read_editor(what):
                if what == 'reset':
                    Path(str(state) + '.reset').write_text('')
                    time.sleep(0.5)
                    return True
                data = json.loads(state.read_text())
                text_pos = data.get('head', '')
                return dict(cursor=data['cursor'], chars=data['chars'], has_selection=data['has_selection'],
                            x_in_head='X' in text_pos)
            case('gtk_editor', read_editor)
            base.stop(editor, 'editor')

        # Chrome, native Wayland, temporary profile.
        page = tmp / 'page.html'
        page.write_text(PAGE)
        chrome_log = tmp / 'chrome.log'
        with open(chrome_log, 'w') as log:
            chrome = subprocess.Popen(['google-chrome', '--ozone-platform=wayland', '--user-data-dir=%s' % (tmp / 'chrome'),
                                       '--no-first-run', '--no-default-browser-check', '--disable-gpu',
                                       '--disable-sync', '--disable-extensions', '--password-store=basic',
                                       '--enable-logging=stderr', '--v=0', '--start-maximized',
                                       '--app=file://%s' % page], env=dict(app_env, WAYLAND_DEBUG=os.environ.get('PROBE_WLDEBUG', '')),
                                      stdout=log, stderr=subprocess.STDOUT)
        procs.append(('chrome', chrome))
        if not base.wait_for(lambda: any(r['kind'] == 'ready' for r in chrome_lines(chrome_log)), 40):
            raise RuntimeError('chrome page did not load')
        time.sleep(2.0)

        def read_chrome(what):
            if what == 'reset':
                # Fresh page state: reload with F5 keeps the same synthetic text.
                xvfb('key', 'F5')
                time.sleep(2.0)
                return True
            time.sleep(0.5)
            rows = chrome_lines(chrome_log)
            last = rows[-1] if rows else {}
            return {k: last.get(k) for k in ('caret', 'len', 'x_at', 'path', 'focus')}
        case('chrome', read_chrome, settle=1.0)
        events = [r for r in chrome_lines(chrome_log) if r['kind'] not in ('tick', 'selectionchange')]
        results['chrome_events'] = events
        return 0
    finally:
        (out / 'results_thumb.json').write_text(json.dumps(results, indent=1, ensure_ascii=False))
        for name, proc in reversed(procs):
            base.stop(proc, name)


def outer(out: Path) -> int:
    out.mkdir(parents=True, exist_ok=True)
    number = base.free_display()
    tmp = Path(tempfile.mkdtemp(prefix='dctt-', dir='/tmp'))
    xvfb_proc = sysbus = None
    try:
        xvfb_proc = subprocess.Popen(['Xvfb', ':%d' % number, '-screen', '0', '1600x1000x24', '-nolisten', 'tcp'],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if not base.wait_for(lambda: Path('/tmp/.X11-unix/X%d' % number).exists(), 10):
            raise RuntimeError('Xvfb did not start')
        sysbus = subprocess.Popen(['dbus-daemon', '--session', '--nofork', '--nopidfile',
                                   '--address=unix:path=%s/sysbus' % tmp],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if not base.wait_for(lambda: (tmp / 'sysbus').exists(), 10):
            raise RuntimeError('private system bus did not start')
        home = tmp / 'home'
        for path in (home, tmp / 'rt'):
            path.mkdir(mode=0o700)
        env = dict(PATH='/usr/bin:/bin', HOME=str(home), LANG='C.UTF-8',
                   XDG_RUNTIME_DIR=str(tmp / 'rt'), XDG_CONFIG_HOME=str(home / '.config'),
                   XDG_DATA_HOME=str(home / '.local/share'), XDG_CACHE_HOME=str(home / '.cache'),
                   XDG_CURRENT_DESKTOP='GNOME', GSETTINGS_BACKEND='keyfile', NO_AT_BRIDGE='1',
                   LIBGL_ALWAYS_SOFTWARE='1', DISPLAY=':%d' % number, PROBE_XVFB=':%d' % number,
                   DBUS_SYSTEM_BUS_ADDRESS='unix:path=%s/sysbus' % tmp, PROBE_TMP=str(tmp),
                   PROBE_OUT=str(out), MUTTER_DEBUG_DUMMY_MODE_SPECS='1600x1000',
                   **{k: v for k, v in os.environ.items() if k in ('PROBE_BUTTONS', 'PROBE_SKIP_GTK', 'PROBE_WLDEBUG')})
        result = subprocess.run(['dbus-run-session', '--', sys.executable, __file__, '--inside'],
                                env=env, timeout=300)
        return result.returncode
    finally:
        base.stop(sysbus, 'private system bus')
        base.stop(xvfb_proc, 'Xvfb')
        for name in ('shell.log', 'chrome.log'):
            if (tmp / name).exists():
                shutil.copy(tmp / name, out / ('thumb_' + name))
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    if '--inside' in sys.argv:
        sys.exit(inner())
    if '--out' not in sys.argv:
        sys.exit('usage: probe_thumb_nested.py --out DIR')
    sys.exit(outer(Path(sys.argv[sys.argv.index('--out') + 1]).resolve()))
