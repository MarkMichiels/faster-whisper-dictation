"""Driver for probe_wayland_nested.py: runs inside the private nested session only.

The real fork code drives the nested GNOME Shell: App.start (rewrite mode),
context_capture with the Wayland ShellClient, selection_release with the
shell focus, the Ctrl+Home shortcut and the badge of the extension, and the
real Wayland paste (CopyQ stub fails, xclip writes the nested clipboard).
Faked: the recorder (no audio) and the transcription/typing loops (no model,
no LLM call). Keys and mouse reach the nested apps through xdotool on the
private Xvfb; the same keys go as synthetic evdev events into the backend's
InputHub, which never opens a device here. All text is synthetic.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
XVFB = os.environ.get('PROBE_XVFB', '')
OUT = Path(os.environ.get('PROBE_OUT', '.'))
# DISPLAY is the nested shell's own Xwayland, verified by the probe through its
# lock file and process tree (mutter may pick :0 when nothing else holds it).
if (not os.environ.get('PROBE_NESTED_X') or os.environ.get('DISPLAY') != os.environ['PROBE_NESTED_X']
        or XVFB in ('', ':0', ':1', ':2') or not os.environ.get('PROBE_TMP')
        or os.environ.get('XDG_SESSION_TYPE') != 'wayland' or os.environ.get('WAYLAND_DISPLAY')):
    sys.exit('refusing: driver runs only inside the private nested session')

# evdev codes (Linux input-event-codes.h); be AZERTY: physical Q (16) gives 'a'.
L_CTRL, R_CTRL, L_ALT, Q_KEY, BTN_LEFT = 29, 97, 56, 16, 272
JOURNAL: list[tuple[str, dict]] = []


def xdo(*args: str) -> str:
    """xdotool, always on the private Xvfb that hosts the nested shell window."""
    return subprocess.run(['xdotool', *args], capture_output=True, text=True, timeout=10, check=False,
                          env=dict(os.environ, DISPLAY=XVFB)).stdout.strip()


def shot(name: str) -> None:
    subprocess.run(['import', '-display', XVFB, '-window', 'root', str(OUT / name)],
                   timeout=10, check=False, capture_output=True)


class FakeRecorder:
    SAMPLE_RATE = 16000

    def __init__(self, transcription_queue, **kw):
        self._raw_frames = []
        self.recording = False

    def start(self):
        self.recording = True

    def stop(self):
        self.recording = False


class Harness:
    def __init__(self):
        sys.path.insert(0, str(ROOT))
        os.chdir(ROOT)
        import dictation
        import rewrite
        import selection_release
        assert dictation.WAYLAND, 'XDG_SESSION_TYPE=wayland expected'
        self.d = dictation
        dictation.diagnostic = lambda kind, **details: JOURNAL.append((kind, details))
        dictation.StreamingRecorder = FakeRecorder
        app = dictation.App.__new__(dictation.App)
        app.args = types.SimpleNamespace(silence_ms=1000, auto_stop_silence=0, min_chunk_s=1, max_chunk_s=10,
                                         eager_prob=0.5, max_time=600, language='nl')
        app.active, app.session, app.timer = False, None, None
        app.replayer = dictation.KeyboardReplayer()
        app.beep = lambda k, wait=True: JOURNAL.append(('beep', dict(sound=k)))
        app.rewrite_config = rewrite.RewriteConfig(endpoint='https://llm.invalid/v1', model='none')
        app.rewrite_reason = None
        app._transcription_loop = lambda session=None: None
        app._typing_loop = lambda session=None: session.done.set()
        # The listener wiring of App.run (Right-Ctrl double tap), minus hub.run().
        listener = dictation.MultiTapKeyListener(app.start, app.stop, key=dictation.keyboard.Key.ctrl_r,
                                                 default_language='nl', shift_language='en')
        app.input_trace = selection_release.InputTrace(focus=app._focus_reader())
        listener.trace = app.input_trace
        app.key_listener = listener
        app.input_trace.set_known(keys=True)
        app._start_input_mouse_listener()
        self.hub = listener.hub
        self.hub.provider = lambda: {}   # never a real device; run() is never called
        self.hub.key_listener = listener
        self.app = app
        self.shell = app.replayer.shell

    # Synthetic evdev into the backend, mirrored as X keys on the private Xvfb.
    def ev(self, code: int, value: int, device: str = 'kbd') -> None:
        self.hub.event(device, types.SimpleNamespace(type=1, code=code, value=value))

    def ctrl_a(self) -> None:
        self.ev(L_CTRL, 1)
        self.ev(Q_KEY, 1)
        self.ev(Q_KEY, 0)
        self.ev(L_CTRL, 0)
        xdo('key', 'ctrl+a')

    def drag(self, x0: int, y0: int, x1: int, y1: int) -> None:
        xdo('mousemove', str(x0), str(y0))
        time.sleep(0.1)
        self.ev(BTN_LEFT, 1, 'mouse')
        xdo('mousedown', '1')
        for step in range(1, 6):
            xdo('mousemove', str(x0 + (x1 - x0) * step // 5), str(y0 + (y1 - y0) * step // 5))
            time.sleep(0.05)
        self.ev(BTN_LEFT, 0, 'mouse')
        xdo('mouseup', '1')

    def chord(self, hold_s: float, modifier: int = L_ALT, badge_shot: str | None = None) -> dict:
        """Left-Alt (or Left-Ctrl) held, Right-Ctrl double tap, release after hold_s."""
        mark = len(JOURNAL)
        x_mod = 'Alt_L' if modifier == L_ALT else 'Control_L'
        self.ev(modifier, 1)
        xdo('keydown', x_mod)
        for _ in range(2):
            self.ev(R_CTRL, 1)
            self.ev(R_CTRL, 0)
        session = self.app.session
        if badge_shot:
            time.sleep(0.4)
            shot(badge_shot)
        time.sleep(max(0.0, hold_s - (0.4 if badge_shot else 0.0)))
        self.ev(modifier, 0)
        xdo('keyup', x_mod)
        deadline = time.monotonic() + 4
        while time.monotonic() < deadline and not any(k == 'selection_release' for k, _ in JOURNAL[mark:]):
            time.sleep(0.05)
        time.sleep(0.3)
        session = self.app.session
        return dict(mark=mark, session=session)

    def finish(self, run: dict, paste: str | None = None) -> dict:
        session = run['session']
        self.app.stop()
        if session is not None:
            session.done.wait(3)
        delivery = None
        if paste and session is not None:
            # The paste of the rewritten text, through the real Wayland replayer.
            self.app.replayer.target_wid = session.target_wid
            delivery = self.app.replayer.type_text(paste, echo=False, rewrite_mode=True)
            time.sleep(0.8)
        lines = JOURNAL[run['mark']:]
        rel = next((d for k, d in lines if k == 'selection_release'), None)
        ctx = next((d for k, d in lines if k == 'rewrite_context'), None)
        start = next((d for k, d in lines if k == 'session_start'), {})
        return dict(mode=start.get('mode'), keys=start.get('keys'),
                    context=ctx, selection_release=rel,
                    badge=[d for k, d in lines if k == 'context_badge'],
                    paste=None if delivery is None else {k: delivery.get(k) for k in
                                                         ('ok', 'path', 'paste', 'ownership', 'shortcut')})


def editor_state(path: str) -> dict:
    return json.loads(Path(path).read_text())


def reset(path: str) -> None:
    Path(path + '.reset').touch()
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline and Path(path + '.reset').exists():
        time.sleep(0.05)
    time.sleep(0.3)


def run_cases(explore: bool) -> int:
    h = Harness()
    windows = json.loads(os.environ['PROBE_WINDOWS'])
    editors = json.loads(os.environ['PROBE_EDITORS'])
    A, B, T = windows['editor_A']['id'], windows['editor_B']['id'], windows['terminal']['id']
    results = []

    def focus(wid: str) -> None:
        h.shell.activate(wid)
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline and h.shell.focus()[0] != wid:
            time.sleep(0.05)
        time.sleep(0.4)

    def case(name: str, fn) -> None:
        print('case', name, flush=True)
        try:
            res = fn()
        except Exception as exc:  # recorded, the other cases still run
            res = dict(error='%s: %s' % (type(exc).__name__, exc))
        res['case'] = name
        results.append(res)
        print(json.dumps(res, ensure_ascii=False), flush=True)

    check = subprocess.run([sys.executable, str(ROOT / 'context_capture.py'), '--probe'],
                           capture_output=True, text=True, timeout=10, check=False)
    print('x probe:', check.returncode, check.stdout.strip()[:300], check.stderr.strip()[-600:], flush=True)
    if explore:
        # Quick check of the paste route: real xclip fallback + shell Ctrl+V.
        focus(A)
        reset(editors['A'])
        h.app.replayer.target_wid = A
        res = h.app.replayer.type_text('XTEST ', echo=False)
        time.sleep(1.0)
        print('paste:', {k: res.get(k) for k in ('ok', 'path', 'paste')}, editor_state(editors['A']), flush=True)
        shot('explore.png')
        return 0

    def a_ctrl_a(release: bool, label: str):
        h.d.RELEASE_SELECTION = release
        focus(A)
        reset(editors['A'])
        h.ctrl_a()
        time.sleep(1.5)
        before = editor_state(editors['A'])
        run = h.chord(0.6, badge_shot='%s_badge.png' % label)
        after_release = editor_state(editors['A'])
        res = h.finish(run, paste='XHERSCHREVEN ')
        after = editor_state(editors['A'])
        shot('%s_after.png' % label)
        h.d.RELEASE_SELECTION = True
        res.update(editor_before=before, editor_after_release=after_release, editor_after_paste=after)
        return res

    def b_mouse(label='B'):
        focus(A)
        reset(editors['A'])
        h.drag(20, 98, 470, 98)
        time.sleep(1.0)
        before = editor_state(editors['A'])
        run = h.chord(0.6)
        res = h.finish(run)
        res.update(editor_before=before, editor_after=editor_state(editors['A']))
        shot('%s_after.png' % label)
        return res

    # B first: no Ctrl+A in the trace yet. B2 after a Ctrl+A: the click counts.
    case('B_mouse_selection', b_mouse)
    case('A_ctrl_a_release', lambda: a_ctrl_a(True, 'A'))
    case('A2_ctrl_a_release_off', lambda: a_ctrl_a(False, 'A2'))
    case('B2_mouse_selection_after_ctrl_a', lambda: b_mouse('B2'))

    def c_old():
        focus(A)
        reset(editors['A'])
        h.ctrl_a()
        time.sleep(21.5)
        run = h.chord(0.6)
        res = h.finish(run)
        res.update(editor_after=editor_state(editors['A']))
        return res
    case('C_old_selection', c_old)

    def d_elsewhere():
        focus(B)
        reset(editors['B'])
        h.ctrl_a()
        time.sleep(0.5)
        focus(A)
        reset(editors['A'])
        run = h.chord(0.6)
        res = h.finish(run)
        res.update(editor_A_after=editor_state(editors['A']), editor_B_after=editor_state(editors['B']))
        return res
    case('D_selection_in_other_program', d_elsewhere)

    def e_terminal():
        focus(T)
        shot('E_terminal_before.png')
        h.drag(3, 68, 360, 68)
        time.sleep(1.0)
        run = h.chord(0.6, modifier=L_CTRL)
        res = h.finish(run)
        shot('E_terminal_after.png')
        return res
    case('E_terminal_left_ctrl', e_terminal)

    def f_alt_held():
        focus(A)
        reset(editors['A'])
        h.ctrl_a()
        time.sleep(1.0)
        run = h.chord(3.0)
        res = h.finish(run)
        res.update(editor_after=editor_state(editors['A']))
        return res
    case('F_alt_held_3s', f_alt_held)

    (OUT / 'results_nested.json').write_text(json.dumps(results, ensure_ascii=False, indent=1))
    return 0
