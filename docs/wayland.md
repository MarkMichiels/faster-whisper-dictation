# GNOME Wayland kernel path

The session type selects the backend once at process startup. X11 keeps pynput,
xdotool and the existing clipboard policy. `XDG_SESSION_TYPE=wayland` selects
read-only evdev input and the GNOME Dictation Helper D-Bus interface. It does
not import pynput or use XTest, uinput, input grabs or root.

This is an **experimental kernel path**, not a completed Wayland release.
A real Wayland login/application acceptance test is still required. Metadata
declares GNOME 46–50; that is not a claim that all five versions were tested.

## Requirements and limitations

- GNOME Shell, `gdbus`, and `evdev` (`requirements-wayland.txt`).
- Read access to relevant `/dev/input/event*` devices, normally via group
  `input`. This is keylogger-level access for every process of that user.
  The installer checks membership but never changes groups or udev rules.
- CopyQ running under Xwayland, with the existing xclip fallback. Clipboard
  writes, MIME ownership checks, history and paste-only delivery are inherited
  unchanged. No automatic Return, typed-text fallback or arbitrary shortcut.
- Only Right-Ctrl double-tap is supported on Wayland; custom double keys and
  combo shortcuts fail explicitly. Left-Shift chooses the alternate language,
  Left-Alt requests rewrite, Left-Ctrl requests terminal rewrite. Right-hand
  modifiers do not select modes. Repeat presses are ignored.
- The explicit evdev keymap contract defaults to Belgian AZERTY (`be`).
  `DICTATION_EVDEV_LAYOUT=us` selects US QWERTY. Only these two values are
  accepted. These are Linux keycodes: Ctrl+A uses KEY_Q (16) for `be`, KEY_A
  (30) for `us`; Ctrl+C uses KEY_C (46). This is not layout autodetection and
  does not follow GNOME layout switching. Use a single matching layout while
  testing; other layouts require a future xkbcommon implementation.
- Thumb buttons are read, **not suppressed**: browser back/forward still
  happens. `button8`/`x1` map to BTN_SIDE, `button9`/`x2` to BTN_EXTRA;
  `button10` and `button11` map to BTN_FORWARD and BTN_BACK. An empty mouse
  button setting disables mouse triggers. No xinput helpers run on Wayland.
- Hotplug is rescanned every second. Disconnect resets held state and pending
  taps; duplicate held modifiers across devices are reference-counted. Losing
  every device after startup waits for hotplug; no readable devices at startup
  fails with an input-access message. SYN_DROPPED clears state and ignores
  events through SYN_REPORT; a fresh press is required afterwards.
- The context badge and automatic selection release are deliberately disabled.
  Context capture still uses the existing Xwayland best-effort reader; PRIMARY
  age/owner metadata for native windows is not yet reliable.

## Extension interface and safety

UUID: `dictation-helper@faster-whisper-dictation`.
Bus/interface: `org.gnome.Shell.Extensions.DictationHelper`.
Object: `/org/gnome/Shell/Extensions/DictationHelper`.

| Method | Signature | Result |
| --- | --- | --- |
| GetFocus | `() -> (s, s)` | Meta.Window id as string, wm_class; empty strings if no focus |
| Activate | `(s) -> (b)` | Request activation of an existing numeric id |
| SendShortcut | `(s) -> ()` | Only `paste`, `paste_terminal`, `ctrl_home`; everything else errors |
| ShowBadge | `(u) -> ()` | No-op placeholder |
| SelectionInfo | `() -> (b)` | Always false/unknown placeholder |

No selection text or window titles cross this interface. Every process on the
session bus can request these methods, including paste into the focused app.
This is not a security boundary against another process in the same account.
Python verifies target focus after activation and again before sending paste;
missing extension, stale id or failed focus refuses paste while keeping the
clipboard/history recovery path. The last focus check is not atomic with the
shortcut. A command receipt is not application acknowledgement.

The Shell creates a Clutter virtual keyboard. The allowlist is a Map, not a
prototype-bearing object lookup. Only the keyvals for Ctrl+V, Ctrl+Shift+V and
Ctrl+Home exist in it. Pressed keys are released in reverse order in a finally
block. Disabling unexports D-Bus and drops the virtual device.

## Installation and rollout (manual approval required)

These commands are **not** part of offline tests. Installing changes user
extensions and systemd configuration and starts/restarts dictation. Review the
branch before authorizing this on a real desktop. Do not run two dictation
instances at once.

1. Keep the current X11 checkout/service as the rollback. Check `id -nG`, the
   selected GNOME layout, available `gdbus`, and CopyQ's Xwayland setup.
   Group changes, if needed, are a separate administrator decision plus logout.
2. Review `bash install.sh --check --wayland`. A first install is expected to
   report not current. This mode makes no persistent changes.
3. After explicit approval, run `bash install.sh --wayland`. The explicit flag
   allows preparing from X11; without it, extension installation happens only
   when the installer itself runs under Wayland. This installs evdev in the
   checkout venv and copies the two extension files. It never changes input
   groups. It renders a Wayland unit with no xinput pre/post hooks.
4. Log out/in to load a newly installed extension. Choose Ubuntu (Wayland),
   not Ubuntu on Xorg. Run
   `gnome-extensions enable dictation-helper@faster-whisper-dictation`.
   After first installation activation is an explicit post-login step; the
   installer does not claim the newly copied extension is already enabled.
5. Ensure the user service environment has the login's Xwayland DISPLAY,
   WAYLAND_DISPLAY and XDG_SESSION_TYPE. The rendered Wayland unit explicitly
   sets XDG_SESSION_TYPE=wayland. Restart the service only with approval after
   the extension is enabled. `bash install.sh --check` should now pass.
6. Complete the acceptance matrix below before upgrading other machines.
   Repeat on each target GNOME version; shell-version metadata is insufficient.

To roll back after approval: log into Ubuntu on Xorg, reinstall the previous
X11 checkout's unit and disable the helper extension. The Wayland installer
flag must not be used for rollback. No rollback was executed during development.

## Ubuntu 24.04 Wayland login acceptance

Use synthetic, non-private text. Record only success/failure and counts, never
clipboard contents, real input traces or journal transcription payloads.

1. Confirm `echo "$XDG_SESSION_TYPE"` reports `wayland`. Call GetFocus with
   `gdbus call --session --dest org.gnome.Shell.Extensions.DictationHelper
   --object-path /org/gnome/Shell/Extensions/DictationHelper
   --method org.gnome.Shell.Extensions.DictationHelper.GetFocus` (one line).
   Native Chrome/Firefox, Writer and a terminal must each report distinct ids
   and the expected class. Check the terminal class against TERMINAL_CLASSES.
2. In a blank document, double-tap Right-Ctrl, dictate a harmless test sentence,
   then tap once to stop. Exactly one session starts and one paste lands there.
   Repeat with Left-Shift, Left-Alt and Left-Ctrl held at the second tap; check
   language, rewrite and terminal mode respectively. Repeat with right-hand
   modifiers to confirm they do not change modes.
3. Start in a native app, switch focus to a second blank app while recording,
   then stop. Focus must return and paste only in the first app. Close the
   target before stopping: no paste into the second app; text stays recoverable
   in CopyQ. Disable the extension: no synthetic paste, no typing fallback.
4. In a terminal with a harmless empty prompt, verify Ctrl+Shift+V is selected,
   not Ctrl+V, and no Return is sent. Do not use a privileged shell or commands.
   Verify both plain and rich clipboard ownership gates with test content.
5. Exercise the configured thumb buttons once each. Expect browser navigation
   too; use an empty disposable tab. Disable these triggers if unacceptable.
   Unplug/replug a keyboard and mouse while idle, and with a start modifier
   held: no stuck modifier or spurious double-tap. Try zero devices then replug.
6. For the keymap trace unit test, Ctrl+A corresponds to physical Q on Belgian
   AZERTY. No selection release or badge is expected in this part. Do not treat
   missing native PRIMARY age/owner metadata as tested functionality.
7. Keep the X11 rollback until repeated real native-app pastes and CopyQ
   recovery have passed. Only then authorize the next machine/OS trial.

## Reproducible offline checks

Run from the checkout, with its Python dependencies available. The unittest
suite AST-loads desktop logic and uses fake evdev/D-Bus: it does not open real
input devices, audio, clipboard or displays.

```sh
env -u DISPLAY -u WAYLAND_DISPLAY -u DBUS_SESSION_BUS_ADDRESS \
  XDG_SESSION_TYPE=x11 python -m unittest discover -s tests
node tests/test_wayland_extension.mjs
python tests/probe_wayland_dbus.py
```

The last probe runs actual GJS/Gio on a private D-Bus, with a fake compositor:
this verifies XML marshalling and Python transport, **not** GNOME loading.
The optional `python tests/probe_wayland_shell.py` requires working unprivileged
bubblewrap namespaces. It hides host input/DRM/runtime sockets, uses a private
HOME and D-Bus, and runs headless GNOME with hard deadlines and exact-child-PID
cleanup. It fails closed if isolation is unavailable. It shares the network
namespace only because some hosts disallow creating a network namespace; the
probe does not make network requests. Do not bypass its device isolation.
