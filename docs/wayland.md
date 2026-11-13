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
- Tap timing uses the kernel time of each evdev event, not the moment the
  reader handles it, so a first tap that stops a running session (the reader
  is busy in `stop()`) cannot push the second tap outside `keys.tap_window_s`.
  A second Right-Ctrl press that comes later than the window but within
  1 s is journalled as `double_tap_slow` (interval, first hold, gap, held
  left modifiers; no other key), because a missed start is otherwise silent.
- Optional key trace for diagnosing missed taps, **off by default**:
  `DICTATION_KEY_TRACE=1` journals one `key_trace` line per event of the
  modifiers and Right-Ctrl only (`ctrl_l`, `ctrl_r`, `shift_l`, `shift_r`,
  `alt_l`, `alt_r`, `meta_l`, `meta_r`): value (1 press, 0 release, 2 repeat),
  kernel time, read latency and what the hub did (`passed`, `ignored_repeat`,
  `ignored_already_down`, `ignored_not_down`, `ignored_held_elsewhere`,
  `ignored_desynced`), plus `syn_dropped` and device add/remove. Every other
  key is never journalled, not even as a count. For a user service, enable it
  for one run with `systemctl --user set-environment DICTATION_KEY_TRACE=1`
  and a service restart; remove it with `systemctl --user unset-environment
  DICTATION_KEY_TRACE` and another restart.
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
- Rewrite reply context, Ctrl+A selection release and the context badge work
  through the helper; see *Selection context* below. The xclip clipboard
  fallback offers `UTF8_STRING` on Wayland (a bare `text/plain` written
  through Xwayland does not paste into native GTK apps).

## Extension interface and safety

UUID: `dictation-helper@faster-whisper-dictation`.
Bus/interface: `org.gnome.Shell.Extensions.DictationHelper`.
Object: `/org/gnome/Shell/Extensions/DictationHelper`.

| Method | Signature | Result |
| --- | --- | --- |
| GetFocus | `() -> (s, s)` | Meta.Window id as string, wm_class; empty strings if no focus |
| Activate | `(s) -> (b)` | Request activation of an existing numeric id |
| SendShortcut | `(s) -> ()` | Only `paste`, `paste_terminal`, `ctrl_home`; everything else errors |
| ShowBadge | `(u) -> ()` | Show "Context opgepakt: N tekens" for 1.5 s, top right of the focused window's monitor; only the number is sent |
| SelectionInfo | `(s) -> (s, d, s, s, b)` | For `primary` or `clipboard` only: state (`set`, `cleared`, `unknown`), seconds since the owner last changed (-1 if unknown), window id focused at that change, window id focused now, and whether both belong to the same process |

No selection text, MIME types or window titles cross this interface. Every process on the
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

## Selection context

The helper listens to `Meta.Selection` `owner-changed` for PRIMARY and
CLIPBOARD and records, per selection, a monotonic time, the focused window id
and its process id. It never reads or stores the selection content.
`context_capture` still reads owner, TARGETS and text from Xwayland (mutter
mirrors both selections there); age and origin come from `SelectionInfo`.
The reasons are logged in the `rewrite_context` journal line with
`backend: wayland`:

| Reason | Meaning |
| --- | --- |
| `shell_unavailable` | Helper missing, disabled or an old version without `SelectionInfo` |
| `no_change_seen` | No owner change since the helper started (fail closed) |
| `empty` | No owner, or the selection was cleared |
| `copyq_owner` | CopyQ owns the selection (also for PRIMARY) |
| `too_old` | Same limits as X11: PRIMARY 20 s, CLIPBOARD 120 s (configurable) |
| `focus_elsewhere` | PRIMARY changed while another program had focus |

The PRIMARY focus rule is stricter than on X11: the selection must have been
made while the program that has focus at the start had focus (same process).
Selecting in one program and starting in another therefore gives no context.
Workaround: copy with Ctrl+C; CLIPBOARD has no focus rule, only its age limit.

Selection release after Ctrl+A follows the X11 rules (`docs/rewrite_mode.md`).
The Ctrl+A trace comes from evdev with the `be`/`us` keymap; every button code
(touchpad taps included) counts as a mouse press. The focused-window id at
Ctrl+A, at the start and while waiting comes from `GetFocus`; "same X client"
becomes "PRIMARY changed in the focused window". Ctrl+Home goes through the
`ctrl_home` shortcut of the helper. Terminals are recognised by their Wayland
app id (for example `org.gnome.terminal`, `org.gnome.ptyxis`, `kitty`).

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
4. Log out/in to load a newly installed **or updated** extension: GNOME on
   Wayland cannot reload extension code in a running session, so an updated
   helper (for example the one adding `SelectionInfo`) is active only after
   the next login. Until then capture reports `shell_unavailable` and no
   selection is released. Choose Ubuntu (Wayland),
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

To turn off only the selection features, set `rewrite.context.release_selection:
false` and/or `rewrite.context.badge: false` in the settings file and restart
the service. To return to the previous code, check out the previous commit,
run `bash install.sh --wayland`, log out/in and restart the service.

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
6. Reply context, with synthetic text: in a mail or document, press Ctrl+A and
   start with Left-Alt + double-tap Right-Ctrl. Expect the high pip, the badge
   and, after releasing Alt, the caret at the top; the rewrite is pasted above
   the original. A mouse selection gives context but is left as is
   (`no_select_all`). A selection older than 20 s gives `too_old`. A selection
   made in another program gives `focus_elsewhere`. In a terminal the context
   is taken but nothing is sent (`terminal`). Check only the journal metadata.
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

`python tests/probe_wayland_nested.py --out DIR` runs the real GNOME Shell
(`--nested --wayland`) inside its own Xvfb, with a private session bus, a
private stand-in system bus (no logind or display manager), a stub for the
systemd user manager (the shell starts an X11 services target before
Xwayland is usable), a private HOME with only the helper enabled, and CopyQ
shadowed by a failing stub. Two synthetic GTK4 editors and kitty run inside.
The real fork code (App.start, context capture, selection release, badge,
paste) runs against it; keys reach the apps via xdotool on that Xvfb and the
same keys go as synthetic evdev events into the input hub, which opens no
device. The shell runs under strace; the result lists the `/dev` paths it
opened. Mutter may take display `:0` when free; the probe accepts the nested
display only when its lock file belongs to a child of the nested shell.
Run it with `DISPLAY`, `WAYLAND_DISPLAY`, `DBUS_SESSION_BUS_ADDRESS` and
`XAUTHORITY` unset.
The optional `python tests/probe_wayland_shell.py` requires working unprivileged
bubblewrap namespaces. It hides host input/DRM/runtime sockets, uses a private
HOME and D-Bus, and runs headless GNOME with hard deadlines and exact-child-PID
cleanup. It fails closed if isolation is unavailable. It shares the network
namespace only because some hosts disallow creating a network namespace; the
probe does not make network requests. Do not bypass its device isolation.
