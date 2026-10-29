[Home](../README.md) / Rebuild and operate

# Rebuild and operate

## Supported evidence, not a universal installer

The reference setup is an existing Ubuntu Xorg desktop using NVIDIA CUDA,
`large-v3`, `float16`, a working microphone and clipboard tools. A fresh OS,
a second machine, every keyboard/mouse, other Linux desktops, Windows and
macOS have not been validated as an equivalent end-to-end setup. Offline unit
tests do not establish hardware compatibility or recognition speed.

Python **3.12** is the reference environment. The installer asks the venv's
`bin/python3` for its `sysconfig` package directory, so CUDA library discovery
also follows other Python minor versions. That does not establish dependency
compatibility with every Python version.

You need:

| Component | Purpose / condition |
|---|---|
| Ubuntu graphical X11 session and systemd user manager | Global input, window targeting and login lifecycle |
| Python, venv and pip | A dedicated environment in the source directory |
| PortAudio development/runtime libraries | PyAudio input and sounddevice output; microphone must open at 16 kHz mono |
| faster-whisper and CTranslate2 | Local Whisper inference; no transcription API |
| ONNX Runtime / bundled Silero VAD interface | Speech/pause detection in streaming mode |
| NVIDIA driver, compatible cuBLAS and cuDNN, sufficient free VRAM | Required for the installer's CUDA profile; not installed by the script |
| `xdotool`, `xprop` | Active-window ID, focus restoration and terminal classification |
| CopyQ (recommended), `xclip` | Clipboard history plus clipboard; fallback without history |
| `xinput`, `udevadm` | Mouse button remapping and device-add watcher |
| `assets/` beside the Python file | Start/stop WAV sounds; launch from the source directory |

`requirements.txt` pins `av==15.1.0` and `faster-whisper==1.2.0`, but leaves other
packages unpinned or in ranges. It is **not a lockfile**. It still specifies
`tokenizers>=0.13,<0.15`; availability/build compatibility of these older versions
on a fresh Python installation is not demonstrated here. A dependency-resolution
or build failure is a rebuild blocker to diagnose, not a reason to install with
`--no-deps`.
There is no complete, validated fresh-machine CUDA/dependency recipe in this
snapshot. Preserve an existing working environment until its replacement works.

PyTorch is **not** listed or imported by this app; installing it is not the
repository's CUDA setup procedure. `nvidia-smi` working proves the driver is
visible, not that CTranslate2 can load its required cuDNN/cuBLAS libraries.
Match those libraries to the **actually resolved CTranslate2 version** and the
GPU/driver on the target; no single CUDA minor version or VRAM minimum is
certified here. Obtain the working environment's package/version inventory if
exact reproduction is required.

## 1. Obtain and inspect the source

Use an agreed source snapshot of this fork, or a checkout pinned to the agreed
revision. Keep `dictation.py`, `coverage_ledger.py`, `requirements.txt`, scripts,
`assets/`, `tests/`, documentation and `LICENSE` together. Do not substitute a
clone of upstream: it does not contain this setup.

For a history-free handover, a maintainer can export committed source with:

```sh
git archive --format=tar --output=dictation-source.tar HEAD
```

This exports tracked files, not `.git`, an untracked venv or model cache. It is
**not a privacy scrub of tracked contents**; review the exported tree separately.
The app and installer work without Git metadata, but `DICTATION_REVISION` then
becomes `unknown`. Keep the agreed source revision in the handover manifest.

## 2. Check the target before starting anything

From the logged-in desktop, inspect:

```sh
python3 --version
printf 'session=%s DISPLAY=%s\n' "${XDG_SESSION_TYPE:-unset}" "${DISPLAY:-unset}"
systemctl --user show-environment | grep -E '^(DISPLAY|XAUTHORITY)='
command -v xdotool xprop xclip copyq xinput udevadm
nvidia-smi
systemctl --user status spraakherkenning.service
ps -C python3 -o pid,args
```

An absent service is expected on a new installation. Stop at an unexpected
existing instance, missing tools/libraries, Wayland, or missing X-session
credentials. The unit inherits `DISPLAY`/`XAUTHORITY` from the user manager; it
does not select a display itself. Choose an Xorg login session rather than
assuming that Xwayland provides global input for native Wayland applications.
Do not start another copy alongside a service.

If installing on Ubuntu with appropriate administrator permission, the usual
system-package starting point is:

```sh
sudo apt install python3-venv python3-dev build-essential portaudio19-dev \
    xdotool x11-utils xclip copyq xinput udev
```

This supplies desktop/build tools, **not** the NVIDIA/CUDA stack. Start CopyQ in
the same graphical user session and arrange its own autostart if desired; the
dictation installer does not configure CopyQ history or autostart for you.

## 3. Build a dedicated environment

From the source directory, with Python 3.12 selected as `python3`:

```sh
python3 -m venv venv
venv/bin/python3 -m pip install -r requirements.txt
venv/bin/python3 -m pip check
venv/bin/python3 -m pip freeze > environment-packages.txt
```

Retain that inventory locally for reproduction; it is not a new dependency
lock. On failure, keep the resolver's error and reconcile the declared package
constraints before proceeding. Never replace a working venv in place without a
return path. If one already exists, `install.sh` skips dependency installation;
rerunning the installer does **not** upgrade its packages.

Resolve the CUDA runtime prerequisites before GPU startup. The installer only
adds existing `nvidia/cudnn/lib` and `nvidia/cublas/lib` directories under the
venv interpreter's `sysconfig.get_path("purelib")` to the unit's
`LD_LIBRARY_PATH`, in that order. Detection runs after venv/dependency creation.
Missing directories are allowed, including both absent; they do not abort the
installer. The script neither installs NVIDIA runtime wheels explicitly nor
discovers arbitrary CUDA installations. Ensure compatible libraries are present
in those venv locations or available through the system loader before startup.
An empty generated path is not proof that GPU inference will work.

You can also let `./install.sh` create a missing venv and install requirements,
but that route immediately proceeds to service installation and startup. Use the
manual steps above when you need to validate dependencies and CUDA first.
Offline regression tests cover bootstrap and generated paths using fake venv,
pip and systemctl commands; a complete clean-machine GPU setup remains unproven.

Model names are passed to `WhisperModel` without an explicit cache directory
or model revision pin. A name such as `large-v3` can trigger a Hugging Face
download; a converted local model directory is also accepted by `-m`. Package
installation and initial model acquisition need network access unless their
bytes are already available. No model download is part of the offline unit test.

## 4. Choose one launch method

### User service (reference GPU profile)

Read `install.sh` before running it. Back up an existing unit, enablement links,
legacy autostart file and mouse button maps if they matter to your return path.
Then, when ready to change the desktop setup:

```sh
./install.sh
./install.sh --check
systemctl --user status spraakherkenning.service
```

The installer:

1. Removes the exact legacy `faster-whisper-dictation.desktop` autostart file.
2. Builds a missing `venv` and installs requirements; skips an existing venv.
3. Overwrites `~/.config/systemd/user/spraakherkenning.service`.
4. Removes the legacy `default.target` enablement link and enables the unit for
   `graphical-session.target`; `PartOf` links stopping to graphical logout.
5. Creates `~/.config/faster-whisper-dictation/config.yaml` **once** with the
   installer's profile (`large-v3`, `cuda`, `float16`, language `nl`, no short
   recording limit); an existing file is never overwritten.
6. Reloads systemd and **starts or restarts** the service. The generated command
   is `venv/bin/python3 dictation.py` without options: the settings come from
   that file (`DICTATION_CONFIG` in the unit). See [Configuration](configuration.md).
7. Runs mouse remapping before the process, and starts its device-add watcher.

The unit uses `Restart=on-failure` with a five-second delay, `Nice=10` and
`OOMScoreAdjust=200`. There is no pattern-based orphan-process kill or global
singleton lock. Keep only one launcher enabled; systemd manages its own cgroup,
not unrelated manually started processes.

`--check` compares the generated unit (including source revision and detected
library paths), venv executable, the presence of `config.yaml`, enablement
links and active status. It creates
and removes a temporary comparison file, but does not install or restart
anything. `CURRENT` is **not** a microphone, GPU decode, CopyQ, mouse-map or
application-paste acceptance test.

### Manual diagnostic / CPU launch

Use this only instead of the service. After stopping an existing service and
confirming there is no separate instance, run from the source directory:

```sh
# Same settings as the service (reads ~/.config/faster-whisper-dictation/config.yaml)
venv/bin/python3 dictation.py

# Alternative: CPU profile (not the installer's profile, speed not equivalent)
venv/bin/python3 dictation.py -m small -v cpu -c int8 -l en
```

Run **one**, not both. The Python application needs the same working runtime
library paths as the service if your CUDA libraries are in the venv. The Python
command does not run the mouse remap/watcher automatically. `start_dictation.sh`
prefers an enabled service, otherwise starts the app with the settings file and
a fallback `DISPLAY=:1`; it is not a portable CPU launcher. `run.sh` is not a
second autostart mechanism to enable beside systemd.

## Configure the application

Every option below can also be set in the settings file
`~/.config/faster-whisper-dictation/config.yaml`; an option given on the command
line wins. The file, its keys, precedence and journal lines are described in
[Configuration](configuration.md). Defaults below are those of the code, used
when neither the file nor the command line sets a value. Quote hotkey strings so
the shell does not interpret angle brackets.

| Option | Default | Meaning |
|---|---|---|
| `-m`, `--model-name` | `base` | Model name/ID or converted local directory |
| `-v`, `--device` | `cpu` | Inference device; service uses `cuda` |
| `-c`, `--compute-type` | `int8` | Service uses `float16` |
| `-l`, `--language` | auto-detect | Forced base language; service uses `nl` |
| `--shift-language` | `en` | Left-Shift alternate for double-tap starts |
| `-d`, `--double-key` | Linux Right Ctrl | e.g. `-d '<ctrl_r>'`; some CLI help still says Right Super |
| `-k`, `--key-combo` | unset on Linux | e.g. `-k '<ctrl>+<alt>+a'` toggles capture; `-d` takes precedence if also supplied |
| `--mouse-buttons` | `button8,button9,button10,button11` | Comma-separated pynput names; `--mouse-buttons ''` disables listener |
| `-t`, `--max-time` | `30` seconds | `0` removes this short timer, **not** the streaming 30-minute cap |
| `--auto-stop-silence` | `10` seconds | Streaming silence timeout; `0` disables it |
| `--silence-ms` | `1000` | Full-pause chunk boundary |
| `--min-chunk-s` | `2.5` | Eager cut floor; `0` disables eager cuts |
| `--max-chunk-s` | `10` | Eager split search ceiling; not a total paste-latency guarantee |
| `--eager-prob` | `0.35` | Quiet-enough probability for an early cut |
| `--paste-min-s` | `12` | Accumulate eager chunks before decoding; full pause/stop flushes earlier; `0` decodes per chunk |
| `--context-chars` | `500` | Rolling prompt tail; `0` disables |
| `--transcribe-vad` | off | Additional VAD inside decoding; can change recognition boundaries |
| `--no-transcribe-vad` | no-op | Compatibility option only |
| `--no-hallucination-filter` | filter enabled | Disables ordinary phrase filtering, not all energy gates or recovery filters |
| `--no-space-rule-runs` | spacing enabled | Disables spacing of repeated Markdown-like symbols |
| `--ellipsis-style` | `space` | `space`, `single` or `keep` |
| `--batch-mode` | off | Legacy record-then-decode path, without the streaming ledger/stop check |

For service changes, prefer `config.yaml` and a restart. Unit changes go in the
tracked template in `install.sh` (regenerate the unit), or in an explicit systemd
override. A rerun overwrites
direct edits to the generated unit. Overrides are outside `--check`'s comparison
of the generated file; inspect `systemctl --user cat spraakherkenning.service`
for the effective configuration. Nothing translates the Dutch recovery labels;
the stop status line is `output.stop_status` in the settings file.

## Operate and verify

```sh
systemctl --user start spraakherkenning.service
systemctl --user stop spraakherkenning.service
systemctl --user restart spraakherkenning.service
systemctl --user status spraakherkenning.service
journalctl --user -u spraakherkenning.service -f
```

A temporary stop does not disable next-login autostart. To disable that:
`systemctl --user disable --now spraakherkenning.service`.
Stopping the service does not restore X mouse maps; save each device's original
`xinput get-button-map DEVICE_ID` before remapping and restore that exact map
with `xinput set-button-map` only after the watcher is stopped and the device
identity checked. Disabling the listener alone also does not restore maps.

Use the [daily controls](../README.md#daily-controls-linux-default) in a disposable
local editor field. For a new target's acceptance check:

1. Verify a single instance and the selected model/device/language in startup logs.
2. Dictate a non-sensitive phrase with Right Ctrl; stop and wait for `Session ended.`.
   Confirm the actual editor text, not just a success return code.
3. Repeat with Left Shift held at double-tap, then start unshifted again to verify
   the default language returns.
4. Test both thumb buttons: no browser navigation, no immediate double toggle.
   Replug the mouse and verify the watcher remaps it again.
5. Verify only the new test text in CopyQ. Do not export or overwrite old history.
6. Inspect `stop_summary`, capture errors and delivery receipts. Confirm autostart
   on a later logout/login separately if required.

The start sound follows recorder-thread startup, not a hardware-ready handshake.
The stop sound is **not** completion of transcription or recovery. Let finishing
work drain before stopping/restarting the service; service shutdown or Ctrl+C
is not a guaranteed graceful recovery flush. Preserve existing code/environment
and unit backups until this acceptance check succeeds.

Files outside the repository that a rebuild needs and the repository cannot
restore: `~/.config/faster-whisper-dictation/config.yaml` (your settings; the
installer only recreates its own profile), an older `rewrite.env` if you still use
one, and the rewrite key file with any style or term files its paths point to.
Back them up yourself; keep the key file private (`chmod 600`).
