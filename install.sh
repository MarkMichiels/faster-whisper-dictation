#!/bin/bash
# Install faster-whisper-dictation as a systemd user service.
# Run this once on each machine after cloning the repo.
#
# This is the ONLY autostart mechanism. The previous .desktop autostart
# approach caused duplicate instances because both GNOME session and
# systemd-xdg-autostart-generator would each launch a copy.

set -euo pipefail

CHECK_ONLY=false
case "${1:-}" in
    "") ;;
    --check) CHECK_ONLY=true ;;
    *)
        echo "Usage: $0 [--check]" >&2
        exit 2
        ;;
esac

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SERVICE_NAME="spraakherkenning"
SERVICE_DIR="$HOME/.config/systemd/user"
SERVICE_FILE="$SERVICE_DIR/$SERVICE_NAME.service"
VENV_DIR="$SCRIPT_DIR/venv"
GRAPHICAL_WANTS_DIR="$SERVICE_DIR/graphical-session.target.wants"
GRAPHICAL_LINK="$GRAPHICAL_WANTS_DIR/$SERVICE_NAME.service"
LEGACY_DEFAULT_LINK="$SERVICE_DIR/default.target.wants/$SERVICE_NAME.service"
# Settings live in one YAML file (docs/configuration.md). The installer creates
# it once with its own choices below; an existing file is never touched.
CONFIG_DIR="$HOME/.config/faster-whisper-dictation"
CONFIG_FILE="$CONFIG_DIR/config.yaml"
REVISION="$(git -C "$SCRIPT_DIR" rev-parse --short=12 HEAD 2>/dev/null || printf 'unknown')"

detect_ld_path() {
    local site_packages library lib_dir
    LD_PATH=""
    # A missing venv is normal for --check on a fresh clone. Ask the actual
    # service interpreter, not a hardcoded Python minor version or stale glob.
    if [ ! -x "$VENV_DIR/bin/python3" ]; then
        return 0
    fi
    site_packages="$("$VENV_DIR/bin/python3" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
    for library in cudnn cublas; do
        lib_dir="$site_packages/nvidia/$library/lib"
        if [ -d "$lib_dir" ]; then
            LD_PATH="${LD_PATH:+$LD_PATH:}$lib_dir"
        fi
    done
    # Missing optional wheel directories must not trip set -e.
    return 0
}

render_seed_config() {
    cat << 'YAML'
# faster-whisper-dictation settings, created once by install.sh and never
# overwritten by it. These are the installer's choices; every other setting
# keeps its default. All keys with their defaults: config.example.yaml in the
# repository; reference: docs/configuration.md. Never put a secret here.
# Apply a change with: systemctl --user restart spraakherkenning
model:
  name: large-v3
  device: cuda
  compute_type: float16
language:
  default: nl
recording:
  max_time_s: 0
YAML
}

seed_config() {
    # -L as well: a symlink, even a dangling one, is the user's choice.
    if [ -e "$CONFIG_FILE" ] || [ -L "$CONFIG_FILE" ]; then
        echo "Settings file exists, left unchanged: $CONFIG_FILE"
        return 0
    fi
    mkdir -p "$CONFIG_DIR"
    local seed
    seed="$(mktemp "$CONFIG_DIR/.config.yaml.XXXXXX")"
    render_seed_config > "$seed"
    chmod 0644 "$seed"
    # No-clobber move: a file that appeared meanwhile still wins.
    if ! mv -n "$seed" "$CONFIG_FILE" || [ -e "$seed" ]; then
        rm -f "$seed"
        echo "Settings file exists, left unchanged: $CONFIG_FILE"
        return 0
    fi
    echo "Created settings file: $CONFIG_FILE"
}

render_service() {
    cat << UNIT
[Unit]
Description=GPU-Accelerated Dutch Speech Recognition
# Start with the real graphical login. PartOf stops dictation at logout without
# pulling graphical-session.target into the early default.target transaction.
After=graphical-session.target
PartOf=graphical-session.target

ConditionPathExists=$SCRIPT_DIR/dictation.py

[Service]
Type=simple
WorkingDirectory=$SCRIPT_DIR

# CUDA libraries for faster-whisper GPU inference
Environment=LD_LIBRARY_PATH=$LD_PATH

# DISPLAY and XAUTHORITY come from the login session: on Xorg,
# /etc/X11/Xsession.d/20dbus_xdg-runtime pushes both into the systemd user
# manager before graphical-session.target starts. A hardcoded DISPLAY=:1 only
# matched a password login after an Xorg greeter; with automatic login the
# session gets :0 and the listener would attach to the wrong server or none.

# Records which checked-out revision generated the installed unit. This lets
# setup_workspace.sh detect a pull whose install step was interrupted.
Environment=DICTATION_REVISION=$REVISION

# Flush stdout per line so every transcription is immediately recoverable
# from the journal (Python block-buffers stdout under systemd otherwise).
Environment=PYTHONUNBUFFERED=1

# Settings: model, language and every other option come from this YAML file
# (created once by install.sh). Command-line options would override it.
Environment=DICTATION_CONFIG=%h/.config/faster-whisper-dictation/config.yaml

# Older route for the rewrite mode, still supported: DICTATION_REWRITE_*
# variables. They win over config.yaml. The leading '-' keeps the service
# starting when the file does not exist.
EnvironmentFile=-%h/.config/faster-whisper-dictation/rewrite.env

# No pattern-based pre-kill: systemd owns the lifecycle and stops every
# process in this unit's cgroup. A pkill -f pattern also matched unrelated
# shells and agent sessions whose command line merely mentioned dictation.py.


# Detach the thumb buttons from browser back/forward before the listener starts,
# so a press that toggles dictation no longer navigates the focused app.
ExecStartPre=-$SCRIPT_DIR/free_mouse_thumb_buttons.sh

# X forgets a button map when the device is re-created, so a replugged mouse
# would silently get browser back/forward again. The watcher re-applies the map
# on every udev input-add event and exits with the service.
ExecStartPost=-/bin/bash -c '$SCRIPT_DIR/watch_mouse_thumb_buttons.sh &'

ExecStart=$VENV_DIR/bin/python3 dictation.py

Restart=on-failure
RestartSec=5
Nice=10
OOMScoreAdjust=200

[Install]
WantedBy=graphical-session.target
UNIT
}

unit_is_current() {
    local expected_unit
    expected_unit="$(mktemp)"
    render_service > "$expected_unit"

    if [ ! -x "$VENV_DIR/bin/python3" ]; then
        echo "NOT CURRENT: virtual environment is missing" >&2
        rm -f "$expected_unit"
        return 1
    fi
    if [ ! -f "$SERVICE_FILE" ] || ! cmp -s "$expected_unit" "$SERVICE_FILE"; then
        echo "NOT CURRENT: installed unit differs from revision $REVISION" >&2
        rm -f "$expected_unit"
        return 1
    fi
    rm -f "$expected_unit"

    # Without its settings file the flagless unit would run the code defaults
    # (base model on CPU, language auto-detect): rerun install.sh to create it.
    if [ ! -e "$CONFIG_FILE" ]; then
        echo "NOT CURRENT: settings file is missing: $CONFIG_FILE" >&2
        return 1
    fi

    if [ ! -L "$GRAPHICAL_LINK" ] || [ "$(readlink -f "$GRAPHICAL_LINK")" != "$SERVICE_FILE" ]; then
        echo "NOT CURRENT: service is not enabled for graphical-session.target" >&2
        return 1
    fi
    if [ -e "$LEGACY_DEFAULT_LINK" ] || [ -L "$LEGACY_DEFAULT_LINK" ]; then
        echo "NOT CURRENT: legacy default.target link still exists" >&2
        return 1
    fi
    if ! systemctl --user is-active --quiet "$SERVICE_NAME.service"; then
        echo "NOT CURRENT: service is not active" >&2
        return 1
    fi

    return 0
}

if $CHECK_ONLY; then
    detect_ld_path
    if unit_is_current; then
        echo "CURRENT: revision $REVISION is installed, enabled for the graphical session, and active"
        exit 0
    fi
    exit 1
fi

# --- Cleanup legacy autostart ---
LEGACY_DESKTOP="$HOME/.config/autostart/faster-whisper-dictation.desktop"
if [ -f "$LEGACY_DESKTOP" ]; then
    rm -f "$LEGACY_DESKTOP"
    echo "Removed legacy GNOME autostart: $LEGACY_DESKTOP"
fi

# --- Create venv if missing ---
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    pip install -r "$SCRIPT_DIR/requirements.txt"
    echo "Virtual environment created and dependencies installed."
else
    echo "Virtual environment already exists."
fi

# Discover libraries only after a new venv and its dependencies exist.
detect_ld_path

# --- Settings file (once; never overwritten) ---
# Before the unit: the flagless service must find its settings at first start.
seed_config

# --- Install systemd user service ---
mkdir -p "$SERVICE_DIR"
rendered_unit="$(mktemp)"
trap 'rm -f "$rendered_unit"' EXIT
render_service > "$rendered_unit"
install -m 0644 "$rendered_unit" "$SERVICE_FILE"

echo "Installed: $SERVICE_FILE"

# --- Enable and start ---
# Versions before 2026-09-20 enabled the service for default.target and used
# Requires=graphical-session.target. On Ubuntu this could start the graphical
# target too early; GNOME then stopped it during session setup and dictation was
# left inactive. Remove that exact legacy link before enabling the new target.
if [ -e "$LEGACY_DEFAULT_LINK" ] || [ -L "$LEGACY_DEFAULT_LINK" ]; then
    rm -f "$LEGACY_DEFAULT_LINK"
    echo "Removed legacy default.target link: $LEGACY_DEFAULT_LINK"
fi
systemctl --user daemon-reload
systemctl --user enable "$SERVICE_NAME.service"
echo "Enabled: $SERVICE_NAME.service (graphical-session.target)"

# Start, or restart if already running, so re-running install.sh always applies
# the latest unit definition and pulled code (idempotent on every machine).
if systemctl --user is-active --quiet "$SERVICE_NAME.service"; then
    systemctl --user restart "$SERVICE_NAME.service"
    echo "Restarted: $SERVICE_NAME.service (applied unit + code changes)"
else
    systemctl --user start "$SERVICE_NAME.service"
    echo "Started: $SERVICE_NAME.service"
fi

echo ""
echo "Done. Dictation will autostart on login via systemd."
echo "Installed revision: $REVISION"
echo "Settings: $CONFIG_FILE (reference: docs/configuration.md)"
echo ""
echo "Useful commands:"
echo "  systemctl --user status $SERVICE_NAME   # Check status"
echo "  systemctl --user restart $SERVICE_NAME   # Restart after config change"
echo "  systemctl --user stop $SERVICE_NAME      # Stop temporarily"
echo "  journalctl --user -u $SERVICE_NAME -f    # Follow logs"
