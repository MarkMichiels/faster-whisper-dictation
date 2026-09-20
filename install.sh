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
REVISION="$(git -C "$SCRIPT_DIR" rev-parse --short=12 HEAD 2>/dev/null || printf 'unknown')"

detect_ld_path() {
    local cudnn_lib="$VENV_DIR/lib/python3.12/site-packages/nvidia/cudnn/lib"
    local cublas_lib="$VENV_DIR/lib/python3.12/site-packages/nvidia/cublas/lib"

    LD_PATH=""
    [ -d "$cudnn_lib" ] && LD_PATH="$cudnn_lib"
    [ -d "$cublas_lib" ] && LD_PATH="${LD_PATH:+$LD_PATH:}$cublas_lib"
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

# DISPLAY needed for pynput global hotkey listener on X11
Environment=DISPLAY=:1

# Records which checked-out revision generated the installed unit. This lets
# setup_workspace.sh detect a pull whose install step was interrupted.
Environment=DICTATION_REVISION=$REVISION

# Flush stdout per line so every transcription is immediately recoverable
# from the journal (Python block-buffers stdout under systemd otherwise).
Environment=PYTHONUNBUFFERED=1

# Kill any orphan instances before starting
ExecStartPre=/bin/bash -c 'pkill -f "python.*dictation\\\\.py" 2>/dev/null; sleep 0.5; true'


# Detach the thumb buttons from browser back/forward before the listener starts,
# so a press that toggles dictation no longer navigates the focused app.
ExecStartPre=-$SCRIPT_DIR/free_mouse_thumb_buttons.sh

ExecStart=$VENV_DIR/bin/python3 dictation.py -m large-v3 -v cuda -c float16 -l nl -t 0

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

detect_ld_path

if $CHECK_ONLY; then
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

# The venv may have been created above, so detect CUDA paths again.
detect_ld_path

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
echo ""
echo "Useful commands:"
echo "  systemctl --user status $SERVICE_NAME   # Check status"
echo "  systemctl --user restart $SERVICE_NAME   # Restart after config change"
echo "  systemctl --user stop $SERVICE_NAME      # Stop temporarily"
echo "  journalctl --user -u $SERVICE_NAME -f    # Follow logs"
