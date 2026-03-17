#!/bin/bash
# Install faster-whisper-dictation as a systemd user service.
# Run this once on each machine after cloning the repo.
#
# This is the ONLY autostart mechanism. The previous .desktop autostart
# approach caused duplicate instances because both GNOME session and
# systemd-xdg-autostart-generator would each launch a copy.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SERVICE_NAME="spraakherkenning"
SERVICE_DIR="$HOME/.config/systemd/user"
SERVICE_FILE="$SERVICE_DIR/$SERVICE_NAME.service"
VENV_DIR="$SCRIPT_DIR/venv"

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

# --- Detect CUDA library paths ---
CUDNN_LIB="$VENV_DIR/lib/python3.12/site-packages/nvidia/cudnn/lib"
CUBLAS_LIB="$VENV_DIR/lib/python3.12/site-packages/nvidia/cublas/lib"
LD_PATH=""
[ -d "$CUDNN_LIB" ] && LD_PATH="$CUDNN_LIB"
[ -d "$CUBLAS_LIB" ] && LD_PATH="${LD_PATH:+$LD_PATH:}$CUBLAS_LIB"

# --- Install systemd user service ---
mkdir -p "$SERVICE_DIR"

cat > "$SERVICE_FILE" << UNIT
[Unit]
Description=GPU-Accelerated Dutch Speech Recognition
# Wait for graphical session (DISPLAY must be available for pynput)
After=graphical-session.target
Requires=graphical-session.target

ConditionPathExists=$SCRIPT_DIR/dictation.py

[Service]
Type=simple
WorkingDirectory=$SCRIPT_DIR

# CUDA libraries for faster-whisper GPU inference
Environment=LD_LIBRARY_PATH=$LD_PATH

# DISPLAY needed for pynput global hotkey listener (X11)
Environment=DISPLAY=:1

# Kill any orphan instances before starting
ExecStartPre=/bin/bash -c 'pkill -f "python.*dictation\\\\.py" 2>/dev/null; sleep 0.5; true'

ExecStart=$VENV_DIR/bin/python3 dictation.py -m large-v3 -v cuda -c float16 -l nl -t 0

Restart=on-failure
RestartSec=5
Nice=10
OOMScoreAdjust=200

[Install]
WantedBy=default.target
UNIT

echo "Installed: $SERVICE_FILE"

# --- Enable and start ---
systemctl --user daemon-reload
systemctl --user enable "$SERVICE_NAME.service"
echo "Enabled: $SERVICE_NAME.service"

# Start if not already running
if systemctl --user is-active --quiet "$SERVICE_NAME.service"; then
    echo "Service already running. Use 'systemctl --user restart $SERVICE_NAME' to apply changes."
else
    systemctl --user start "$SERVICE_NAME.service"
    echo "Started: $SERVICE_NAME.service"
fi

echo ""
echo "Done. Dictation will autostart on login via systemd."
echo ""
echo "Useful commands:"
echo "  systemctl --user status $SERVICE_NAME   # Check status"
echo "  systemctl --user restart $SERVICE_NAME   # Restart after config change"
echo "  systemctl --user stop $SERVICE_NAME      # Stop temporarily"
echo "  journalctl --user -u $SERVICE_NAME -f    # Follow logs"
