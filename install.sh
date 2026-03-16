#!/bin/bash
# Install faster-whisper-dictation autostart
# Run this once on each machine after cloning the repo.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DESKTOP_SRC="$SCRIPT_DIR/faster-whisper-dictation.desktop"
DESKTOP_DST="$HOME/.config/autostart/faster-whisper-dictation.desktop"

# Ensure autostart directory exists
mkdir -p "$HOME/.config/autostart"

# Install .desktop file (copy, not symlink — avoids executable bit issues)
cp "$DESKTOP_SRC" "$DESKTOP_DST"
chmod 644 "$DESKTOP_DST"

echo "Installed: $DESKTOP_DST"

# Create venv if missing
if [ ! -d "$SCRIPT_DIR/venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$SCRIPT_DIR/venv"
    source "$SCRIPT_DIR/venv/bin/activate"
    pip install -r "$SCRIPT_DIR/requirements.txt"
    echo "Virtual environment created and dependencies installed."
else
    echo "Virtual environment already exists."
fi

echo "Done. Dictation will autostart on next login."
echo "To start now: $SCRIPT_DIR/start_dictation.sh &"
