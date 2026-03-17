#!/bin/bash
# Manual launcher for faster-whisper-dictation.
#
# Preferred method: systemctl --user start spraakherkenning
# This script is for manual/debugging use only.

set -e

# Prevent duplicate instances
if pgrep -f "python.*dictation\.py" > /dev/null 2>&1; then
    echo "Dictation already running (PID $(pgrep -f 'python.*dictation\.py'))."
    echo "Use: systemctl --user status spraakherkenning"
    exit 0
fi

# If systemd service is available, prefer that
if systemctl --user is-enabled spraakherkenning.service &>/dev/null; then
    echo "Starting via systemd service..."
    systemctl --user start spraakherkenning.service
    sleep 1
    systemctl --user status spraakherkenning.service --no-pager
    exit 0
fi

# Fallback: direct launch (e.g. no systemd, macOS, testing)
export DISPLAY="${DISPLAY:-:1}"
cd "$(dirname "$0")"
source venv/bin/activate
exec python3 dictation.py -m large-v3 -v cuda -c float16 -l nl -t 0
