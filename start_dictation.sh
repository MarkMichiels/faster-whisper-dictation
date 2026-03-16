#!/bin/bash
# GPU-Accelerated Dictation for Desktop AI Workstation
# Uses large-v3 model with CUDA acceleration

# Prevent duplicate instances (match any dictation.py process)
if pgrep -f "python.*dictation\.py" > /dev/null 2>&1; then
    echo "Dictation already running (PID $(pgrep -f 'python.*dictation\.py')), exiting."
    exit 0
fi

# Ensure DISPLAY is set (needed for pynput keyboard listener).
# When launched via .desktop autostart, GNOME sets this automatically.
# When launched via SSH/Sero, it must be provided.
export DISPLAY="${DISPLAY:-:1}"

# Navigate to the script's directory to ensure paths are correct.
cd "$(dirname "$0")"

# Activate the virtual environment.
source venv/bin/activate

# Start with GPU acceleration, streaming mode, no hard time limit
exec python3 dictation.py -m large-v3 -v cuda -c float16 -l nl -t 0
