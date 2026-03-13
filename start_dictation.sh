#!/bin/bash
# GPU-Accelerated Dictation for Desktop AI Workstation
# RTX 5070 Ti with large-v3 model

# Prevent duplicate instances
if pgrep -f "dictation.py.*large-v3" > /dev/null 2>&1; then
    echo "Dictation already running (PID $(pgrep -f 'dictation.py.*large-v3')), exiting."
    exit 0
fi

# Navigate to the script's directory to ensure paths are correct.
cd "$(dirname "$0")"

# Activate the virtual environment.
source venv/bin/activate

# Start with GPU acceleration, streaming mode, no hard time limit
python3 dictation.py -m large-v3 -v cuda -c float16 -l nl -t 0
