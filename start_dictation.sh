#!/bin/bash
# GPU-Accelerated Dictation for Desktop AI Workstation
# RTX 5070 Ti with large-v3 model

# Navigate to the script's directory to ensure paths are correct.
cd "$(dirname "$0")"

# Activate the virtual environment.
source venv/bin/activate

# Start with GPU acceleration, streaming mode, no hard time limit
python3 dictation.py -m large-v3 -v cuda -c float16 -l nl -t 0
