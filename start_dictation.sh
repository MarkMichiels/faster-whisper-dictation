#!/bin/bash
# Wrapper script to start faster-whisper-dictation

# Navigate to the script's directory to ensure paths are correct.
# This makes the script runnable from anywhere.
cd "$(dirname "$0")"

# Activate the virtual environment.
source venv/bin/activate

# Start the dictation script with the 'small' model and Dutch language.
# It will run in the background.
python3 dictation.py -m small -l nl
