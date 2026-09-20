#!/bin/bash
# Detach the mouse thumb buttons from browser back/forward.
#
# The thumb buttons toggle dictation (see MouseToggleListener in dictation.py),
# but X also delivers them to the focused application as buttons 8 and 9, which
# every browser-based app reads as "go back" and "go forward". Pressing one to
# dictate therefore navigated the app away from the current page.
#
# Buttons 10 and 11 carry no default meaning in X, so remapping the physical
# thumb buttons onto them keeps the dictation trigger working while no
# application acts on the press. dictation.py listens for both pairs, so
# running or skipping this script never breaks the trigger.
#
# X forgets a button map when the device is re-plugged, so this runs at login
# and can be re-run by hand afterwards.

set -euo pipefail

DEVICE_NAME="${1:-SINOWEALTH Wired Gaming Mouse}"

export DISPLAY="${DISPLAY:-:1}"

device_id="$(xinput list --id-only "$DEVICE_NAME" 2>/dev/null | head -1 || true)"
if [ -z "$device_id" ]; then
    echo "Mouse '$DEVICE_NAME' not found — button map unchanged." >&2
    exit 0
fi

# Identity mapping for buttons 1-7, then 8->10 and 9->11.
xinput set-button-map "$device_id" 1 2 3 4 5 6 7 10 11
echo "Thumb buttons on '$DEVICE_NAME' (id $device_id) no longer send browser back/forward."
