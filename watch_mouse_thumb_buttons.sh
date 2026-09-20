#!/bin/bash
# Re-apply the dictation thumb-button map whenever an input device appears.
#
# free_mouse_thumb_buttons.sh runs once when the dictation service starts, which
# covers boot and login. X forgets a button map when the device is re-created,
# so unplugging and replugging the mouse -- or a USB reset after suspend --
# silently restored browser back/forward on the thumb buttons, and dictation
# started navigating the focused app again.
#
# This waits on udev's device-add events rather than polling: udevadm monitor
# blocks until the kernel reports a change, so the watcher costs nothing while
# idle. Each event re-runs the remap script, which is idempotent.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REMAP="$SCRIPT_DIR/free_mouse_thumb_buttons.sh"

export DISPLAY="${DISPLAY:-:1}"

# Settle before the first pass: at service start the mouse is normally already
# present, and X needs the device to exist before a map can be set.
"$REMAP" || true

# --subsystem-match=input keeps the stream to input devices; udev still emits a
# few events per physical device, so the remap runs a handful of times per plug.
# That is harmless and far simpler than debouncing.
udevadm monitor --udev --subsystem-match=input 2>/dev/null | while read -r line; do
    case "$line" in
        *"add"*"/input/"*)
            # X creates its device slightly after udev announces it.
            sleep 1
            "$REMAP" || true
            ;;
    esac
done
