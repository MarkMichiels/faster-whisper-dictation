#!/bin/bash
# Detach mouse thumb buttons from browser back/forward on every attached mouse.
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
# No device is named here on purpose: machines have different mice, and the
# same machine can gain one at any time. Every real pointer with at least nine
# buttons is remapped, and the map is built from that device's own button
# count, so a mouse with extra buttons keeps them. A pointer with fewer than
# nine buttons has no thumb buttons to free and is left alone, as are X's
# virtual core and XTEST pointers, which are not physical devices.
#
# Pass a device name or id to limit the run to one device.

set -euo pipefail

export DISPLAY="${DISPLAY:-:1}"

# Index of the first button to divert, and where the diverted pair lands.
FIRST_THUMB=8
DIVERT_BASE=10

button_count() {
    # "Buttons supported: N" appears once per pointer in the long listing.
    xinput list --long "$1" 2>/dev/null |
        sed -n 's/.*Buttons supported: \([0-9]\+\).*/\1/p' | head -1
}

is_real_pointer() {
    # Master pointers are X's own aggregate, and the XTEST pointer is the
    # virtual device that synthesises clicks for tools like xdotool. Remapping
    # XTEST would silently rewrite every synthetic click on the session, so
    # both are excluded and only physical slave pointers are touched.
    xinput list --short "$1" 2>/dev/null | grep -q 'slave  pointer' || return 1
    case "$(xinput list --name-only "$1" 2>/dev/null)" in
        *XTEST*) return 1 ;;
    esac
    return 0
}

remap_device() {
    local id="$1" name count map i
    name="$(xinput list --name-only "$id" 2>/dev/null || echo "id $id")"
    count="$(button_count "$id")"

    if [ -z "$count" ] || [ "$count" -lt $((FIRST_THUMB + 1)) ]; then
        return 0
    fi

    # Identity for buttons below the thumb pair, then divert the rest upward.
    # Building the map from the device's own count keeps any button beyond the
    # thumb pair addressable instead of silently dropping it.
    map=""
    for i in $(seq 1 "$count"); do
        if [ "$i" -lt $FIRST_THUMB ]; then
            map="$map $i"
        else
            map="$map $((DIVERT_BASE + i - FIRST_THUMB))"
        fi
    done

    # shellcheck disable=SC2086 -- map is a deliberate space-separated list.
    if xinput set-button-map "$id" $map 2>/dev/null; then
        echo "Thumb buttons on '$name' (id $id) no longer send browser back/forward."
    else
        echo "Could not remap '$name' (id $id) — button map unchanged." >&2
    fi
}

target="${1:-}"
if [ -n "$target" ]; then
    ids="$(xinput list --id-only "$target" 2>/dev/null || true)"
    if [ -z "$ids" ]; then
        echo "Pointer '$target' not found — button map unchanged." >&2
        exit 0
    fi
else
    ids="$(xinput list --id-only 2>/dev/null || true)"
fi

found=false
for id in $ids; do
    is_real_pointer "$id" || continue
    found=true
    remap_device "$id"
done

if ! $found; then
    echo "No mouse with thumb buttons found — button map unchanged." >&2
fi
