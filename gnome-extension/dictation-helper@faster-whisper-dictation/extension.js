import Clutter from 'gi://Clutter';
import Gio from 'gi://Gio';
import GLib from 'gi://GLib';
import Meta from 'gi://Meta';
import St from 'gi://St';
import {Extension} from 'resource:///org/gnome/shell/extensions/extension.js';
import * as Main from 'resource:///org/gnome/shell/ui/main.js';

const BUS = 'org.gnome.Shell.Extensions.DictationHelper';
const PATH = '/org/gnome/Shell/Extensions/DictationHelper';
const XML = `<node><interface name="${BUS}">
<method name="GetFocus"><arg type="s" direction="out"/><arg type="s" direction="out"/></method>
<method name="Activate"><arg type="s" direction="in"/><arg type="b" direction="out"/></method>
<method name="SendShortcut"><arg type="s" direction="in"/></method>
<method name="ShowBadge"><arg type="u" direction="in"/></method>
<method name="SetStatus"><arg type="s" direction="in"/><arg type="s" direction="in"/><arg type="s" direction="in"/><arg type="u" direction="in"/></method>
<method name="ClearStatus"/>
<method name="SelectionInfo"><arg type="s" direction="in"/><arg type="s" direction="out"/><arg type="d" direction="out"/><arg type="s" direction="out"/><arg type="s" direction="out"/><arg type="b" direction="out"/></method>
</interface></node>`;

// Deliberately no arbitrary key, text, command, window title or selection API.
const SHORTCUTS = new Map([
    ['paste', [Clutter.KEY_Control_L, Clutter.KEY_v]],
    ['paste_terminal', [Clutter.KEY_Control_L, Clutter.KEY_Shift_L, Clutter.KEY_v]],
    ['ctrl_home', [Clutter.KEY_Control_L, Clutter.KEY_Home]],
]);
// Selection metadata only: when the owner last changed and which window had
// focus then. Never the selection content or its MIME types.
const SELECTIONS = new Map([
    ['primary', Meta.SelectionType.SELECTION_PRIMARY],
    ['clipboard', Meta.SelectionType.SELECTION_CLIPBOARD],
]);
const BADGE_MS = 1500;
const BADGE_MARGIN = 16;
const BADGE_STYLE = 'background-color: #1f6f3f; color: #ffffff; font-weight: bold; ' +
    'font-size: 15px; padding: 9px 14px; border-radius: 6px;';

function thousands(count) {
    return String(Math.max(0, Math.floor(count))).replace(/\B(?=(\d{3})+(?!\d))/g, '.');
}

function badgeText(chars) {
    return `Context opgepakt: ${thousands(chars)} tekens`;
}

// Status box: what the dictation is doing now. Every word on it comes from
// these tables; the caller sends only codes and one count, never text.
const STATUS_MARGIN = 8;
const STATUS_BASE = 'color: #ffffff; font-weight: bold; font-size: 15px; ' +
    'padding: 8px 14px; border-radius: 6px; background-color: ';
const STATUS_COLOURS = new Map([
    ['recording', '#a51d2d'], ['busy', '#1c5fb4'], ['ok', '#1f6f3f'],
    ['neutral', '#3d3846'], ['error', '#c64600'],
]);
const STATUS_MODES = new Map([
    ['dictate', 'Dictatie'],
    ['rewrite', 'Herschrijven met opmaak'],
    ['rewrite_line', 'Herschrijven als één regel'],
    ['dictate_fallback', 'Dictatie (herschrijven niet beschikbaar)'],
]);
const STATUS_LANGUAGES = new Map([['en', 'Engels'], ['nl', 'Nederlands'], ['fr', 'Frans'], ['de', 'Duits']]);
// text, colour, ms until it hides (stale guard for open phases), clock shown.
const STATUS_PHASES = new Map([
    ['recording', ['● opnemen', 'recording', 1_900_000, true]],
    ['recording_silent', ['● opnemen, maar de microfoon geeft alleen stilte', 'error', 1_900_000, true]],
    ['finishing', ['afronden…', 'busy', 900_000, true]],
    ['rewriting', ['herschrijven…', 'busy', 900_000, true]],
    ['pasted', ['geplakt ✓', 'ok', 1500, false]],
    ['nothing', ['niets herkend, niets geplakt', 'neutral', 3000, false]],
    ['mic_silent', ['microfoon gaf alleen stilte, niets opgenomen', 'error', 6000, false]],
    ['paste_refused', ['plakken geweigerd, tekst staat in CopyQ/klembord', 'error', 6000, false]],
    ['raw_pasted', ['herschrijven mislukt, ruwe tekst geplakt', 'error', 6000, false]],
]);

function statusText(phase, mode, language, chars, seconds) {
    const [label, , , clock] = STATUS_PHASES.get(phase);
    let head = STATUS_MODES.get(mode);
    if (language)
        head += ` · ${STATUS_LANGUAGES.get(language) ?? language.toUpperCase()}`;
    if (chars > 0)
        head += ` · context: ${thousands(chars)} tekens`;
    let text = `${head} — ${label}`;
    if (clock) {
        const s = Math.max(0, Math.floor(seconds));
        text += ` ${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
    }
    return text;
}

export default class DictationHelper extends Extension {
    enable() {
        this._keyboard = Clutter.get_default_backend().get_default_seat()
            .create_virtual_device(Clutter.InputDeviceType.KEYBOARD_DEVICE);
        this._changes = new Map();
        this._selection = global.display.get_selection();
        this._selectionId = this._selection.connect('owner-changed',
            (_selection, type, source) => this._ownerChanged(type, source));
        this._badge = null;
        this._badgeTimeout = 0;
        this._status = null;
        this._statusState = null;
        this._statusTimeout = 0;
        this._statusTick = 0;
        this._object = Gio.DBusExportedObject.wrapJSObject(XML, this);
        this._object.export(Gio.DBus.session, PATH);
        this._busId = Gio.bus_own_name_on_connection(Gio.DBus.session, BUS,
            Gio.BusNameOwnerFlags.NONE, null, null);
    }

    disable() {
        this._object?.unexport();
        this._object = null;
        if (this._busId)
            Gio.bus_unown_name(this._busId);
        this._busId = 0;
        if (this._selectionId)
            this._selection.disconnect(this._selectionId);
        this._selectionId = 0;
        this._selection = null;
        this._changes = null;
        this._hideBadge();
        this.ClearStatus();
        this._keyboard = null;
    }

    _ownerChanged(type, source) {
        if (![...SELECTIONS.values()].includes(type))
            return;
        const window = global.display.focus_window;
        this._changes.set(type, {
            at: GLib.get_monotonic_time(),
            owned: Boolean(source),
            window: window ? String(window.get_id()) : '',
            pid: window ? window.get_pid() : 0,
        });
    }

    GetFocus() {
        const window = global.display.focus_window;
        return window ? [String(window.get_id()), window.get_wm_class() ?? ''] : ['', ''];
    }

    Activate(id) {
        if (!/^\d+$/.test(id))
            return false;
        const window = global.get_window_actors().map(actor => actor.meta_window)
            .find(candidate => String(candidate.get_id()) === id);
        if (!window)
            return false;
        window.activate(global.get_current_time());
        return true;
    }

    SendShortcut(name) {
        const keys = SHORTCUTS.get(name);
        if (!keys)
            throw new Error('Shortcut not allowed');
        const pressed = [];
        try {
            for (const key of keys) {
                this._keyboard.notify_keyval(GLib.get_monotonic_time(), key, Clutter.KeyState.PRESSED);
                pressed.push(key);
            }
        } finally {
            for (const key of pressed.reverse())
                this._keyboard.notify_keyval(GLib.get_monotonic_time(), key, Clutter.KeyState.RELEASED);
        }
    }

    // (state, age_s, window at change, focus window now, same application).
    // state: 'set', 'cleared' or 'unknown' (no change seen since enable).
    SelectionInfo(name) {
        const type = SELECTIONS.get(name);
        if (type === undefined)
            throw new Error('Selection not allowed');
        const focus = global.display.focus_window;
        const focusId = focus ? String(focus.get_id()) : '';
        const change = this._changes.get(type);
        if (!change)
            return ['unknown', -1, '', focusId, false];
        const age = (GLib.get_monotonic_time() - change.at) / 1e6;
        if (!change.owned)
            return ['cleared', age, change.window, focusId, false];
        const sameApp = Boolean(focus && change.window &&
            (change.window === focusId || (change.pid > 0 && focus.get_pid() === change.pid)));
        return ['set', age, change.window, focusId, sameApp];
    }

    // Only the number of characters arrives here; the label never takes input.
    ShowBadge(chars) {
        this._hideBadge();
        const label = new St.Label({text: badgeText(chars), style: BADGE_STYLE, reactive: false,
            can_focus: false, track_hover: false});
        Main.layoutManager.uiGroup.add_child(label);
        const area = global.display.get_monitor_geometry(this._monitor());
        const [, width] = label.get_preferred_width(-1);
        label.set_position(area.x + Math.max(0, area.width - width - BADGE_MARGIN), area.y + BADGE_MARGIN);
        this._badge = label;
        this._badgeTimeout = GLib.timeout_add(GLib.PRIORITY_DEFAULT, BADGE_MS, () => {
            this._badgeTimeout = 0;
            this._hideBadge();
            return GLib.SOURCE_REMOVE;
        });
    }

    _monitor() {
        const focus = global.display.focus_window;
        const monitor = focus ? focus.get_monitor() : global.display.get_current_monitor();
        return Math.max(0, monitor);
    }

    // Only allowlisted codes and one count arrive here; anything else is refused.
    // The box stays until the next call, its own hide time, or ClearStatus.
    SetStatus(phase, mode, language, chars) {
        if (!STATUS_PHASES.has(phase) || !STATUS_MODES.has(mode) || !/^([a-z]{2,3})?$/.test(language))
            throw new Error('Status not allowed');
        const now = GLib.get_monotonic_time();
        const previous = this._statusState;
        // The clock restarts when the phase or mode changes, not on a repeat.
        const since = previous && previous.phase === phase && previous.mode === mode
            ? previous.since : now;
        this._statusState = {phase, mode, language, chars: Math.max(0, Math.floor(chars)), since};
        if (!this._status) {
            this._status = new St.Label({text: '', reactive: false, can_focus: false, track_hover: false});
            Main.layoutManager.uiGroup.add_child(this._status);
        }
        const [, colour, hideMs, clock] = STATUS_PHASES.get(phase);
        this._status.set_style(STATUS_BASE + STATUS_COLOURS.get(colour) + ';');
        this._drawStatus();
        if (this._statusTimeout)
            GLib.source_remove(this._statusTimeout);
        this._statusTimeout = GLib.timeout_add(GLib.PRIORITY_DEFAULT, hideMs, () => {
            this._statusTimeout = 0;
            this.ClearStatus();
            return GLib.SOURCE_REMOVE;
        });
        if (clock && !this._statusTick) {
            this._statusTick = GLib.timeout_add(GLib.PRIORITY_DEFAULT, 1000, () => {
                this._drawStatus();
                return GLib.SOURCE_CONTINUE;
            });
        } else if (!clock && this._statusTick) {
            GLib.source_remove(this._statusTick);
            this._statusTick = 0;
        }
    }

    ClearStatus() {
        for (const id of [this._statusTimeout, this._statusTick]) {
            if (id)
                GLib.source_remove(id);
        }
        this._statusTimeout = this._statusTick = 0;
        this._status?.destroy();
        this._status = null;
        this._statusState = null;
    }

    // Top right of the focused window's monitor, under the top bar (work area).
    _drawStatus() {
        const state = this._statusState;
        const seconds = (GLib.get_monotonic_time() - state.since) / 1e6;
        this._status.set_text(statusText(state.phase, state.mode, state.language, state.chars, seconds));
        const area = Main.layoutManager.getWorkAreaForMonitor(this._monitor());
        const [, width] = this._status.get_preferred_width(-1);
        this._status.set_position(area.x + Math.max(0, area.width - width - STATUS_MARGIN),
            area.y + STATUS_MARGIN);
    }

    _hideBadge() {
        if (this._badgeTimeout)
            GLib.source_remove(this._badgeTimeout);
        this._badgeTimeout = 0;
        this._badge?.destroy();
        this._badge = null;
    }
}
