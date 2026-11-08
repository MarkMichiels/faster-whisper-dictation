import Clutter from 'gi://Clutter';
import Gio from 'gi://Gio';
import GLib from 'gi://GLib';
import {Extension} from 'resource:///org/gnome/shell/extensions/extension.js';

const BUS = 'org.gnome.Shell.Extensions.DictationHelper';
const PATH = '/org/gnome/Shell/Extensions/DictationHelper';
const XML = `<node><interface name="${BUS}">
<method name="GetFocus"><arg type="s" direction="out"/><arg type="s" direction="out"/></method>
<method name="Activate"><arg type="s" direction="in"/><arg type="b" direction="out"/></method>
<method name="SendShortcut"><arg type="s" direction="in"/></method>
<method name="ShowBadge"><arg type="u" direction="in"/></method>
<method name="SelectionInfo"><arg type="b" direction="out"/></method>
</interface></node>`;

// Deliberately no arbitrary key, text, command, window title or selection API.
const SHORTCUTS = new Map([
    ['paste', [Clutter.KEY_Control_L, Clutter.KEY_v]],
    ['paste_terminal', [Clutter.KEY_Control_L, Clutter.KEY_Shift_L, Clutter.KEY_v]],
    ['ctrl_home', [Clutter.KEY_Control_L, Clutter.KEY_Home]],
]);

export default class DictationHelper extends Extension {
    enable() {
        this._keyboard = Clutter.get_default_backend().get_default_seat()
            .create_virtual_device(Clutter.InputDeviceType.KEYBOARD_DEVICE);
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
        this._keyboard = null;
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

    ShowBadge(_count) { /* Implemented in the selection/badge follow-up. */ }
    SelectionInfo() { return false; } // Unknown, never permission to release.
}
