// Offline JS contract/lifecycle tests. No GNOME, display or input access.
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';

const source = readFileSync(new URL('../gnome-extension/dictation-helper@faster-whisper-dictation/extension.js', import.meta.url), 'utf8');
const events = [];
let exported = false;
let owned = false;
let active = false;
let failOn = null;
const keyboard = {notify_keyval(time, key, state) {
    if (failOn === key && state === 1)
        throw new Error('injection unavailable');
    events.push([key, state]);
}};
const Clutter = {
    KEY_Control_L: 1, KEY_Shift_L: 2, KEY_v: 3, KEY_Home: 4,
    KeyState: {PRESSED: 1, RELEASED: 0}, InputDeviceType: {KEYBOARD_DEVICE: 1},
    get_default_backend: () => ({get_default_seat: () => ({create_virtual_device: () => keyboard})}),
};
const window = {get_id: () => 42, get_wm_class: () => 'editor', activate() { active = true; }};
const context = vm.createContext({Clutter, Gio: {
    DBus: {session: {}}, BusNameOwnerFlags: {NONE: 0},
    DBusExportedObject: {wrapJSObject: () => ({export() {exported = true;}, unexport() {exported = false;}})},
    bus_own_name_on_connection() {owned = true; return 9;},
    bus_unown_name(id) {assert.equal(id, 9); owned = false;},
}, GLib: {get_monotonic_time: () => 100}, Extension: class {},
global: {display: {focus_window: window}, get_window_actors: () => [{meta_window: window}], get_current_time: () => 100},
});
// Only GI/resource imports and the ES export wrapper are replaced. Methods
// under test are exactly the extension source, not a copied implementation.
vm.runInContext(source.replace(/^import .*;\n/gm, '').replace('export default class', 'globalThis.Helper = class'), context);
const helper = new context.Helper();
helper.enable();
assert.equal(exported && owned, true);
assert.equal(JSON.stringify(helper.GetFocus()), '["42","editor"]');
assert.equal(helper.Activate('42'), true);
assert.equal(active, true);
assert.equal(helper.Activate('missing'), false);
assert.equal(helper.Activate('43'), false);
assert.equal(helper.SelectionInfo(), false);
helper.ShowBadge(5);
for (const [name, expected] of [
    ['paste', [[1, 1], [3, 1], [3, 0], [1, 0]]],
    ['paste_terminal', [[1, 1], [2, 1], [3, 1], [3, 0], [2, 0], [1, 0]]],
    ['ctrl_home', [[1, 1], [4, 1], [4, 0], [1, 0]]],
]) {
    events.length = 0;
    helper.SendShortcut(name);
    assert.deepEqual(events, expected);
}
for (const name of ['Return', 'enter', 'toString', '__proto__', '', 'paste;Return']) {
    events.length = 0;
    assert.throws(() => helper.SendShortcut(name), /not allowed/);
    assert.deepEqual(events, []);
}
failOn = 3;
events.length = 0;
assert.throws(() => helper.SendShortcut('paste_terminal'), /unavailable/);
assert.deepEqual(events, [[1, 1], [2, 1], [2, 0], [1, 0]]);
helper.disable();
assert.equal(exported || owned, false);
assert.equal(helper._keyboard, null);
console.log('PASS: JS focus/activation, strict allowlist, reverse release/failure cleanup, lifecycle, stubs');
