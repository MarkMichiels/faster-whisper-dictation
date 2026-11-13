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
let now = 100;
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
const Meta = {SelectionType: {SELECTION_PRIMARY: 0, SELECTION_CLIPBOARD: 1, SELECTION_DND: 2}};
const handlers = new Map();
const selection = {
    connect(name, fn) { assert.equal(name, 'owner-changed'); handlers.set(7, fn); return 7; },
    disconnect(id) { handlers.delete(id); },
};
const changeOwner = (type, source) => [...handlers.values()].forEach(fn => fn(selection, type, source));
const timeouts = new Map();
let nextTimeout = 1;
const GLib = {
    get_monotonic_time: () => now, PRIORITY_DEFAULT: 0, SOURCE_REMOVE: false, SOURCE_CONTINUE: true,
    timeout_add(_priority, ms, fn) { const id = nextTimeout++; timeouts.set(id, [ms, fn]); return id; },
    source_remove(id) { timeouts.delete(id); },
};
const stage = [];
class Label {
    constructor(props) { Object.assign(this, props); }
    get_preferred_width() { return [0, 300]; }
    set_position(x, y) { this.x = x; this.y = y; }
    set_text(text) { this.text = text; }
    set_style(style) { this.style = style; }
    destroy() { this.destroyed = true; stage.splice(stage.indexOf(this), 1); }
}
// Work area = monitor minus a 32 px top bar.
const Main = {layoutManager: {uiGroup: {add_child(actor) { stage.push(actor); }},
    getWorkAreaForMonitor: i => (i === 1 ? {x: 1920, y: 32, width: 2560, height: 1408}
        : {x: 0, y: 32, width: 1920, height: 1048})}};
const editor = {get_id: () => 42, get_wm_class: () => 'editor', get_pid: () => 500,
    get_monitor: () => 1, activate() { active = true; }};
const popout = {get_id: () => 43, get_wm_class: () => 'editor', get_pid: () => 500, get_monitor: () => 1};
const other = {get_id: () => 44, get_wm_class: () => 'viewer', get_pid: () => 600, get_monitor: () => 0};
const display = {
    focus_window: editor,
    get_selection: () => selection,
    get_current_monitor: () => 0,
    get_monitor_geometry: i => (i === 1 ? {x: 1920, y: 0, width: 2560, height: 1440} : {x: 0, y: 0, width: 1920, height: 1080}),
};
const context = vm.createContext({Clutter, Meta, GLib, St: {Label}, Main, Gio: {
    DBus: {session: {}}, BusNameOwnerFlags: {NONE: 0},
    DBusExportedObject: {wrapJSObject: () => ({export() {exported = true;}, unexport() {exported = false;}})},
    bus_own_name_on_connection() {owned = true; return 9;},
    bus_unown_name(id) {assert.equal(id, 9); owned = false;},
}, Extension: class {},
global: {display, get_window_actors: () => [{meta_window: editor}], get_current_time: () => 100},
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

// Selection metadata: nothing seen yet, then owner changes with focus at that moment.
const info = name => JSON.stringify(helper.SelectionInfo(name));
assert.equal(info('primary'), '["unknown",-1,"","42",false]');
now = 1_000_000;
changeOwner(Meta.SelectionType.SELECTION_PRIMARY, {});
now = 3_500_000;
assert.equal(info('primary'), '["set",2.5,"42","42",true]');
assert.equal(info('clipboard'), '["unknown",-1,"","42",false]');
display.focus_window = popout;  // another window of the same program
assert.equal(info('primary'), '["set",2.5,"42","43",true]');
display.focus_window = other;   // another program
assert.equal(info('primary'), '["set",2.5,"42","44",false]');
changeOwner(Meta.SelectionType.SELECTION_PRIMARY, {});  // made while the viewer had focus
display.focus_window = editor;
assert.equal(info('primary'), '["set",0,"44","42",false]');
display.focus_window = null;    // no focus now
assert.equal(info('primary'), '["set",0,"44","",false]');
display.focus_window = editor;
changeOwner(Meta.SelectionType.SELECTION_PRIMARY, null);
assert.equal(info('primary'), '["cleared",0,"42","42",false]');
changeOwner(Meta.SelectionType.SELECTION_DND, {});    // not tracked
changeOwner(Meta.SelectionType.SELECTION_CLIPBOARD, {});
assert.equal(info('clipboard'), '["set",0,"42","42",true]');
display.focus_window = null;
changeOwner(Meta.SelectionType.SELECTION_CLIPBOARD, {});  // change without any focus
display.focus_window = editor;
assert.equal(info('clipboard'), '["set",0,"","42",false]');
for (const name of ['dnd', 'PRIMARY', '', 'toString', '__proto__'])
    assert.throws(() => helper.SelectionInfo(name), /not allowed/);

// Badge: count only, top right of the focus window's monitor, not reactive, one at a time.
helper.ShowBadge(4312);
assert.equal(stage.length, 1);
const badge = stage[0];
assert.equal(badge.text, 'Context opgepakt: 4.312 tekens');
assert.equal(badge.reactive, false);
assert.equal(badge.can_focus, false);
assert.deepEqual([badge.x, badge.y], [1920 + 2560 - 300 - 16, 16]);
helper.ShowBadge(7);
assert.equal(stage.length, 1);
assert.equal(badge.destroyed, true);
assert.equal(stage[0].text, 'Context opgepakt: 7 tekens');
assert.equal(timeouts.size, 1);
const [[id, [ms, fire]]] = [...timeouts.entries()];
assert.equal(ms, 1500);
assert.equal(fire(), false);  // SOURCE_REMOVE: GLib drops the source
timeouts.delete(id);
assert.equal(stage.length, 0);
display.focus_window = null;
helper.ShowBadge(1234567);
assert.equal(stage[0].text, 'Context opgepakt: 1.234.567 tekens');
assert.deepEqual([stage[0].x, stage[0].y], [1920 - 300 - 16, 16]);
display.focus_window = editor;


// Status box: codes and one count only, fixed Dutch words, under the top bar.
for (const [id, [, fn]] of [...timeouts.entries()]) { fn(); timeouts.delete(id); }   // badge hides
assert.equal(stage.length, 0);
now = 10_000_000;
helper.SetStatus('recording', 'dictate', '', 0);
assert.equal(stage.length, 1);
const box = stage[0];
assert.equal(box.text, 'Dictatie — ● opnemen 0:00');
assert.equal(box.reactive, false);
assert.equal(box.can_focus, false);
assert.equal(box.track_hover, false);
assert.match(box.style, /#a51d2d/);
assert.deepEqual([box.x, box.y], [1920 + 2560 - 300 - 8, 32 + 8]);
const timers = () => [...timeouts.values()].map(([ms]) => ms).sort((a, b) => a - b);
assert.deepEqual(timers(), [1000, 1_900_000]);   // clock tick + stale guard
now += 7_000_000;
const tick = [...timeouts.values()].find(([ms]) => ms === 1000)[1];
assert.equal(tick(), true);   // SOURCE_CONTINUE
assert.equal(box.text, 'Dictatie — ● opnemen 0:07');
helper.SetStatus('recording', 'dictate', '', 0);   // repeat keeps the clock
assert.equal(box.text, 'Dictatie — ● opnemen 0:07');
assert.equal(stage.length, 1);
helper.SetStatus('recording', 'dictate', 'en', 0);
assert.equal(box.text, 'Dictatie · Engels — ● opnemen 0:07');
helper.SetStatus('recording', 'rewrite', '', 4312);   // mode change restarts the clock
assert.equal(box.text, 'Herschrijven met opmaak · context: 4.312 tekens — ● opnemen 0:00');
helper.SetStatus('recording', 'rewrite_line', 'en', 0);
assert.equal(box.text, 'Herschrijven als één regel · Engels — ● opnemen 0:00');
helper.SetStatus('recording', 'dictate_fallback', 'xx', 0);
assert.equal(box.text, 'Dictatie (herschrijven niet beschikbaar) · XX — ● opnemen 0:00');
helper.SetStatus('recording_silent', 'dictate', '', 0);
assert.match(box.text, /microfoon geeft alleen stilte/);
assert.match(box.style, /#c64600/);
helper.SetStatus('finishing', 'rewrite', '', 4312);
assert.equal(box.text, 'Herschrijven met opmaak · context: 4.312 tekens — afronden… 0:00');
assert.match(box.style, /#1c5fb4/);
helper.SetStatus('rewriting', 'rewrite', '', 4312);
assert.equal(box.text, 'Herschrijven met opmaak · context: 4.312 tekens — herschrijven… 0:00');
assert.deepEqual(timers(), [1000, 900_000]);
helper.SetStatus('pasted', 'rewrite', '', 4312);
assert.equal(box.text, 'Herschrijven met opmaak · context: 4.312 tekens — geplakt ✓');
assert.match(box.style, /#1f6f3f/);
assert.deepEqual(timers(), [1500]);   // no clock, short hide
const [[hideId, [, hide]]] = [...timeouts.entries()];
assert.equal(hide(), false);   // SOURCE_REMOVE: GLib drops the source
timeouts.delete(hideId);
assert.equal(stage.length, 0);
assert.equal(timeouts.size, 0);
for (const [phase, ms, words] of [['raw_pasted', 6000, 'herschrijven mislukt, ruwe tekst geplakt'],
    ['paste_refused', 6000, 'plakken geweigerd'], ['mic_silent', 6000, 'microfoon gaf alleen stilte'],
    ['nothing', 3000, 'niets herkend']]) {
    helper.SetStatus(phase, 'dictate', '', 0);
    assert.ok(stage[0].text.includes(words), phase);
    assert.deepEqual(timers(), [ms]);
}
helper.ClearStatus();
assert.equal(stage.length, 0);
assert.equal(timeouts.size, 0);
helper.ClearStatus();   // idempotent
// Strict allowlist: no free text can ever reach the label.
for (const args of [['Hallo wereld', 'dictate', '', 0], ['recording', 'Geachte heer', '', 0],
    ['recording', 'dictate', 'Beste Jan', 0], ['recording', 'dictate', 'EN', 0],
    ['toString', 'dictate', '', 0], ['recording', '__proto__', '', 0], ['recording', 'dictate', 'english', 0]])
    assert.throws(() => helper.SetStatus(...args), /not allowed/);
assert.equal(stage.length, 0);
// Focus on the other monitor: box on that monitor; still there at disable -> removed.
display.focus_window = other;
helper.SetStatus('recording', 'dictate', '', 0);
assert.deepEqual([stage[0].x, stage[0].y], [1920 - 300 - 8, 32 + 8]);
display.focus_window = editor;
helper.ShowBadge(12);   // the old badge still exists beside it
assert.equal(stage.length, 2);

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
assert.equal(handlers.size, 0);
assert.equal(stage.length, 0);
assert.equal(timeouts.size, 0);
assert.ok(!/get_text|get_content|transfer_async|read_bytes/.test(source), 'no selection content API');
console.log('PASS: JS focus/activation, selection change metadata, badge, status box, strict allowlist, reverse release/failure cleanup, lifecycle');
