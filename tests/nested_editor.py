"""Synthetic GTK4 editor for the nested GNOME proof (probe_wayland_nested.py).

    python3 nested_editor.py --app-id org.example.DictationProbeA --state STATE.json

One maximized window with a text view holding a synthetic two-paragraph
message. On every change of text, caret or selection it writes its state
(offsets, counts, the first characters of the synthetic text) to STATE.json.
When STATE.json.reset appears it restores the text, caret at the end.
It runs only inside the private nested session of the probe.
"""
import argparse
import json
import os

import gi

gi.require_version('Gtk', '4.0')
from gi.repository import GLib, Gtk  # noqa: E402

TEXT = ('Beste, dit is een synthetisch testbericht over de levering van vrijdag.\n'
        'Tweede alinea van het synthetische bericht, alleen voor de proef.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--app-id', required=True)
    parser.add_argument('--state', required=True)
    parser.add_argument('--title', default='Synthetic editor')
    args = parser.parse_args()
    app = Gtk.Application(application_id=args.app_id)

    def write_state(buffer, *_):
        has_sel = buffer.get_has_selection()
        bounds = buffer.get_selection_bounds()
        cursor = buffer.get_iter_at_mark(buffer.get_insert()).get_offset()
        text = buffer.get_text(buffer.get_start_iter(), buffer.get_end_iter(), False)
        state = dict(chars=len(text), cursor=cursor, has_selection=has_sel,
                     selection=[bounds[0].get_offset(), bounds[1].get_offset()] if has_sel else None,
                     head=text[:24], original_intact=TEXT in text)
        tmp = args.state + '.tmp'
        with open(tmp, 'w') as fh:
            json.dump(state, fh)
        os.replace(tmp, args.state)

    def activate(app):
        window = Gtk.ApplicationWindow(application=app, title=args.title)
        view = Gtk.TextView(wrap_mode=Gtk.WrapMode.WORD, left_margin=20, top_margin=20)
        buffer = view.get_buffer()
        buffer.set_text(TEXT)
        buffer.place_cursor(buffer.get_end_iter())
        buffer.connect('changed', write_state)
        buffer.connect('mark-set', write_state)
        def poll_reset():
            if os.path.exists(args.state + '.reset'):
                os.unlink(args.state + '.reset')
                buffer.set_text(TEXT)
                buffer.place_cursor(buffer.get_end_iter())
                write_state(buffer)
            return GLib.SOURCE_CONTINUE
        GLib.timeout_add(100, poll_reset)
        window.set_child(view)
        window.maximize()
        window.present()
        view.grab_focus()
        write_state(buffer)

    app.connect('activate', activate)
    app.run([])


if __name__ == '__main__':
    main()
