"""Stand-in for the systemd user manager on the probe's private session bus.

GNOME Shell starts `gnome-session-x11-services-ready.target` through
org.freedesktop.systemd1 before it lets Xwayland finish. The private bus of
probe_wayland_nested.py has no systemd, so this stub answers StartUnit with a
job that completes at once ('done') and starts nothing. It never runs on the
real session bus: it refuses unless the probe's private bus is set.
"""
import os
import sys

from gi.repository import Gio, GLib

# dbus-run-session gives the probe its own bus; the login session bus lives in /run/user.
if (not os.environ.get('PROBE_TMP') or not os.environ.get('DBUS_SESSION_BUS_ADDRESS')
        or '/run/user/' in os.environ['DBUS_SESSION_BUS_ADDRESS']):
    sys.exit('refusing: private probe bus only')

XML = """<node><interface name='org.freedesktop.systemd1.Manager'>
<method name='StartUnit'><arg type='s' direction='in'/><arg type='s' direction='in'/>
<arg type='o' direction='out'/></method>
<method name='StopUnit'><arg type='s' direction='in'/><arg type='s' direction='in'/>
<arg type='o' direction='out'/></method>
<method name='Subscribe'/>
<signal name='JobRemoved'><arg type='u'/><arg type='o'/><arg type='s'/><arg type='s'/></signal>
</interface></node>"""
PATH = '/org/freedesktop/systemd1'
counter = [0]


def on_call(conn, sender, path, iface, method, params, invocation):
    if method in ('StartUnit', 'StopUnit'):
        counter[0] += 1
        unit = params.unpack()[0]
        job = '%s/job/%d' % (PATH, counter[0])
        invocation.return_value(GLib.Variant('(o)', (job,)))
        print('stub systemd: %s %s' % (method, unit), flush=True)

        def done(n=counter[0], job=job, unit=unit):
            conn.emit_signal(None, PATH, 'org.freedesktop.systemd1.Manager', 'JobRemoved',
                             GLib.Variant('(uoss)', (n, job, unit, 'done')))
            return False
        GLib.timeout_add(50, done)
    elif method == 'Subscribe':
        invocation.return_value(None)
    else:
        invocation.return_dbus_error('org.freedesktop.systemd1.NoSuchUnit', 'stub')


def main():
    bus = Gio.bus_get_sync(Gio.BusType.SESSION)
    info = Gio.DBusNodeInfo.new_for_xml(XML).interfaces[0]
    bus.register_object(PATH, info, on_call)
    Gio.bus_own_name_on_connection(bus, 'org.freedesktop.systemd1', 0, None, None)
    loop = GLib.MainLoop()
    GLib.timeout_add_seconds(900, loop.quit)  # hard limit
    loop.run()


if __name__ == '__main__':
    main()
