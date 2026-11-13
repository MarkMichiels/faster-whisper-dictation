"""Missed first start under evdev: tap timing, slow-tap journal, opt-in key trace.

Synthetic evdev streams with kernel timestamps and a fake clock; no devices,
no desktop IO.
"""
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import wayland_io as w
from test_wayland import listeners

CTRL_L, CTRL_R, ALT_L, SHIFT_L, KEY_Q, KEY_A = 29, 97, 56, 42, 16, 30


class Clock:
    def __init__(self, now):
        self.now = now

    def time(self):
        return self.now


def ev(code, value, t, type_=1):
    sec = int(t)
    return types.SimpleNamespace(type=type_, code=code, value=value,
                                 sec=sec, usec=int(round((t - sec) * 1e6)))


class MissedStartTests(unittest.TestCase):
    T0 = 1794571790.0

    def setUp(self):
        self.ns = listeners()
        self.clock = Clock(self.T0)
        self.ns['time'] = self.clock
        self.journal = []
        self.ns['diagnostic'] = lambda kind, **info: self.journal.append((kind, info))
        self.start, self.stop = Mock(), Mock()
        self.tap = self.ns['MultiTapKeyListener'](self.start, self.stop,
            key=w.Key.ctrl_r, default_language='nl', shift_language='en')
        self.hub = w.InputHub(provider=lambda: {}, selector=Mock(), layout='be', trace=False)
        self.hub.key_listener = self.tap

    def key(self, code, value, at, processed=None):
        """Deliver one event stamped `at`; the reader handles it at `processed`."""
        self.clock.now = self.T0 + (at if processed is None else processed)
        self.hub.event('/dev/input/event6', ev(code, value, self.T0 + at))

    def ctrl_a(self, at):
        self.key(CTRL_L, 1, at)
        self.key(KEY_Q, 1, at + .08)   # physical Q is A on Belgian AZERTY
        self.key(KEY_Q, 0, at + .15)
        self.key(CTRL_L, 0, at + .33)

    def slow(self):
        return [info for kind, info in self.journal if kind == 'double_tap_slow']

    def test_slow_double_tap_after_ctrl_a_with_alt_is_journalled(self):
        """Ctrl+A, then Left-Alt and a second tap 0.40 s after the first: no
        start (as before), but no longer silent. A retry inside the window starts."""
        self.ctrl_a(0)
        self.key(ALT_L, 1, 9.0)
        self.key(ALT_L, 2, 9.5)          # autorepeat of a held Alt is ignored
        self.key(CTRL_R, 1, 10.00)
        self.key(CTRL_R, 0, 10.12)
        self.key(CTRL_R, 1, 10.40)
        self.start.assert_not_called()
        self.assertEqual(self.stop.call_count, 2)
        [info] = self.slow()
        self.assertEqual(info['interval_s'], .4)
        self.assertEqual(info['first_hold_s'], .12)
        self.assertEqual(info['gap_s'], .28)
        self.assertEqual(info['window_s'], .3)
        self.assertEqual(info['clock'], 'event')
        self.assertTrue(info['alt'])
        self.assertFalse(info['ctrl'])   # Left-Ctrl of Ctrl+A is released
        self.assertEqual(set(info), {'interval_s', 'window_s', 'clock', 'shift', 'alt',
                                     'ctrl', 'first_hold_s', 'gap_s'})
        # Second attempt, inside the window: a rewrite start.
        self.key(CTRL_R, 0, 10.50)
        self.key(CTRL_R, 1, 10.70)
        self.start.assert_called_once()
        language, rewrite, keys = self.start.call_args.args
        self.assertEqual((language, rewrite), ('nl', True))
        self.assertAlmostEqual(keys['alt_held_s'], 1.7, places=3)
        self.assertAlmostEqual(keys['ctrl_released_s_ago'], 10.37, places=3)
        self.assertEqual(len(self.slow()), 1)

    def test_no_slow_tap_line_for_normal_use(self):
        # A quick double tap, a stop much later and a lone tap: nothing journalled.
        self.key(CTRL_R, 1, 0)
        self.key(CTRL_R, 0, .1)
        self.key(CTRL_R, 1, .2)
        self.key(CTRL_R, 0, .3)
        self.start.assert_called_once()
        self.key(CTRL_R, 1, 20)
        self.key(CTRL_R, 0, 20.1)
        self.key(CTRL_R, 1, 25)
        self.assertEqual(self.slow(), [])

    def test_tap_timing_uses_kernel_time_when_reader_was_busy(self):
        """A quick double tap whose first tap stops a running session: stop()
        keeps the reader busy for 0.5 s (recorder join, beep). The second press
        waits in the kernel buffer. Timed on processing, that is a 0.5 s gap
        and a silently missed start; timed on the event, 0.18 s and a start."""
        def busy_stop():
            self.clock.now += .5
        self.stop.side_effect = busy_stop
        self.key(CTRL_R, 1, 0)
        self.key(CTRL_R, 0, .08, processed=.5)
        self.key(CTRL_R, 1, .18, processed=.5)
        self.start.assert_called_once()
        self.assertEqual(self.slow(), [])

    def test_implausible_kernel_time_falls_back_to_now(self):
        self.tap.event_time = self.T0 - 3600
        self.assertEqual(self.tap._now(), self.T0)
        self.tap.event_time = None
        self.assertEqual(self.tap._now(), self.T0)

    def test_hub_sets_and_clears_event_time(self):
        seen = []
        self.tap.on_press = lambda key: seen.append(self.tap.event_time)
        self.key(CTRL_R, 1, 1.25)
        self.assertAlmostEqual(seen[0], self.T0 + 1.25, places=5)
        self.assertIsNone(self.tap.event_time)
        def boom(key):
            raise RuntimeError('x')
        self.tap.on_release = boom
        with self.assertRaises(RuntimeError):
            self.key(CTRL_R, 0, 1.3)
        self.assertIsNone(self.tap.event_time)
        # Synthetic releases (device loss) carry no kernel time.
        self.tap.on_press = lambda key: seen.append(self.tap.event_time)
        self.hub.event('x', types.SimpleNamespace(type=1, code=CTRL_R, value=1))
        self.assertIsNone(seen[-1])


class KeyTraceTests(unittest.TestCase):
    T0 = 1794571790.0

    def hub(self, trace):
        log = Mock()
        hub = w.InputHub(provider=lambda: {}, selector=Mock(), trace=trace, log=log)
        hub.key_listener = Mock()
        return hub, log

    def feed(self, hub):
        with patch.object(w.time, 'time', return_value=self.T0 + .002):
            for code, value in ((CTRL_L, 1), (KEY_Q, 1), (KEY_Q, 0), (KEY_A, 1), (KEY_A, 0),
                                (CTRL_L, 0), (ALT_L, 1), (ALT_L, 2), (CTRL_R, 1), (CTRL_R, 1),
                                (CTRL_R, 0), (SHIFT_L, 0), (2, 1), (2, 0)):
                hub.event('/dev/input/event6', ev(code, value, self.T0))
            hub.event('/dev/input/event6', ev(3, 0, self.T0, type_=0))      # SYN_DROPPED
            hub.event('/dev/input/event6', ev(CTRL_R, 1, self.T0))         # while desynced
            hub.event('/dev/input/event6', ev(KEY_Q, 1, self.T0))          # while desynced

    def test_off_by_default_and_silent(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop('DICTATION_KEY_TRACE', None)
            hub, log = self.hub(None)
            self.assertFalse(hub.trace)
            self.feed(hub)
            log.assert_not_called()
        with patch.dict(os.environ, {'DICTATION_KEY_TRACE': '1'}):
            self.assertTrue(w.InputHub(provider=lambda: {}, selector=Mock()).trace)

    def test_trace_only_modifiers_and_right_ctrl(self):
        hub, log = self.hub(True)
        self.feed(hub)
        lines = [call.kwargs for call in log.call_args_list]
        for call in log.call_args_list:
            self.assertEqual(call.args, ('key_trace',))
        allowed = set(w.InputHub.TRACE_CODES.values()) | {'syn_dropped'}
        self.assertTrue(lines)
        for line in lines:
            self.assertIn(line['key'], allowed)
            self.assertEqual(line['device'], 'event6')
            self.assertLessEqual(set(line), {'key', 'value', 'device', 'action', 't',
                                             'read_latency_ms'})
        actions = [(l['key'], l.get('value'), l['action']) for l in lines]
        self.assertIn(('alt_l', 2, 'ignored_repeat'), actions)
        self.assertIn(('ctrl_r', 1, 'ignored_already_down'), actions)
        self.assertIn(('shift_l', 0, 'ignored_not_down'), actions)
        self.assertIn(('syn_dropped', None, 'released_all'), actions)
        self.assertIn(('ctrl_r', 1, 'ignored_desynced'), actions)
        self.assertEqual(lines[0]['read_latency_ms'], 2.0)
        # No trace of Q, A or 1 in any form.
        self.assertEqual(len([a for a in actions if a[0] == 'ctrl_l']), 2)
        self.assertNotIn('16', repr(lines))
        self.assertNotIn("'a'", repr(lines))

    def test_device_lines(self):
        class Dev:
            def capabilities(self):
                return {1: [97]}
            def close(self):
                pass
        log = Mock()
        provider = Mock(return_value={'/dev/input/event6': Dev})
        hub = w.InputHub(provider=provider, selector=Mock(), trace=True, log=log)
        hub.rescan()
        provider.return_value = {}
        hub.rescan()
        actions = [c.kwargs['action'] for c in log.call_args_list]
        self.assertEqual(actions, ['enabled', 'device_removed'])
        self.assertEqual(log.call_args_list[0].kwargs['devices'], ['event6'])


if __name__ == '__main__':
    unittest.main()
