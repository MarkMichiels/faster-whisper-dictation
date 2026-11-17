"""WP61: synthetic/offline only; no capture backend or live graph accessed."""
import contextlib
import io
import json
import queue
import threading
import types
import unittest
from unittest.mock import patch

import numpy as np
from test_recovery import load_logic
import capture_metadata as meta

R = meta.RATE


def tone(seconds, db):
    t = np.arange(int(seconds * R)) / R
    return (np.sin(2*np.pi*180*t)*np.sqrt(2)*10**(db/20)).astype(np.float32)


def entries(out):
    return [json.loads(line.removeprefix('DCT-COVERAGE '))
            for line in out.getvalue().splitlines() if line.startswith('DCT-COVERAGE ')]


class ObservationTests(unittest.TestCase):
    def run_levels(self, levels):
        obs = meta.InputObservation()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            for i, db in enumerate(levels):
                obs.observe(tone(1, db), i*R, 0, [], i+1, .5, 10, i == len(levels)-1)
        return entries(out)

    def test_weak_and_normal_levels_no_content(self):
        for db, voiced in [(-69.4, 0), (-24, .99)]:
            rows = self.run_levels([db]*5)
            windows = [r for r in rows if r['kind'] == 'vad_observation']
            self.assertEqual(len(windows), 5)
            for i, row in enumerate(windows):
                self.assertEqual(row['window_start'], i*R)
                self.assertEqual(row['window_end'], (i+1)*R)
                self.assertAlmostEqual(row['audio']['rms_dbfs'], db, delta=.1)
                self.assertAlmostEqual(row['audio']['peak_dbfs'], db+3, delta=.1)
                self.assertAlmostEqual(row['audio']['voiced_s'], voiced, delta=.01)
                self.assertIsNone(row['last_speech_end'])
                self.assertEqual(set(row), {'timestamp', 'kind', 'capture_id', 'buffer_start', 'buffer_end',
                    'processed_end', 'last_speech_end', 'vad_speech_seconds',
                    'trailing_silence_seconds', 'threshold', 'auto_stop_silence_s',
                    'stop', 'window_start', 'window_end', 'audio'})
                json.dumps(row, allow_nan=False)
            warnings = [r for r in rows if r['kind'] == 'input_level_warning']
            self.assertEqual(len(warnings), int(db < -45))
            if warnings:
                self.assertEqual(warnings[0]['end'], 3*R)
                self.assertFalse(windows[2]['stop'])  # warning precedes stop

    def test_short_pauses_and_speech_never_warn(self):
        rows = self.run_levels([-24, -69, -69, -24, -69, -24])
        self.assertFalse(any(r['kind'] == 'input_level_warning' for r in rows))

    def test_trim_cadence_stop_override_and_vad_duration(self):
        obs = meta.InputObservation()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            obs.observe(tone(1, -24), 0, 100, [{'start':0, 'end':R}], 0, .5, 10, False)
            obs.observe(tone(.5, -24), R, 100, [{'start':0, 'end':R//2}], 0, .5, 10, False)
            obs.observe(tone(.6, -24), R, 100, [], .6, .5, 10, True)
            obs.observe(tone(1, -24), R, 100, [{'start':0, 'end':R}], 0, .5, 10, False)
        rows = entries(out)
        self.assertEqual(len(rows), 2)  # cadence, but always log at stop
        self.assertEqual(rows[0]['vad_speech_seconds'], 1)
        self.assertEqual(rows[0]['last_speech_end'], R)
        self.assertEqual(rows[1]['processed_end'], R+100)
        self.assertEqual(rows[1]['buffer_end'], int(1.6*R))
        self.assertTrue(rows[1]['stop'])

    def test_digital_silence_and_log_failure(self):
        obs = meta.InputObservation()
        with patch.object(meta, 'diagnostic', side_effect=OSError('broken journal')):
            for i in range(4):
                obs.observe(np.zeros(R), i*R, 0, [], i+1, .5, 10, False)
        self.assertTrue(obs.warned)

    def test_start_resets_state_without_threads_or_microphone(self):
        logic = load_logic()
        obj = logic['StreamingRecorder'].__new__(logic['StreamingRecorder'])
        obj._input_observation = meta.InputObservation()
        old = obj._input_observation
        old.warned = True
        with patch.object(threading, 'Thread'):
            obj.start()
        self.assertIsNot(old, obj._input_observation)
        self.assertNotEqual(old.capture_id, obj._input_observation.capture_id)
        self.assertFalse(obj._input_observation.warned)
        self.assertIsNone(obj._input_observation.last_log_end)

    def test_recording_ticks_warn_before_autostop_and_preserve_decisions(self):
        for db, speaking in [(-69.4, False), (-24, False), (-24, True)]:
            logic = load_logic()
            obj = logic['StreamingRecorder'].__new__(logic['StreamingRecorder'])
            obj._buffer_offset = 0
            obj._vad_processed_samples = 0
            obj.vad_options = types.SimpleNamespace(threshold=.5)
            obj.auto_stop_silence_s = 10
            obj.on_auto_stop = lambda: None
            obj.min_chunk_samples = 0
            obj.silence_ms = 1000
            obj._trim_buffer = lambda: None
            obj._input_observation = meta.InputObservation()
            logic['get_speech_timestamps'] = lambda a, *args, **kw: ([{'start':0, 'end':len(a)}] if speaking else [])
            out = io.StringIO()
            with contextlib.redirect_stdout(out), patch.object(threading, 'Thread'):
                for tick in range(2, 36):
                    audio = tone(tick*.3, db)
                    obj._get_all_audio_fp32 = lambda: audio
                    stopped = obj._check_and_send_chunks()
                    self.assertEqual(bool(stopped), not speaking and tick*.3 > 10)
                    if stopped:
                        break
            rows = entries(out)
            warnings = [r for r in rows if r['kind'] == 'input_level_warning']
            self.assertEqual(len(warnings), int(db < -45))
            if warnings:
                self.assertLess(warnings[0]['end']/R, 4)
                self.assertLess(warnings[0]['end'], rows[-1]['buffer_end'])
            self.assertEqual(rows[-1]['stop'], not speaking)

    def test_real_stop_path_and_observer_failure_do_not_change_stop(self):
        for broken in (False, True):
            logic = load_logic()
            obj = logic['StreamingRecorder'].__new__(logic['StreamingRecorder'])
            obj._get_all_audio_fp32 = lambda: tone(10.5, -69.4)
            obj._buffer_offset = R
            obj._vad_processed_samples = 0
            obj.vad_options = types.SimpleNamespace(threshold=.5)
            obj.auto_stop_silence_s = 10
            obj.on_auto_stop = lambda: None
            logic['get_speech_timestamps'] = lambda *a, **kw: []
            obj._input_observation = meta.InputObservation()
            if broken:
                obj._input_observation.observe = lambda *a: 1/0
            out = io.StringIO()
            with contextlib.redirect_stdout(out), patch.object(threading, 'Thread') as thread:
                self.assertTrue(obj._check_and_send_chunks())
                thread.return_value.start.assert_called_once()
            if not broken:
                row = [r for r in entries(out) if r['kind'] == 'vad_observation'][0]
                self.assertTrue(row['stop'])
                self.assertEqual(row['audio']['rms_dbfs'], -69.4)
                self.assertEqual(row['buffer_end'], int(11.5*R))


class CaptureSourceTests(unittest.TestCase):
    def test_graph_only_own_capture_links_not_default_or_other_clients(self):
        def node(i, **props):
            return dict(id=i, type='PipeWire:Interface:Node', info=dict(props=props))
        graph = [node(1, **{'application.process.id': '42', 'media.class':'Stream/Input/Audio'}),
                 node(2, **{'node.name':'alsa_input.test', 'node.description':'Test mic', 'device.id':7}),
                 node(3, **{'application.process.id':'99', 'media.class':'Stream/Input/Audio'}),
                 dict(type='PipeWire:Interface:Link', info={'input-node-id':1, 'output-node-id':2}),
                 dict(type='PipeWire:Interface:Link', info={'input-node-id':3, 'output-node-id':99})]
        row = meta.pipewire_capture(42, graph)
        self.assertEqual(row['status'], 'matched')
        self.assertEqual(row['routes'][0]['sources'][0]['description'], 'Test mic')
        self.assertEqual(len(row['routes']), 1)
        self.assertEqual(meta.pipewire_capture(100, graph)['status'], 'unknown_or_ambiguous')
        self.assertFalse(row['physical_route_proven'])

    def test_pw_dump_is_bounded_and_never_logs_raw_graph(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), patch.object(meta.subprocess, 'run', return_value=types.SimpleNamespace(stdout='[]')) as run:
            meta.log_pipewire_capture()
        self.assertEqual(run.call_args.kwargs['timeout'], 1)
        self.assertEqual(run.call_args.args[0], ['pw-dump'])
        row = entries(out)[0]
        self.assertEqual(set(row), {'timestamp', 'kind', 'capture_id', 'status', 'routes', 'physical_route_proven'})
        self.assertEqual(row['routes'], [])

    def test_portaudio_and_missing_pipewire_are_nonfatal(self):
        pa = types.SimpleNamespace(
            get_default_input_device_info=lambda: dict(index=4, name='default', hostApi=0, defaultSampleRate=48000),
            get_host_api_info_by_index=lambda i: dict(name='ALSA'))
        out = io.StringIO()
        with contextlib.redirect_stdout(out), patch.object(meta.subprocess, 'run', side_effect=FileNotFoundError):
            meta.log_capture_device(pa, R)
            meta.log_pipewire_capture()
            meta.log_capture_device(None, R)
        rows = entries(out)
        self.assertEqual(rows[0]['index'], 4)
        self.assertEqual(rows[0]['requested_rate'], R)
        self.assertEqual(rows[0]['host_api'], 'ALSA')
        self.assertFalse(rows[0]['physical_route_proven'])
        self.assertEqual(rows[1]['status'], 'unavailable')
        self.assertEqual(rows[2]['status'], 'unavailable')


if __name__ == '__main__':
    unittest.main()
