"""Two loss paths: a lasting CUDA OOM at stop, and a microphone that passes only zeros.

OOM is simulated with exceptions; no model, GPU, microphone or desktop is used.
"""
import contextlib
import io
import threading
import time
import types
import unittest
from unittest.mock import Mock

import numpy as np

import coverage_ledger as coverage
from test_recovery import load_logic
from test_silence import int16, lines, noise, speech

R = coverage.RATE
OOM = RuntimeError('CUDA failed with error out of memory')


def stop_session(ranges, gpu, cpu_text='hersteld op de processor', budget_s=None):
    """One ordinary session whose ranges all failed with a CUDA OOM;
    runs the real stop check through App._typing_loop."""
    ns = load_logic()  # fresh classes: changing a constant here stays local
    ns['diagnostic'] = coverage.diagnostic
    ns['OomRecovery'].GPU_WAIT_S = 0.01
    if budget_s is not None:
        ns['OomRecovery'].BUDGET_S = budget_s
    app = ns['App'].__new__(ns['App'])
    ledger = coverage.CoverageLedger()
    start = 0
    for seconds in ranges:
        ledger.capture(int16(speech(seconds)).tobytes())
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        for seconds in ranges:
            ledger.outcome(start, start + int(seconds * R), error=str(OOM))
            start += int(seconds * R)
    app.ledger = ledger
    app.recorder = None
    app.active = True
    app.typing_queue = __import__('queue').Queue()
    app.typing_queue.put(None)
    app.transcription_queue = None
    gpu_calls, cpu_loads, cpu_calls = [], [], []

    def gpu_decode(audio, recovery=False):
        gpu_calls.append(len(audio))
        result = gpu(len(gpu_calls))
        if isinstance(result, Exception):
            raise result
        return result
    app.transcription_worker = types.SimpleNamespace(transcribe_chunk=gpu_decode)

    def load():
        cpu_loads.append(1)
        def transcribe(audio, recovery=False):
            cpu_calls.append(recovery)
            return cpu_text
        return types.SimpleNamespace(transcribe_chunk=transcribe)
    app._load_cpu_recovery_worker = load
    pasted = []
    app.replayer = types.SimpleNamespace(type_text=lambda t: pasted.append(t) or dict(
        ok=True, path='copyq', paste='shortcut_sent_not_acknowledged'))
    with contextlib.redirect_stdout(out):
        app._typing_loop()
    return dict(ns=ns, ledger=ledger, out=out, pasted=pasted, gpu_calls=gpu_calls,
                cpu_loads=cpu_loads, cpu_calls=cpu_calls)


class OomRecoveryTests(unittest.TestCase):
    def test_short_oom_cleared_by_longer_gpu_retry_needs_no_cpu(self):
        run = stop_session([9.0], lambda n: OOM if n == 1 else 'terug op de gpu')
        self.assertEqual(run['cpu_loads'], [])
        self.assertEqual(len(run['gpu_calls']), 2)
        retry = lines(run['out'], 'oom_recovery')
        self.assertEqual([(e['route'], e['ok']) for e in retry], [('gpu_retry', True)])
        self.assertEqual(retry[0]['wait_s'], run['ns']['OomRecovery'].GPU_WAIT_S)
        self.assertTrue(any('terug op de gpu' in p for p in run['pasted']))

    def test_lasting_oom_decodes_every_window_once_on_the_cpu(self):
        # 25 s of failed audio: three 10 s windows; one GPU retry in total, then CPU only.
        run = stop_session([25.0], lambda n: OOM)
        self.assertEqual(len(run['gpu_calls']), 2)
        self.assertEqual(run['cpu_loads'], [1])
        self.assertEqual(run['cpu_calls'], [True, True, True])
        events = lines(run['out'], 'oom_recovery')
        self.assertEqual([(e['route'], e['ok']) for e in events],
                         [('gpu_retry', False)] + [('cpu', True)] * 3)
        self.assertEqual(len(lines(run['out'], 'correction_delivery')), 1)
        self.assertTrue(any('hersteld op de processor' in p for p in run['pasted']))
        summary = lines(run['out'], 'oom_recovery_summary')[0]
        self.assertEqual(summary['cpu_ok'], 3)
        self.assertFalse(summary['cpu_still_running'])
        # Model released after the stop check, audio released as before.
        self.assertEqual(len(lines(run['out'], 'oom_recovery_released')), 1)
        self.assertEqual(len(lines(run['out'], 'audio_released')), 1)
        self.assertEqual(run['ledger'].frames, [])

    def test_wait_is_bounded_and_abandoned_window_is_reported(self):
        ns = load_logic()
        ns['diagnostic'] = coverage.diagnostic
        cls = ns['OomRecovery']
        cls.BUDGET_S, cls.GPU_WAIT_S = 0.5, 0.05
        released, loads = threading.Event(), []

        def load():
            loads.append(1)
            return types.SimpleNamespace(
                transcribe_chunk=lambda audio, recovery=False: time.sleep(1.5) or 'te laat')
        recovery = cls(Mock(side_effect=OOM), load)
        original_release = recovery._release
        recovery._release = lambda: (original_release(), released.set())
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            t0 = time.monotonic()
            with self.assertRaises(cls.BudgetExceeded):
                recovery.decode(np.zeros(10 * R, np.float32))
            elapsed = time.monotonic() - t0
            # The next window is refused at once: one CPU decode at a time.
            with self.assertRaises(cls.BudgetExceeded):
                recovery.decode(np.zeros(10 * R, np.float32))
            recovery.close()
            self.assertTrue(released.wait(5))  # released when the late thread ends
        self.assertLess(elapsed, 0.5 + 0.2)
        self.assertEqual(loads, [1])
        reasons = [e.get('reason') for e in lines(out, 'oom_recovery') if e['route'] == 'cpu']
        self.assertEqual(reasons, ['deadline', 'budget'])
        self.assertTrue(lines(out, 'oom_recovery_summary')[0]['cpu_still_running'])
        self.assertIsNone(recovery.cpu_worker)

    def test_budget_exhaustion_leaves_range_unresolved_and_visible(self):
        run = stop_session([9.0], lambda n: OOM, budget_s=0.0)
        self.assertEqual(run['cpu_loads'], [])
        self.assertEqual(len(lines(run['out'], 'recovery_failed')), 1)
        self.assertIn('niet terug te vinden', run['pasted'][-1])
        self.assertEqual(run['ledger'].frames, [])

    def test_non_oom_error_gets_no_retry_and_no_cpu(self):
        ns = load_logic()
        ns['diagnostic'] = coverage.diagnostic
        gpu, loader, sleep = Mock(side_effect=RuntimeError('decoder error')), Mock(), Mock()
        recovery = ns['OomRecovery'](gpu, loader, sleep=sleep)
        with self.assertRaisesRegex(RuntimeError, 'decoder error'):
            recovery.decode(np.zeros(R, np.float32))
        gpu.assert_called_once()
        loader.assert_not_called()
        sleep.assert_not_called()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            recovery.close()
        self.assertEqual(out.getvalue(), '')

    def test_no_oom_means_no_extra_work(self):
        run = stop_session([9.0], lambda n: 'gewoon herkend')
        self.assertEqual(len(run['gpu_calls']), 1)
        self.assertEqual(run['cpu_loads'], [])
        self.assertFalse(lines(run['out'], 'oom_recovery'))
        self.assertFalse(lines(run['out'], 'oom_recovery_summary'))

    def test_slow_cpu_window_is_not_started_when_it_cannot_fit(self):
        ns = load_logic()
        ns['diagnostic'] = coverage.diagnostic
        now = [0.0]
        loader = Mock(return_value=types.SimpleNamespace(
            transcribe_chunk=lambda audio, recovery=False: now.__setitem__(0, now[0] + 25) or 'x'))
        recovery = ns['OomRecovery'](Mock(side_effect=OOM), loader, clock=lambda: now[0],
                                     sleep=lambda s: None)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(recovery.decode(np.zeros(R, np.float32)), 'x')    # t=25 of 60
            self.assertEqual(recovery.decode(np.zeros(R, np.float32)), 'x')    # t=50, 10 s left
            with self.assertRaises(recovery.BudgetExceeded):
                recovery.decode(np.zeros(R, np.float32))                       # needs 25 s
            recovery.close()
        self.assertEqual(loader.call_count, 1)
        self.assertEqual(lines(out, 'oom_recovery_summary')[0]['budget_refused'], 1)

    def test_cpu_worker_is_cpu_int8_local_only_with_the_session_settings(self):
        ns = load_logic()
        calls = []
        ns['TranscriptionWorker'] = lambda *a, **kw: calls.append((a, kw)) or 'worker'
        app = ns['App'].__new__(ns['App'])
        app.args = types.SimpleNamespace(model_name='large-v3')
        app.transcription_worker = types.SimpleNamespace(
            language='en', vad_filter=False, filter_hallucinations=True,
            space_rule_runs=True, ellipsis_style='space')
        self.assertEqual(app._load_cpu_recovery_worker(), 'worker')
        (args, kw), = calls
        self.assertEqual(args, ('large-v3', 'cpu', 'int8', 'en'))
        self.assertEqual(kw['context_chars'], 0)
        self.assertEqual(kw['model_options'], dict(cpu_threads=ns['OomRecovery'].CPU_THREADS,
                                                    local_files_only=True))


def frames(samples, size=1024):
    data = int16(samples).tobytes() if samples.dtype != np.int16 else samples.tobytes()
    return [data[i:i + size * 2] for i in range(0, len(data), size * 2)]


class DigitalSilenceTests(unittest.TestCase):
    def capture(self, audio, callback=None):
        ledger = coverage.CoverageLedger()
        ledger.on_digital_silence = callback
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            for frame in frames(audio):
                ledger.capture(frame)
        return ledger, out

    def test_only_zeros_alerts_once_after_two_seconds(self):
        alert = Mock()
        ledger, out = self.capture(np.zeros(10 * R, np.int16), alert)
        alert.assert_called_once()
        event, = lines(out, 'digital_silence')
        self.assertEqual(set(event), {'timestamp', 'kind', 'start_s', 'zero_run_s', 'threshold_s'})
        self.assertEqual(event['start_s'], 0.0)
        self.assertGreaterEqual(event['zero_run_s'], coverage.DIGITAL_SILENCE_S)
        self.assertLess(event['zero_run_s'], coverage.DIGITAL_SILENCE_S + 0.1)
        self.assertTrue(ledger.digital_silence)

    def test_ordinary_silence_above_zero_never_alerts(self):
        rng = np.random.default_rng(3)
        quiet_room = noise(12, -62)
        # Dither of about one LSB: many single zeros, never a long run.
        dither = rng.integers(-1, 2, 12 * R).astype(np.int16)
        for audio in (quiet_room, dither, speech(12, -48)):
            alert = Mock()
            ledger, out = self.capture(audio, alert)
            alert.assert_not_called()
            self.assertFalse(lines(out, 'digital_silence'))
            self.assertFalse(ledger.digital_silence)

    def test_short_zero_runs_like_device_start_do_not_alert(self):
        audio = np.concatenate([np.zeros(int(0.21 * R), np.int16), int16(noise(3)),
                                np.zeros(int(1.9 * R), np.int16), int16(speech(2))])
        alert = Mock()
        ledger, out = self.capture(audio, alert)
        alert.assert_not_called()
        self.assertAlmostEqual(ledger.longest_zero_run / R, 1.9, delta=0.07)

    def test_dropout_mid_session_alerts_with_its_start_across_frames(self):
        head = int16(speech(3.3))           # does not end on a frame boundary
        audio = np.concatenate([head, np.zeros(3 * R, np.int16), int16(speech(1))])
        alert = Mock()
        _, out = self.capture(audio, alert)
        alert.assert_called_once()
        self.assertAlmostEqual(lines(out, 'digital_silence')[0]['start_s'], 3.3, delta=0.01)

    def test_failing_alert_never_stops_capture(self):
        ledger, out = self.capture(np.zeros(4 * R, np.int16), Mock(side_effect=OSError('no sound')))
        self.assertEqual(ledger.samples, 4 * R)
        self.assertEqual(lines(out, 'digital_silence_alert_failed')[0]['error'], 'OSError')

    def test_stop_summary_records_it_without_pasting_anything(self):
        ledger, out = self.capture(np.zeros(10 * R, np.int16))
        pasted = []
        with contextlib.redirect_stdout(out):
            ledger.outcome(0, 10 * R)  # nothing decoded from zeros
            ledger.finish(lambda a: '', lambda t: pasted.append(t) or dict(ok=True))
        summary, = lines(out, 'stop_summary')
        self.assertTrue(summary['digital_silence'])
        self.assertEqual(summary['longest_zero_run_s'], 10.0)
        self.assertEqual(pasted, [])

    def test_app_plays_three_equal_tones_off_the_recording_thread(self):
        ns = load_logic()
        sounds = ns['rewrite_sounds'](ns['tone'](600, 50))
        self.assertIn('mic_silent', sounds)
        others = [v for k, v in sounds.items() if k != 'mic_silent']
        self.assertTrue(all(len(v) != len(sounds['mic_silent']) or not np.array_equal(v, sounds['mic_silent'])
                            for v in others))
        app = ns['App'].__new__(ns['App'])
        played = []
        done = threading.Event()
        app.beep = lambda k, wait=True: (played.append((k, wait, threading.current_thread())), done.set())
        app._mic_silent_alert()
        self.assertTrue(done.wait(2))
        self.assertEqual(played[0][:2], ('mic_silent', False))
        self.assertIsNot(played[0][2], threading.current_thread())


if __name__ == '__main__':
    unittest.main()
