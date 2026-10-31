"""WP12 offline regressions: invented phrases, synthetic levels, mocked GPU query."""
import contextlib
import io
import os
import subprocess
import unittest
from unittest.mock import patch

import numpy as np

import coverage_ledger as coverage
from test_recovery import logic, worker
import test_silence
from test_silence import speech, lines


class FilterOOMTests(unittest.TestCase):
    def session(self, audio, text):
        return test_silence.PipelineSilenceTests().run_session([audio], [text, text])

    def test_standalone_thanks_dropped_at_every_level_and_duration(self):
        for phrase in logic['THANK_PHRASES']:
            text = phrase.upper() + '!'
            for duration, level in ((0.344, -48), (1.544, -24), (2.0, -60), (10.5, -46)):
                with self.subTest(phrase=phrase, duration=duration, level=level):
                    pasted, ledger, _, _ = self.session(speech(duration, level), text)
                    self.assertEqual(pasted, [])
                    self.assertEqual(ledger.entries[0]['discard_reason'],
                                     'known_silence_artifact')

    def test_thanks_dropped_without_audio_but_filter_can_be_disabled(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(logic['sanitize_transcript']('Dank u wel.'), '')
        self.assertEqual(logic['transcript_discard_reason']('Dank u wel.', None),
                         'known_silence_artifact')
        self.assertIsNone(logic['transcript_discard_reason']('Dank u wel.', None, False))
        self.assertEqual(logic['sanitize_transcript']('Dank u wel.', drop_phrases=False),
                         'Dank u wel.')
        self.assertEqual(self.session(np.zeros(24000, np.float32), 'Dank u wel.')[0], [])

    def test_thanks_inside_longer_sentence_survives(self):
        for text in ('Dank je wel voor de hulp met de blauwe deur.',
                     'Ik zeg dank u wel en ga verder.', 'Thank you for watching the clock.'):
            self.assertFalse(logic['is_standalone_thanks'](text))
            self.assertIsNone(logic['transcript_discard_reason'](text, None))
            self.assertEqual(logic['sanitize_transcript'](text), text)
            self.assertEqual(self.session(speech(2), text)[0], [text])

    def test_recovery_drops_standalone_thanks_but_keeps_longer_sentence(self):
        for text in ('Dank u wel.', 'Dankjewel.'):
            _, _, out, _ = test_silence.PipelineSilenceTests().run_session(
                [speech(9.5)], ['', '', text])
            self.assertFalse(lines(out, 'correction_delivery'))
            self.assertTrue(lines(out, 'recovery_discarded'))
        text = 'Dank je wel voor de hulp met de blauwe deur.'
        _, _, out, _ = test_silence.PipelineSilenceTests().run_session(
            [speech(9.5)], ['', '', text])
        self.assertEqual(len(lines(out, 'correction_delivery')), 1)
        for caption in ('ZANG EN MUZIEK', 'Ondertiteling door de Amara.org gemeenschap'):
            self.assertEqual(self.session(speech(1.5), caption)[0], [])

    def test_short_exact_repeat_on_near_silence_is_rejected(self):
        stats = dict(seconds=2.456, rms_dbfs=-62.5, peak_dbfs=-44.2, voiced_s=0.0)
        text = 'De groene deur sluit. De groene deur sluit.'
        self.assertEqual(logic['transcript_discard_reason'](text, stats),
                         'silent_repeated_phrase')
        # Each conjunct matters: protect quiet real words and deliberate repetition.
        for changed in (dict(rms_dbfs=-51.5, peak_dbfs=-28.5, voiced_s=0.06),
                        dict(rms_dbfs=-55), dict(voiced_s=0.032), dict(seconds=8.0)):
            self.assertIsNone(logic['transcript_discard_reason'](text, dict(stats, **changed)))
        for real in ('De groene deur sluit.', 'Ja. Ja.',
                     'De groene deur sluit. De blauwe deur sluit.'):
            self.assertIsNone(logic['transcript_discard_reason'](real, stats))
        # Synthetic noise with a single tiny spike reproduces the measured levels.
        audio = np.full(round(2.456 * coverage.RATE), 10 ** (-62.5 / 20), np.float32)
        audio[0] = 10 ** (-44.2 / 20)
        pasted, ledger, _, _ = self.session(audio, text)
        self.assertEqual(pasted, [])
        self.assertEqual(ledger.entries[0]['discard_reason'], 'silent_repeated_phrase')

    def test_cuda_oom_emits_one_bounded_snapshot_and_no_retry(self):
        obj = worker([RuntimeError('CUDA failed with error out of memory')])
        out = io.StringIO()
        with patch.dict(logic, subprocess=subprocess, os=os, diagnostic=coverage.diagnostic):
            with patch.object(subprocess, 'run', return_value=subprocess.CompletedProcess(
                    [], 0, stdout='gpu-1, 123, other-worker, 7000 MiB\n', stderr='')) as run:
                with contextlib.redirect_stdout(out), self.assertRaises(RuntimeError):
                    obj.transcribe_chunk(np.zeros(16000))
        self.assertEqual(len(obj.calls), 1)
        run.assert_called_once()
        self.assertLessEqual(run.call_args.kwargs['timeout'], 2)
        event = lines(out, 'cuda_oom_snapshot')[0]
        self.assertEqual(event['status'], 'ok')
        self.assertIn('7000 MiB', event['compute_apps'])
        self.assertEqual(event['pid'], os.getpid())

    def test_oom_snapshot_failure_is_bounded_and_preserves_original_error(self):
        for failure in (subprocess.TimeoutExpired('nvidia-smi', 1), FileNotFoundError()):
            obj = worker([RuntimeError('CUDA out of memory')])
            out = io.StringIO()
            with patch.dict(logic, subprocess=subprocess, os=os, diagnostic=coverage.diagnostic):
                with patch.object(subprocess, 'run', side_effect=failure) as run:
                    with contextlib.redirect_stdout(out), self.assertRaisesRegex(RuntimeError, 'CUDA'):
                        obj.transcribe_chunk(np.zeros(16000))
            run.assert_called_once()
            snapshots = lines(out, 'cuda_oom_snapshot')
            self.assertEqual(len(snapshots), 1)
            self.assertEqual(snapshots[0]['status'], 'unavailable')

    def test_oom_during_retry_and_recovery_also_logged_once(self):
        for outputs, recovery in ((['***', RuntimeError('CUDA out of memory')], False),
                                  ([RuntimeError('CUDA out of memory')], True)):
            obj = worker(outputs)
            out = io.StringIO()
            with patch.dict(logic, subprocess=subprocess, os=os, diagnostic=coverage.diagnostic):
                with patch.object(subprocess, 'run', return_value=subprocess.CompletedProcess(
                        [], 1, stdout='x' * 9000, stderr='e' * 1000)) as run:
                    with contextlib.redirect_stdout(out):
                        if recovery:
                            with self.assertRaises(RuntimeError):
                                obj.transcribe_chunk(np.zeros(16000), recovery=True)
                        else:
                            obj.transcribe_chunk(np.zeros(16000))
            run.assert_called_once()
            event = lines(out, 'cuda_oom_snapshot')[0]
            self.assertEqual(event['status'], 'query_failed')
            self.assertEqual(len(event['compute_apps']), 8192)
            self.assertEqual(len(event['stderr']), 512)
            self.assertTrue(event['truncated'])

    def test_non_oom_does_not_query_gpu(self):
        obj = worker([RuntimeError('decoder error')])
        with patch.dict(logic, subprocess=subprocess, os=os, diagnostic=coverage.diagnostic):
            with patch.object(subprocess, 'run') as run:
                with self.assertRaises(RuntimeError):
                    obj.transcribe_chunk(np.zeros(16000))
                run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
