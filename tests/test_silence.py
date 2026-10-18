"""WP7: silence hallucinations, meaningful corrections, per-range audio diagnosis.

Literal strings come from the 17/10 19:32-19:35 journal. No model, mic or desktop.
"""
import contextlib
import io
import json
import queue
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import coverage_ledger as coverage
from test_recovery import load_logic, worker, logic

R = coverage.RATE
rng = np.random.default_rng(7)


def noise(seconds, dbfs=-62.0):
    """Idle receiver / quiet room: Gaussian noise at the given RMS level."""
    return (rng.standard_normal(int(seconds * R)) * 10 ** (dbfs / 20)).astype(np.float32)


def speech(seconds, dbfs=-24.0):
    """Stand-in for voice energy: a modulated tone at a speaking level."""
    t = np.arange(int(seconds * R)) / R
    return (np.sin(2 * np.pi * 180 * t) * np.sqrt(2) * 10 ** (dbfs / 20)).astype(np.float32)


def int16(audio):
    return np.clip(np.round(audio * 32768), -32768, 32767).astype(np.int16)


def lines(out, kind):
    return [json.loads(l.removeprefix('DCT-COVERAGE ')) for l in out.getvalue().splitlines()
            if l.startswith('DCT-COVERAGE ') and '"kind": "%s"' % kind in l]


class PhraseFilterTests(unittest.TestCase):
    def test_sound_captions_and_credits_are_artefacts(self):
        bad = ['ZANG EN MUZIEK', 'Zang en muziek.', '[MUZIEK]', '(Muziek speelt)', '[Gelach]',
               '*zucht*', '\u266a\u266a', 'MUZIEK. ZANG.', 'Muziek en applaus', 'TV Gelderland 2023',
               'Ondertiteling door de Amara.org gemeenschap',
               'Ondertitels ingevoegd door de Amara.org gemeenschap',
               'Subtitles by the Amara.org community', '* * *', '. . .']
        for text in bad:
            self.assertTrue(logic['is_hallucination'](text), text)

    def test_real_short_speech_survives(self):
        good = ['Er is een, ja.', 'Ja.', 'Nee.', 'OK', 'NATO', 'Zang en muziek waren goed vandaag.',
                'Dat is muziek en dank u wel.', 'Ondertiteling is belangrijk.',
                '(zie bijlage drie voor de details)', 'In deze momenten vind ik het zo leuk dat ze er zijn.']
        for text in good:
            self.assertFalse(logic['is_hallucination'](text), text)
            self.assertEqual(logic['sanitize_transcript'](text), text)


class AudioStatsTests(unittest.TestCase):
    def test_levels_and_digital_silence(self):
        stats = coverage.audio_stats(int16(speech(2.0, -24.0)))
        self.assertAlmostEqual(stats['rms_dbfs'], -24.0, delta=0.3)
        self.assertGreater(stats['voiced_s'], 1.9)
        self.assertLess(stats['zero_fraction'], 0.01)  # only zero crossings
        # A wireless dropout: 6 s of exact zeros inside 12.3 s of speech.
        dropout = np.concatenate([speech(3.0), np.zeros(6 * R, np.float32), speech(3.3)])
        stats = coverage.audio_stats(int16(dropout))
        self.assertAlmostEqual(stats['zero_fraction'], 6 / 12.3, delta=0.01)
        self.assertEqual(stats['longest_zero_s'], 6.0)
        self.assertFalse(coverage.silent_audio(stats))
        silent = coverage.audio_stats(np.zeros(10 * R, np.int16))
        self.assertEqual((silent['rms_dbfs'], silent['peak_dbfs'], silent['zero_fraction']), (-120.0, -120.0, 1.0))
        json.dumps(silent, allow_nan=False)  # No -inf in the journal.

    def test_silence_threshold_spares_one_short_word(self):
        self.assertTrue(coverage.silent_audio(coverage.audio_stats(noise(10.5))))
        click = noise(10.5)
        click[R:R + 16] = 0.9  # 1 ms click lifts one 32 ms frame, not five
        self.assertTrue(coverage.silent_audio(coverage.audio_stats(click)))
        ja = noise(10.5)
        ja[5 * R:5 * R + int(0.3 * R)] += speech(0.3, -30.0)  # one quiet "ja"
        self.assertFalse(coverage.silent_audio(coverage.audio_stats(ja)))
        self.assertFalse(coverage.silent_audio(coverage.audio_stats(noise(0.5, -120))))  # too short to judge

    def test_ledger_slices_frames_without_joining_wrongly(self):
        ledger = coverage.CoverageLedger()
        audio = int16(np.concatenate([noise(1.0), speech(1.0)]))
        for part in np.array_split(audio, 7):
            ledger.capture(part.tobytes())
        np.testing.assert_array_equal(ledger.audio(12345, 23456), audio[12345:23456])
        with contextlib.redirect_stdout(io.StringIO()):
            quiet = ledger.outcome(0, R)
            loud = ledger.outcome(R, 2 * R, 'woorden')
        self.assertTrue(coverage.silent_audio(quiet['audio']))
        self.assertFalse(coverage.silent_audio(loud['audio']))


class PipelineSilenceTests(unittest.TestCase):
    def run_session(self, chunks, outputs):
        """chunks: float32 arrays queued as consecutive ranges; outputs: model texts."""
        app = load_logic()['App'].__new__(load_logic()['App'])
        ledger = coverage.CoverageLedger()
        for chunk in chunks:
            ledger.capture(int16(chunk).tobytes())
        app.ledger = ledger
        app.args = types.SimpleNamespace(paste_min_s=0)
        app.transcription_worker = worker(outputs)
        app.transcription_queue, app.typing_queue = queue.Queue(), queue.Queue()
        start = 0
        for chunk in chunks:
            app.transcription_queue.put((chunk, True, start))
            start += len(chunk)
        app.transcription_queue.put(None)
        pasted = []
        app.replayer = types.SimpleNamespace(type_text=lambda t: pasted.append(t) or dict(
            ok=True, path='copyq', paste='shortcut_sent_not_acknowledged'))
        app.recorder = None
        app.active = True
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            app._transcription_loop()
            app._typing_loop()
        return pasted, ledger, out, app

    def test_journal_case_zang_en_muziek_on_silence_is_not_pasted(self):
        # 19:32:56 auto-stop after 10.5 s of silence; first pass and retry both said it.
        pasted, ledger, out, _ = self.run_session([noise(10.496)], ['ZANG EN MUZIEK', 'ZANG EN MUZIEK'])
        self.assertEqual(pasted, [])
        entry = ledger.entries[0]
        self.assertEqual((entry['outcome'], entry['raw']), ('artifact', 'ZANG EN MUZIEK'))
        self.assertTrue(entry['recovery']['silent'])  # stop check did not decode silence again
        self.assertEqual(lines(out, 'stop_summary')[0]['silent_ranges'], 1)

    def test_unknown_hallucination_on_silence_dropped_by_audio_gate(self):
        pasted, ledger, out, app = self.run_session([noise(10.5)], ['Bedankt allemaal.', 'Bedankt allemaal.'])
        self.assertEqual(pasted, [])
        self.assertEqual(ledger.entries[0]['discard_reason'], 'silent_audio')
        self.assertEqual(ledger.entries[0]['raw'], 'Bedankt allemaal.')
        self.assertIn('[filter] Dropped text on silent audio', out.getvalue())
        self.assertEqual(app.transcription_worker._context, '')

    def test_short_real_speech_with_status_line(self):
        with patch.object(coverage, 'STOP_STATUS', True):
            ja = noise(2.0)
            ja[R:R + int(0.3 * R)] += speech(0.3, -30.0)
            pasted, ledger, _, _ = self.run_session([ja], ['Ja.'])
        self.assertEqual(pasted, ['Ja.'])  # Clean session: no check line.

    def test_very_quiet_short_word_is_still_pasted(self):
        # Below the voiced threshold, but peaks near speech level and the text
        # is not suspect for 2 s: the gate must not eat it.
        pasted, ledger, _, _ = self.run_session([speech(2.0, -48.0)], ['Ja.'])
        self.assertTrue(coverage.silent_audio(ledger.entries[0]['audio']))
        self.assertEqual(pasted[0], 'Ja.')

    def test_short_unknown_text_on_dead_silence_dropped(self):
        pasted, ledger, _, _ = self.run_session([noise(3.5, -70.0)], ['Hallo.'])
        self.assertEqual(pasted, [])
        self.assertEqual(ledger.entries[0]['discard_reason'], 'silent_audio')


class CorrectionTests(unittest.TestCase):
    def setUp(self):
        patcher = patch.object(coverage, 'STOP_STATUS', True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def session(self, texts, seconds=11.648):
        """One or more delivered ranges of audible audio (last one is the suspect)."""
        ledger = coverage.CoverageLedger()
        ledger.capture(int16(speech(seconds * len(texts))).tobytes())
        n = int(seconds * R)
        with contextlib.redirect_stdout(io.StringIO()):
            for i, text in enumerate(texts):
                e = ledger.outcome(i * n, (i + 1) * n, text, text)
                if text:
                    ledger.delivered(e, dict(ok=True, path='copyq'))
        return ledger

    def finish(self, ledger, recovered):
        pasted = []
        out = io.StringIO()
        # 11.6 s is checked in two <=10 s windows; the words sit in the first.
        windows = iter([recovered])
        with contextlib.redirect_stdout(out):
            corrections = ledger.finish(lambda audio: next(windows, ''),
                                        lambda t: pasted.append(t) or dict(ok=True),
                                        is_artifact=logic['is_hallucination'])
        return corrections, pasted, out

    def test_journal_case_repeat_with_pause_marks_is_not_a_correction(self):
        # 19:34:29: 'Er is een, ja.' on 11.6 s; stop check found 'dat is een ja * * *'.
        for recovered in ('dat is een ja * * *', 'dat is een ja ***', 'Er is een, ja.'):
            ledger = self.session(['Er is een, ja.'])
            corrections, pasted, out = self.finish(ledger, recovered)
            self.assertEqual(corrections, [], recovered)
            self.assertEqual(pasted, [])  # Duplicate only: nothing new, no check line.
            record = lines(out, 'recovery_outcome')[0]
            self.assertTrue(record['duplicate'])
            self.assertNotIn('*', record['text'])

    def test_repeat_of_neighbour_text_is_not_a_correction(self):
        ledger = self.session(['Ik zie dat ik iets gemist heb maar dan de inhoud zelf komt niet mee.', 'Weinig.'])
        corrections, _, _ = self.finish(ledger, 'de inhoud zelf komt niet mee. . .')
        self.assertEqual(corrections, [])

    def test_only_pause_marks_or_artefact_left_is_unresolved(self):
        for recovered in ('* * *', '. . .', 'ZANG EN MUZIEK', '[MUZIEK]', ''):
            ledger = self.session(['Er is een, ja.'])
            corrections, pasted, out = self.finish(ledger, recovered)
            self.assertEqual(corrections, [], recovered)
            self.assertRegex(pasted[0], r'^\[Controle: 1 stuk nagekeken, 1 niet terug te vinden, ')
            self.assertEqual(len(lines(out, 'unresolved')), 1)

    def test_new_words_still_pasted_clean(self):
        ledger = self.session(['Er is een, ja.'])
        corrections, pasted, _ = self.finish(
            ledger, 'Er is een nieuwe pomp besteld bij de leverancier in Gent, ja * * *')
        self.assertEqual(corrections, ['[Correctie: in deze opname is waarschijnlijk dit gemist: '
                                       '"Er is een nieuwe pomp besteld bij de leverancier in Gent, ja"]'])
        self.assertEqual(pasted[0], corrections[0])
        self.assertRegex(pasted[1], r'1 correctie')

    def test_clean_correction_and_duplicate_rule(self):
        clean = coverage.clean_correction
        self.assertEqual(clean('maar ik zie niet een. . .'), 'maar ik zie niet een')
        self.assertEqual(clean('een \u2026 twee --- drie ***'), 'een twee drie')
        self.assertEqual(clean('* * *'), '')
        dup = coverage.repeats_delivered
        self.assertTrue(dup('dat is een ja', 'Er is een, ja.'))
        self.assertTrue(dup('ja', 'Er is een, ja.'))
        # Shared small words alone never make a duplicate.
        self.assertFalse(dup('de pomp is stuk', 'Het is de vraag of de klant komt.'))
        self.assertFalse(dup('iets', ''))


if __name__ == '__main__':
    unittest.main()
