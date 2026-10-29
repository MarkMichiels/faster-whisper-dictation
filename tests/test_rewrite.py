"""Rewrite mode, offline: fake key events, fake xclip/probe, fake LLM, fake paste.

Never touches the real clipboard, CopyQ, X selections or any network endpoint.
"""
import ast
import contextlib
import io
import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import context_capture  # noqa: E402
import coverage_ledger as coverage  # noqa: E402
import rewrite  # noqa: E402
from test_recovery import load_logic, worker  # noqa: E402

R = coverage.RATE
ENDPOINT = 'https://llm.invalid/v1/chat/completions'


def app_logic():
    logic = load_logic()
    logic.update(rewrite=rewrite, context_capture=context_capture,
                 HELD_FOR_REWRITE=coverage.HELD_FOR_REWRITE, diagnostic=coverage.diagnostic,
                 CoverageLedger=coverage.CoverageLedger, MAX_SECONDS=coverage.MAX_SECONDS, RATE=R)
    return logic


def listener_class():
    tree = ast.parse((ROOT / 'dictation.py').read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MultiTapKeyListener')
    fake_keyboard = types.SimpleNamespace(Key=types.SimpleNamespace(cmd_r=key('cmd_r', 65516)))
    ns = dict(keyboard=fake_keyboard, time=time)
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'listener', 'exec'), ns)
    return ns['MultiTapKeyListener']


def key(name, vk):
    return types.SimpleNamespace(name=name, vk=vk)


CTRL_R, CTRL = key('ctrl_r', 65508), key('ctrl', 65507)
ALT, ALT_GR = key('alt', 65513), key('alt_gr', 65027)
SHIFT = key('shift', 65505)
META_L = key(None, 0xffe7)


def llm_answer(text, field='tekst', finish='stop'):
    return {'choices': [{'message': {'content': json.dumps({field: text})}, 'finish_reason': finish}],
            'usage': {'prompt_tokens': 700, 'completion_tokens': 60, 'cost': 0.0003},
            'provider': 'FakeProvider'}


def config(**kw):
    kw.setdefault('endpoint', ENDPOINT)
    kw.setdefault('model', 'fake/model')
    return rewrite.RewriteConfig(**kw)


class FakeReplayer:
    def __init__(self, events, windows=('win-start', 'win-stop')):
        self.events = events
        self.windows = list(windows)
        self.target_wid = None

    def window_id(self):
        return self.windows.pop(0) if self.windows else 'win-later'

    def add_to_history(self, text):
        self.events.append(('history', text))
        return 0

    def type_text(self, text, echo=True):
        if echo:
            print(text)
        self.events.append(('paste', text, self.target_wid))
        return dict(ok=True, path='copyq', paste='shortcut_sent_not_acknowledged')


RAW_PARTS = ['Eerste deel van het dictaat over de pomp.', 'Tweede deel over de levering van vrijdag.']


class RewriteTestCase(unittest.TestCase):
    def pipeline(self, answer=None, outputs=None, context=None, cfg=None, http=None,
                 paste_wid='win-stop', language='nl', chunks=2):
        """Run one rewrite session through the real App loops; return (events, journal, session, bodies)."""
        logic = app_logic()
        events, bodies = [], []
        outputs = outputs or list(RAW_PARTS)
        app = logic['App'].__new__(logic['App'])
        app.args = types.SimpleNamespace(paste_min_s=0)
        app.transcription_worker = worker(outputs)
        app.replayer = FakeReplayer(events)
        app.beep = lambda k, wait=True: events.append(('beep', k))
        app.rewrite_config = cfg if cfg is not None else config()
        app.rewrite_reason = None
        app.active = False
        ledger = coverage.CoverageLedger()
        ledger.capture(np.full(chunks * 10 * R, 8000, dtype=np.int16).tobytes())
        session = app.Session(queue.Queue(), queue.Queue(), ledger, types.SimpleNamespace(_raw_frames=[]),
                              None, language, 'win-start', rewrite=True)
        session.context = context
        session.paste_wid = paste_wid
        app.session = session
        for i in range(chunks):
            session.transcription_queue.put((np.full(10 * R, 0.1, dtype=np.float32), True, i * 10 * R))
        session.transcription_queue.put(None)

        def fake_http(cfg, body, timeout):
            bodies.append(body)
            events.append(('llm',))
            if http is not None:
                return http(cfg, body, timeout)
            return llm_answer(answer if answer is not None else 'Herschreven tekst over pomp en levering.')
        out = io.StringIO()
        with patch.object(rewrite, 'http_call', fake_http), contextlib.redirect_stdout(out):
            app._transcription_loop(session)
            app._typing_loop(session)
        return events, out.getvalue(), session, bodies


class KeyTests(unittest.TestCase):
    """Test 1: Left-Alt held at the second tap selects rewrite mode."""

    def tap_twice(self, listener, held=()):
        listener.last_press_time = 0  # a new gesture, well apart from the previous one
        for k in held:
            listener.on_press(k)
        for _ in range(2):
            listener.on_press(CTRL_R)
            listener.on_press(CTRL)  # pynput's generic duplicate event
            listener.on_release(CTRL_R)
        for k in held:
            listener.on_release(k)

    def test_alt_shift_combinations(self):
        cls = listener_class()
        calls = []
        listener = cls(lambda *a: calls.append(a), lambda: calls.append('stop'),
                       key=CTRL_R, default_language='nl', shift_language='en')
        self.tap_twice(listener)
        self.tap_twice(listener, (ALT,))
        self.tap_twice(listener, (SHIFT, ALT))
        self.tap_twice(listener, (SHIFT,))
        self.tap_twice(listener)  # Alt released: no rewrite remembered
        self.tap_twice(listener, (ALT_GR,))  # Right-Alt/AltGr is not the rewrite key
        starts = [c[:2] for c in calls if c != 'stop']
        self.assertEqual(starts, [('nl', False), ('nl', True), ('en', True), ('en', False), ('nl', False),
                                  ('nl', False)])
        keys = [c[2] for c in calls if c != 'stop']
        self.assertEqual({k: v for k, v in keys[2].items() if not k.endswith('_s')},
                         {'shift': True, 'alt': True})
        self.assertIn('shift_held_s', keys[2])
        self.assertIn('alt_released_s_ago', keys[3])

    def test_alt_after_shift_arrives_as_meta_l(self):
        # X reports Left-Alt as Meta_L (no pynput name) while Shift is held;
        # measured with pynput and this keymap on a private X display.
        cls = listener_class()
        calls = []
        listener = cls(lambda *a: calls.append(a), lambda: None,
                       key=CTRL_R, default_language='nl', shift_language='en')
        self.tap_twice(listener, (SHIFT, META_L))  # Shift first, then Alt
        self.assertEqual(calls[-1][:2], ('en', True))
        self.assertTrue(calls[-1][2].get('alt_as_meta'))
        self.assertFalse(listener.alt_held)
        # Alt first, Shift, then Alt released first: the release is Meta_L too.
        listener.on_press(ALT)
        listener.on_press(SHIFT)
        listener.on_release(META_L)
        listener.on_release(SHIFT)
        self.assertFalse(listener.alt_held)  # never stuck in rewrite mode
        self.tap_twice(listener)
        self.assertEqual(calls[-1][:2], ('nl', False))

    def test_alt_tracked_despite_generic_skip(self):
        cls = listener_class()
        listener = cls(lambda *a: None, lambda: None, key=CTRL_R, default_language='nl')
        listener.on_press(ALT)  # name 'alt' is also in the generic skip list
        self.assertTrue(listener.alt_held)
        listener.on_release(ALT)
        self.assertFalse(listener.alt_held)

    def test_session_language_selects_prompt_block(self):
        cfg = config()
        nl, field_nl = rewrite.build_messages('een dictaat', 'nl', None, cfg)
        en, field_en = rewrite.build_messages('a dictation', 'en', None, cfg)
        self.assertIn('Belgisch-Nederlands', nl[0]['content'])
        self.assertTrue(en[0]['content'].startswith('Task: rewrite'))
        self.assertNotIn('Belgisch', en[0]['content'])
        self.assertEqual((field_nl, field_en), ('tekst', 'text'))
        self.assertIn('<dictaat>\neen dictaat\n</dictaat>', nl[1]['content'])
        self.assertIn('<dictation>\na dictation\n</dictation>', en[1]['content'])
        self.assertIn('Er is geen context.', nl[1]['content'])


class FlowTests(RewriteTestCase):
    def test_no_paste_during_session_one_paste_at_end(self):
        """Test 2 and 5: nothing pasted while dictating; raw to CopyQ before the LLM; one paste."""
        events, journal, session, bodies = self.pipeline()
        kinds = [e[0] for e in events]
        self.assertEqual(kinds, ['history', 'llm', 'paste', 'beep'])
        raw = ' '.join(RAW_PARTS)
        self.assertEqual(events[0], ('history', raw))
        self.assertEqual(events[2][1], 'Herschreven tekst over pomp en levering.')
        self.assertEqual(events[3], ('beep', 'rewrite_done'))
        self.assertIn('"rewrite_outcome"', journal)
        # Held ranges are delivered (not degraded) and not re-recovered at stop.
        self.assertNotIn('delivery_degraded', journal)
        summary = json.loads(next(l for l in journal.splitlines() if '"stop_summary"' in l)
                             .removeprefix('DCT-COVERAGE '))
        self.assertEqual((summary['suspects'], summary['original_delivery_failures']), (0, 0))
        self.assertEqual(session.ledger.frames, [])
        self.assertTrue(session.done.is_set())

    def test_recovered_gap_in_place_for_llm(self):
        """Test 3 (pipeline): a decoder error recovered at stop lands in place, no [Correctie]."""
        outputs = [RuntimeError('OOM'), 'Tweede deel van het dictaat.', 'Eerste deel teruggevonden bij stop.']
        events, journal, session, bodies = self.pipeline(outputs=outputs)
        prompt = bodies[0]['messages'][1]['content']
        self.assertIn('<dictaat>\nEerste deel teruggevonden bij stop. Tweede deel van het dictaat.\n</dictaat>',
                      prompt)
        self.assertNotIn('[Correctie', prompt)
        self.assertEqual(events[0], ('history', 'Eerste deel teruggevonden bij stop. Tweede deel van het dictaat.'))
        self.assertEqual(len([e for e in events if e[0] == 'paste']), 1)

    def test_ordered_text_ledger(self):
        """Test 3 (ledger): inline finish, gap text in place, nothing delivered."""
        obj = coverage.CoverageLedger()
        obj.capture(np.full(3 * R, 8000, dtype=np.int16).tobytes())
        held = dict(ok=True, path=coverage.HELD_FOR_REWRITE)
        obj.delivered(obj.outcome(0, R, 'eerste stuk'), held)
        obj.outcome(R, 2 * R, error='OOM')
        obj.delivered(obj.outcome(2 * R, 3 * R, 'derde stuk.'), held)
        delivered = []
        with contextlib.redirect_stdout(io.StringIO()):
            out = obj.finish(lambda a: 'gemiste woorden', lambda t: delivered.append(t) or held, inline=True)
        self.assertEqual((out, delivered), ([], []))
        self.assertEqual(obj.ordered_text(), 'eerste stuk gemiste woorden derde stuk.')

    def test_unresolved_range_appends_check_line_after_rewrite(self):
        """Design §2 step 8: a range that cannot be found back keeps the [Controle] line."""
        outputs = [RuntimeError('OOM'), 'Tweede deel van het dictaat met genoeg woorden.', '']
        events, journal, session, bodies = self.pipeline(outputs=outputs, answer='Tweede deel, herschreven.')
        paste = next(e for e in events if e[0] == 'paste')
        self.assertTrue(paste[1].startswith('Tweede deel, herschreven.\n\n[Controle: '))
        self.assertIn('niet terug te vinden', paste[1])
        self.assertNotIn('[Controle', bodies[0]['messages'][1]['content'])


class FailureTests(RewriteTestCase):
    """Test 4: every failure pastes the raw text and logs rewrite_failed with a reason."""

    RAW = 'dit is een gedicteerde tekst van precies twaalf woorden voor de controle hier'

    def run_rewrite(self, http, cfg=None):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            result = rewrite.rewrite(self.RAW, 'nl', None, cfg or config(), call=http)
        return result, out.getvalue()

    def assertFallback(self, http, reason, cfg=None):
        result, journal = self.run_rewrite(http, cfg)
        self.assertFalse(result.ok)
        self.assertEqual(result.text, self.RAW)
        self.assertEqual(result.reason, reason)
        record = json.loads(next(l for l in journal.splitlines() if '"rewrite_failed"' in l)
                            .removeprefix('DCT-COVERAGE '))
        self.assertEqual(record['reason'], reason)
        self.assertNotIn('gedicteerde', journal)

    def test_timeout(self):
        def slow(cfg, body, timeout):
            time.sleep(3)
            return llm_answer('te laat')
        t0 = time.monotonic()
        self.assertFallback(slow, 'timeout', config(timeout_s=1))
        self.assertLess(time.monotonic() - t0, 2.5)

    def test_exception_and_http_errors(self):
        def boom(cfg, body, timeout):
            raise ConnectionResetError('reset')
        self.assertFallback(boom, 'exception')

        def http_500(cfg, body, timeout):
            raise rewrite.RewriteError('http_500')
        self.assertFallback(http_500, 'http_500')

        self.assertFallback(lambda *a: {'error': {'code': 402, 'message': 'x'}}, 'provider_error')
        self.assertFallback(lambda *a: {'choices': []}, 'invalid_response')
        self.assertFallback(lambda *a: {'choices': [{'message': {'content': 'geen json'}}]}, 'invalid_json')
        self.assertFallback(lambda *a: llm_answer('Een korte tekst.', finish='length'), 'answer_truncated')

    def test_suspicious_answers(self):
        self.assertFallback(lambda *a: llm_answer('   '), 'empty_answer')
        self.assertFallback(lambda *a: llm_answer(' '.join(['woord'] * 20)), 'answer_too_long')
        self.assertFallback(lambda *a: llm_answer('Ja.'), 'answer_too_short')  # 1/12 < 0.15
        self.assertFallback(lambda *a: llm_answer('Hier is de tekst: een gedicteerde tekst.'), 'answer_has_intro')
        self.assertFallback(lambda *a: llm_answer('Here is the text you wanted.', field='tekst'), 'answer_has_intro')

    def test_quotes_and_fences_stripped(self):
        result, _ = self.run_rewrite(lambda *a: llm_answer('```\nEen nette tekst van zes woorden.\n```'))
        self.assertEqual((result.ok, result.text), (True, 'Een nette tekst van zes woorden.'))
        result, _ = self.run_rewrite(lambda *a: llm_answer('\u201cEen nette tekst van zes woorden.\u201d'))
        self.assertEqual(result.text, 'Een nette tekst van zes woorden.')

    def test_pipeline_failure_pastes_raw_with_error_tone(self):
        def boom(cfg, body, timeout):
            raise rewrite.RewriteError('http_503')
        events, journal, session, bodies = self.pipeline(http=boom, context='Een mail waarop geantwoord wordt.')
        raw = ' '.join(RAW_PARTS)
        self.assertEqual([e for e in events if e[0] == 'paste'], [('paste', raw, 'win-stop')])
        self.assertEqual(events[-1], ('beep', 'rewrite_failed'))
        self.assertIn('"reason": "http_503"', journal)
        self.assertNotIn('[', raw)  # no marker inside the pasted text
        self.assertIsNone(session.context)  # cleared, also after a failure


class RestartTests(unittest.TestCase):
    def test_restart_during_rewrite_is_accepted_and_ordered(self):
        """Test 6: B records while A waits for the LLM; B pastes after A's single paste."""
        logic = app_logic()
        release, in_llm = threading.Event(), threading.Event()
        events, recorders = [], []

        class FakeRecorder:
            SAMPLE_RATE = R
            def __init__(self, transcription_queue, ledger=None, **kw):
                self.queue, self.ledger, self._raw_frames = transcription_queue, ledger, []
                self.name = 'AB'[len(recorders)]
                recorders.append(self)
            def start(self):
                events.append('record ' + self.name)
                self.ledger.capture(np.full(2 * R, 8000, dtype=np.int16).tobytes())
            def stop(self):
                value = 0.25 if self.name == 'A' else 1.0
                for start in (0, R):
                    self.queue.put((np.full(R, value, dtype=np.float32), True, start))
                self.queue.put(None)

        class FakeWorker:
            language = 'nl'
            def reset_context(self):
                pass
            def transcribe_chunk(self, audio, recovery=False):
                name = 'A' if audio[0] < 0.5 else 'B'
                events.append('decode ' + name)
                return 'tekst %s met een paar woorden' % name

        class Replayer(FakeReplayer):
            def type_text(self, text, echo=True):
                events.append('paste %s @%s' % (text.strip(), self.target_wid))
                return dict(ok=True, path='copyq')
            def add_to_history(self, text):
                events.append('history')
                return 0

        def fake_http(cfg, body, timeout):
            in_llm.set()
            release.wait(5)
            return llm_answer('Herschreven A.')

        logic.update(StreamingRecorder=FakeRecorder,
                     context_capture=types.SimpleNamespace(capture=lambda **kw: context_capture.Context(None)))
        app = logic['App'].__new__(logic['App'])
        app.args = types.SimpleNamespace(paste_min_s=0, silence_ms=1000, auto_stop_silence=0,
                                         min_chunk_s=1, max_chunk_s=10, eager_prob=0.5, max_time=60)
        app.active, app.session, app.timer = False, None, None
        app.beep = lambda *a, **kw: None
        app.rewrite_config, app.rewrite_reason = config(timeout_s=10), None
        app.transcription_worker = FakeWorker()
        app.replayer = Replayer([], windows=['win-A', 'win-A-stop', 'win-B'])
        out = io.StringIO()
        with patch.object(rewrite, 'http_call', fake_http), contextlib.redirect_stdout(out):
            self.assertTrue(app.start('nl', True))
            session_a = app.session
            self.assertTrue(app.stop())
            self.assertTrue(in_llm.wait(5))  # A is rewriting
            self.assertTrue(app.start('nl'))  # not refused
            session_b = app.session
            self.assertTrue(app.stop())
            self.assertNotIn('decode B', events)
            release.set()
            self.assertTrue(session_b.done.wait(5))
        self.assertTrue(session_a.done.is_set())
        self.assertIn('"start_while_finishing"', out.getvalue())
        pastes = [e for e in events if e.startswith('paste')]
        self.assertEqual(pastes, ['paste Herschreven A. @win-A-stop',
                                  'paste tekst B met een paar woorden @win-B',
                                  'paste tekst B met een paar woorden @win-B'])
        self.assertLess(events.index('record B'), events.index('paste Herschreven A. @win-A-stop'))
        self.assertFalse(app.active)


class PrivacyTests(RewriteTestCase):
    def test_journal_has_no_rewritten_text_or_context(self):
        """Test 7 and part of 9: no rewritten text or context in any output line."""
        with tempfile.TemporaryDirectory() as tmp:
            style = Path(tmp) / 'style.md'
            style.write_text('Spreek de lezer aan met u.\n')
            terms = Path(tmp) / 'terms.txt'
            terms.write_text('Voorbeeldstad\n')
            cfg = config(style_file=str(style), terms_file=str(terms))
            events, journal, session, bodies = self.pipeline(
                answer='UNIEKE-HERSCHRIJVING over de pomp en de levering.',
                context='CONTEXT-MARKER-XYZ van een andere afzender met een vraag.', cfg=cfg)
        self.assertEqual(events[2][1], 'UNIEKE-HERSCHRIJVING over de pomp en de levering.')
        self.assertNotIn('UNIEKE-HERSCHRIJVING', journal)
        self.assertNotIn('CONTEXT-MARKER-XYZ', journal)
        system, user = (m['content'] for m in bodies[0]['messages'])
        self.assertIn('Voorkeuren van de gebruiker:\nSpreek de lezer aan met u.', system)
        self.assertIn('Schrijf deze termen exact zo:\nVoorbeeldstad', system)
        self.assertTrue(system.endswith('Geef JSON terug met de velden "mail", "aanhef", "tekst" en "afsluiting".'))
        self.assertIn('<context>\nCONTEXT-MARKER-XYZ', user)
        self.assertIn('</context>\n\n<dictaat>\n', user)
        self.assertTrue(user.endswith('</dictaat>'))
        body = bodies[0]
        self.assertEqual(body['provider']['zdr'], True)
        self.assertEqual(body['provider']['data_collection'], 'deny')
        self.assertEqual(body['reasoning'], {'enabled': False})
        self.assertEqual(body['response_format']['json_schema']['schema']['required'],
                         ['mail', 'aanhef', 'tekst', 'afsluiting'])

    def test_without_style_file_prompt_is_plain_v2(self):
        messages, _ = rewrite.build_messages('tekst', 'nl', None, config(style_file='/nonexistent/style.md'))
        self.assertEqual(messages[0]['content'], rewrite.PROMPT_NL)

    def test_word_factor_only_for_long_dictations(self):
        cfg = config(word_factor=0.45)
        short = rewrite.system_prompt('nl', cfg, 100)
        long = rewrite.system_prompt('nl', cfg, 155)
        self.assertEqual(short, rewrite.PROMPT_NL)
        self.assertIn('krimpen. Gebruik hoogstens 70 woorden.', long)
        self.assertIn('third. Use at most 70 words.', rewrite.system_prompt('en', cfg, 155))
        self.assertEqual(rewrite.system_prompt('nl', config(), 155), rewrite.PROMPT_NL)


def mail_answer(fields, finish='stop'):
    return {'choices': [{'message': {'content': json.dumps(fields)}, 'finish_reason': finish}],
            'usage': {'prompt_tokens': 700, 'completion_tokens': 60, 'cost': 0.0003},
            'provider': 'FakeProvider'}


class EmailTests(unittest.TestCase):
    """A clear email gets greeting, body and sign-off; anything else stays body only."""

    RAW = ('ja eh dus de levering van de pompen mag vrijdag komen en de factuur volgt later '
           'groetjes Voornaam')

    CONTEXT = 'Van: Bart <bart@voorbeeld.invalid>\n\nBeste,\n\nWanneer komen de pompen?\n\nGroeten,\nBart'

    def run_rewrite(self, fields, cfg=None, language='nl', raw=None, context=CONTEXT):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            result = rewrite.rewrite(self.RAW if raw is None else raw, language, context, cfg or config(),
                                     call=lambda *a: mail_answer(fields))
        return result, out.getvalue()

    ANSWER = {'mail': True, 'aanhef': 'Dag Els,', 'tekst': 'De pompen komen vrijdag.',
              'afsluiting': 'Groetjes, Voornaam'}

    def test_no_email_without_context_unless_dictated(self):
        result, journal = self.run_rewrite(self.ANSWER, context=None)
        # 'Dag Els,' was not dictated, 'groetjes Voornaam' was: only that stays.
        self.assertEqual(result.text, 'De pompen komen vrijdag.\n\nGroetjes, Voornaam')
        self.assertIn('"email": false', journal)
        self.assertIn('"email_blocked": "no_context"', journal)
        result, journal = self.run_rewrite(self.ANSWER, context=None,
                                           raw='dit is een mail ' + self.RAW)
        self.assertTrue(result.text.startswith('Dag Els,\n\n'))
        self.assertIn('"email": true', journal)
        self.assertNotIn('email_blocked', journal)

    def test_no_email_when_context_is_not_an_email(self):
        for context in ('Kan iemand de pompen nakijken voor vrijdag? Dat zou fijn zijn.',
                        'Release notes 2.3: faster start-up and a smaller download size.'):
            result, journal = self.run_rewrite(self.ANSWER, context=context)
            self.assertIn('"email_blocked": "context_not_email"', journal)
            self.assertFalse(result.text.startswith('Dag Els'))

    def test_context_shape_of_an_email(self):
        yes = ('Van: Jan <jan@voorbeeld.invalid>\nOnderwerp: x', 'Dear Sam,\n\nCould you send it?',
               'Op ma 3 nov. schreef Jan:\n> tekst', 'De stalen zijn klaar.\n\nMet vriendelijke groeten,\nJan',
               'Thanks for the quote.\n\nKind regards,\nSam', 'contact: jan@voorbeeld.invalid')
        no = ('Kan iemand de pompen nakijken?', 'Dagelijkse controle van reactor twee om tien uur.',
              'Beschrijving van de taak in twee zinnen. Niets meer.')
        for text in yes:
            self.assertTrue(rewrite.context_is_email(text), text)
        for text in no:
            self.assertFalse(rewrite.context_is_email(text), text)
        self.assertIsNone(rewrite.email_block('stuur een e-mail naar Jan', None))
        self.assertIsNone(rewrite.email_block('write an email to Sam', None))
        self.assertEqual(rewrite.email_block('Dag Sam, de pompen komen', None), 'no_context')

    def test_schema_and_prompt_ask_for_email_fields(self):
        for lang, flag in (('nl', 'mail'), ('en', 'email')):
            messages, field = rewrite.build_messages('x', lang, None, config())
            schema = rewrite.response_format(field)['json_schema']['schema']
            self.assertEqual(schema['properties'][flag], {'type': 'boolean'})
            self.assertEqual(schema['required'][2], field)
            self.assertIn('"%s"' % flag, messages[0]['content'])

    def test_email_is_composed_with_signature(self):
        cfg = config(signature='Voornaam Achternaam')
        result, journal = self.run_rewrite(
            {'mail': True, 'aanhef': 'Beste Katrien,', 'tekst': 'De pompen mogen vrijdag geleverd worden.',
             'afsluiting': 'Vriendelijke groeten,'}, cfg)
        self.assertTrue(result.ok)
        self.assertEqual(result.text, 'Beste Katrien,\n\nDe pompen mogen vrijdag geleverd worden.'
                                      '\n\nVriendelijke groeten,\nVoornaam Achternaam')
        self.assertIn('"email": true', journal)
        self.assertNotIn('Katrien', journal)
        self.assertNotIn('Achternaam', journal)

    def test_email_without_signature_has_sign_off_line_only(self):
        result, _ = self.run_rewrite({'mail': True, 'aanhef': '', 'tekst': 'De pompen mogen vrijdag komen.',
                                      'afsluiting': ''})
        self.assertEqual(result.text, 'Beste,\n\nDe pompen mogen vrijdag komen.\n\nVriendelijke groeten,')
        result, _ = self.run_rewrite({'email': True, 'greeting': '', 'text': 'The pumps may arrive Friday.',
                                      'closing': ''}, language='en')
        self.assertEqual(result.text, 'Hello,\n\nThe pumps may arrive Friday.\n\nBest regards,')

    def test_dictated_name_is_not_doubled(self):
        cfg = config(signature='Voornaam Achternaam')
        for closing in ('Groetjes, Voornaam', 'Groetjes,\nVoornaam'):
            result, _ = self.run_rewrite({'mail': True, 'aanhef': 'Dag Els,', 'tekst': 'De pompen komen vrijdag.',
                                          'afsluiting': closing}, cfg)
            self.assertTrue(result.text.endswith('\n\n' + closing), closing)
            self.assertEqual(result.text.count('Voornaam'), closing.count('Voornaam'))

    def test_undictated_name_under_closing_is_replaced_by_signature(self):
        # The model took a name from the context; it was not dictated.
        answer = {'mail': True, 'aanhef': 'Beste Bart,', 'tekst': 'De pompen komen vrijdag.',
                  'afsluiting': 'Vriendelijke groeten,\nUitDeContext'}
        result, _ = self.run_rewrite(answer, config(signature='Voornaam Achternaam'))
        self.assertTrue(result.text.endswith('\n\nVriendelijke groeten,\nVoornaam Achternaam'))
        result, _ = self.run_rewrite(answer)
        self.assertTrue(result.text.endswith('\n\nVriendelijke groeten,'))
        self.assertNotIn('UitDeContext', result.text)

    def test_not_an_email_stays_body_only(self):
        result, journal = self.run_rewrite({'mail': False, 'aanhef': 'Beste,', 'tekst': 'Pompen vrijdag, factuur later.',
                                            'afsluiting': 'Groeten,'}, config(signature='Voornaam'))
        self.assertEqual(result.text, 'Pompen vrijdag, factuur later.')
        self.assertIn('"email": false', journal)

    def test_length_check_judges_body_only(self):
        # 16 dictated words, body of 15: an added greeting and sign-off must not trip the 1.2x bound.
        body = ' '.join(['woord'] * 15)
        result, _ = self.run_rewrite({'mail': True, 'aanhef': 'Beste collega van de aankoopdienst,', 'tekst': body,
                                      'afsluiting': 'Met heel vriendelijke groeten en tot binnenkort,'},
                                     config(signature='Voornaam Achternaam'))
        self.assertTrue(result.ok)
        result, _ = self.run_rewrite({'mail': True, 'aanhef': 'Beste,', 'tekst': 'Ja.', 'afsluiting': 'Groeten,'})
        self.assertEqual((result.ok, result.reason, result.text), (False, 'answer_too_short', self.RAW))

    def test_signature_from_environment(self):
        env = dict(DICTATION_REWRITE_ENDPOINT=ENDPOINT, DICTATION_REWRITE_MODEL='m')
        self.assertEqual(rewrite.load_config(env)[0].signature, '')
        env['DICTATION_REWRITE_SIGNATURE'] = ' Voornaam Achternaam\\nAfdeling '
        self.assertEqual(rewrite.load_config(env)[0].signature, 'Voornaam Achternaam\nAfdeling')


class AppStart:
    """Run App.start/stop with fake recorder, worker, replayer and capture."""

    def start_app(self, cfg, reason, capture=None, windows=('win-start', 'win-stop')):
        logic = app_logic()
        events = []

        class FakeRecorder:
            SAMPLE_RATE = R
            def __init__(self, transcription_queue, ledger=None, **kw):
                self.queue, self.ledger, self._raw_frames = transcription_queue, ledger, []
            def start(self):
                events.append(('record',))
                self.ledger.capture(np.full(2 * R, 8000, dtype=np.int16).tobytes())
            def stop(self):
                for start in (0, R):
                    self.queue.put((np.full(R, 0.5, dtype=np.float32), True, start))
                self.queue.put(None)

        class FakeWorker:
            language = 'nl'
            def reset_context(self):
                pass
            def transcribe_chunk(self, audio, recovery=False):
                return 'gewone tekst'

        def default_capture(**kw):
            events.append(('capture',))
            return context_capture.Context(None)
        logic.update(StreamingRecorder=FakeRecorder,
                     context_capture=types.SimpleNamespace(capture=capture or default_capture))
        app = logic['App'].__new__(logic['App'])
        app.args = types.SimpleNamespace(paste_min_s=0, silence_ms=1000, auto_stop_silence=0,
                                         min_chunk_s=1, max_chunk_s=10, eager_prob=0.5, max_time=60)
        app.active, app.session, app.timer = False, None, None
        app.beep = lambda k, wait=True: events.append(('beep', k))
        app.rewrite_config, app.rewrite_reason = cfg, reason
        app.transcription_worker = FakeWorker()
        app.replayer = FakeReplayer(events, windows=list(windows))
        return app, events


class ConfigTests(AppStart, RewriteTestCase):
    """Test 8: missing configuration or key: raw text, clear journal line, ordinary mode unchanged."""

    def test_load_config(self):
        self.assertEqual(rewrite.load_config({}), (None, 'not_configured'))
        self.assertEqual(rewrite.load_config({'DICTATION_REWRITE_ENDPOINT': ENDPOINT})[1], 'model_missing')
        self.assertEqual(rewrite.load_config({'DICTATION_REWRITE_MODEL': 'm'})[1], 'endpoint_missing')
        self.assertEqual(rewrite.load_config({'DICTATION_REWRITE_ENDPOINT': 'http://llm.invalid/v1',
                                              'DICTATION_REWRITE_MODEL': 'm'})[1], 'endpoint_not_https')
        self.assertEqual(rewrite.load_config({'DICTATION_REWRITE_ENDPOINT': ENDPOINT, 'DICTATION_REWRITE_MODEL': 'm',
                                              'DICTATION_REWRITE_KEY_FILE': '/nonexistent/key'})[1],
                         'key_file_unreadable')
        with tempfile.TemporaryDirectory() as tmp:
            keyfile = Path(tmp) / 'key.env'
            keyfile.write_text('SOME_API_KEY=abc123\n')
            cfg, reason = rewrite.load_config({
                'DICTATION_REWRITE_ENDPOINT': ENDPOINT, 'DICTATION_REWRITE_MODEL': 'm',
                'DICTATION_REWRITE_KEY_FILE': str(keyfile), 'DICTATION_REWRITE_TIMEOUT_S': '20',
                'DICTATION_REWRITE_PROVIDER_ORDER': 'together, fireworks'})
            self.assertIsNone(reason)
            self.assertEqual((cfg.timeout_s, cfg.provider_order), (20.0, ('together', 'fireworks')))
            self.assertEqual(rewrite.read_key(str(keyfile)), 'abc123')

    def test_key_sent_only_as_header_never_logged(self):
        with tempfile.TemporaryDirectory() as tmp:
            keyfile = Path(tmp) / 'key.env'
            keyfile.write_text('SOME_API_KEY=sk-test-SECRETVALUE\n')
            seen = {}

            class Response(io.BytesIO):
                def __enter__(self):
                    return self
                def __exit__(self, *a):
                    return False

            def urlopen(request, timeout):
                seen['auth'] = request.get_header('Authorization')
                seen['timeout'] = timeout
                return Response(json.dumps(llm_answer('Een nette korte herschrijving.')).encode())
            out = io.StringIO()
            cfg = config(key_file=str(keyfile))
            with contextlib.redirect_stdout(out):
                result = rewrite.rewrite('een gedicteerde tekst van een paar woorden', 'nl', None, cfg,
                                         call=lambda c, b, t: rewrite.http_call(c, b, t, urlopen=urlopen))
            self.assertTrue(result.ok)
            self.assertEqual(seen['auth'], 'Bearer sk-test-SECRETVALUE')
            self.assertNotIn('SECRETVALUE', out.getvalue())

    def test_key_vanished_after_start_pastes_raw(self):
        events, journal, session, bodies = self.pipeline(cfg=config(key_file='/nonexistent/key'))
        self.assertEqual([e for e in events if e[0] == 'paste'][0][1], ' '.join(RAW_PARTS))
        self.assertIn('"reason": "key_file_unreadable"', journal)
        self.assertEqual(bodies, [])  # no call without a key
        self.assertEqual(events[-1], ('beep', 'rewrite_failed'))

    def test_rewrite_hotkey_without_config_dictates_normally(self):
        app, events = self.start_app(None, 'not_configured')
        out = io.StringIO()
        with patch.object(rewrite, 'http_call', lambda *a: self.fail('no call')), contextlib.redirect_stdout(out):
            self.assertTrue(app.start('nl', True))
            session = app.session
            app.stop()
            self.assertTrue(session.done.wait(5))
        self.assertFalse(session.rewrite)
        self.assertIn('"rewrite_unavailable"', out.getvalue())
        self.assertIn('"reason": "not_configured"', out.getvalue())
        self.assertIn(('beep', 'rewrite_unavailable'), events)
        self.assertNotIn(('capture',), events)
        self.assertEqual([e for e in events if e[0] == 'paste'],
                         [('paste', 'gewone tekst', 'win-start'), ('paste', ' gewone tekst', 'win-start')])

    def test_ordinary_mode_unchanged_with_config(self):
        app, events = self.start_app(config(), None)
        with patch.object(rewrite, 'http_call', lambda *a: self.fail('no call')), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(app.start('nl'))
            session = app.session
            app.stop()
            self.assertTrue(session.done.wait(5))
        self.assertEqual([e[0] for e in events], ['record', 'beep', 'beep', 'paste', 'paste'])
        self.assertEqual(events[1:3], [('beep', 'start_recording'), ('beep', 'finish_recording')])
        self.assertIsNone(session.paste_wid)  # stop window only matters in rewrite mode


class ContextTests(AppStart, unittest.TestCase):
    """Test 9: source choice, refusal rules, captured once, cleared, never logged."""

    def fake_run(self, meta, texts, calls=None):
        def run(argv, **kw):
            if calls is not None:
                calls.append(argv)
            self.assertIn('timeout', kw)
            if '--probe' in argv:
                if meta is None:
                    raise subprocess.TimeoutExpired(argv, kw['timeout'])
                return types.SimpleNamespace(returncode=0, stdout=json.dumps(meta).encode())
            self.assertEqual(argv[0], 'xclip')
            selection = argv[argv.index('-selection') + 1]
            self.assertEqual(argv[-1], '-o')  # read-only: never writes a selection
            return types.SimpleNamespace(returncode=0, stdout=texts[selection].encode())
        return run

    MAIL = 'Beste, kunnen jullie de levering voor vrijdag bevestigen? Groeten.'

    def sel(self, age=10, targets=('TARGETS', 'TIMESTAMP', 'UTF8_STRING')):
        return dict(owner=True, targets=list(targets), age_s=age)

    def test_primary_wins_over_clipboard(self):
        calls = []
        run = self.fake_run({'primary': self.sel(), 'clipboard': self.sel()},
                            {'primary': self.MAIL, 'clipboard': 'Iets anders gekopieerd, lang genoeg.'}, calls)
        ctx = context_capture.capture(run)
        self.assertEqual((ctx.source, ctx.text), ('primary', self.MAIL))
        self.assertEqual(sum('xclip' in c for c in calls), 1)  # clipboard not even read

    def test_clipboard_fallback_and_refusals(self):
        cases = [
            ({'primary': self.sel(age=301), 'clipboard': self.sel(age=30)}, 'clipboard', {'primary': 'too_old'}),
            ({'primary': dict(owner=False), 'clipboard': self.sel(age=121)}, 'none',
             {'primary': 'empty', 'clipboard': 'too_old'}),
            ({'primary': self.sel(age=None), 'clipboard': self.sel(
                targets=('TARGETS', 'application/x-copyq-owner'))}, 'none',
             {'primary': 'no_timestamp', 'clipboard': 'copyq_owner'}),
            ({'primary': self.sel(targets=('TARGETS', 'x-kde-passwordManagerHint')),
              'clipboard': self.sel(targets=('application/x-copyq-hidden',))}, 'none',
             {'primary': 'secret_hint', 'clipboard': 'secret_hint'}),
        ]
        for meta, source, reasons in cases:
            with self.subTest(reasons=reasons):
                ctx = context_capture.capture(self.fake_run(meta, {'primary': self.MAIL, 'clipboard': self.MAIL}))
                self.assertEqual(ctx.source, source)
                self.assertEqual(ctx.reasons, reasons)
                self.assertEqual(ctx.text is None, source == 'none')
        ctx = context_capture.capture(self.fake_run(None, {}))
        self.assertEqual((ctx.text, ctx.reasons['primary']), (None, 'probe_failed'))

    def test_text_refusals_and_truncation(self):
        both = lambda: {'primary': self.sel(), 'clipboard': dict(owner=False)}  # noqa: E731
        for text, reason in (('sk-abcdefghijklmnop1234 is mijn sleutel', 'secret_pattern'),
                             ('-----BEGIN OPENSSH PRIVATE KEY-----\nabc', 'secret_pattern'),
                             ('Hunter2Passw0rd!xyz', 'secret_pattern'),
                             ('AKIAIOSFODNN7EXAMPLE en nog wat tekst', 'secret_pattern'),
                             ('kort woord', 'too_short')):
            with self.subTest(reason=reason, text=text[:8]):
                ctx = context_capture.capture(self.fake_run(both(), {'primary': text}))
                self.assertIsNone(ctx.text)
                self.assertEqual(ctx.reasons['primary'], reason)
        long_mail = 'Bovenste bericht. ' + 'middelste draad ' * 2000 + 'Onderste bericht.'
        ctx = context_capture.capture(self.fake_run(both(), {'primary': long_mail}), max_chars=200)
        self.assertTrue(ctx.truncated)
        # Both ends survive: the message being answered may sit on top or at the bottom.
        self.assertTrue(ctx.text.startswith('Bovenste bericht.'))
        self.assertTrue(ctx.text.endswith('Onderste bericht.'))
        self.assertEqual(ctx.text.count('\n[ingekort]\n'), 1)
        self.assertLessEqual(len(ctx.text), 200 + len('\n[ingekort]\n'))
        self.assertEqual((ctx.chars, ctx.original_chars), (len(ctx.text), len(long_mail)))
        self.assertEqual(set(ctx.journal()),
                         {'source', 'chars', 'age_s', 'truncated', 'original_chars', 'reasons'})
        fits = 'Past precies binnen de grens. ' * 6
        ctx = context_capture.capture(self.fake_run(both(), {'primary': fits}), max_chars=len(fits.strip()))
        self.assertEqual((ctx.text, ctx.truncated), (fits.strip(), False))

    def test_default_limit_keeps_a_long_thread_whole(self):
        both = {'primary': self.sel(), 'clipboard': dict(owner=False)}
        thread = 'Een lange draad met vele berichten. ' * 2500  # ~90,000 characters
        ctx = context_capture.capture(self.fake_run(both, {'primary': thread}))
        self.assertEqual(context_capture.DEFAULT_MAX_CHARS, 100_000)
        self.assertEqual((ctx.text, ctx.truncated), (thread.strip(), False))
        ctx = context_capture.capture(self.fake_run(both, {'primary': thread}), max_chars=5000)
        self.assertTrue(ctx.truncated)
        self.assertLessEqual(ctx.chars, 5000 + len('\n[ingekort]\n'))

    def test_limit_setting(self):
        name = rewrite.CONTEXT_MAX_CHARS_ENV
        base = {'DICTATION_REWRITE_ENDPOINT': ENDPOINT, 'DICTATION_REWRITE_MODEL': 'm'}
        load = lambda value: rewrite.load_config(dict(base, **{name: value}))[0]  # noqa: E731
        default = context_capture.DEFAULT_MAX_CHARS
        cfg = rewrite.load_config(base)[0]
        self.assertEqual((cfg.context_max_chars, cfg.context_max_chars_invalid), (default, None))
        for value, limit in ((' ', default), ('30000', 30000), ('250_000', 250000)):
            with self.subTest(value=value):
                cfg = load(value)
                self.assertEqual((cfg.context_max_chars, cfg.context_max_chars_invalid), (limit, None))
        for value, reason in (('veel', 'not_a_number'), ('1e5', 'not_a_number'), ('12.5', 'not_a_number'),
                              ('999', 'out_of_range'), ('-5', 'out_of_range'), ('500001', 'out_of_range')):
            with self.subTest(value=value):
                cfg = load(value)  # an invalid limit never disables rewrite mode
                self.assertEqual((cfg.context_max_chars, cfg.context_max_chars_invalid), (default, reason))

    def test_configured_limit_reaches_capture(self):
        seen = []

        def capture(**kw):
            seen.append(kw)
            return context_capture.Context(None)
        app, events = self.start_app(config(context_max_chars=30000), None, capture=capture)
        with patch.object(rewrite, 'http_call', lambda *a: llm_answer('Een korte tekst.')), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(app.start('nl', True))
            session = app.session
            app.stop()
            self.assertTrue(session.done.wait(5))
        self.assertEqual(seen, [{'max_chars': 30000}])

    def test_invalid_limit_is_journaled_without_its_value(self):
        app = app_logic()['App'].__new__(app_logic()['App'])
        name = rewrite.CONTEXT_MAX_CHARS_ENV
        base = {'DICTATION_REWRITE_ENDPOINT': ENDPOINT, 'DICTATION_REWRITE_MODEL': 'fake/model'}
        for value, reason in (('geheim-12345', 'not_a_number'), ('99999999', 'out_of_range')):
            with self.subTest(reason=reason):
                app.rewrite_config, app.rewrite_reason = rewrite.load_config(dict(base, **{name: value}))
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    app._announce_rewrite()
                lines = out.getvalue()
                self.assertNotIn(value, lines)
                records = [json.loads(line.split(' ', 1)[1]) for line in lines.splitlines()
                           if line.startswith('DCT-COVERAGE ')]
                self.assertEqual([(r['kind'], r['setting'], r['reason'], r['default']) for r in records],
                                 [('rewrite_setting_invalid', name, reason, 100_000)])
                self.assertIn('up to 100000 characters (%s invalid: %s, default used)' % (name, reason), lines)
        app.rewrite_config, app.rewrite_reason = rewrite.load_config(dict(base, **{name: '30000'}))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            app._announce_rewrite()
        self.assertEqual(out.getvalue().strip(),
                         'Rewrite mode: model fake/model, time-out 30s, reply context up to 30000 characters.')

    def test_captured_once_at_start_and_used_at_stop(self):
        selection = {'text': 'MAIL-BIJ-START: kunnen jullie de levering bevestigen?'}
        calls = []

        def capture(**kw):
            calls.append(selection['text'])
            return context_capture.Context(selection['text'], 'primary', len(selection['text']), 3.0)
        app, events = self.start_app(config(), None, capture=capture)
        bodies = []

        def http(cfg, body, timeout):
            bodies.append(body)
            return llm_answer('Een korte herschreven bevestiging.')
        out = io.StringIO()
        with patch.object(rewrite, 'http_call', http), contextlib.redirect_stdout(out):
            self.assertTrue(app.start('nl', True))
            session = app.session
            selection['text'] = 'GEWIJZIGDE-SELECTIE na klik in het antwoordveld'
            app.stop()
            self.assertTrue(session.done.wait(5))
        self.assertEqual(len(calls), 1)
        user = bodies[0]['messages'][1]['content']
        self.assertIn('MAIL-BIJ-START', user)
        self.assertNotIn('GEWIJZIGDE-SELECTIE', user)
        self.assertIsNone(session.context)  # cleared at session end
        # Captured before any paste; paste goes to the window active at stop.
        self.assertLess(events.index(('record',)), events.index(('beep', 'rewrite_start_context')))
        pastes = [e for e in events if e[0] == 'paste']
        self.assertEqual(pastes, [('paste', 'Een korte herschreven bevestiging.', 'win-stop')])
        self.assertEqual(session.target_wid, 'win-start')
        self.assertEqual(session.paste_wid, 'win-stop')
        self.assertNotIn('MAIL-BIJ-START', out.getvalue())
        record = json.loads(next(l for l in out.getvalue().splitlines() if '"rewrite_context"' in l)
                            .removeprefix('DCT-COVERAGE '))
        self.assertEqual((record['source'], record['chars']), ('primary', len('MAIL-BIJ-START: kunnen jullie '
                                                                             'de levering bevestigen?')))

    def test_start_tone_differs_with_and_without_context(self):
        app, events = self.start_app(config(), None)  # default capture: no context
        with patch.object(rewrite, 'http_call', lambda *a: llm_answer('Gewone tekst herschreven.')), \
                contextlib.redirect_stdout(io.StringIO()):
            app.start('nl', True)
            session = app.session
            app.stop()
            self.assertTrue(session.done.wait(5))
        self.assertIn(('beep', 'rewrite_start'), events)
        self.assertNotIn(('beep', 'rewrite_start_context'), events)
        logic = app_logic()
        start = np.zeros((3553, 2), dtype=np.float32)
        sounds = logic['rewrite_sounds'](start)
        self.assertGreater(len(sounds['rewrite_start_context']), len(sounds['rewrite_start']))
        self.assertEqual(sounds['rewrite_failed'].shape[1], 2)
        self.assertFalse(np.array_equal(sounds['rewrite_done'][:100], sounds['rewrite_failed'][:100]))


class SourceTests(unittest.TestCase):
    def test_unit_reads_optional_rewrite_environment(self):
        installer = (ROOT / 'install.sh').read_text()
        unit = installer[installer.index('render_service() {'):installer.index('unit_is_current() {')]
        self.assertIn('EnvironmentFile=-%h/.config/faster-whisper-dictation/rewrite.env', unit)

    def test_no_personal_paths_or_new_dependencies(self):
        for name in ('rewrite.py', 'context_capture.py', 'tests/test_rewrite.py'):
            source = (ROOT / name).read_text()
            self.assertNotIn('/' + 'home/', source)
        for name in ('rewrite.py', 'context_capture.py'):
            self.assertNotIn('notify-' + 'send', (ROOT / name).read_text())
            tree = ast.parse((ROOT / name).read_text())
            modules = {a.name.split('.')[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
            modules |= {n.module.split('.')[0] for n in ast.walk(tree)
                        if isinstance(n, ast.ImportFrom) and n.module}
            self.assertLessEqual(modules, {'__future__', 'json', 'os', 're', 'threading', 'time', 'urllib',
                                           'dataclasses', 'typing', 'coverage_ledger', 'context_capture', 'subprocess', 'sys',
                                           'select', 'Xlib'})


if __name__ == '__main__':
    unittest.main()


class EnglishRewriteTests(AppStart, unittest.TestCase):
    """Left-Shift in rewrite mode: recognition stays Dutch, the rewrite is written in English."""

    def run_start(self, language, rewrite_mode, keys=None):
        app, events = self.start_app(config(), None)
        app.args.language = 'nl'
        bodies = []

        def http(cfg, body, timeout):
            bodies.append(body)
            return llm_answer('A short English text.', field='text')
        out = io.StringIO()
        with patch.object(rewrite, 'http_call', http), contextlib.redirect_stdout(out):
            self.assertTrue(app.start(language, rewrite_mode, keys))
            session = app.session
            app.stop()
            self.assertTrue(session.done.wait(5))
        records = [json.loads(line.split(' ', 1)[1]) for line in out.getvalue().splitlines()
                   if line.startswith('DCT-COVERAGE ')]
        return app, session, records, bodies

    def test_shift_alt_recognises_dutch_writes_english(self):
        keys = {'shift': True, 'shift_held_s': 0.8, 'alt': True, 'alt_held_s': 0.6}
        app, session, records, bodies = self.run_start('en', True, keys)
        start = next(r for r in records if r['kind'] == 'session_start')
        self.assertEqual((start['language'], start['output_language'], start['keys']), ('nl', 'en', keys))
        self.assertEqual(app.transcription_worker.language, 'nl')
        self.assertEqual(session.output_language, 'en')
        self.assertTrue(bodies[0]['messages'][0]['content'].startswith('Task: rewrite'))
        self.assertIn('always write the result in English', bodies[0]['messages'][0]['content'])
        outcome = next(r for r in records if r['kind'] == 'rewrite_outcome')
        self.assertEqual((outcome['language'], outcome['spoken']), ('en', 'nl'))

    def test_alt_only_and_ordinary_shift_unchanged(self):
        app, session, records, bodies = self.run_start('nl', True, {'shift': False, 'alt': True})
        start = next(r for r in records if r['kind'] == 'session_start')
        self.assertEqual((start['language'], start['output_language']), ('nl', 'nl'))
        self.assertIn('Belgisch-Nederlands', bodies[0]['messages'][0]['content'])
        app, session, records, bodies = self.run_start('en', False, {'shift': True, 'alt': False})
        start = next(r for r in records if r['kind'] == 'session_start')
        self.assertEqual(start['language'], 'en')  # ordinary Shift dictation: forced English, as before
        self.assertNotIn('output_language', start)
        self.assertEqual(app.transcription_worker.language, 'en')
        self.assertEqual(bodies, [])

    def test_primary_age_limit(self):
        self.assertLessEqual(context_capture.PRIMARY_MAX_AGE_S, 20)
        self.assertEqual(context_capture.judge('primary', dict(owner=True, targets=[], age_s=25),
                                               context_capture.PRIMARY_MAX_AGE_S), 'too_old')
        self.assertIsNone(context_capture.judge('primary', dict(owner=True, targets=[], age_s=8),
                                                context_capture.PRIMARY_MAX_AGE_S))
