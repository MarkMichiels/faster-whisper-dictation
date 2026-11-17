"""LLM log, offline: made-up text only, temporary directories only.

Covers the secret filter, the password-window skip, retention (text wiped,
metadata kept, FTS cleaned, wiped text gone from the file), permissions,
settings, the rewrite hook, and that record() never blocks on a slow disk or a
locked database.
"""
import contextlib
import io
import os
import sqlite3
import stat
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import llm_log  # noqa: E402
import rewrite  # noqa: E402
import settings  # noqa: E402

DAY = 86400


def wait_for(predicate, timeout=5.0):
    """Event-style wait for the writer thread (no fixed sleeps)."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.01)
    return False


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = os.path.join(self.tmp.name, 'data', 'faster-whisper-dictation')
        self.path = os.path.join(self.dir, 'llm_log.db')
        self.events = []
        self.logs = []

    def tearDown(self):
        for log in self.logs:
            log.close()
        self.tmp.cleanup()

    def diag(self, kind, **details):
        self.events.append((kind, details))

    def make(self, **kw):
        log = llm_log.LlmLog(self.path, diagnostic=self.diag, **kw)
        self.logs.append(log)
        return log

    def rows(self, sql='SELECT * FROM calls ORDER BY id', args=()):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            return [dict(r) for r in conn.execute(sql, args)]
        finally:
            conn.close()


class SecretFilterTest(unittest.TestCase):
    CASES = [
        'mijn sleutel is sk-or-v1-0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b',
        'token ghp_A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8',
        'google AIzaSyA-1234567890abcdefghijklmnopqrstu',
        'aws AKIAABCDEFGHIJKLMNOP klaar',
        'header Bearer abcdefghijklmnop1234567890',
        'jwt eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dBjftJeZ4CVPmB92K27uhbUJU1p1r',
        'hex 0123456789abcdef0123456789abcdef0123',
        'b64 QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVowMTIzNDU2Nzg5YWJj==',
        'wachtwoord: Zomer2026!',
        'password=hunter2xyz',
    ]

    def test_every_pattern_becomes_secret_mark(self):
        for text in self.CASES:
            out, n = llm_log.filter_secrets(text)
            self.assertGreaterEqual(n, 1, text)
            self.assertIn(llm_log.SECRET_MARK, out, text)
        out, _ = llm_log.filter_secrets('wachtwoord: Zomer2026!')
        self.assertEqual(out, 'wachtwoord: [geheim]')
        self.assertNotIn('Zomer2026', out)

    def test_private_key_block(self):
        text = 'voor\n-----BEGIN OPENSSH PRIVATE KEY-----\nb3BlbnNzaC1rZXk\nAAAA\n-----END OPENSSH PRIVATE KEY-----\nna'
        out, n = llm_log.filter_secrets(text)
        self.assertEqual(n, 1)
        self.assertEqual(out, 'voor\n[geheim]\nna')

    def test_ordinary_text_untouched(self):
        text = ('De vergadering van donderdag 14 maart om 10u30 gaat niet door. '
                'Bel me op 0470 12 34 56 of mail jan.peeters@voorbeeld.invalid. Factuur 2026-0042.')
        self.assertEqual(llm_log.filter_secrets(text), (text, 0))
        self.assertEqual(llm_log.filter_secrets(None), (None, 0))

    def test_same_prefixes_as_context_capture(self):
        import context_capture
        for prefix in ('sk-', 'sk_live_', 'ghp_', 'gho_', 'github_pat_', 'glpat-', 'xoxb-', 'AKIA', 'AIza'):
            sample = prefix + 'Abcdefgh12345678'
            self.assertTrue(context_capture._SECRET_RE.search(sample))
            self.assertEqual(llm_log.filter_secrets(sample)[0], llm_log.SECRET_MARK, prefix)

    def test_password_windows(self):
        for name in ('org.keepassxc.keepassxc', 'Bitwarden', '1password', 'pinentry-gnome3',
                     'gcr-prompter', 'org.gnome.World.Secrets'):
            self.assertTrue(llm_log.is_password_window(name), name)
        for name in ('', None, 'org.gnome.texteditor', 'firefox', 'kitty', 'thunderbird'):
            self.assertFalse(llm_log.is_password_window(name), name)


class WriterTest(Base):
    def test_row_written_filtered_and_searchable(self):
        log = self.make().start()
        self.assertTrue(log.record(kind='rewrite', mode='opmaak', language='nl', model='m', status='ok',
                                   delivered='pasted', context='Beste Jan, kan je de offerte sturen?',
                                   input='ja dus de sleutel is sk-abcdefgh12345678 en de offerte komt',
                                   output='De offerte komt morgen.', window_class='org.gnome.texteditor',
                                   system_prompt='Opdracht: verzonnen'))
        self.assertTrue(wait_for(lambda: log.counts['written'] == 1))
        row, = self.rows()
        self.assertEqual(row['kind'], 'rewrite')
        self.assertNotIn('sk-abcdefgh', row['input'])
        self.assertEqual(row['secrets_replaced'], 1)
        self.assertEqual(row['prompt_version'], llm_log.prompt_version('Opdracht: verzonnen'))
        self.assertIsNone(row['text_purged_at'])
        hits = self.rows("SELECT rowid FROM calls_fts WHERE calls_fts MATCH 'offerte'")
        self.assertEqual(len(hits), 1)
        self.assertFalse(self.rows("SELECT rowid FROM calls_fts WHERE calls_fts MATCH 'abcdefgh12345678'"))
        # The journal never carries text.
        self.assertFalse(any('offerte' in repr(d) for _, d in self.events))

    def test_password_window_and_secret_context_skip_everything(self):
        log = self.make().start()
        self.assertFalse(log.record(kind='rewrite', input='geheim dictaat', window_class='KeePassXC'))
        self.assertFalse(log.record(kind='rewrite', input='geheim dictaat', password_field=True))
        self.assertTrue(log.record(kind='rewrite', input='gewoon dictaat'))
        self.assertTrue(wait_for(lambda: log.counts['written'] == 1))
        self.assertEqual([r['input'] for r in self.rows()], ['gewoon dictaat'])
        self.assertEqual(log.counts['skipped_password'], 2)

    def test_permissions_private_and_status_file(self):
        log = self.make().start()
        log.record(kind='rewrite', input='a')
        self.assertTrue(wait_for(lambda: log.counts['written'] == 1))
        self.assertEqual(stat.S_IMODE(os.stat(self.dir).st_mode), 0o700)
        for name in os.listdir(self.dir):
            self.assertEqual(stat.S_IMODE(os.stat(os.path.join(self.dir, name)).st_mode), 0o600, name)
        self.assertEqual(self.rows("PRAGMA journal_mode")[0]['journal_mode'], 'wal')
        import json
        with open(os.path.join(self.dir, llm_log.STATUS_NAME)) as fh:
            status = json.load(fh)
        self.assertEqual(status['written'], 1)
        self.assertNotIn('a', [v for v in status.values() if isinstance(v, str) and len(v) == 1])
        ok, lines = llm_log.check(self.path, True)
        self.assertTrue(ok, lines)
        os.chmod(self.path, 0o644)
        ok, lines = llm_log.check(self.path, True)
        self.assertFalse(ok)
        self.assertIn('WRONG', lines[0])
        self.assertTrue(llm_log.check(self.path, False)[0])

    def test_context_max_chars(self):
        log = self.make(context_max_chars=100).start()
        log.record(kind='rewrite', context='woord ' * 200, input='x')
        self.assertTrue(wait_for(lambda: log.counts['written'] == 1))
        context = self.rows()[0]['context']
        self.assertLess(len(context), 130)
        self.assertIn('[ingekort]', context)


class RetentionTest(Base):
    def test_text_wiped_metadata_kept_fts_clean_file_clean(self):
        now = time.time()
        conn = llm_log.connect(self.path)
        cols = ', '.join(llm_log.COLUMNS)
        for age, word in ((100, 'zeppelinoud'), (800, 'zeppelinoeroud'), (5, 'zeppelinnieuw')):
            epoch = now - age * DAY
            conn.execute('INSERT INTO calls(%s) VALUES (%s)' % (cols, ', '.join('?' * len(llm_log.COLUMNS))),
                         (llm_log._iso(epoch), epoch, 'rewrite', 'opmaak', 'nl', 'm', 'p', 'v', 10, 1, 1,
                          0.0, 'ok', None, 'pasted', 'context ' + word, 'input ' + word, 'output ' + word,
                          0, None, None))
        result = llm_log.purge(conn, 90, 730, now)
        conn.close()
        self.assertEqual(result, dict(text_wiped=1, rows_deleted=1))
        rows = self.rows()
        self.assertEqual(len(rows), 2)
        old, new = rows
        self.assertIsNone(old['input'])
        self.assertIsNone(old['context'])
        self.assertIsNone(old['output'])
        self.assertIsNotNone(old['text_purged_at'])
        self.assertEqual((old['model'], old['ms'], old['status']), ('m', 10, 'ok'))
        self.assertEqual(new['input'], 'input zeppelinnieuw')
        for word, hits in (('zeppelinoud', 0), ('zeppelinoeroud', 0), ('zeppelinnieuw', 1)):
            self.assertEqual(len(self.rows('SELECT rowid FROM calls_fts WHERE calls_fts MATCH ?', (word,))),
                             hits, word)
        # secure_delete + optimize + checkpoint: the wiped words are gone from every file.
        for name in os.listdir(self.dir):
            with open(os.path.join(self.dir, name), 'rb') as fh:
                data = fh.read()
            self.assertNotIn(b'zeppelinoud', data, name)
            self.assertNotIn(b'zeppelinoeroud', data, name)
        # Idempotent.
        conn = llm_log.connect(self.path)
        self.assertEqual(llm_log.purge(conn, 90, 730, now), dict(text_wiped=0, rows_deleted=0))
        self.assertEqual(llm_log.purge(conn, 0, 0, now + 9999 * DAY), dict(text_wiped=0, rows_deleted=0))
        conn.close()

    def test_writer_runs_purge_at_start_and_then_daily(self):
        now = [time.time()]
        log = self.make(clock=lambda: now[0], purge_interval_s=0.05).start()
        log.record(kind='rewrite', input='oud dictaat', epoch=now[0] - 100 * DAY)
        self.assertTrue(wait_for(lambda: log.counts['written'] == 1))
        self.assertTrue(wait_for(lambda: (log.last_purge or {}).get('text_wiped') == 1))
        self.assertIsNone(self.rows()[0]['input'])
        self.assertIn('llm_log_purged', [k for k, _ in self.events])


class NeverBlocksTest(Base):
    def test_slow_disk_record_returns_at_once(self):
        log = self.make()
        gate = threading.Event()
        real = log._write
        log._write = lambda row: (gate.wait(5), real(row))
        log.start()
        times = []
        for i in range(200):
            t0 = time.perf_counter()
            log.record(kind='rewrite', input='tekst %d' % i)
            times.append(time.perf_counter() - t0)
        self.assertLess(max(times), 0.05)
        gate.set()
        self.assertTrue(wait_for(lambda: log.counts['written'] == 200, 10))

    def test_full_queue_drops_with_number_only(self):
        log = self.make(queue_size=2)   # not started: nothing drains the queue
        results = [log.record(kind='rewrite', input='geheime inhoud') for _ in range(3)]
        self.assertEqual(results, [True, True, False])
        kind, details = self.events[-1]
        self.assertEqual((kind, details['reason'], details['dropped']), ('llm_log_dropped', 'queue_full', 1))
        self.assertNotIn('geheime', repr(self.events))

    def test_locked_database_drops_row_without_blocking(self):
        conn = llm_log.connect(self.path)
        conn.execute('BEGIN EXCLUSIVE')
        try:
            log = self.make(busy_timeout_ms=100).start()
            t0 = time.perf_counter()
            self.assertTrue(log.record(kind='rewrite', input='op slot'))
            self.assertLess(time.perf_counter() - t0, 0.05)
            self.assertTrue(wait_for(lambda: log.counts['dropped'] == 1, 5))
            self.assertEqual(self.events[-1][1]['reason'], 'locked')
        finally:
            conn.execute('ROLLBACK')
            conn.close()
        log.record(kind='rewrite', input='weer vrij')
        self.assertTrue(wait_for(lambda: log.counts['written'] == 1, 5))

    def test_unwritable_directory_drops(self):
        os.makedirs(os.path.dirname(self.dir), exist_ok=True)
        with open(self.dir, 'w') as fh:   # a file where the directory should be
            fh.write('x')
        log = self.make().start()
        self.assertTrue(log.record(kind='rewrite', input='x'))
        self.assertTrue(wait_for(lambda: log.counts['dropped'] == 1, 5))


class SettingsTest(unittest.TestCase):
    def test_defaults_and_yaml(self):
        loaded = settings.load({'XDG_CONFIG_HOME': '/nonexistent'})
        self.assertEqual([loaded.get(k) for k in ('log.enabled', 'log.path', 'log.text_days', 'log.meta_days',
                                                  'log.context_max_chars')], [False, None, 90, 730, 0])
        self.assertIsNone(llm_log.from_settings(loaded))
        with tempfile.TemporaryDirectory() as tmp:
            cfg = os.path.join(tmp, 'config.yaml')
            db = os.path.join(tmp, 'd', 'log.db')
            with open(cfg, 'w') as fh:
                fh.write('log:\n  enabled: true\n  path: %s\n  text_days: 30\n  meta_days: 0\n'
                         '  context_max_chars: 2000\n' % db)
            loaded = settings.load({}, cfg)
            log = llm_log.from_settings(loaded)
            try:
                self.assertEqual((log.path, log.text_days, log.meta_days, log.context_max_chars),
                                 (db, 30, 0, 2000))
            finally:
                log.close()
            with open(cfg, 'w') as fh:
                fh.write('log:\n  text_days: -1\n')
            loaded = settings.load({}, cfg)
            self.assertEqual(loaded.get('log.text_days'), 90)
        self.assertTrue(llm_log.default_path({'XDG_DATA_HOME': '/x'}).startswith('/x/faster-whisper-dictation/'))

    def test_cli_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = os.path.join(tmp, 'config.yaml')
            with open(cfg, 'w') as fh:
                fh.write('log:\n  enabled: true\n  path: %s\n' % os.path.join(tmp, 'log', 'llm_log.db'))
            run = lambda: subprocess.run([sys.executable, str(ROOT / 'llm_log.py'), '--check', '--config', cfg],
                                         capture_output=True, text=True, timeout=30)
            self.assertEqual(run().returncode, 0)
            conn = llm_log.connect(os.path.join(tmp, 'log', 'llm_log.db'))
            conn.close()
            self.assertEqual(run().returncode, 0)
            os.chmod(os.path.join(tmp, 'log'), 0o755)
            result = run()
            self.assertEqual(result.returncode, 1)
            self.assertIn('WRONG', result.stderr)


class RewriteHookTest(unittest.TestCase):
    def fake_call(self, config, body, timeout):
        import json
        answer = {'kern': 'k', 'punten': ['p'], 'mail': False, 'aanhef': '', 'afsluiting': '',
                  'tekst': 'De vergadering gaat niet door.'}
        return {'choices': [{'message': {'content': json.dumps(answer)}, 'finish_reason': 'stop'}],
                'provider': 'fakeprov', 'usage': {'prompt_tokens': 11, 'completion_tokens': 7, 'cost': 0.001}}

    def test_result_carries_meta_and_answer(self):
        config = rewrite.RewriteConfig(endpoint='https://llm.invalid/v1', model='fake/model')
        with contextlib.redirect_stdout(io.StringIO()) as out:
            result = rewrite.rewrite('ja dus de vergadering eh die gaat niet door', 'nl', None, config,
                                     call=self.fake_call)
        self.assertTrue(result.ok)
        self.assertEqual(result.answer, 'De vergadering gaat niet door.')
        self.assertEqual((result.meta['model'], result.meta['provider'], result.meta['tokens_in']),
                         ('fake/model', 'fakeprov', 11))
        self.assertIn('beknopt', result.system_prompt)
        self.assertEqual(len(llm_log.prompt_version(result.system_prompt)), 12)
        self.assertNotIn('vergadering', out.getvalue())
        with contextlib.redirect_stdout(io.StringIO()):
            none = rewrite.rewrite('x', 'nl', None, None, 'not_configured')
        self.assertIsNone(none.meta)

    def test_deliver_rewrite_queues_one_row_after_paste(self):
        from test_rewrite import app_logic
        logic = app_logic()
        app = logic['App'].__new__(logic['App'])
        app.replayer = Mock()
        app.replayer.add_to_history.return_value = 0
        app.replayer.last_window_class = 'org.gnome.texteditor'
        order = []
        app.replayer.type_text.side_effect = lambda *a, **k: (order.append('paste'), dict(ok=True))[1]
        app.beep = Mock()
        app.transcription_worker = types.SimpleNamespace(language='nl')
        app.llm_log = Mock()
        app.llm_log.record.side_effect = lambda **row: order.append('log')
        session = types.SimpleNamespace(language='nl', output_language='nl', context='Beste Jan',
                                        target_wid='start', paste_wid='stop', terminal_mode=False,
                                        context_secret=False)
        result = rewrite.RewriteResult('Klaar.', True, meta=dict(model='m', seconds=1.234),
                                       answer='Klaar.', system_prompt='Opdracht: verzonnen prompt')
        with patch.object(rewrite, 'rewrite', return_value=result), contextlib.redirect_stdout(io.StringIO()):
            app._deliver_rewrite(session, 'ruw dictaat', [])
        self.assertEqual(order, ['paste', 'log'])
        row = app.llm_log.record.call_args.kwargs
        self.assertEqual((row['kind'], row['mode'], row['ms'], row['status'], row['delivered'], row['input'],
                          row['context'], row['output'], row['window_class'], row['password_field'],
                          row['system_prompt']),
                         ('rewrite', 'opmaak', 1234, 'ok', 'pasted', 'ruw dictaat', 'Beste Jan', 'Klaar.',
                          'org.gnome.texteditor', False, 'Opdracht: verzonnen prompt'))
        # A log that raises never breaks delivery.
        app.llm_log.record.side_effect = RuntimeError('boom')
        with patch.object(rewrite, 'rewrite', return_value=result), contextlib.redirect_stdout(io.StringIO()):
            app._deliver_rewrite(session, 'ruw dictaat', [])
        # No call attempted -> no row.
        app.llm_log.record.reset_mock(side_effect=True)
        with patch.object(rewrite, 'rewrite', return_value=rewrite.RewriteResult('r', False, 'not_configured')), \
                contextlib.redirect_stdout(io.StringIO()):
            app._deliver_rewrite(session, 'ruw dictaat', [])
        app.llm_log.record.assert_not_called()


if __name__ == '__main__':
    unittest.main()
