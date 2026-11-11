"""Offline tests for the replacement list (recognition.replacements_file).

Neutral example terms only; a real list is private and never belongs here.
"""
import contextlib
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import coverage_ledger  # noqa: E402
import replacements  # noqa: E402
import settings  # noqa: E402
from test_recovery import logic, worker  # noqa: E402
from test_settings import dictation_functions, write  # noqa: E402

RULES = """\
# Neutral example list.
fubar|foo ?bar -> FooBar
npq → MPQ
wolk ?(?:kode|code) -> Wolk Code
zeroo -> Zerro
diepzoek -> DeepZoek
provinco|provinkko -> Provinco
"""

CONTROL = [
    'De cloud staat in de provincie en de pc is traag.',
    'Een zero waste plan maakt me diep ziek.',
    'Foo en bar zijn twee woorden; een foobaria ook niet.',
    'Wolk boven de npqr-straat, zero kode, diepzoeken en provincoos.',
    'De PC, de Cloud, ZERO, Diep ziek, Provincie.',
    '',
]


def make(text=RULES):
    log = []
    rules = replacements.parse(text, lambda kind, **d: log.append((kind, d)))
    return rules, log


class RuleTests(unittest.TestCase):
    def test_each_rule_changes_only_its_own_form(self):
        rules, _ = make()
        cases = {
            'Ik werk bij Fubar en foo bar en FOOBAR.': 'Ik werk bij FooBar en FooBar en FooBar.',
            'De NPQ en de npq.': 'De MPQ en de MPQ.',
            'Open wolk code en Wolkkode.': 'Open Wolk Code en Wolk Code.',
            'Zeroo!': 'Zerro!',
            'Vraag het aan diepzoek.': 'Vraag het aan DeepZoek.',
            'provinkko, Provinco.': 'Provinco, Provinco.',
        }
        for before, after in cases.items():
            self.assertEqual(rules.apply(before), after)
        # One rule at a time: a rule never touches another rule's sentence.
        for rule in rules.rules:
            single = replacements.ReplacementList([rule])
            for before, after in cases.items():
                changed = single.apply(before)
                if changed != before:
                    self.assertEqual(changed, after, (rule[0], before))

    def test_control_sentences_unchanged(self):
        rules, log = make()
        for sentence in CONTROL:
            self.assertEqual(rules.apply(sentence), sentence)
        self.assertEqual([k for k, _ in log if k == 'replacement'], [])

    def test_whole_words_only(self):
        rules, _ = make('ab -> AB\n')
        self.assertEqual(rules.apply('ab abc cab ab-c a_ab ab.'), 'AB abc cab AB-c a_ab AB.')
        self.assertEqual(rules.apply('éab abé'), 'éab abé')

    def test_replacement_is_literal(self):
        rules, _ = make('x1 -> \\1 & \\g<0>\n')
        self.assertEqual(rules.apply('a x1 b'), 'a \\1 & \\g<0> b')

    def test_rules_apply_in_file_order(self):
        rules, _ = make('aa -> bb\nbb -> cc\n')
        self.assertEqual(rules.apply('aa'), 'cc')

    def test_journal_has_rule_id_and_count_only(self):
        rules, log = make()
        rules.apply('fubar en foo bar, npq.')
        self.assertEqual(log, [('replacement', {'rule': 'L2', 'count': 2}),
                               ('replacement', {'rule': 'L3', 'count': 1})])

    def test_matching_the_right_form_is_no_change(self):
        rules, log = make('foo ?bar -> foobar\n')
        self.assertEqual(rules.apply('foobar en FooBar en foo bar'), 'foobar en foobar en foobar')
        self.assertEqual(log, [('replacement', {'rule': 'L1', 'count': 2})])
        log.clear()
        self.assertEqual(rules.apply('alleen foobar'), 'alleen foobar')
        self.assertEqual(log, [])

    def test_bad_lines_are_skipped_with_line_number(self):
        rules, log = make('geen pijl\n-> leeg\nx ->  \n(open -> y\na? -> b\n | -> c\nok -> OK\n')
        self.assertEqual([r[0] for r in rules.rules], ['L7'])
        self.assertEqual(log, [
            ('replacements_rule_skipped', {'line': 1, 'reason': 'no_arrow'}),
            ('replacements_rule_skipped', {'line': 2, 'reason': 'no_arrow'}),
            ('replacements_rule_skipped', {'line': 3, 'reason': 'no_arrow'}),
            ('replacements_rule_skipped', {'line': 4, 'reason': 'bad_regex'}),
            ('replacements_rule_skipped', {'line': 5, 'reason': 'matches_empty'}),
            ('replacements_rule_skipped', {'line': 6, 'reason': 'matches_empty'}),
        ])
        self.assertEqual(rules.apply('ok a'), 'OK a')


class LoadTests(unittest.TestCase):
    def load(self, path):
        log = []
        rules = replacements.load(path, lambda kind, **d: log.append((kind, d)))
        return rules, log

    def test_no_path_is_off_and_silent(self):
        for path in (None, ''):
            rules, log = self.load(path)
            self.assertFalse(rules)
            self.assertEqual(rules.apply('fubar npq'), 'fubar npq')
            self.assertEqual(log, [])

    def test_unusable_file_warns_and_continues(self):
        tmp = tempfile.mkdtemp()
        latin = write(tmp, '', 'latin.txt')
        Path(latin).write_bytes('caf\xe9 -> x\n'.encode('latin-1'))
        big = write(tmp, 'a -> b\n' * 50_000, 'big.txt')
        unreadable = write(tmp, 'a -> b\n', 'locked.txt')
        os.chmod(unreadable, 0)
        cases = [(os.path.join(tmp, 'absent.txt'), 'missing'), (tmp, 'not_a_file'),
                 (latin, 'not_utf8'), (big, 'too_large')]
        if not os.access(unreadable, os.R_OK):   # root can read anything
            cases.append((unreadable, 'cannot_read'))
        for path, reason in cases:
            with self.subTest(reason=reason):
                rules, log = self.load(path)
                self.assertFalse(rules)
                self.assertEqual(log, [('replacements_unavailable', {'reason': reason})])
                self.assertNotIn(tmp, json.dumps(log))

    def test_empty_file_and_tilde(self):
        home = tempfile.mkdtemp()
        Path(home, 'list.txt').write_text('\ufeff# only a comment\n\n')
        old = os.environ.get('HOME')
        os.environ['HOME'] = home
        try:
            rules, log = self.load('~/list.txt')
        finally:
            if old is None:
                del os.environ['HOME']
            else:
                os.environ['HOME'] = old
        self.assertFalse(rules)
        self.assertEqual(log, [('replacements_loaded', {'rules': 0})])

    def test_loaded_journal_has_no_rule_text(self):
        rules, log = self.load(write(tempfile.mkdtemp(), RULES, 'list.txt'))
        self.assertEqual(len(rules), 6)
        self.assertEqual(log, [('replacements_loaded', {'rules': 6})])


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.saved = logic.get('REPLACEMENTS')

    def tearDown(self):
        logic['REPLACEMENTS'] = self.saved

    def test_applied_after_sanitize_before_context_raw_kept(self):
        logic['REPLACEMENTS'], log = make()
        obj = worker(['Zeg het tegen fubar, dat is goed en echt.', 'Daarna de npq.'])
        self.assertEqual(obj.transcribe_chunk(np.zeros(16000)), 'Zeg het tegen FooBar, dat is goed en echt.')
        self.assertEqual(obj.last_raw, 'Zeg het tegen fubar, dat is goed en echt.')
        self.assertEqual(obj.transcribe_chunk(np.zeros(16000)), 'Daarna de MPQ.')
        # The rolling context carries the corrected spelling, never the wrong one.
        self.assertIn('FooBar', obj.calls[1]['initial_prompt'])
        self.assertNotIn('fubar', obj.calls[1]['initial_prompt'])
        self.assertIn('MPQ', obj._context)
        self.assertEqual(log, [('replacement', {'rule': 'L2', 'count': 1}),
                               ('replacement', {'rule': 'L3', 'count': 1})])

    def test_recovery_decode_is_corrected_too(self):
        logic['REPLACEMENTS'], _ = make()
        obj = worker(['Een herstelde zin over fubar.'])
        self.assertEqual(obj.transcribe_chunk(np.zeros(16000), recovery=True),
                         'Een herstelde zin over FooBar.')

    def test_artefact_filter_still_first(self):
        logic['REPLACEMENTS'], _ = make('tv gelderland 2021 -> echt\n')
        obj = worker(['TV Gelderland 2021', 'TV Gelderland 2021'])
        self.assertEqual(obj.transcribe_chunk(np.zeros(16000)), '')

    def test_empty_or_missing_list_is_unchanged_behaviour(self):
        texts = ['Een normale zin met fubar en npq.', 'Zeg het tegen fubar...', 'Dank u wel.', '***']
        logic['REPLACEMENTS'] = None
        baseline = []
        for text in texts:
            obj = worker([text, text])
            baseline.append((obj.transcribe_chunk(np.zeros(16000)), obj._context, len(obj.calls)))
        for empty in (replacements.ReplacementList(), replacements.load(None),
                      replacements.load('/nonexistent/list.txt')):
            logic['REPLACEMENTS'] = empty
            for text, expected in zip(texts, baseline):
                obj = worker([text, text])
                self.assertEqual((obj.transcribe_chunk(np.zeros(16000)), obj._context, len(obj.calls)),
                                 expected)

    def test_instance_list_wins_over_module_list(self):
        logic['REPLACEMENTS'], _ = make()
        obj = worker(['npq en fubar.'])
        obj.replacements, _ = make('npq -> QQQ\n')
        self.assertEqual(obj.transcribe_chunk(np.zeros(16000)), 'QQQ en fubar.')


class SettingsTests(unittest.TestCase):
    def test_key_default_and_yaml(self):
        self.assertIsNone(settings.load({}, '/nonexistent/config.yaml').get('recognition.replacements_file'))
        loaded = settings.load({}, write(tempfile.mkdtemp(), 'recognition:\n  replacements_file: ~/r.txt\n'))
        self.assertEqual(loaded.get('recognition.replacements_file'), '~/r.txt')
        self.assertEqual(loaded.events, [])

    def test_apply_settings_loads_list_or_warns(self):
        stub = lambda **kw: type('Stub', (), kw)  # noqa: E731
        ns = dictation_functions(
            'apply_settings', MultiTapKeyListener=stub(TAP_WINDOW=0.3),
            MouseToggleListener=stub(COALESCE_WINDOW=0.4),
            KeyboardReplayer=stub(TERMINAL_CLASSES={'kitty'}))
        tmp = tempfile.mkdtemp()
        rules = write(tmp, RULES, 'list.txt')
        for path, expected_rules, kind in ((rules, 6, 'replacements_loaded'),
                                           (os.path.join(tmp, 'gone.txt'), 0, 'replacements_unavailable')):
            config = write(tmp, 'recognition:\n  replacements_file: %s\n' % path)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                ns['apply_settings'](settings.load({}, config), {})
            self.assertEqual(len(ns['REPLACEMENTS']), expected_rules)
            self.assertIn('"kind": "%s"' % kind, out.getvalue())
            self.assertNotIn(tmp, out.getvalue())
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            ns['apply_settings'](settings.load({}, '/nonexistent/config.yaml'), {})
        self.assertFalse(ns['REPLACEMENTS'])
        self.assertNotIn('replacements', out.getvalue())


INSTALLER = (ROOT / 'install.sh').read_text()
FUNCTIONS = INSTALLER[INSTALLER.index('detect_ld_path() {'):INSTALLER.index('unit_is_current() {')]


class InstallerTests(unittest.TestCase):
    def seed(self, tmp, replacements_file):
        config_dir = os.path.join(tmp, 'cfg')
        code = ('set -euo pipefail\nCONFIG_DIR=%s\nCONFIG_FILE="$CONFIG_DIR/config.yaml"\nREPLACEMENTS_FILE=%s\n'
                % (shlex.quote(config_dir), shlex.quote(replacements_file)) + FUNCTIONS + '\nseed_config\n')
        result = subprocess.run(['bash', '-c', code], text=True, capture_output=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        return os.path.join(config_dir, 'config.yaml'), result.stdout

    def test_new_config_gets_key_only_when_given(self):
        tmp = tempfile.mkdtemp()
        config, _ = self.seed(tmp, '')
        self.assertNotIn('recognition', Path(config).read_text())
        tmp = tempfile.mkdtemp()
        odd = os.path.join(tmp, "it's: here.txt")
        Path(odd).write_text(RULES)
        config, _ = self.seed(tmp, odd)
        loaded = settings.load({}, config)
        self.assertEqual(loaded.events, [])
        self.assertEqual(loaded.get('recognition.replacements_file'), odd)
        self.assertEqual(loaded.get('model.name'), 'large-v3')

    def test_existing_config_untouched_with_hint(self):
        tmp = tempfile.mkdtemp()
        os.makedirs(os.path.join(tmp, 'cfg'))
        config = write(os.path.join(tmp, 'cfg'), 'model:\n  name: small\n')
        _, out = self.seed(tmp, '/some/list.txt')
        self.assertEqual(Path(config).read_text(), 'model:\n  name: small\n')
        self.assertIn('replacements_file: /some/list.txt', out)

    def test_missing_list_fails_before_any_change(self):
        tmp = tempfile.mkdtemp()
        env = dict(os.environ, HOME=tmp, PATH='/usr/bin:/bin')
        result = subprocess.run(['bash', str(ROOT / 'install.sh'), '--replacements-file=%s/absent.txt' % tmp],
                                env=env, text=True, capture_output=True, timeout=15)
        self.assertEqual(result.returncode, 2)
        self.assertIn('not found', result.stderr)
        self.assertEqual(os.listdir(tmp), [])


if __name__ == '__main__':
    unittest.main()
