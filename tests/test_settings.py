"""Offline tests for settings.py: schema, precedence, error paths, journal, migration."""
import argparse
import ast
import contextlib
import io
import json
import os
import re
import sys
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import context_capture  # noqa: E402
import coverage_ledger  # noqa: E402
import rewrite  # noqa: E402
import settings  # noqa: E402

DICTATION = (ROOT / 'dictation.py').read_text()
TREE = ast.parse(DICTATION)


def dictation_functions(*names, **ns):
    """Selected top-level functions of dictation.py, without importing it (no X needed)."""
    nodes = [n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name in names]
    namespace = dict(argparse=argparse, settings=settings, os=os, **ns)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'dictation.py', 'exec'), namespace)
    return namespace


def class_attr(cls, name):
    for node in TREE.body:
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for item in node.body:
                if isinstance(item, ast.Assign) and item.targets[0].id == name:
                    return ast.literal_eval(item.value)
    raise KeyError(name)


def module_constant(name):
    for node in TREE.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], 'id', None) == name:
            return ast.literal_eval(node.value)
    raise KeyError(name)


CLI = dictation_functions('build_parser', 'parse_args')


def write(tmp, text, name='config.yaml'):
    path = Path(tmp) / name
    path.write_text(text)
    return str(path)


def journal(loaded):
    lines = []
    settings.report(loaded, lambda kind, **details: lines.append((kind, details)))
    return lines


class SchemaTests(unittest.TestCase):
    def test_defaults_equal_the_code(self):
        loaded = settings.load({}, '/nonexistent/config.yaml')
        parser = CLI['build_parser']()
        for setting in settings.SCHEMA:
            if setting.dest:
                with self.subTest(setting.path):
                    self.assertEqual(settings.cli_defaults(loaded)[setting.dest], parser.get_default(setting.dest))
        get = loaded.get
        self.assertEqual(get('keys.tap_window_s'), class_attr('MultiTapKeyListener', 'TAP_WINDOW'))
        self.assertEqual(get('mouse.coalesce_s'), class_attr('MouseToggleListener', 'COALESCE_WINDOW'))
        self.assertEqual(get('output.stop_status'), coverage_ledger.STOP_STATUS)
        self.assertEqual(get('sounds.start'), module_constant('SOUND_START'))
        self.assertEqual(get('sounds.stop'), module_constant('SOUND_STOP'))
        self.assertEqual(get('rewrite.timeout_s'), rewrite.DEFAULT_TIMEOUT_S)
        self.assertEqual(get('rewrite.context.max_chars'), context_capture.DEFAULT_MAX_CHARS)
        self.assertEqual(get('rewrite.context.primary_max_age_s'), context_capture.PRIMARY_MAX_AGE_S)
        self.assertEqual(get('rewrite.context.clipboard_max_age_s'), context_capture.CLIPBOARD_MAX_AGE_S)
        limit = settings.BY_PATH['rewrite.context.max_chars']
        self.assertEqual((limit.low, limit.high), rewrite.CONTEXT_MAX_CHARS_RANGE)
        self.assertIsNone(module_constant('REWRITE_ENVIRON'))
        self.assertIsNone(module_constant('CONTEXT_MAX_AGES'))

    def test_every_cli_dest_exists_and_secret_pattern_in_step(self):
        dests = {a.dest for a in CLI['build_parser']()._actions}
        for setting in settings.SCHEMA:
            if setting.dest:
                self.assertIn(setting.dest, dests)
        self.assertEqual(settings._SECRET_VALUE_RE.pattern, context_capture._SECRET_RE.pattern)

    def test_example_file_matches_schema_and_loads_as_defaults(self):
        self.assertEqual((ROOT / 'config.example.yaml').read_text(), settings.example())
        loaded = settings.load({}, str(ROOT / 'config.example.yaml'))
        self.assertEqual(loaded.status, 'yaml')
        self.assertEqual(loaded.events, [])
        self.assertEqual(set(loaded.sources.values()), {'default'})

    def test_uncommented_example_lines_are_valid(self):
        text = re.sub(r'^(\s+)# (\w+: )', r'\1\2', settings.example(), flags=re.M)
        loaded = settings.load({}, write(tempfile.mkdtemp(), text))
        self.assertEqual(loaded.events, [])
        self.assertEqual(set(loaded.sources.values()), {'yaml'})
        self.assertEqual(loaded.values, {s.path: s.default for s in settings.SCHEMA})

    def test_doc_table_matches_schema(self):
        doc = (ROOT / 'docs/configuration.md').read_text()
        table = doc.split('<!-- settings-table:begin -->\n')[1].split('<!-- settings-table:end -->')[0]
        self.assertEqual(table, settings.doc_table())


class PrecedenceTests(unittest.TestCase):
    YAML = """\
model:
  name: large-v3
  device: cuda
  compute_type: float16
language:
  default: nl
recording:
  max_time_s: 0
output:
  hallucination_filter: false
  stop_status: false
mouse:
  buttons: []
rewrite:
  model: yaml/model
  endpoint: https://llm.invalid/v1/chat/completions
"""

    def test_no_file_is_exactly_the_old_behaviour(self):
        env = {'DICTATION_REWRITE_MODEL': 'm', 'DICTATION_REWRITE_ENDPOINT': 'https://x.invalid', 'OTHER': '1',
               'DICTATION_REWRITE_TIMEOUT_S': '7'}
        loaded = settings.load(env, '/nonexistent/config.yaml')
        self.assertEqual(loaded.status, 'defaults')
        self.assertEqual(settings.rewrite_environ(loaded, env),
                         {k: v for k, v in env.items() if k.startswith('DICTATION_REWRITE_')})
        self.assertEqual(rewrite.load_config(settings.rewrite_environ(loaded, env)), rewrite.load_config(env))
        args = CLI['parse_args'](['-m', 'large-v3', '-l', 'nl'], {'DICTATION_CONFIG': '/nonexistent/c.yaml'})
        self.assertEqual((args.model_name, args.device, args.language, args.max_time), ('large-v3', 'cpu', 'nl', 30))

    def test_yaml_then_env_then_cli(self):
        path = write(tempfile.mkdtemp(), self.YAML)
        env = {'DICTATION_CONFIG': path, 'DICTATION_REWRITE_MODEL': 'env/model'}
        args = CLI['parse_args'](['-c', 'int8'], env)
        self.assertEqual((args.model_name, args.device, args.compute_type, args.language, args.max_time),
                         ('large-v3', 'cuda', 'int8', 'nl', 0))
        self.assertTrue(args.no_hallucination_filter)
        self.assertEqual(args.mouse_buttons, '')
        sources = args.settings.sources
        self.assertEqual(sources['model.name'], 'yaml')
        self.assertEqual(sources['model.compute_type'], 'cli')
        self.assertEqual(sources['rewrite.model'], 'env')
        self.assertEqual(sources['streaming.silence_ms'], 'default')
        self.assertEqual(settings.rewrite_environ(args.settings, env)['DICTATION_REWRITE_MODEL'], 'env/model')
        self.assertIn(('config_override', {'key': 'rewrite.model', 'source': 'env'}), args.settings.events)
        self.assertIn(('config_override', {'key': 'model.compute_type', 'source': 'cli'}), args.settings.events)
        # --config wins over DICTATION_CONFIG; a store_true option can still switch on.
        other = write(tempfile.mkdtemp(), 'model:\n  name: small\n')
        args = CLI['parse_args'](['--config', other, '--transcribe-vad'], env)
        self.assertEqual((args.model_name, args.transcribe_vad), ('small', True))
        self.assertEqual(args.settings.sources['transcription.vad_filter'], 'cli')

    def test_empty_env_variable_counts_as_absent(self):
        path = write(tempfile.mkdtemp(), self.YAML)
        env = {'DICTATION_REWRITE_MODEL': '  '}
        loaded = settings.load(env, path)
        self.assertEqual(settings.rewrite_environ(loaded, env)['DICTATION_REWRITE_MODEL'], 'yaml/model')

    def test_rewrite_yaml_reaches_load_config(self):
        tmp = tempfile.mkdtemp()
        key = Path(tmp) / 'key.env'
        key.write_text('KEY=value\n')
        path = write(tmp, """\
rewrite:
  endpoint: https://llm.invalid/v1/chat/completions
  model: m
  key_file: %s
  timeout_s: 12
  word_factor: 0.5
  provider_order: [a, b]
  signature: |
    First Last
    Team
  context:
    max_chars: 30000
    primary_max_age_s: 45
""" % key)
        loaded = settings.load({}, path)
        cfg, reason = rewrite.load_config(settings.rewrite_environ(loaded, {}))
        self.assertIsNone(reason)
        self.assertEqual((cfg.model, cfg.timeout_s, cfg.word_factor, cfg.provider_order, cfg.signature,
                          cfg.context_max_chars, cfg.key_file),
                         ('m', 12.0, 0.5, ('a', 'b'), 'First Last\nTeam', 30000, str(key)))
        self.assertEqual(loaded.get('rewrite.context.primary_max_age_s'), 45)

    def test_tilde_in_file_paths_is_expanded(self):
        loaded = settings.load({}, write(tempfile.mkdtemp(), 'rewrite:\n  style_file: ~/style.md\n'))
        self.assertEqual(settings.rewrite_environ(loaded, {})['DICTATION_REWRITE_STYLE_FILE'],
                         os.path.expanduser('~/style.md'))

    def test_https_gate_stays_in_code(self):
        path = write(tempfile.mkdtemp(), 'rewrite:\n  endpoint: http://llm.invalid/v1\n  model: m\n')
        self.assertEqual(rewrite.load_config(settings.rewrite_environ(settings.load({}, path), {})),
                         (None, 'endpoint_not_https'))

    def test_apply_settings(self):
        stub = lambda **kw: type('Stub', (), kw)  # noqa: E731
        ns = dictation_functions(
            'apply_settings', MultiTapKeyListener=stub(TAP_WINDOW=0.3),
            MouseToggleListener=stub(COALESCE_WINDOW=0.4),
            KeyboardReplayer=stub(TERMINAL_CLASSES={'kitty'}),
            SOUND_START=None, SOUND_STOP=None, REWRITE_ENVIRON=None, CONTEXT_MAX_AGES=None)
        path = write(tempfile.mkdtemp(), """\
keys: {tap_window_s: 0.5}
mouse: {coalesce_s: 1}
output: {stop_status: false, extra_terminal_classes: [foot]}
sounds: {start: ~/a.wav}
rewrite: {context: {clipboard_max_age_s: 60}}
""")
        before = coverage_ledger.STOP_STATUS
        try:
            ns['apply_settings'](settings.load({}, path), {})
            self.assertFalse(coverage_ledger.STOP_STATUS)
        finally:
            coverage_ledger.STOP_STATUS = before
        self.assertEqual(ns['MultiTapKeyListener'].TAP_WINDOW, 0.5)
        self.assertEqual(ns['MouseToggleListener'].COALESCE_WINDOW, 1)
        self.assertEqual(ns['KeyboardReplayer'].TERMINAL_CLASSES, {'kitty', 'foot'})
        self.assertEqual(ns['SOUND_START'], os.path.expanduser('~/a.wav'))
        self.assertEqual(ns['SOUND_STOP'], 'assets/beepbeep.wav')
        self.assertEqual(ns['CONTEXT_MAX_AGES'], (20, 60))
        self.assertEqual(ns['REWRITE_ENVIRON'], {})


class ErrorPathTests(unittest.TestCase):
    SECRET = 'sk-or-v1-0123456789abcdef'

    def load(self, text):
        return settings.load({}, write(tempfile.mkdtemp(), text))

    def assert_no_values(self, loaded, *values):
        text = json.dumps(journal(loaded))
        for value in values:
            self.assertNotIn(value, text)

    def test_missing_file(self):
        loaded = settings.load({}, '/nonexistent/config.yaml')
        self.assertEqual(journal(loaded), [('config_loaded', dict(source='defaults', file_exists=False, keys=0,
                                                                  invalid=0, sources={}))])

    def test_yaml_error_reports_position_only(self):
        loaded = self.load('model:\n  name: [unclosed-%s\n' % 'privateword')
        self.assertEqual(loaded.status, 'unreadable')
        kind, details = loaded.events[0]
        self.assertEqual((kind, details['reason']), ('config_unreadable', 'yaml_error'))
        self.assertIsInstance(details['line'], int)
        self.assertEqual(set(loaded.sources.values()), {'default'})
        self.assert_no_values(loaded, 'privateword')

    def test_top_level_not_a_mapping(self):
        loaded = self.load('- a\n- b\n')
        self.assertEqual(loaded.events, [('config_unreadable', {'reason': 'not_a_mapping'})])

    def test_unknown_wrong_type_out_of_range_choice(self):
        loaded = self.load("""\
streaming:
  silense_ms: 3
  silence_ms: "fast"
  eager_prob: 7
output:
  ellipsis_style: dots
recording:
  batch_mode: "yes"
  max_time_s: 1.5
model: large
""")
        self.assertEqual(sorted(loaded.events, key=repr), sorted([
            ('config_unknown_key', {'key': 'streaming.silense_ms'}),
            ('config_invalid', {'key': 'streaming.silence_ms', 'reason': 'wrong_type'}),
            ('config_invalid', {'key': 'streaming.eager_prob', 'reason': 'out_of_range'}),
            ('config_invalid', {'key': 'output.ellipsis_style', 'reason': 'not_a_choice'}),
            ('config_invalid', {'key': 'recording.batch_mode', 'reason': 'wrong_type'}),
            ('config_invalid', {'key': 'recording.max_time_s', 'reason': 'wrong_type'}),
            ('config_invalid', {'key': 'model', 'reason': 'not_a_mapping'}),
        ], key=repr))
        self.assertEqual(set(loaded.sources.values()), {'default'})
        self.assert_no_values(loaded, 'fast', 'dots')

    def test_secret_name_and_value_refused(self):
        loaded = self.load('rewrite:\n  api_key: %s\n  model: %s\ntoken: abc\n' % (self.SECRET, self.SECRET))
        self.assertEqual(sorted(loaded.events, key=repr), [
            ('config_secret_refused', {'key': 'rewrite.api_key'}),
            ('config_secret_refused', {'key': 'rewrite.model'}),
            ('config_secret_refused', {'key': 'token'}),
        ])
        self.assertEqual(settings.rewrite_environ(loaded, {}), {})
        self.assert_no_values(loaded, self.SECRET)
        self.assertEqual(journal(loaded)[-1][1]['invalid'], 3)

    def test_null_group_and_empty_file(self):
        self.assertEqual(self.load('model:\nrewrite:\n').events, [])
        self.assertEqual(self.load('').status, 'yaml')

    def test_journal_has_sources_not_values(self):
        loaded = self.load('rewrite:\n  model: vendor/secret-model-name\n  signature: Private Person\n')
        last = journal(loaded)[-1]
        self.assertEqual(last[0], 'config_loaded')
        self.assertEqual(last[1]['sources'], {'rewrite.model': 'yaml', 'rewrite.signature': 'yaml'})
        self.assert_no_values(loaded, 'secret-model-name', 'Private Person')


class MigrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.key = Path(self.tmp) / 'key.env'
        self.key.write_text('KEY=never-read-this-value\n')
        self.env_file = write(self.tmp, """\
# comment
DICTATION_REWRITE_ENDPOINT=https://llm.invalid/v1/chat/completions
DICTATION_REWRITE_MODEL="vendor/model"
DICTATION_REWRITE_KEY_FILE=%s
DICTATION_REWRITE_TIMEOUT_S=30
DICTATION_REWRITE_PROVIDER_ORDER=a, b
DICTATION_REWRITE_SIGNATURE=First Last\\nTeam
DICTATION_REWRITE_CONTEXT_MAX_CHARS=100_000
""" % self.key, 'rewrite.env')
        self.config = str(Path(self.tmp) / 'config.yaml')

    def run_main(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = settings.main(['--migrate-env', '--env-file', self.env_file, '--config', self.config, *argv])
        return code, out.getvalue(), err.getvalue()

    def test_preview_equals_env_behaviour_and_never_reads_key(self):
        code, block, _ = self.run_main()
        self.assertEqual(code, 0)
        self.assertNotIn('never-read-this-value', block)
        Path(self.config).write_text(block)
        env = settings.read_env_file(self.env_file)
        loaded = settings.load({}, self.config)
        self.assertEqual(loaded.events, [])
        self.assertEqual(rewrite.load_config(settings.rewrite_environ(loaded, {})), rewrite.load_config(env))

    def test_write_appends_once_and_never_creates(self):
        code, _, err = self.run_main('--write')
        self.assertEqual(code, 1)
        self.assertIn('run install.sh first', err)
        self.assertFalse(os.path.exists(self.config))
        seed = 'model:\n  name: large-v3\n'
        Path(self.config).write_text(seed)
        self.assertEqual(self.run_main('--write')[0], 0)
        text = Path(self.config).read_text()
        self.assertTrue(text.startswith(seed))
        self.assertEqual(settings.load({}, self.config).get('rewrite.model'), 'vendor/model')
        self.assertEqual(settings.load({}, self.config).get('model.name'), 'large-v3')
        self.assertEqual(self.run_main('--write')[0], 1)
        self.assertEqual(Path(self.config).read_text(), text)


if __name__ == '__main__':
    unittest.main()
