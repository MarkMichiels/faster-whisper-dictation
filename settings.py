"""User settings from one YAML file, on top of the defaults in the code.

File: $DICTATION_CONFIG, else $XDG_CONFIG_HOME/faster-whisper-dictation/config.yaml
(default ~/.config/faster-whisper-dictation/config.yaml); --config PATH wins over both.
Reference: docs/configuration.md. Example: config.example.yaml.

Precedence, high to low: command line > environment variable > config.yaml >
default in code. Only the existing DICTATION_REWRITE_* variables count as
environment variables; there is no variable per key.

The service always starts: an unreadable file means all defaults, an invalid
key means the default for that key. The journal names keys, sources and reasons,
never a value (a value can be a path, a name or a mistyped secret).

Not configurable here, on purpose: audio formats, helper time-outs, recovery
heuristics and the safety and privacy gates (secret filter on the reply context,
https-only endpoint, zero-data-retention provider flags). A configuration file
must not be able to switch a safety rule off.

Command line (no service needed):
  python3 settings.py --doc          Markdown reference table (docs/configuration.md)
  python3 settings.py --example      the commented example file (config.example.yaml)
  python3 settings.py --migrate-env [--env-file PATH] [--write]
                                     turn rewrite.env into a `rewrite:` block; --write
                                     appends it to an existing config.yaml that has no
                                     `rewrite:` section yet, and never overwrites a file
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

CONFIG_ENV = 'DICTATION_CONFIG'
APP_DIR = 'faster-whisper-dictation'

# _SECRET_RE in context_capture.py, kept in sync by a test: a value with a token
# prefix or a PEM header is refused instead of passed on.
_SECRET_VALUE_RE = re.compile(r'-----BEGIN|(?<![\w-])(?:sk-|sk_live_|ghp_|gho_|github_pat_|glpat-|xox[abprs]-|AKIA|AIza)')
# Unknown key names that look like a credential. Refused, never passed on.
_SECRET_NAME_RE = re.compile(r'(^|_)(api_?key|key|token|password|passwd|secret|bearer)$', re.I)


@dataclass(frozen=True)
class Setting:
    path: str                     # YAML path, e.g. 'streaming.silence_ms'
    kind: str                     # str | str? | int | float | bool | list
    default: Any
    help: str
    dest: Optional[str] = None    # argparse dest when there is a CLI option
    flag: Optional[str] = None    # CLI option as shown in the docs
    invert: bool = False          # YAML true == CLI flag absent (--no-... options)
    env: Optional[str] = None     # legacy environment variable (rewrite only)
    choices: tuple = ()
    low: Optional[float] = None
    high: Optional[float] = None


def _s(*args, **kwargs) -> Setting:
    return Setting(*args, **kwargs)


_RW = 'DICTATION_REWRITE_'

# One table: defaults here are exactly the code defaults. argparse defaults,
# validation, docs/configuration.md and config.example.yaml all come from it.
SCHEMA: tuple[Setting, ...] = (
    _s('model.name', 'str', 'base', 'Whisper model: tiny ... large-v3, or a CTranslate2 model path or ID.',
       dest='model_name', flag='-m'),
    _s('model.device', 'str', 'cpu', 'Inference device: cpu, cuda or auto.', dest='device', flag='-v'),
    _s('model.compute_type', 'str', 'int8', 'Precision: int8, float16 or float32.',
       dest='compute_type', flag='-c'),
    _s('language.default', 'str?', None, 'Recognition language, e.g. nl; null = auto-detect.',
       dest='language', flag='-l'),
    _s('language.shift', 'str', 'en', 'Language for a session started with Left-Shift held.',
       dest='shift_language', flag='--shift-language'),
    _s('keys.double_key', 'str?', None, 'Key to double-tap; null = <ctrl_r> on Linux, <cmd_r> on macOS.',
       dest='double_key', flag='-d'),
    _s('keys.key_combo', 'str?', None, 'Key combination that toggles instead, e.g. <ctrl>+<alt>+a; '
       'null = <win>+z on Windows.', dest='key_combo', flag='-k'),
    _s('keys.tap_window_s', 'float', 0.3, 'Most seconds between the two taps of a double tap.',
       low=0.05, high=2),
    _s('mouse.buttons', 'list', ['button8', 'button9', 'button10', 'button11'],
       'pynput mouse buttons that toggle dictation; [] disables.', dest='mouse_buttons', flag='--mouse-buttons'),
    _s('mouse.coalesce_s', 'float', 0.4, 'Presses within this many seconds count as one.', low=0, high=5),
    _s('recording.max_time_s', 'int', 30, 'Recording limit in seconds; 0 = only the 30-minute session cap.',
       dest='max_time', flag='-t', low=0),
    _s('recording.auto_stop_silence_s', 'int', 10, 'Stop after this many seconds of silence; 0 = never.',
       dest='auto_stop_silence', flag='--auto-stop-silence', low=0),
    _s('recording.batch_mode', 'bool', False, 'Original batch mode instead of streaming.',
       dest='batch_mode', flag='--batch-mode'),
    _s('streaming.silence_ms', 'int', 1000, 'Pause in milliseconds that ends a speech chunk.',
       dest='silence_ms', flag='--silence-ms', low=1),
    _s('streaming.min_chunk_s', 'float', 2.5, 'Eager-flush floor in seconds; 0 disables eager flushing.',
       dest='min_chunk_s', flag='--min-chunk-s', low=0),
    _s('streaming.max_chunk_s', 'float', 10, 'Eager-flush ceiling in seconds.',
       dest='max_chunk_s', flag='--max-chunk-s', low=0.5),
    _s('streaming.eager_prob', 'float', 0.35, 'Speech probability at or below which a pause is cut early.',
       dest='eager_prob', flag='--eager-prob', low=0, high=1),
    _s('streaming.paste_min_s', 'float', 12.0, 'Seconds of audio merged before one paste; 0 = per chunk.',
       dest='paste_min_s', flag='--paste-min-s', low=0),
    _s('streaming.context_chars', 'int', 500, 'Characters of earlier text given to Whisper as prompt; 0 = off.',
       dest='context_chars', flag='--context-chars', low=0),
    _s('transcription.vad_filter', 'bool', False, 'Gate decoding with VAD (off: the phrase filter is used).',
       dest='transcribe_vad', flag='--transcribe-vad'),
    _s('output.hallucination_filter', 'bool', True, 'Drop chunks that are only a known silence artefact.',
       dest='no_hallucination_filter', flag='--no-hallucination-filter', invert=True),
    _s('output.space_rule_runs', 'bool', True, "Type '***' as '* * *' so it never renders as Markdown.",
       dest='no_space_rule_runs', flag='--no-space-rule-runs', invert=True),
    _s('output.ellipsis_style', 'str', 'space', "How '...' is typed: space, single or keep.",
       dest='ellipsis_style', flag='--ellipsis-style', choices=('space', 'single', 'keep')),
    _s('output.stop_status', 'bool', True, 'Paste a [Controle: ...] line after a session with a correction.'),
    _s('output.extra_terminal_classes', 'list', [], 'Extra window classes that paste with Ctrl+Shift+V '
       '(added to the built-in list).'),
    _s('sounds.start', 'str', 'assets/granted-04.wav', 'Start sound (WAV); relative to the working directory.'),
    _s('sounds.stop', 'str', 'assets/beepbeep.wav', 'Stop sound (WAV); relative to the working directory.'),
    _s('rewrite.endpoint', 'str?', None, 'Chat completions URL, https (or loopback); rewrite mode needs it.',
       env=_RW + 'ENDPOINT'),
    _s('rewrite.model', 'str?', None, 'Model id; rewrite mode needs it.', env=_RW + 'MODEL'),
    _s('rewrite.key_file', 'str?', None, 'Path of the file holding the API key; never the key itself.',
       env=_RW + 'KEY_FILE'),
    _s('rewrite.style_file', 'str?', None, 'Path of a short preferences file sent with every call.',
       env=_RW + 'STYLE_FILE'),
    _s('rewrite.terms_file', 'str?', None, 'Path of a term list, one term per line.', env=_RW + 'TERMS_FILE'),
    _s('rewrite.timeout_s', 'float', 30, 'Hard total time-out in seconds.', env=_RW + 'TIMEOUT_S', low=1, high=120),
    _s('rewrite.word_factor', 'float', 0, 'Word maximum for long dictations as factor x words; 0 = off.',
       env=_RW + 'WORD_FACTOR', low=0, high=1),
    _s('rewrite.provider_order', 'list', [], 'OpenRouter provider preference, e.g. [together, fireworks].',
       env=_RW + 'PROVIDER_ORDER'),
    _s('rewrite.signature', 'str', '', 'Name under the sign-off of an email; may span lines.',
       env=_RW + 'SIGNATURE'),
    _s('rewrite.context.max_chars', 'int', 100_000, 'Most characters of reply context sent.',
       env=_RW + 'CONTEXT_MAX_CHARS', low=1_000, high=500_000),
    _s('rewrite.context.primary_max_age_s', 'float', 20, 'Oldest highlighted text (PRIMARY) used as context.',
       low=0, high=3600),
    _s('rewrite.context.clipboard_max_age_s', 'float', 120, 'Oldest clipboard text used as context.',
       low=0, high=3600),
)
BY_PATH = {s.path: s for s in SCHEMA}
GROUPS = tuple(dict.fromkeys(s.path.split('.')[0] for s in SCHEMA))


@dataclass
class Loaded:
    """Merged settings plus metadata for the journal (never values)."""
    values: dict[str, Any]
    sources: dict[str, str]                       # path -> default | yaml | env | cli
    path: str
    status: str = 'defaults'                      # defaults | yaml | unreadable
    events: list[tuple[str, dict]] = field(default_factory=list)

    def get(self, path: str) -> Any:
        return self.values[path]

    def note(self, kind: str, **details: Any) -> None:
        self.events.append((kind, details))


def config_path(environ=None, explicit: Optional[str] = None) -> str:
    env = os.environ if environ is None else environ
    if explicit:
        return os.path.expanduser(explicit)
    if env.get(CONFIG_ENV):
        return os.path.expanduser(env[CONFIG_ENV])
    base = env.get('XDG_CONFIG_HOME') or os.path.join(os.path.expanduser('~'), '.config')
    return os.path.join(base, APP_DIR, 'config.yaml')


def _check(setting: Setting, value: Any) -> tuple[Any, Optional[str]]:
    """(normalised value, None) or (None, reason)."""
    kind = setting.kind
    if kind == 'bool':
        return (value, None) if isinstance(value, bool) else (None, 'wrong_type')
    if kind in ('int', 'float'):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None, 'wrong_type'
        if kind == 'int' and not float(value).is_integer():
            return None, 'wrong_type'
        value = int(value) if kind == 'int' else value
        if (setting.low is not None and value < setting.low) or (setting.high is not None and value > setting.high):
            return None, 'out_of_range'
        return value, None
    if kind == 'list':
        if not isinstance(value, list) or not all(isinstance(v, str) and v.strip() for v in value):
            return None, 'wrong_type'
        value = [v.strip() for v in value]
    elif value is None and kind == 'str?':
        return None, None
    elif not isinstance(value, str):
        return None, 'wrong_type'
    elif kind == 'str?' and not value.strip():
        return None, None
    if setting.choices and value not in setting.choices:
        return None, 'not_a_choice'
    texts = value if isinstance(value, list) else [value]
    if any(_SECRET_VALUE_RE.search(t) for t in texts):
        return None, 'looks_like_secret'
    return value, None


def _flatten(data: dict, prefix: str, loaded: Loaded, out: dict) -> None:
    for key, value in data.items():
        path = '%s%s' % (prefix, key)
        if not isinstance(key, str):
            loaded.note('config_unknown_key', key=prefix + '?')
            continue
        if path in BY_PATH:
            out[path] = value
        elif any(p.startswith(path + '.') for p in BY_PATH):
            if isinstance(value, dict):
                _flatten(value, path + '.', loaded, out)
            elif value is not None:   # a group whose keys are all commented out is null
                loaded.note('config_invalid', key=path, reason='not_a_mapping')
        elif _SECRET_NAME_RE.search(key):
            loaded.note('config_secret_refused', key=path)
        else:
            loaded.note('config_unknown_key', key=path)


def read_yaml(path: str, loaded: Loaded) -> dict[str, Any]:
    """Flat {path: raw value} from the file, or {} with the reason noted."""
    try:
        with open(path, encoding='utf-8') as fh:
            text = fh.read()
    except FileNotFoundError:
        return {}
    except (OSError, UnicodeDecodeError):
        loaded.status = 'unreadable'
        loaded.note('config_unreadable', reason='cannot_read')
        return {}
    try:
        import yaml
    except ImportError:
        loaded.status = 'unreadable'
        loaded.note('config_unreadable', reason='pyyaml_missing')
        return {}
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        # Position only: PyYAML's message quotes the offending line.
        mark = getattr(exc, 'problem_mark', None)
        loaded.status = 'unreadable'
        loaded.note('config_unreadable', reason='yaml_error',
                    line=mark.line + 1 if mark else None, column=mark.column + 1 if mark else None)
        return {}
    loaded.status = 'yaml'
    if data is None:
        return {}
    if not isinstance(data, dict):
        loaded.status = 'unreadable'
        loaded.note('config_unreadable', reason='not_a_mapping')
        return {}
    flat: dict[str, Any] = {}
    _flatten(data, '', loaded, flat)
    return flat


def load(environ=None, explicit_path: Optional[str] = None) -> Loaded:
    """Defaults, then config.yaml, then the legacy environment variables."""
    env = os.environ if environ is None else environ
    path = config_path(env, explicit_path)
    loaded = Loaded(values={s.path: s.default for s in SCHEMA},
                    sources={s.path: 'default' for s in SCHEMA}, path=path)
    for key, raw in read_yaml(path, loaded).items():
        value, problem = _check(BY_PATH[key], raw)
        if problem == 'looks_like_secret':
            loaded.note('config_secret_refused', key=key)
        elif problem:
            loaded.note('config_invalid', key=key, reason=problem)
        else:
            loaded.values[key] = value
            loaded.sources[key] = 'yaml'
    for setting in SCHEMA:
        # An empty variable counts as absent, as rewrite.load_config always did.
        if setting.env and (env.get(setting.env) or '').strip():
            if loaded.sources[setting.path] == 'yaml':
                loaded.note('config_override', key=setting.path, source='env')
            loaded.sources[setting.path] = 'env'
    return loaded


def cli_defaults(loaded: Loaded) -> dict[str, Any]:
    """argparse defaults (dest -> value) for every setting with a CLI option."""
    out = {}
    for s in SCHEMA:
        if not s.dest:
            continue
        value = loaded.values[s.path]
        if s.invert:
            value = not value
        elif s.kind == 'list':
            value = ','.join(value)
        out[s.dest] = value
    return out


def mark_cli(loaded: Loaded, explicit_dests) -> None:
    """Record which settings an explicit command-line option decided."""
    for s in SCHEMA:
        if s.dest and s.dest in explicit_dests:
            if loaded.sources[s.path] != 'default':
                loaded.note('config_override', key=s.path, source='cli')
            loaded.sources[s.path] = 'cli'


def _env_text(setting: Setting, value: Any) -> str:
    if setting.kind == 'list':
        return ','.join(value)
    if setting.path == 'rewrite.signature':
        # rewrite._signature splits on a literal backslash-n.
        return '\\n'.join(value.splitlines())
    if setting.path.endswith('_file'):
        return os.path.expanduser(value)
    return str(value)


def rewrite_environ(loaded: Loaded, environ=None) -> dict[str, str]:
    """DICTATION_REWRITE_* view for rewrite.load_config: YAML values with the
    real environment on top. Without config.yaml this equals the environment."""
    env = os.environ if environ is None else environ
    out = {}
    for s in SCHEMA:
        if not s.env:
            continue
        if (env.get(s.env) or '').strip():
            out[s.env] = env[s.env]
        elif loaded.sources[s.path] == 'yaml' and loaded.values[s.path] not in (None, '', []):
            out[s.env] = _env_text(s, loaded.values[s.path])
    return out


def report(loaded: Loaded, log: Callable[..., None]) -> None:
    """Journal lines: key names, sources and reasons; never a value."""
    for kind, details in loaded.events:
        log(kind, **details)
    chosen = {k: v for k, v in loaded.sources.items() if v != 'default'}
    log('config_loaded', source=loaded.status, file_exists=os.path.exists(loaded.path),
        keys=sum(1 for v in chosen.values() if v == 'yaml'),
        invalid=sum(1 for k, _ in loaded.events if k in ('config_invalid', 'config_secret_refused')),
        sources=chosen)


# --- documentation and example, generated from SCHEMA -----------------------

def _yaml_value(value: Any) -> str:
    if value is None:
        return 'null'
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, list):
        return '[%s]' % ', '.join(value)
    if isinstance(value, str):
        return '""' if value == '' else value
    return repr(value)


def _range(s: Setting) -> str:
    if s.choices:
        return ' | '.join(s.choices)
    if s.low is not None or s.high is not None:
        low = '' if s.low is None else _yaml_value(s.low)
        high = '' if s.high is None else _yaml_value(s.high)
        return '%s–%s' % (low, high) if high else '≥ %s' % low
    return ''


def doc_table() -> str:
    rows = ['| Key | Type | Default | Range | CLI | Environment | Effect |',
            '|---|---|---|---|---|---|---|']
    for s in SCHEMA:
        kind = {'str?': 'string or null', 'str': 'string', 'list': 'list of strings'}.get(s.kind, s.kind)
        rows.append('| `%s` | %s | `%s` | %s | %s | %s | %s |' % (
            s.path, kind, _yaml_value(s.default), _range(s),
            '`%s`' % s.flag if s.flag else '', '`%s`' % s.env if s.env else '', s.help))
    return '\n'.join(rows) + '\n'


EXAMPLE_HEADER = """\
# faster-whisper-dictation settings.
# Copy to ~/.config/faster-whisper-dictation/config.yaml and uncomment what you change.
# Every value shown is the default. Reference: docs/configuration.md
# Precedence: command line > DICTATION_REWRITE_* environment > this file > default.
# Never put a secret here: rewrite.key_file is the path of the file holding the key.
# A change takes effect after: systemctl --user restart spraakherkenning
"""


def example() -> str:
    lines = [EXAMPLE_HEADER.rstrip('\n')]
    for group in GROUPS:
        lines += ['', '%s:' % group]
        sub = None
        for s in SCHEMA:
            parts = s.path.split('.')
            if parts[0] != group:
                continue
            indent = '  '
            if len(parts) == 3:
                if sub != parts[1]:
                    sub = parts[1]
                    lines.append('  %s:' % sub)
                indent = '    '
            lines.append('%s# %s: %s    # %s' % (indent, parts[-1], _yaml_value(s.default), s.help))
    return '\n'.join(lines) + '\n'


# --- migration from rewrite.env ---------------------------------------------

def read_env_file(path: str) -> dict[str, str]:
    """KEY=value lines as systemd's EnvironmentFile reads them (simple subset)."""
    out = {}
    with open(path, encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith(('#', ';')) or '=' not in line:
                continue
            key, value = line.split('=', 1)
            key = key.strip()
            if key.startswith('export '):
                key = key[len('export '):].strip()
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in '"\'':
                value = value[1:-1]
            out[key] = value
    return out


def migrate_block(env_values: dict[str, str]) -> str:
    """A `rewrite:` YAML block from DICTATION_REWRITE_* values. Never opens the key file."""
    import yaml
    data: dict[str, Any] = {}
    for s in SCHEMA:
        raw = (env_values.get(s.env or '') or '').strip() if s.env else ''
        if not raw:
            continue
        if s.kind == 'list':
            value: Any = [p.strip() for p in raw.split(',') if p.strip()]
        elif s.kind in ('int', 'float'):
            number = float(raw.replace('_', ''))
            value = int(number) if number.is_integer() else number
        elif s.path == 'rewrite.signature':
            value = '\n'.join(p.strip() for p in raw.split('\\n'))
        else:
            value = raw
        node = data
        parts = s.path.split('.')
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    if not data:
        return ''
    return '# Migrated from rewrite.env by settings.py --migrate-env.\n' + yaml.safe_dump(
        data, sort_keys=False, allow_unicode=True, default_flow_style=False)


def _has_rewrite_section(path: str) -> bool:
    import yaml
    with open(path, encoding='utf-8') as fh:
        data = yaml.safe_load(fh)
    return isinstance(data, dict) and 'rewrite' in data


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='faster-whisper-dictation settings helper')
    parser.add_argument('--doc', action='store_true', help='print the reference table')
    parser.add_argument('--example', action='store_true', help='print config.example.yaml')
    parser.add_argument('--migrate-env', action='store_true', help='rewrite.env -> YAML block')
    parser.add_argument('--env-file', help='rewrite.env to migrate (default: next to config.yaml)')
    parser.add_argument('--config', help='config.yaml path (default: see docs/configuration.md)')
    parser.add_argument('--write', action='store_true',
                        help='append the block to an existing config.yaml without a rewrite: section')
    args = parser.parse_args(argv)
    if args.doc:
        sys.stdout.write(doc_table())
        return 0
    if args.example:
        sys.stdout.write(example())
        return 0
    if not args.migrate_env:
        parser.print_help()
        return 2
    path = config_path(explicit=args.config)
    env_file = args.env_file or os.path.join(os.path.dirname(path), 'rewrite.env')
    try:
        block = migrate_block(read_env_file(env_file))
    except FileNotFoundError:
        print('No rewrite.env at %s: nothing to migrate.' % env_file, file=sys.stderr)
        return 1
    if not block:
        print('%s holds no DICTATION_REWRITE_* setting: nothing to migrate.' % env_file, file=sys.stderr)
        return 1
    if not args.write:
        sys.stdout.write(block)
        return 0
    if not os.path.exists(path):
        print('%s does not exist; run install.sh first (it creates the file once).' % path, file=sys.stderr)
        return 1
    if _has_rewrite_section(path):
        print('%s already has a rewrite: section; nothing written.' % path, file=sys.stderr)
        return 1
    with open(path, 'a', encoding='utf-8') as fh:
        fh.write('\n' + block)
    print('Appended a rewrite: section to %s. rewrite.env still wins until it is renamed.' % path)
    return 0


if __name__ == '__main__':
    sys.exit(main())
