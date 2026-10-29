# Configuration

All settings live in one optional YAML file. Without it the application runs
with the defaults in the code; command-line options still work and win.

## Where the file is

| Route | Path |
|---|---|
| Default | `$XDG_CONFIG_HOME/faster-whisper-dictation/config.yaml`, usually `~/.config/faster-whisper-dictation/config.yaml` |
| Environment | `DICTATION_CONFIG=/path/to/config.yaml` |
| Command line | `dictation.py --config /path/to/config.yaml` |

The installed service sets `DICTATION_CONFIG` to
`%h/.config/faster-whisper-dictation/config.yaml`. `install.sh` creates that
file **once**, with the installer's choices (`large-v3`, `cuda`, `float16`,
language `nl`, no short recording limit); it never overwrites an existing file
or symlink. `install.sh --check` reports `NOT CURRENT` when the file is missing,
because the service starts without command-line options and would otherwise fall
back to the code defaults (`base` on CPU, language auto-detect).

[`config.example.yaml`](../config.example.yaml) lists every key, commented out,
with its default. Copy it, or add only the keys you change. A change takes effect
after `systemctl --user restart spraakherkenning`; the file is not watched.
Restart only when no dictation is running.

## Precedence

From high to low:

1. **Command line** (`-m`, `-l`, `--silence-ms`, ...).
2. **Environment variable**: only the existing `DICTATION_REWRITE_*` variables,
   for example from the older `rewrite.env` (see below). There is no variable per
   key. An empty variable counts as absent.
3. **`config.yaml`**.
4. **Default in the code.**

Options such as `--no-hallucination-filter` or `--transcribe-vad` can only switch
*to* the non-default on the command line; in the YAML file they are plain
booleans (`output.hallucination_filter: false`, `transcription.vad_filter: true`).

## Errors never stop the service

| Situation | Behaviour | Journal line (`DCT-COVERAGE`, kind) |
|---|---|---|
| No file | all defaults | `config_loaded` with `source=defaults` |
| Unreadable, or not valid YAML | all defaults | `config_unreadable` with `reason=yaml_error` and line/column only |
| Top level is not a mapping | all defaults | `config_unreadable` with `reason=not_a_mapping` |
| PyYAML not installed | all defaults | `config_unreadable` with `reason=pyyaml_missing` |
| Unknown key (typo) | ignored | `config_unknown_key` with the key path |
| Wrong type, out of range, not an allowed choice | default for that key | `config_invalid` with key and reason |
| Key name or value that looks like a credential | ignored, never passed on | `config_secret_refused` with the key path |
| Same key set by a higher source | the higher source wins | `config_override` with key and `source=env` or `source=cli` |
| Every start | – | `config_loaded`: `source` (defaults, yaml, unreadable), `file_exists`, number of keys from the file, number refused, and `sources`: for each setting not at its default, which source decided it (`yaml`, `env`, `cli`) |

The journal never shows a value from this file: only key paths, sources and
reasons. The usual start lines (`Loading Whisper model ...`, `Streaming dictation
mode (... language: nl)`, `Rewrite mode: model ...`) still show the effective
model, language, rewrite model and context limit.

```sh
journalctl --user -u spraakherkenning -b | grep -E 'config_|Rewrite mode|language:'
```

The rewrite mode keeps its own checks after merging: an endpoint that is not
https (or loopback), a missing model or an unreadable key file make the mode
unavailable with the same reasons as before (`endpoint_not_https`, ...).

## Secrets

Never put a key, token or password in this file. `rewrite.key_file` is the path
of a file that holds the key; the key is read from it per request and never
logged. Keep that key file at `chmod 600`. The settings file itself holds no
secret and may be readable by others.

## What is deliberately not configurable

Audio formats (16 kHz, 512/1024 frames), helper-program time-outs, the recovery
heuristics, the voice-activity and level thresholds, prompts and the silence
artefact list stay in the code. So do the safety and privacy gates: the secret
filter on the reply context, the https-only endpoint rule and the
zero-data-retention / no-data-collection provider flags. A configuration file
must not be able to switch those off.

## Keys

Generated from `settings.py` (`python3 settings.py --doc`); a test keeps this
table and `config.example.yaml` in step with the code.

<!-- settings-table:begin -->
| Key | Type | Default | Range | CLI | Environment | Effect |
|---|---|---|---|---|---|---|
| `model.name` | string | `base` |  | `-m` |  | Whisper model: tiny ... large-v3, or a CTranslate2 model path or ID. |
| `model.device` | string | `cpu` |  | `-v` |  | Inference device: cpu, cuda or auto. |
| `model.compute_type` | string | `int8` |  | `-c` |  | Precision: int8, float16 or float32. |
| `language.default` | string or null | `null` |  | `-l` |  | Recognition language, e.g. nl; null = auto-detect. |
| `language.shift` | string | `en` |  | `--shift-language` |  | Language for a session started with Left-Shift held. |
| `keys.double_key` | string or null | `null` |  | `-d` |  | Key to double-tap; null = <ctrl_r> on Linux, <cmd_r> on macOS. |
| `keys.key_combo` | string or null | `null` |  | `-k` |  | Key combination that toggles instead, e.g. <ctrl>+<alt>+a; null = <win>+z on Windows. |
| `keys.tap_window_s` | float | `0.3` | 0.05–2 |  |  | Most seconds between the two taps of a double tap. |
| `mouse.buttons` | list of strings | `[button8, button9, button10, button11]` |  | `--mouse-buttons` |  | pynput mouse buttons that toggle dictation; [] disables. |
| `mouse.coalesce_s` | float | `0.4` | 0–5 |  |  | Presses within this many seconds count as one. |
| `recording.max_time_s` | int | `30` | ≥ 0 | `-t` |  | Recording limit in seconds; 0 = only the 30-minute session cap. |
| `recording.auto_stop_silence_s` | int | `10` | ≥ 0 | `--auto-stop-silence` |  | Stop after this many seconds of silence; 0 = never. |
| `recording.batch_mode` | bool | `false` |  | `--batch-mode` |  | Original batch mode instead of streaming. |
| `streaming.silence_ms` | int | `1000` | ≥ 1 | `--silence-ms` |  | Pause in milliseconds that ends a speech chunk. |
| `streaming.min_chunk_s` | float | `2.5` | ≥ 0 | `--min-chunk-s` |  | Eager-flush floor in seconds; 0 disables eager flushing. |
| `streaming.max_chunk_s` | float | `10` | ≥ 0.5 | `--max-chunk-s` |  | Eager-flush ceiling in seconds. |
| `streaming.eager_prob` | float | `0.35` | 0–1 | `--eager-prob` |  | Speech probability at or below which a pause is cut early. |
| `streaming.paste_min_s` | float | `12.0` | ≥ 0 | `--paste-min-s` |  | Seconds of audio merged before one paste; 0 = per chunk. |
| `streaming.context_chars` | int | `500` | ≥ 0 | `--context-chars` |  | Characters of earlier text given to Whisper as prompt; 0 = off. |
| `transcription.vad_filter` | bool | `false` |  | `--transcribe-vad` |  | Gate decoding with VAD (off: the phrase filter is used). |
| `output.hallucination_filter` | bool | `true` |  | `--no-hallucination-filter` |  | Drop chunks that are only a known silence artefact. |
| `output.space_rule_runs` | bool | `true` |  | `--no-space-rule-runs` |  | Type '***' as '* * *' so it never renders as Markdown. |
| `output.ellipsis_style` | string | `space` | space | single | keep | `--ellipsis-style` |  | How '...' is typed: space, single or keep. |
| `output.stop_status` | bool | `true` |  |  |  | Paste a [Controle: ...] line after a session with a correction. |
| `output.extra_terminal_classes` | list of strings | `[]` |  |  |  | Extra window classes that paste with Ctrl+Shift+V (added to the built-in list). |
| `sounds.start` | string | `assets/granted-04.wav` |  |  |  | Start sound (WAV); relative to the working directory. |
| `sounds.stop` | string | `assets/beepbeep.wav` |  |  |  | Stop sound (WAV); relative to the working directory. |
| `rewrite.endpoint` | string or null | `null` |  |  | `DICTATION_REWRITE_ENDPOINT` | Chat completions URL, https (or loopback); rewrite mode needs it. |
| `rewrite.model` | string or null | `null` |  |  | `DICTATION_REWRITE_MODEL` | Model id; rewrite mode needs it. |
| `rewrite.key_file` | string or null | `null` |  |  | `DICTATION_REWRITE_KEY_FILE` | Path of the file holding the API key; never the key itself. |
| `rewrite.style_file` | string or null | `null` |  |  | `DICTATION_REWRITE_STYLE_FILE` | Path of a short preferences file sent with every call. |
| `rewrite.terms_file` | string or null | `null` |  |  | `DICTATION_REWRITE_TERMS_FILE` | Path of a term list, one term per line. |
| `rewrite.timeout_s` | float | `30` | 1–120 |  | `DICTATION_REWRITE_TIMEOUT_S` | Hard total time-out in seconds. |
| `rewrite.word_factor` | float | `0` | 0–1 |  | `DICTATION_REWRITE_WORD_FACTOR` | Word maximum for long dictations as factor x words; 0 = off. |
| `rewrite.provider_order` | list of strings | `[]` |  |  | `DICTATION_REWRITE_PROVIDER_ORDER` | OpenRouter provider preference, e.g. [together, fireworks]. |
| `rewrite.signature` | string | `""` |  |  | `DICTATION_REWRITE_SIGNATURE` | Name under the sign-off of an email; may span lines. |
| `rewrite.context.max_chars` | int | `100000` | 1000–500000 |  | `DICTATION_REWRITE_CONTEXT_MAX_CHARS` | Most characters of reply context sent. |
| `rewrite.context.primary_max_age_s` | float | `20` | 0–3600 |  |  | Oldest highlighted text (PRIMARY) used as context. |
| `rewrite.context.clipboard_max_age_s` | float | `120` | 0–3600 |  |  | Oldest clipboard text used as context. |
<!-- settings-table:end -->

Notes:

- `language.default: null` means auto-detect; `keys.double_key` and
  `keys.key_combo` at `null` keep the platform defaults (`<ctrl_r>` on Linux).
- `output.extra_terminal_classes` adds window classes to the built-in terminal
  list (kitty, gnome-terminal, konsole, xterm, alacritty, terminator, st,
  wezterm, tilix, urxvt, rxvt); it never replaces it.
- Sound paths and the rewrite file paths may start with `~`.
- `rewrite.signature` may be a multi-line YAML string; each line goes on its own
  line under the sign-off.

## The older `rewrite.env`

Before this file existed, the rewrite mode was configured only through
`DICTATION_REWRITE_*` variables in `~/.config/faster-whisper-dictation/rewrite.env`,
which the service reads as an `EnvironmentFile`. That still works and, being an
environment variable, wins over `config.yaml`.

To move it into `config.yaml`:

```sh
venv/bin/python3 settings.py --migrate-env          # show the rewrite: block
venv/bin/python3 settings.py --migrate-env --write  # append it to config.yaml
mv ~/.config/faster-whisper-dictation/rewrite.env ~/.config/faster-whisper-dictation/rewrite.env.bak
systemctl --user restart spraakherkenning           # only when no dictation runs
```

`--write` only appends to an existing `config.yaml` that has no `rewrite:`
section yet; it never creates or overwrites the file. The key file is never
opened, only its path is copied. As long as `rewrite.env` is in place, its
variables still win; the journal then shows `config_override ... source=env`.
