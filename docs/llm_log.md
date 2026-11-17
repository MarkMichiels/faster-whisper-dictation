# LLM log

An optional, local and searchable record of every call that goes to a language
model, so past rewrites can be found again and compared (which prompt, which
model, what came back). Off by default: set `log.enabled: true` in
`config.yaml` and restart the service.

Only rewrite calls are logged today. The content table already has columns for
a later research mode that runs a Pi agent session (`pi_session_id`,
`pi_session_file`); such a session's full transcript stays in Pi's own session
file and is not copied, and it gets no `llm_call` record (the session file is
already its record).

## Where and how

| Item | Value |
|---|---|
| File | `log.path`, default `$XDG_DATA_HOME/faster-whisper-dictation/llm_log.db` (usually `~/.local/share/...`) |
| Permissions | directory created `0700`; database, `-wal`, `-shm` and status file `0600`. An existing directory is never re-moded: point `log.path` into a private directory. |
| Engine | SQLite, WAL mode, `secure_delete=ON`; two tables plus an FTS5 index (`unicode61 remove_diacritics 2`) over `context`, `input`, `output` of `llm_content`, with external content and triggers |
| Health | `llm_log_status.json` next to the database: rows written, dropped, skipped, secrets replaced, last write, last retention run, last error. Numbers and times only. |
| Check | `install.sh --check` (and `python3 llm_log.py --check`) reports `off`, `on`, or `WRONG permissions` and then exits non-zero |

## Two tables

Every rewrite attempt writes one row in each table, in one transaction, under
the same `request_id`. An attempt is also logged when it failed or was never
sent (for example an unreadable key). No row is written when no call was
attempted (rewrite not configured).

### `llm_call`: the content-free call record

Exactly the 36 columns of the shared `llm-call/1` record format (flat SQLite
form), so a usage reader can count calls, tokens and cost without ever touching
text. It holds no text at all: no prompt, context, dictation or answer.

| Column | Value written |
|---|---|
| `schema` | always `llm-call/1` |
| `request_id` | local id of the attempt (random hex), primary key, same as in `llm_content` |
| `ts` | start of the attempt, UTC with zone (`2026-01-02T09:05:32.690Z`) |
| `producer` | `log.producer`, default `DCT` |
| `caller`, `step` | `dictation`, `rewrite` |
| `traffic` | `real` (work started by a person dictating) |
| `provider`, `model` | serving provider and model as the endpoint reported them; `unknown` when not known (the configured model when no answer came back) |
| `transport` | access route from the endpoint host: `openrouter`, `openai`, `anthropic`, or the host name |
| `response_id` | the provider's response id (`gen-...`), NULL without an answer; unique when present |
| `execution_status` | `ok` when the endpoint answered (also when this application then rejected the answer), `error_after_send` when it was sent but failed (HTTP error, timeout, provider error, invalid response), `error_before_send` when it was never sent |
| `http_status`, `error` | numeric HTTP code when known; short failure code such as `http_429`, `timeout`, `key_unreadable` (NULL when `ok`) |
| `duration_s` | seconds from request to outcome |
| `prompt_tokens`, `completion_tokens`, `cached_tokens`, `cache_write_tokens`, `reasoning_tokens` | as the provider reported them; cached and reasoning tokens are part of the prompt and completion totals. A missing counter stays NULL, never 0. |
| `usage_complete` | 1 only when both prompt and completion tokens were reported |
| `cost_usd`, `price_basis` | cost as the provider reported it with basis `provider`; otherwise NULL with basis `not_sent` (never sent), `rejected` (provider error or HTTP 4xx) or `unknown` |
| `content_store`, `content_key` | `sqlite` and the `request_id`: the text is in `llm_content` |
| `input_sha256`, `input_bytes` | sha256 and size of the full request messages (JSON) as built for the endpoint, before the secret filter |
| `output_sha256`, `output_bytes` | sha256 and size of the raw model answer, NULL without an answer |
| other contract columns | (`provider_request_id`, `session_id`, `parent_session_id`, `correlation_id`, `assignment_*`, `route`) present and NULL for rewrites |

### `llm_content`: text and what only this application knows

| Column | Meaning |
|---|---|
| `request_id` | same as `llm_call.request_id` |
| `ts`, `epoch` | time of the call (local ISO with offset, and seconds for retention) |
| `kind`, `mode`, `language` | `rewrite`; `opmaak` (rich) or `regel` (one-line); written language |
| `prompt_version` | 12-hex hash of the system prompt sent |
| `status`, `reason` | `ok`, `failed` or `timeout`, and the failure reason (no text) |
| `delivered` | `pasted` or `none` |
| `context`, `input`, `output` | context as sent, raw dictation, the model's composed answer (also when it was rejected and the raw text was pasted), after the secret filter |
| `secrets_replaced` | how many secrets the filter replaced in this row |
| `pi_session_id`, `pi_session_file` | research sessions only |
| `text_purged_at` | set when retention wiped the text |

## Never in the paste path

The dictation thread only calls `LlmLog.record()`, a `put_nowait` on a bounded
queue (1000 rows). One daemon writer thread filters, writes and runs retention.
Measured added wait in the rewrite path, both tables written: about 2 µs
median and p99, a few tenths of a millisecond at most, also with the writer
blocked (slow disk) or the database locked by another process. A full queue, a locked database (2 s busy timeout, in the
writer) or any write error drops that row with a journal line
`llm_log_dropped` holding a reason and a count, never text.

## Privacy gates (not configurable)

- **Secret filter** before every write, on all three text columns: private-key
  and certificate blocks, prefixed tokens (`sk-`, `ghp_`, `github_pat_`,
  `glpat-`, `xox?-`, `AKIA`, `AIza`, `ya29.`, `hf_`, `npm_`, ...), JWTs,
  `Bearer`/`Basic` values, the value after `password:`/`wachtwoord=`/`token:`
  and similar labels, hex runs of 32+ characters and base64-like runs of 40+
  characters with letters and digits. Each becomes `[geheim]`. This also hides
  some harmless long identifiers (commit hashes, document ids); that is
  accepted.
- **Password windows**: no row at all when the paste window's class looks like a
  password manager or secret prompt (KeePassXC, Bitwarden, 1Password, Secrets,
  Seahorse, pinentry, gcr-prompter, polkit, ...) or when a context source was
  refused because a password manager had marked it secret.
  Not covered: a password field inside an ordinary application (for example a
  web login form). Detecting that needs the AT-SPI role of the focused widget,
  which the service environment does not provide today.
- Never logged: audio, request headers, the endpoint, the API key.
- The journal keeps its rule: metadata only, never text.

## Retention

| Setting | Default | Effect |
|---|---|---|
| `log.text_days` | 90 | after this many days `context`, `input` and `output` in `llm_content` become NULL and `text_purged_at` is set; the `llm_call` record with its sha256 stays as proof that the call happened; 0 = keep text |
| `log.meta_days` | 730 | after this many days both the `llm_content` row and the `llm_call` record are deleted; 0 = keep |
| `log.context_max_chars` | 0 | most context characters stored (beginning and end, with `[ingekort]`); 0 = all that was sent |
| `log.producer` | `DCT` | producer code written in every `llm_call` record: 2-6 capitals or digits, starting with a letter; an invalid value falls back to `DCT` with a `config_invalid` journal line |

The job runs inside the service, without a separate timer: once when the writer
thread starts (service start), then every 24 hours. After a run that changed
rows it optimises the FTS index and truncates the WAL, so with
`secure_delete` the wiped text is gone from the database files. Copies made
earlier (backups, snapshots) keep it until they expire. Run it by hand with
`python3 llm_log.py --purge`.

## Searching

Full-text search on the content, joined to the call record for model,
tokens and cost:

```sh
sqlite3 -readonly ~/.local/share/faster-whisper-dictation/llm_log.db \
  "SELECT k.ts, k.model, k.execution_status, k.cost_usd,
          snippet(llm_content_fts, -1, '[', ']', '…', 12)
     FROM llm_content_fts
     JOIN llm_content c ON c.rowid = llm_content_fts.rowid
     JOIN llm_call k ON k.request_id = c.request_id
    WHERE llm_content_fts MATCH 'offerte' ORDER BY c.epoch DESC LIMIT 20;"
```

Usage without text, for example cost per day:

```sh
sqlite3 -readonly ~/.local/share/faster-whisper-dictation/llm_log.db \
  "SELECT substr(ts, 1, 10) AS day, count(*), sum(cost_usd), sum(cost_usd IS NULL)
     FROM llm_call GROUP BY day ORDER BY day DESC LIMIT 14;"
```

## Backups

A live WAL database must not be copied file by file: take a consistent copy
with SQLite's online backup API (`sqlite3 db ".backup copy.db"` or Python's
`Connection.backup`) and back up that copy. Text wiped by retention stays in
older backup snapshots until those expire.
