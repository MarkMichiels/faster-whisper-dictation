# LLM log

An optional, local and searchable record of every call that goes to a language
model, so past rewrites can be found again and compared (which prompt, which
model, what came back). Off by default: set `log.enabled: true` in
`config.yaml` and restart the service.

Only rewrite calls are logged today. The table already has columns for a later
research mode that runs a Pi agent session (`pi_session_id`,
`pi_session_file`); such a session's full transcript stays in Pi's own session
file and is not copied.

## Where and how

| Item | Value |
|---|---|
| File | `log.path`, default `$XDG_DATA_HOME/faster-whisper-dictation/llm_log.db` (usually `~/.local/share/...`) |
| Permissions | directory created `0700`; database, `-wal`, `-shm` and status file `0600`. An existing directory is never re-moded: point `log.path` into a private directory. |
| Engine | SQLite, WAL mode, `secure_delete=ON`, FTS5 index (`unicode61 remove_diacritics 2`) over `context`, `input`, `output` with external content and triggers |
| Health | `llm_log_status.json` next to the database: rows written, dropped, skipped, secrets replaced, last write, last retention run, last error. Numbers and times only. |
| Check | `install.sh --check` (and `python3 llm_log.py --check`) reports `off`, `on`, or `WRONG permissions` and then exits non-zero |

One row per call:

| Column | Meaning |
|---|---|
| `ts`, `epoch` | time of the call (local ISO, and seconds for retention) |
| `kind`, `mode`, `language` | `rewrite`; `opmaak` (rich) or `regel` (one-line); written language |
| `model`, `provider`, `prompt_version` | model id, serving provider, 12-hex hash of the system prompt sent |
| `ms`, `tokens_in`, `tokens_out`, `cost` | as reported by the endpoint |
| `status`, `reason` | `ok`, `failed` or `timeout`, and the failure reason (no text) |
| `delivered` | `pasted` or `none` |
| `context`, `input`, `output` | context as sent, raw dictation, the model's composed answer (also when it was rejected and the raw text was pasted) |
| `secrets_replaced` | how many secrets the filter replaced in this row |
| `text_purged_at` | set when retention wiped the text |

No row is written when no call was attempted (rewrite not configured).

## Never in the paste path

The dictation thread only calls `LlmLog.record()`, a `put_nowait` on a bounded
queue (1000 rows). One daemon writer thread filters, writes and runs retention.
Measured added wait in the rewrite path: about 2 µs median, under 0.1 ms
maximum, also with the writer blocked (slow disk) or the database locked by
another process. A full queue, a locked database (2 s busy timeout, in the
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
| `log.text_days` | 90 | after this many days `context`, `input` and `output` become NULL and `text_purged_at` is set; 0 = keep text |
| `log.meta_days` | 730 | after this many days the whole row is deleted; 0 = keep |
| `log.context_max_chars` | 0 | most context characters stored (beginning and end, with `[ingekort]`); 0 = all that was sent |

The job runs inside the service, without a separate timer: once when the writer
thread starts (service start), then every 24 hours. After a run that changed
rows it optimises the FTS index and truncates the WAL, so with
`secure_delete` the wiped text is gone from the database files. Copies made
earlier (backups, snapshots) keep it until they expire. Run it by hand with
`python3 llm_log.py --purge`.

## Searching

```sh
sqlite3 -readonly ~/.local/share/faster-whisper-dictation/llm_log.db \
  "SELECT c.id, c.ts, c.model, snippet(calls_fts, -1, '[', ']', '…', 12)
     FROM calls_fts JOIN calls c ON c.id = calls_fts.rowid
    WHERE calls_fts MATCH 'offerte' ORDER BY c.id DESC LIMIT 20;"
```

## Backups

A live WAL database must not be copied file by file: take a consistent copy
with SQLite's online backup API (`sqlite3 db ".backup copy.db"` or Python's
`Connection.backup`) and back up that copy. Text wiped by retention stays in
older backup snapshots until those expire.
