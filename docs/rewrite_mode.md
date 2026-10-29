[Home](../README.md) / Rewrite mode

# Rewrite mode (optional, cloud LLM)

Rewrite mode turns one finished dictation into concise written text, for
example a reply to an email you have selected. It is **off unless configured**.
Ordinary dictation never contacts a language model.

## Use

| Action | Control |
|---|---|
| Rewrite in Dutch (Belgian Dutch) | Hold **Left Alt** while double-tapping Right Ctrl |
| Rewrite in English | Hold **Left Shift + Left Alt** while double-tapping Right Ctrl |
| Stop | Tap Right Ctrl once, as usual (or the silence auto-stop) |

The mouse thumb buttons always start ordinary dictation. Right Alt (AltGr) is
not used. Shift and Alt may be pressed in either order; both must be held at
the second tap.

**English output from Dutch speech.** In rewrite mode Left Shift selects only
the language of the *result*. Recognition keeps the configured language
(`-l`, Dutch by default), so you dictate in Dutch and the model writes the text
in English. Ordinary dictation with Left Shift (no Alt) is unchanged: it still
starts English recognition.

Typical flow for a reply:

1. Select the text you reply to (dragging or double-clicking is enough; no copy).
2. Start with Left Alt + double-tap. The selection is read **once**, now.
3. Click into the reply field and dictate. Nothing is pasted while you speak.
4. Stop. The text is rewritten once and pasted into the window that has focus
   **at stop**; that window stays the target while the rewrite runs.

Sounds:

| Sound | Meaning |
|---|---|
| Start sound twice | Rewrite mode, no reply context captured |
| Start sound twice + high pip | Rewrite mode with reply context |
| Short mid tone after stop | Rewritten text pasted |
| Two low tones after stop | Rewrite failed: the raw dictation was pasted instead |
| Two low tones, then start sound | Rewrite mode is not configured: ordinary dictation runs |

## Paste safety

The app sends only the paste shortcut, never Return or keypad Enter. Ordinary
dictation, including corrections and check lines, replaces line-separator runs
with one space. Rewrite mode preserves paragraphs in GUI windows, normalizing
CRLF and CR to LF. In recognized terminal windows it also replaces all line
separators with spaces, including separators in the appended check line and in
raw fallback text. This avoids executing lines in terminals without bracketed
paste. CopyQ's raw recovery history remains unchanged; manually pasting that
history is outside this safeguard.

Terminal detection uses the active window's class and the configured extra
terminal classes (`output.extra_terminal_classes`). An unknown or undetectable terminal
is treated as a GUI window: add its class before using rewrite mode there.
`paste_line_safety` logs only `flattened_line_breaks` (separator characters,
so CRLF counts as two) and `terminal`; it never includes the pasted text.
Application-specific handling of pasted text is outside the app's control.

## What happens at stop

1. The ordinary stop check runs. Text it recovers is placed **where the gap
   was**, not appended as `[Correctie: …]`.
2. The raw text goes into CopyQ history first (`copyq add`, no paste), so it
   survives any later failure.
3. One request to an OpenAI-compatible chat endpoint, with a JSON-schema
   answer and a hard total time-out (default 30 s). A following session can
   start meanwhile; it records at once and pastes after this one.
4. The answer is checked: empty, more than 1.2x or fewer than 0.15x the raw
   word count, an introduction ("Here is…"), a truncated or malformed answer all
   count as failure. Quotes or a code fence around the whole answer are removed.
5. An email is built only when the dictation itself says it is one ("this
   is an email", "mail to ..."), or when the captured context itself has the
   shape of an email (mail headers, an address, a "... wrote:" line, a line
   that opens with a greeting or sign-off) and the dictation answers it.
   Without context a dictated greeting is not enough. The model judges, and a
   check in code refuses an email outside these rules (journal:
   `email_blocked`, then only a dictated greeting or sign-off line is kept).
   An email is built as a complete plain-text email: greeting, the text, sign-off and
   the configured signature. A dictated greeting or sign-off is kept in its own
   words; otherwise "Beste <first name>," / "Dear <first name>," with the name
   from the dictation or the sender of the context email ("Beste," / "Hello,"
   without a name) and "Vriendelijke groeten," / "Best regards,". The signature
   is not added under a sign-off that already carries a dictated name; a name
   under the sign-off that was not dictated (for example taken from the
   context) is dropped. In doubt,
   no greeting or sign-off is added. The length check in step 4 judges the text
   without greeting and sign-off.
6. One paste. On any failure the raw text is pasted, without a marker in the
   text, and the journal gets `rewrite_failed` with the reason.
7. A range that could not be found back still adds the `[Controle: …]` line
   after the text; it is never sent to the model.

## Reply context

At start the app reads, in this order:

- the X11 **PRIMARY** selection, if it is at most 20 seconds old (select
  the text just before you start; an older selection is usually something
  you read earlier, not the message you reply to);
- else **CLIPBOARD**, if it is at most 2 minutes old and not written by CopyQ
  (after an ordinary dictation the clipboard holds your own previous text).

Age is the selection owner's `TIMESTAMP` against the X server clock; an owner
without `TIMESTAMP` counts as too old. No context is taken when the selection
carries a password-manager hint (`x-kde-passwordManagerHint`) or CopyQ's hidden
flag, looks like a secret (private-key header, common API-key prefixes, one long
token with digits), or is shorter than 20 characters. The context lives only
in memory for that session and is cleared when the session ends.

Up to 100,000 characters are sent (roughly 30,000–40,000 tokens; set
`rewrite.context.max_chars` or `DICTATION_REWRITE_CONTEXT_MAX_CHARS` to change it). Longer text keeps its first
and its last half of that limit with `[ingekort]` on a line in between: a mail
thread shows the oldest message on top and the newest at the bottom, a quoted
reply the other way round, so the message being answered can sit at either end.
Reading the selection is not the bottleneck (a million characters read in a few
milliseconds on a test display); the limit bounds cost and waiting time, which
grow with the context.

## Configure

Settings go in `~/.config/faster-whisper-dictation/config.yaml` (see
[Configuration](configuration.md)). Example `rewrite:` section, no secrets in it:

```yaml
rewrite:
  endpoint: https://openrouter.ai/api/v1/chat/completions
  model: deepseek/deepseek-v4.1-flash
  key_file: /path/to/a/file/holding/the/key.env
  timeout_s: 30
  # Optional:
  # provider_order: [together, fireworks]
  # style_file: /path/to/rewrite_style.md
  # terms_file: /path/to/terms.txt
  # word_factor: 0.45
  # signature: Your Name
  # context:
  #   max_chars: 100000
  #   primary_max_age_s: 20
  #   clipboard_max_age_s: 120
```

**Older setups:** `rewrite.env` still works. The service unit reads it as an
optional environment file (`EnvironmentFile=-%h/.config/faster-whisper-dictation/rewrite.env`);
its `DICTATION_REWRITE_*` variables win over `config.yaml`.
`settings.py --migrate-env` turns it into the block above. Each variable and its
YAML key:

| Variable | Meaning |
|---|---|
| `DICTATION_REWRITE_ENDPOINT` (`rewrite.endpoint`) | Chat-completions URL; `https://`, or plain `http://` only on loopback (e.g. a local Ollama) |
| `DICTATION_REWRITE_MODEL` (`rewrite.model`) | Model id |
| `DICTATION_REWRITE_KEY_FILE` (`rewrite.key_file`) | File with the API key, as `NAME=value` or a bare line. Only the path is configured; the key is read per request and never logged |
| `DICTATION_REWRITE_TIMEOUT_S` (`rewrite.timeout_s`) | Hard total time-out, 1–120 s (default 30) |
| `DICTATION_REWRITE_PROVIDER_ORDER` (`rewrite.provider_order`) | Optional OpenRouter provider preference |
| `DICTATION_REWRITE_STYLE_FILE` (`rewrite.style_file`) | Optional few lines of preferences, sent with every request |
| `DICTATION_REWRITE_TERMS_FILE` (`rewrite.terms_file`) | Optional term list the model must spell exactly |
| `DICTATION_REWRITE_WORD_FACTOR` (`rewrite.word_factor`) | Optional: for dictations above 120 words, ask for at most factor x words (default off) |
| `DICTATION_REWRITE_CONTEXT_MAX_CHARS` (`rewrite.context.max_chars`) | Optional: most characters of reply context sent, 1,000–500,000 (default 100,000). An invalid value falls back to the default; the journal then gets `rewrite_setting_invalid` with the reason, never the value |
| `DICTATION_REWRITE_SIGNATURE` (`rewrite.signature`) | Optional name under the sign-off of an email; a literal `\n` starts a new line (max. 200 characters). Without it the sign-off line stands alone |

Every request also sends `provider: {zdr: true, data_collection: "deny",
require_parameters: true}` (OpenRouter: only zero-data-retention endpoints, no
training use) and `reasoning: {enabled: false}`. Other OpenAI-compatible servers
ignore these fields.

After editing the file, restart the service at a moment without a recording.
Without the file, or with an incomplete one, the service starts normally and
logs `Rewrite mode: not available (<reason>)`; a Left-Alt start then logs
`rewrite_unavailable` and dictates as usual.

## Privacy

With a cloud endpoint the dictation **and the captured reply context** (often
someone else's text) leave the machine for the provider. The journal never gets
the rewritten text or the context, only metadata: `rewrite_context` (source,
characters sent and before shortening, age, truncation, refusal reasons), `rewrite_outcome` (model,
provider, seconds, tokens, cost, word ratio, output and spoken language, whether an email was built
or refused) or `rewrite_failed` (reason). `session_start` records the recognition language,
the output language (rewrite mode) and the Shift/Alt state at the second tap (held, for how long, or
how long ago released). The
raw dictation is still logged by the ordinary `DCT-COVERAGE` records, as in
ordinary dictation. CopyQ history receives both the raw and the pasted text.

## Tests

`tests/test_rewrite.py` covers the key combinations, the held-then-paste-once
flow, in-place recovery, every fallback reason, session ordering, the context
rules and the absence of rewritten text, context and key in captured output. It
uses fake key events, a fake xclip/probe, a fake LLM and a fake paste; it never
touches the real clipboard, CopyQ or the network.

`python3 context_capture.py --probe` prints the selection metadata (owner,
TARGETS, age; never text). With an explicit selection name, e.g.
`--probe MY_TEST_SELECTION`, it can be checked against a private test selection
without touching PRIMARY or CLIPBOARD.
