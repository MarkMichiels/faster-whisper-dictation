[Home](../README.md) / Rewrite mode

# Rewrite mode (optional, cloud LLM)

Rewrite mode turns one finished dictation into concise written text, for
example a reply to an email you have selected. It is **off unless configured**.
Ordinary dictation never contacts a language model.

## Use

Hold the modifiers at the **second** Right Ctrl tap (streaming mode):

| Held modifiers | Result |
|---|---|
| None | Ordinary dictation, default language, one plain line, no AI |
| Left Shift | Ordinary English dictation, one plain line, no AI |
| Left Alt | AI rewrite, Dutch, rich HTML plus one-line plain alternative |
| Left Alt + Left Shift | AI rewrite, English, rich HTML plus one-line plain alternative |
| Left Ctrl | AI rewrite, Dutch, one plain line, forced Ctrl+Shift+V |
| Left Ctrl + Left Shift | AI rewrite, English, one plain line, forced Ctrl+Shift+V |
| Left Ctrl + Left Alt | Same as Left Ctrl: terminal mode wins, no HTML |
| Left Ctrl + Left Alt + Left Shift | Same as Left Ctrl + Left Shift |

Stop with one Right Ctrl tap, as usual (or silence auto-stop). Modifiers at
stop cannot change the mode saved at start. Left Ctrl uses vk 65507; the
Right Ctrl trigger uses vk 65508. Left Ctrl events never count as taps.

The mouse thumb buttons always start ordinary dictation. Right Alt (AltGr) is
not used. Shift and Alt may be pressed in either order; both must be held at
the second tap.

**English output from Dutch speech.** In rewrite mode Left Shift selects only
the language of the *result*. Recognition keeps the configured language
(`-l`, Dutch by default), so you dictate in Dutch and the model writes the text
in English. Ordinary dictation with Left Shift (no Alt or Left Ctrl) is unchanged: it still
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
| Two low tones | Rewrite failure (raw text fallback), unavailable rich clipboard (plain fallback), or refused clipboard ownership check (no paste) |
| Two low tones, then start sound | Rewrite mode is not configured: ordinary dictation runs |
| Three equal short tones while recording | The microphone delivers only digital silence (also in ordinary dictation) |

When reply context was taken (from the selection or the clipboard), a small
green badge appears for 1.5 s in the top-right corner of the monitor that
holds the focused window: `Context opgepakt: 4.312 tekens`. It shows only the
length, never any of the text. It takes no focus and clicks pass through it.
Turn it off with `rewrite.context.badge: false`. On GNOME Wayland the status
box (`docs/wayland.md`) shows the count instead (`context: 4.312 tekens`) for
the whole session; the badge appears there only with `output.status_box: false`.

## Paste safety

The app sends only the paste shortcut, never Enter or Shift+Enter. **Every
`text/plain` payload is one line**, including corrections, check lines and raw
fallbacks. All Python line separators (CR, LF, VT, FF, U+001C–U+001E, NEL,
U+2028, U+2029) become spaces. Ordinary and explicit terminal modes never offer
HTML. Raw CopyQ recovery history is unchanged; manually pasting that history
is outside this safeguard.

Rich mode asks the model for limited Markdown in the JSON body, never HTML.
`rich_text.py` parses it with markdown-it-py (raw HTML disabled), derives the
visible plain text from the same tokens and emits only `p`, `ul`, `ol`, `li`,
`strong`, `br`. All text is escaped; there are no attributes, active links,
images, scripts or external resources. Link destinations are dropped and image
alt text is kept. Unsupported blocks fail closed to raw single-line fallback.
Greeting and sign-off become separate literal paragraphs; mail and name gates
still apply. Word checks count visible body words, not markup.

CopyQ first records final plain text in history, then `copyq copy` offers both
`text/plain` and `text/html` together. The target app chooses; a plain field
gets one line. If CopyQ fails, xclip offers only one-line plain text
(`text/plain` on X11, `UTF8_STRING` on Wayland). Two low
tones report lost formatting. No old HTML is reused on a following plain write.

WM_CLASS and `output.extra_terminal_classes` choose **only the shortcut**:
Ctrl+Shift+V for recognized terminals, Ctrl+V otherwise. Explicit Left Ctrl
forces Ctrl+Shift+V even in an unknown terminal. Window detection never chooses
line breaks or HTML permission.

Immediately before the shortcut, the app reads back its MIME payload. The
CopyQ path also checks a per-write owner marker and, for plain mode, absence of
HTML; xclip checks the plain bytes. On mismatch or read error, it refuses the
paste, sounds two low tones and leaves recovery history available. It never
restores an older clipboard over a new writer. The check and the target's paste
are not atomic: a small race remains after the last read. App-specific paste
behaviour is not guaranteed by a successful shortcut receipt.

`paste_line_safety` counts flattened separators (CRLF counts as two).
`paste_delivery` adds only metadata: `mode` (`plain`, `rich`, `terminal`),
`html_offered`, `ownership`, `shortcut`, focus/write/paste receipts. No new
formatting or ownership record contains text.

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
   An email is built from greeting, the text, sign-off and the configured
   signature, with separate greeting and sign-off HTML paragraphs in rich mode. A dictated greeting or sign-off is kept in its own
   words; otherwise "Beste <first name>," / "Dear <first name>," with the name
   from the dictation or the sender of the context email ("Beste," / "Hello,"
   without a name) and "Vriendelijke groeten," / "Best regards,". The signature
   is not added under a sign-off that already carries a dictated name; a name
   under the sign-off that was not dictated (for example taken from the
   context) is dropped. In doubt,
   no greeting or sign-off is added. The length check in step 4 judges the text
   without greeting and sign-off.
6. One paste attempt. On rewrite/render failure the raw text is flattened and
   used instead, without a marker; `rewrite_failed` records the reason. Clipboard
   ownership failure refuses the paste instead of pasting another writer's data.
7. A range that could not be found back still adds the `[Controle: …]` line
   after the text; it is never sent to the model.

## Releasing a Ctrl+A selection

Answering in the reply field itself: press **Ctrl+A** in the reply field,
then start with Left Alt + double-tap. The whole field is taken as context.
Once you release the start modifiers, the app sends **one Ctrl+Home**: the
selection is gone and the caret sits at the top, so the rewritten text is
pasted above the quoted message instead of replacing it.

It does this only when all of these hold; otherwise it does nothing and the
journal line `selection_release` gives the reason (`fired`, `reason`, `wait_s`):

1. The context came from the highlighted text (PRIMARY), not the clipboard.
2. The focused window is no terminal (built-in list plus
   `output.extra_terminal_classes`), and the session is not a Left Ctrl
   (terminal) session.
3. The app heard Ctrl+A (either Ctrl) at most 20 s before the start, in the
   same focused window.
4. Since that Ctrl+A no mouse button was pressed and no other key was typed
   (modifiers, the start key and Ctrl+C/Ctrl+Insert do not count).
5. The selection owner is the same X client as the focused window.
6. The selection's timestamp matches the moment of the Ctrl+A (within
   0.6 s), so the current selection is that one and not a later one.
7. No earlier session is still finishing (it may still paste).
8. After the start, Left Alt, Left Ctrl and Left Shift are released within
   2 s, without another key, a click or a focus change in between.

Why so strict: X11 cannot tell where a selection lies. A message selected in a
read-only pane and a caret placed in the reply field of the same program look
the same; a key there would move your caret. So a mouse selection or a
Shift+arrow selection is never touched; only a Ctrl+A the app saw itself.

Why Ctrl+Home: after Ctrl+A, Left leaves the caret one character before the
end in some editors (LibreOffice Writer); Ctrl+Home puts it at the top in
browsers and Writer alike. Why wait for the release: with Left Alt still held
the key would be Alt+key (a browser goes back in history). The key goes out
through XTest like the paste shortcut, without `--clearmodifiers`. Nothing else
is ever sent: no Enter, Tab or Escape.

Known side effects: after Ctrl+A in a read-only page, Ctrl+Home scrolls to the
top. A spreadsheet treats Ctrl+Home as "go to A1". Turn the feature off with
`rewrite.context.release_selection: false`.

The app keeps a small record of key and mouse presses for this: only times,
the focused window id at Ctrl+A, and whether Ctrl+A was pressed. No key text
and no pointer positions.

### On GNOME Wayland

The same flow works with the Dictation Helper extension (`docs/wayland.md`).
The helper reports when PRIMARY or CLIPBOARD last changed and which window
had focus then; the text is still read through Xwayland. Differences:

- Rule 5 becomes "PRIMARY changed while the focused window had focus"; rule 6
  uses the helper's change time instead of the X selection timestamp.
- Reply context from PRIMARY requires that the selection was made in the
  program that has focus at the start (select the mail, click into the reply
  field of the same program: fine). A selection made in another program is
  refused (`focus_elsewhere`); copy it with Ctrl+C instead.
- Ctrl+Home and the badge come from the helper (virtual keyboard, Shell
  label), not from XTest or a separate window.
- An updated helper is active only after logging out and in again.

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

Rich rendering needs `markdown-it-py==3.0.0` (listed in requirements.txt).
When upgrading an existing installation, install it explicitly **before** the
service restart; `install.sh` installs requirements only for a newly created
venv, not an existing one:

```sh
venv/bin/python -m pip install 'markdown-it-py==3.0.0'
venv/bin/python -m unittest discover -s tests
./install.sh  # only without a recording or a session still finishing
```

Without the parser, rich rewrites fail closed to raw single-line text and the
failure tone. Ordinary and terminal dictation do not load the parser.

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
  #   release_selection: true
  #   badge: true
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
logs `Rewrite mode: not available (<reason>)`; a Left-Alt or Left-Ctrl start then logs
`rewrite_unavailable` and dictates as usual.

## Privacy

With a cloud endpoint the dictation **and the captured reply context** (often
someone else's text) leave the machine for the provider. The journal never gets
the rewritten text or the context, only metadata: `rewrite_context` (source,
characters sent and before shortening, age, truncation, refusal reasons), `rewrite_outcome` (model,
provider, seconds, tokens, cost, word ratio, output and spoken language, whether an email was built
or refused) or `rewrite_failed` (reason). `session_start` records the recognition language,
the output language (rewrite mode), selected mode, and the Shift/Alt/Left-Ctrl state at the second tap (held, for how long, or
how long ago released). The
raw dictation is still logged by the ordinary `DCT-COVERAGE` records, as in
ordinary dictation. CopyQ history receives both the raw and the pasted text.

## Tests

`tests/test_formatting.py` covers the complete eight-row modifier table,
Markdown escaping/tag whitelist, double MIME, clipboard replacement/read errors,
xclip degradation and session modes. All desktop subprocesses and keys are mocked.
`tests/test_rewrite.py` covers the held-then-paste-once
flow, in-place recovery, every fallback reason, session ordering, the context
rules and the absence of rewritten text, context and key in captured output. It
uses fake key events, a fake xclip/probe, a fake LLM and a fake paste; it never
touches the real clipboard, CopyQ or the network.

`python3 context_capture.py --probe` prints the selection metadata (owner,
TARGETS, age; never text). With an explicit selection name, e.g.
`--probe MY_TEST_SELECTION`, it can be checked against a private test selection
without touching PRIMARY or CLIPBOARD.
