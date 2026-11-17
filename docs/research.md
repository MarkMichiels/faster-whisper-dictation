[Home](../README.md) / Research mode

# Research mode (optional, Pi agent session)

Research mode turns one short dictation into a task for an autonomous
Pi coding-agent session (`pi`, package
`@earendil-works/pi-coding-agent`). The session looks the
subject up (files under its working directory, the local knowledge search its
`AGENTS.md` describes, the web) and hands back one short, paste-ready result.
That result replaces the selection, like an AI rewrite. It is **off unless
configured** (`research.enabled: false`); ordinary dictation and rewrite mode do
not change.

## Use

1. Optionally select text in any window: it becomes context, as in rewrite mode.
2. Hold **Right Alt (AltGr)** at the second Right Ctrl tap and dictate the task,
   for example "which pump in the docs ships fastest and what does it cost".
   **Left Shift** as well selects the alternate recognition language.
3. Stop with one Right Ctrl tap (or silence auto-stop). The dictation itself is
   saved to the clipboard history (CopyQ) and nothing is typed.
4. The session runs in the background. You can keep dictating and rewriting
   meanwhile; a session never blocks the dictation chain.
5. When it ends, the result is pasted into the window that was focused when you
   started, or put on the clipboard (see below).

Only one session runs at a time. Starting another one while a session runs is
refused with the status `failed: er loopt al een onderzoek`.

When research mode is not usable (disabled, `pi_bin` or `cwd` missing, a
refused model), AltGr + double tap plays the `rewrite_unavailable` tone and
dictates normally; the journal line `research_unavailable` names the reason.

## Settings

All keys live under `research.` in `config.yaml`; the generated reference is in
[Configuration](configuration.md). The ones you need:

| Key | Use |
|---|---|
| `research.enabled` | `true` to turn the mode on |
| `research.pi_bin` | absolute path of the `pi` executable |
| `research.node_bin` | absolute path of `node` when the service's PATH lacks it (for example with nvm); symlinks such as `dictation-node` are resolved before prepending the target directory to PATH |
| `research.cwd` | working directory of the session; its `AGENTS.md` and skills apply |
| `research.model` | `provider/id`; only `anthropic`, `openai-codex` and `openrouter` are accepted, anything else is refused at load time |
| `research.thinking` | Pi thinking level (default `medium`) |
| `research.notes_dir` | where the session may write its own note, relative to `cwd` (default `research_notes`) |
| `research.proposals` | allow a new file in an `issues/` folder and insert-only edits of a `PLAN.md` under `cwd` |
| `research.timeout_s` | hard stop in seconds (default 600) |
| `research.visible` | show the session in the Pi dashboard |
| `research.escape_cancels` | Escape cancels a running session (default `true`) |

`./install.sh --check` (or `python3 research.py --check`) reports whether the
mode is usable: Pi and node executable, the extension files present, the
settings accepted. A change takes effect after a service restart.

## How a session runs

The service starts Pi as a child in its own process group:

```text
pi --mode json -p --session-id <uuid> --no-mcp -e research_ext/research.ts \
   --exclude-tools <list> --model <model> --thinking <level> \
   --append-system-prompt <research_ext/session_brief.md>
```

The dictation and the selection go to Pi on stdin, never in the argument list.
The service reads Pi's JSON event stream for progress and the result; Pi's own
session file stays the full record.

The only way a result reaches your text field is the extension tool
`deliver_result` (`text`, `summary`, `status` `done` | `no_result` |
`question`). Free prose from the model is never pasted. A session that ends
without `deliver_result` reports `geen resultaat afgeleverd`.

Pi's print mode exits when its event loop is empty. Another loaded extension can
keep a handle open after the session ended; the research extension therefore
exits Pi 3 seconds after `session_shutdown` with an unref'd timer, and the
service stops a session that delivered but did not exit after 10 seconds.

## Layers of protection

The session runs without a person watching, so its limits are enforced in
layers. None of them is complete on its own.

1. **Fixed session brief** (`research_ext/session_brief.md`): the dictation is a
   task, never an approval. No production or live systems, no credentials, no
   account, provider, role or settings changes, no spending beyond the model
   costs, nothing written outside the machine (mail, drafts, trackers, chats,
   drives, push/fetch/pull), no deleting, overwriting or moving, no stopping
   processes. Writable: one note of its own and, with `research.proposals`,
   proposals as a new issue file or one inserted decision line. No code, no
   configuration, no commits. In tests the model itself refused most forbidden
   requests on the basis of this brief.
2. **Tool guard** (`research_ext/policy.mjs`, a `tool_call` hook): a refused call
   returns an error starting with `Blocked by the dictation research guard
   [<rule>]` that the model sees and the service counts. It refuses:
   - writes outside the notes directory, overwriting an existing file,
     edits that change rather than insert (for `PLAN.md`);
   - credential paths in a path argument or a shell command, including a path
     whose real path (through a symlink) is a credential;
   - programs that delete, move, copy, signal, change permissions, control
     services, escalate privileges, open remote shells, inject input or touch
     the clipboard, install packages, or use account and cloud CLIs (including
     `pi` itself);
   - every git command that writes or talks to a remote (`add`, `commit`,
     `push`, `fetch`, ...); reading (`status`, `log`, `diff`, `show`) stays allowed;
   - in-place `sed`/`perl`, `find -exec/-delete`, HTTP writes with `curl`/`wget`,
     shell redirects outside the notes directory or `/tmp`, and scripts with
     file, process or network writes (`subprocess`, `os.remove`, `open(..., 'w')`, ...);
   - tools that act outside the machine, change roles, drive the user's browser
     or wait for a human.
3. **Pi options**: `--no-mcp` loads no MCP servers and `--exclude-tools` does
   not even offer the role, browser, question, cockpit and mail tools.
4. **Settings**: the model provider is checked at load time; the hard
   time-out ends every session.

### Known limits of the denylist

A denylist is never complete. Known gaps:

- Pi's built-in `grep`/`find` tools search from the working directory. If a
  credentials folder there is not excluded (for example by `.gitignore`), a
  search can show matching lines from it.
- A script already on disk runs freely: the guard sees the command, not the
  script's code.
- Shell globs and paths built at run time are not resolved (`cat docs/*`).
- The guard is deliberately broad and refuses some harmless commands (any
  command containing the word `subprocess`, `git branch -a`, `cp` into the
  notes directory). The model then says in its result what it would have needed.
- Cancelling signals the process group only while Pi is still unreaped; a
  process that left the group (`setsid`) is not reached, and there is a small
  race between reaping and signalling.

## Paste or clipboard

The result is pasted like a rich rewrite (replacing the selection, never
pressing Enter) only when all of these hold:

- the status is `done`;
- the dictation is idle: the service waits up to 120 seconds for a running
  dictation or its delivery to finish;
- the focused window is still the one where you started.

Otherwise the result goes to the clipboard and the status box says why
(`ander venster actief`, `dictatie was bezig`, `plakken geweigerd`). A
`question` result always goes to the clipboard. On the clipboard the HTML
flavour keeps paragraphs and lists; the plain flavour is one line, as for
rewrites. A `no_result` explanation is only added to the clipboard history.

Under X11 there is a small race between the focus check and the paste.

## Status box

GNOME Wayland only (the dictation helper extension; there is no status box under
X11, only the tones and the journal). While a session runs, and no dictation is
using the box, it shows:

```text
Onderzoek — bezig 1:12 · stap 4 · bestand lezen
```

The step is a code, never dictated or selected text: a tool name, `thinking`,
`writing`, `blocked`, or for a shell command only a known program name
(`bash:rg`). The end states are `klaar, geplakt ✓` (4 s), `klaar, resultaat op
het klembord (Ctrl+V)` (20 s) and `mislukt: <reason>` (10 s). After an
extension update, log out and back in.

## Cancelling

- AltGr + double tap while a session runs cancels it (nothing is recorded).
- Escape cancels it when `research.escape_cancels` is `true`. Escape is a common
  key; the key still reaches the focused window. Set the switch to `false` if
  sessions get cancelled by accident.
- The hard time-out (`research.timeout_s`) cancels it.

Cancelling sends SIGTERM to the session's process group, then SIGKILL after 5
seconds. Processes are never matched by name.

## Journal and LLM log

The journal (`journalctl --user -u spraakherkenning`) gets codes and counts
only, never dictated, selected or result text: `research_started`,
`research_tool_blocked` (tool and rule), `research_cancel`,
`research_finished` (status, reason, delivered, steps, seconds, blocked
tools, Pi session id, cost, return code), `research_no_exit_after_result`.

With the [LLM log](llm_log.md) enabled, a session adds one `llm_content` row
(`kind=research`, the dictation, the context, the result, `pi_session_id` and
`pi_session_file`) and no `llm_call` row: Pi's session file is the full record.
Password-field context is treated as in rewrite mode.
