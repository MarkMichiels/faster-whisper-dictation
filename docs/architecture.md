[Home](../README.md) / Architecture and fork changes

# Architecture and fork changes

## Purpose and boundaries

The application is a desktop dictation tool, not an audio archive or meeting
transcription service. The reference profile favours recognition context and
recoverable text delivery over word-by-word display. It runs one local Whisper
worker, captures the default microphone only during a dictation session, and
uses the clipboard to enter text into an ordinary desktop application.

## What this fork adds to upstream, and why

Compared with the locally available `upstream/main` baseline for
`doctorguile/faster-whisper-dictation`, which records first and types after
recognition, this source adds:

| Addition | Reason and trade-off |
|---|---|
| VAD streaming with recording, VAD, transcription and typing threads | Show text before a long turn finishes; queues allow capture to continue while decoding/pasting |
| Quiet-point eager cuts and hold-back merge | Bound long unbroken turns while giving Whisper longer context before each paste; latency depends on pauses, merge threshold and decode speed |
| Forced language and rolling prompt tail | Avoid short-fragment language guessing and carry topic across chunk seams; bad context must be scrubbed to prevent repetition cascades |
| One prompt-free retry of a suspect normal decode | Break contaminated-context loops without unlimited retries; cannot guarantee recovery |
| Right-Ctrl trigger, momentary Left-Shift language and mouse toggles | Start without a complex chord or remembered language-toggle state; mouse frees the keyboard hand |
| Mouse remapper and udev watcher | Avoid browser back/forward when using thumb buttons, including after device re-creation; changes X button behaviour beyond this app |
| Clipboard-first output, CopyQ history and X11 focus checks | Reduce per-character input loss and retain text when paste fails; overwrites clipboard and can leave persistent text history |
| Per-session RAM coverage ledger and stop correction path | Distinguish captured-but-undecoded ranges from delivery failures; append recovered material without silently editing earlier prose |
| Energy/caption/repetition gates and punctuation formatting | Reduce invented text on silence and prompt pollution; heuristics can also reject literal or very quiet speech |
| Bounded CUDA OOM snapshot | Show competing GPU allocations at the time of failure; diagnostics do not free memory or repair the model |
| systemd installer with graphical-session lifecycle and revision check | One managed login service, predictable restart and detectable stale units; installer changes the desktop and is GPU-profile-specific |
| Dedicated beep output streams; recorder starts before beep | Avoid one beep cancelling another and avoid waiting for the sound before starting capture; sound does not prove microphone readiness |
| Offline logic/queue regressions | Test loss/recovery behaviour without recording, model downloads or desktop writes; not a hardware acceptance test |

## Files and responsibilities

| File | Responsibility |
|---|---|
| `dictation.py` | CLI; normal and batch apps; audio/VAD threads; model worker; triggers; clipboard and focus handling |
| `coverage_ledger.py` | RAM samples, contiguous outcome ranges, numeric audio statistics, bounded stop recovery, corrections and JSON diagnostics |
| `rewrite.py` | Optional rewrite mode: configuration from the environment, prompt, one chat-completions call (stdlib), answer check and raw-text fallback |
| `context_capture.py` | Optional rewrite mode: one-time PRIMARY/CLIPBOARD reply-context capture with age, origin and secret checks |
| `install.sh` | Render/install/check the systemd user unit; fixed GPU/Dutch launch profile; optional rewrite environment file |
| `free_mouse_thumb_buttons.sh` | Map buttons 8/9 to 10/11 on suitable physical X pointers |
| `watch_mouse_thumb_buttons.sh` | Reapply maps at startup and after udev input-add events |
| `start_dictation.sh`, `run.sh` | Legacy/manual launch helpers; not required alongside the managed service |
| `assets/` | Feedback sounds |
| `tests/` | Synthetic/offline regressions and optional cached-model replay tools |
| `requirements.txt` | Python dependency declarations, not a complete reproducible environment |

## Streaming pipeline

```text
keyboard / mouse -> App.start -> fresh session, target window and queues
                                      |
default microphone -> PyAudio capture -> RAM ledger (sample positions)
                          |
                    shared PCM buffer
                          |
                Silero VAD / quiet-point split
                          |
                  transcription_queue
                          |
                 merge pending audio
                          |
        Whisper + rolling prompt + bounded retry
                          |
           output acceptance / range outcome
                          |
                     typing_queue
                          |
           target focus -> CopyQ / xclip -> paste shortcut
                          |
                   delivery receipt
                          |
stop -> flush tail -> drain queues -> ledger stop check -> RAM release
```

### Capture and chunking

PyAudio reads signed 16-bit, mono PCM at 16,000 samples/second in 1,024-sample
blocks. There is no microphone selector in the CLI: use the system's default
input configuration. Silero evaluates speech and pauses; VAD schedules a cut,
but the stream includes captured samples before/between detected speech ranges
rather than discarding those samples as non-speech.

A full pause (`--silence-ms`, default 1,000 ms) closes a segment. For continuous
speech, eager splitting searches for the lowest smoothed speech probability
after 2.5 seconds and within a nominal ten-second window. A probability at or
below 0.35 allows an early cut; at the ceiling, it uses the quietest available
point even if no real pause exists. **That does not guarantee a word boundary.**
Processed recorder-buffer frames are trimmed while absolute sample offsets
remain attached to queued chunks.

Eager chunks accumulate until at least 12 seconds of audio before one merged
decode; a full-pause boundary or stop flushes earlier. Thus the ten-second eager
ceiling is not a ten-second maximum time-to-text. Lowering `--paste-min-s` can
reduce waiting but removes some decoding context; improvements in speed or
accuracy on a particular machine require measurement.

### Decode, retry and output acceptance

The model is loaded once. Ordinary streaming decode uses beam size 5, the
configured language (or auto-detection), and up to 500 characters of rolling
text as `initial_prompt`. Context resets when a session takes over. Pause runs,
known artefacts, common thanks-only prompts and repetition loops are kept out of
that prompt. The optional decoder-side VAD is off by default because removing
silence can change segment boundaries and recognition.

An empty/known-artifact decode, pause/repetition run, or fewer than half a word
per second on at least eight seconds of audio is suspect. An ordinary suspect
decode gets **one** retry without a prompt, after clearing context. A retry is
not allowed to replace useful first-pass words indiscriminately with a worse
result. A decoder exception propagates to the range's error accounting instead
of being an unlimited in-place retry.

The output gates operate on the whole chunk or whole recovery window:

- Known caption/broadcaster/credit families and punctuation-only output are
  rejected; the code does not do arbitrary substring removal from normal prose.
- Thanks-like phrases require silence evidence to be rejected: at least one
  second of low voiced energy and a peak below -50 dBFS. Audible thanks are not
  categorically discarded.
- A special short-repeat gate rejects exactly two identical multiword phrases
  (6–12 normalised words total) on 1–3 seconds of near-idle audio with zero
  voiced duration and RMS at most -60 dBFS.
- Other text on low-energy audio is rejected when its decode is also suspect
  or its peak is below -50 dBFS. Plausible quiet text is not rejected solely for
  lacking high-energy frames.
- Repeated `* - _ = ~ #` symbols are spaced; ellipses default to `. . .`.
  These are formatting transformations, not a promise that every Markdown
  renderer will treat the result identically.

Audio statistics use 512-sample (32 ms) frames. A frame is counted as voiced
when RMS exceeds -45 dBFS. A range is classified as silent for recovery if it
lasts at least one second with less than 0.15 seconds of voiced frames. These
are signal-level heuristics, not proof of speech absence; gain, quiet speech,
noise and microphone dropouts affect them. Numeric diagnostics include RMS,
peak, voiced duration, zero fraction and longest run of digital zeros.

### Clipboard delivery

The session remembers its starting **window**, not the exact text widget or
cursor position. Before each paste, `xdotool` attempts to activate and verify
that window, up to twice. Window class from `xprop` selects `Ctrl+Shift+V` for
known terminal classes and `Ctrl+V` otherwise.

Delivery first runs both `copyq add -- TEXT` (history) and `copyq copy -- TEXT`
(clipboard). If CopyQ fails, it tries xclip. A successful xclip paste is degraded
because CopyQ history is absent, but is not automatically pasted again. If
focus restoration or clipboard setup fails, the app does **not** send a blind
paste or fall back to character-by-character typing; the old typing helper is
still defined but not called by this path.

Receipts record command success, **not acknowledgement by the receiving app**.
A wrong widget, moved cursor, unusual terminal shortcut or application handling
can still defeat a nominally successful paste. Do not switch fields during a
session and assume targeting is exact.

## Stop and coverage recovery

Manual stop stops and joins the recorder/VAD writers, flushes every remaining
tail (including very short tails), then places a sentinel on the transcription
queue. Transcription drains and forwards its sentinel to typing. Only after
ordinary output drains does the ledger run its stop check.

Outcome ranges must form a contiguous partition of captured sample positions.
Missing queue spans/tails become explicit `queue_gap` errors. Suspects are:

- decoder errors (including queue gaps);
- ranges at least eight seconds long with fewer than half a word per second;
- nonempty recognised text with a failed delivery receipt.

This is **not** a full second transcription or word-for-word comparison. Short
empty/filtered chunks are not automatically all selected. Samples lost before
the app captured them cannot be reconstructed by the ledger.

For each suspect, failed delivery reuses recognised text without another model
call. Otherwise signal-level silence skips decoding, or retained RAM audio is
re-decoded once in windows of at most ten seconds without an initial prompt.
Each recovery window passes caption/energy checks **before** joining; pause-only
recoveries are removed. Recovered text substantially repeating delivered nearby
text is suppressed using ordered matching word runs (60% coverage threshold).
This duplicate heuristic can also suppress a partly new recovery.

Adjacent recovered ranges become one appended correction, quoting up to four
words from neighbouring delivered text as anchors. The current labels are
`[Correctie: ...]` and `[Controle: ...]`, in Dutch regardless of model language.
Earlier output is never edited or automatically relocated. A control line is
pasted only when there are corrections or unresolved ranges; a clean/silence-only
check can be journal-only. `STOP_STATUS` in the ledger controls that line.
A summary distinguishes unresolved ranges, silence, duplicates and correction
paste failures; read those separately rather than treating zero unresolved as
proof of delivery.

The ledger clears retained PCM frames in `finally`; the app clears the
recorder's remaining frames after the check. Releasing references is not secure
memory erasure. Streaming is capped at **30 minutes per session**, by sample
count and wall-clock timer, even with `-t 0`. A journal-only warning occurs after
29 minutes of captured samples. PCM alone costs about 1.92 MB/minute (57.6 MB
at the cap); arrays, queues, multiple finishing sessions and the model add more.

## Consecutive sessions and failure limits

A new start can record once the old recorder has stopped, even while the old
session finishes output/recovery. Fresh queues/ledger/window/language keep their
state separate. New-session transcription waits on the preceding session's
completion event, so normally its output follows the old corrections. That wait
has a **300-second timeout** and proceeds afterwards with a diagnostic: strict
ordering is not guaranteed if the earlier session hangs beyond that limit.
Old silence callbacks address their own recorder rather than stopping a newer
session. Mouse/combo starts do not apply momentary Left-Shift selection and can
inherit the worker's current language.

Input overflow is logged and capture continues, but driver-dropped samples are
already lost. A blocked audio backend can delay stop indefinitely because the
writer joins have no timeout. CUDA exhaustion, failed recovery and focus/paste
failure remain possible. Restart-on-failure handles process exits, not every
failed chunk or hanging thread. Service shutdown can interrupt queued recovery.

`--batch-mode` retains the upstream-style record-then-decode state machine.
It shares the clipboard replayer and some filtering, but has **no streaming
coverage ledger, stop recovery or streaming 30-minute sample cap**. Do not use
batch mode to claim equivalent safety or retention behaviour.

## Mouse mapping and desktop scope

The listener accepts buttons 8/9 and 10/11 and coalesces presses within 0.4
seconds. It does not suppress their native application actions. The helper
therefore maps physical X slave pointers with at least nine buttons: below 8
is unchanged; 8/9 become 10/11 and further buttons shift upward too. Virtual
core/XTEST pointers and devices with fewer buttons are skipped. A device ID or
name argument restricts a manual remap.

The watcher waits on udev input-add events, waits briefly for X to create the
device, then reapplies maps. The service's helper failures are non-fatal; an
active unit does not establish that buttons were remapped. Original maps are
not automatically backed up or restored. Both helper scripts fall back to
`DISPLAY=:1` when absent; the reference service instead relies on inherited
X-session variables. Application-specific bindings to 10/11 can still conflict.
