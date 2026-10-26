[Home](../README.md) / Privacy, tests and diagnosis

# Privacy, offline tests and diagnosis

## Privacy and retention

Recognition runs locally; there is no transcription-service API in this app.
That does **not** make the entire setup network-isolated or text-free on disk.
Package/model acquisition may use the network. The application does not create
an audio archive or write microphone audio files; streaming capture retains PCM
in RAM for the stop check and then clears its audio references. OS swap, crash
handling, memory remnants or other software are outside that guarantee. Batch
mode is a separate path without the ledger's release lifecycle.

| Data | Where it goes |
|---|---|
| Microphone PCM | In-process recorder/queues/ledger; streaming ledger references released after stop checking, no app-created audio file |
| Model weights | Hugging Face cache by library default, or the converted model directory selected with `-m`; retained on disk |
| Current transcript | Intended text application and system clipboard; each new block replaces clipboard content |
| CopyQ history | Explicit `add` plus `copy` calls; history can survive logout/restart depending on CopyQ settings; history limits may evict older items |
| Console/journal | Recognition text, rejected artefacts, errors, range diagnostics and corrections; retention depends on system journal configuration |

**`DCT-COVERAGE` is not metadata-only.** Its JSON can include full `text`, `raw`,
recovered text and quoted correction anchors, as well as times, sample ranges,
audio levels and delivery results. `[transcribe]`, `[filter]` and the replayer's
plain output also print text. The systemd unit sets `PYTHONUNBUFFERED=1`, so
stdout appears promptly in the journal. Model/cache files do not provide a
private dictation archive; journal/clipboard recovery is separate from audio.

On CUDA OOM, `cuda_oom_snapshot` also records this process ID and the
`nvidia-smi` compute-app listing: GPU UUIDs, PIDs, process names and used memory.
Other GPU applications may appear. It uses a one-second subprocess timeout and
truncates the logged stdout/stderr to 8,192/512 characters; this is a diagnostic,
not a memory-allocation fix or authority to stop another process.

Before sensitive use, choose appropriate CopyQ/history, journal retention,
swap and desktop security policies. This repository does not configure them
and has no CLI switch that makes all text logging private or disabled. Do not
paste raw journal/CopyQ exports into bug reports. Share only reviewed, redacted
errors or aggregate metrics. A successful local paste can still enter an
application that synchronises or sends text elsewhere.

## Reproducible offline regression test

Run from the root of this source tree. Only Python with NumPy is needed for
these unit tests; no microphone, desktop session, GPU, model or network request
is required. Use an existing suitable environment. If creating a small test
venv, provision NumPy **before** entering an offline environment (from a local
wheel if necessary).

```sh
# Existing app venv, or substitute a Python interpreter with NumPy installed:
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    venv/bin/python3 -m unittest discover -s tests -v
```

Expected result: a final `OK` and exit code 0, with no skipped failing tests.
There can be synthetic diagnostic output before that. The test source
AST-loads selected functions/classes instead of importing live audio/keyboard
backends. Models, clipboard commands and GPU queries are fakes/mocks; audio is
synthetic, not a recording. The offline environment flags are extra guardrails
for libraries, **not** a general network sandbox.

Coverage includes context cleanup, one-shot prompt recovery, queue/sample-tail
accounting, delivery failures, appended corrections, silent/caption rejection,
quiet audible thanks, short repeated phrases, consecutive-session handling and
bounded OOM diagnostics. Installer tests run a temporary source copy with a
separate HOME and fake venv/pip/systemctl commands; they cover missing venvs,
optional CUDA directories, Python-version-aware paths and rerun/check behavior.
They never install packages or change the real service. Read individual tests
if modifying the heuristics or installer.
Passing this suite proves those cases, not real speech accuracy, CUDA library
compatibility, desktop delivery, autostart, or performance on another GPU/CPU.

To test a source-only handover independently, extract it into a separate
folder, point the same NumPy-equipped interpreter at that tree and run the
command there. Do **not** run its installer or start a second desktop instance.
This also checks that tests and documentation do not depend on `.git`.

### Optional model replay (not the unit test)

`tests/replay_coverage.py` consumes a user-supplied audio file, uses an **already
cached** `large-v3` with CPU/int8 (six threads), simulates the first decoder
failure and runs real app loop methods with fake paste receipts:

```sh
HF_HUB_OFFLINE=1 venv/bin/python3 tests/replay_coverage.py /path/to/your-test.wav
```

It uses `local_files_only=True` and prints aggregate metrics rather than a
transcript. It creates no microphone recording and does not paste into the
desktop, but it does read the chosen file and load a substantial CPU model.
Its recovery assertion requires suitable speech, so arbitrary silence is not a
valid fixture. It is optional and is not necessary for the deterministic suite.
The separate `replay_recovery.py` compares against a hardcoded historical fork
commit and is not the reproducible history-free handover test.

## Troubleshooting without blind restarts

Start with the effective configuration and recent logs, remembering that logs
contain dictated content:

```sh
systemctl --user status spraakherkenning.service
systemctl --user cat spraakherkenning.service
./install.sh --check
journalctl --user -u spraakherkenning.service --since today --no-pager
```

`--check` is non-installing (apart from its temporary comparison file). Use it
from the actual installed source directory, not an unrelated source copy.

| Symptom | Inspect / interpretation |
|---|---|
| Active service, no trigger | X11 vs Wayland, inherited `DISPLAY`/`XAUTHORITY`, selected trigger; `pynput` sees only Xwayland input under Wayland |
| No speech / microphone error | System default input, permissions and `capture_open_error`/`capture_error`; app has no input-device CLI selector |
| Text missing at start | `capture_ready` latency / `capture_start_delay`; the sound is not a capture-ready acknowledgement |
| Text appears late | Full pause, eager/merge settings, pending previous session and inference time; start/stop sounds are not decode completion |
| Silent/odd output | Inspect `outcome.audio`, `discard_reason`, `suspect_decode`, `recovery_discarded`; signal levels are heuristics, not a speech detector oracle |
| Browser navigates on thumb button | Check original/current X button maps and helper logs; listener alone does not suppress back/forward |
| Failure after replug/suspend | Verify device-add watcher and new device's map; do not assume the old X device ID still applies |
| Clipboard filled but editor empty | `delivery.focus_ok`, CopyQ/xclip return codes, paste receipt, cursor/widget and terminal shortcut; no target-app acknowledgement exists |
| CopyQ unavailable | `copyq_failure`, `clipboard_fallback`, `delivery_degraded`; xclip can paste but provides no explicit CopyQ history receipt |
| Extra corrections at end | Intended recovery path; inspect original and correction delivery separately, then manually reconcile the anchored text |
| CUDA out of memory | `decoder_error` and `cuda_oom_snapshot`; inspect free VRAM/other processes without stopping them automatically |
| Repeated service failures | Driver/library load error, dependency environment and model availability; restart-on-failure does not solve bad prerequisites |
| Duplicate process / GPU load | Review process list and launchers; keep one managed instance, do not kill by matching a command-line pattern |

### Reading coverage diagnostics

`DCT-COVERAGE` lines are JSON after the prefix. Useful `kind` values:

- `session_start`, `capture_ready`, `capture_overflow`: capture lifetime and
  possible driver-level loss. Overflowed samples are not in the ledger.
- `outcome`, `delivery`: each captured range's recognition and delivery receipt.
- `stop_check`, `recovery_outcome`, `recovery_discarded`, `unresolved`,
  `recovery_failed`: why and how a suspect range was checked.
- `correction_delivery`, `stop_status_delivery`: appended recovery output and
  its delivery receipt, including failures.
- `stop_summary`: sample/range counts, correction/unresolved/silent/duplicate
  counts, original/correction delivery failures and `check_seconds`.
- `audio_released`: ledger PCM references cleared, not secure erasure.
- `start_while_finishing`, `waited_for_previous`, `previous_session_timeout`:
  new capture while older output drains; the wait limit is 300 seconds.
- `duration_warning`, `duration_limit`, `timer_stop`: session limits;
  warning/limit incidents are journal-only, not desktop popups.
- `cuda_oom_snapshot`: bounded read-only GPU allocation diagnostic on a matching
  CUDA out-of-memory exception; it does not retry or restart by itself.

Check `correction_delivery_failures` as well as `unresolved_ranges`; having
recognised recovery words does not mean their paste succeeded. Likewise,
`shortcut_sent_not_acknowledged` is intentionally weaker than “received”. There
are no independent desktop notifications for these incidents.

For any rebuild result, retain the source revision (or snapshot manifest),
Python/package versions, actual driver/runtime libraries, model identity/cache
state, session type, microphone/mouse configuration and observed editor/CopyQ
results. Keep this inventory separate from dictated content. A unit marked
`CURRENT` and a green unit suite are useful checks, not a clean-machine rebuild
certificate.
