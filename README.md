# Faster Whisper desktop dictation

Dictate into a desktop text field using local speech recognition. A keyboard
trigger or mouse thumb button starts the microphone; speech is decoded in
chunks and pasted into the window where dictation started. You do not need a
cloud transcription account.

This fork builds on [doctorguile/faster-whisper-dictation](https://github.com/doctorguile/faster-whisper-dictation),
[faster-whisper](https://github.com/SYSTRAN/faster-whisper) and
[foges/whisper-dictation](https://github.com/foges/whisper-dictation).
It adds a streaming pipeline, desktop integration and bounded recovery checks.
The original attribution and [MIT license](LICENSE) apply; retain the license
when redistributing a source copy.

## What the setup consists of

- **Recognition:** Python, faster-whisper/CTranslate2 and a Whisper model;
  the installed service selects `large-v3`, NVIDIA CUDA and `float16`.
- **Capture and timing:** PyAudio reads the default microphone. Silero VAD
  finds pauses; queues separate recording, decoding and text delivery.
- **Desktop input/output:** pynput listens for triggers; X11 tools restore the
  starting window. CopyQ stores text in history and sets the clipboard; xclip
  is the clipboard-only fallback. A paste shortcut inserts the text.
- **Recovery:** a RAM-only audio ledger tracks sample ranges, decode outcomes
  and delivery receipts. At stop, suspect ranges can produce appended corrections.
- **Lifecycle:** `install.sh` generates `spraakherkenning.service`, a systemd
  **user** service tied to the graphical session. Mouse helper scripts prevent
  thumb-button dictation from also navigating browser history.

The reference environment is **Ubuntu + an X11/Xorg desktop + NVIDIA/CUDA**.
That is experience with an existing installation, not proof of a clean-machine
rebuild. The code in this snapshot includes newer recovery/filter changes that
have offline regression coverage, not a new full desktop acceptance test.
Windows/macOS branches exist in the code, but this fork's complete paste and
service setup is **not validated there**. Wayland is not an equivalent target.
CPU inference is selectable, but equal speed or desktop parity is not promised.

## Read in this order

1. [Rebuild and operate](docs/rebuild.md): requirements, installation, configuration,
   daily controls and a small acceptance check.
2. [Architecture and fork changes](docs/architecture.md): pipeline, reasons for
   the additions, stop checks and known limits.
3. [Privacy, offline tests and diagnosis](docs/privacy_testing.md): what remains
   on disk, safe verification and troubleshooting.

These documents travel with a source-only copy: they do not require the fork's
Git history, local evidence files or a private setup repository. Obtain **this
fork's agreed snapshot**, not just the upstream repository linked above.

## Daily controls (Linux default)

| Action | Control |
|---|---|
| Start | Double-tap **Right Ctrl** (within 0.3 seconds) |
| Stop recording | Tap **Right Ctrl** once |
| Start in the alternate language | Hold **Left Shift** while double-tapping Right Ctrl |
| Start/stop by mouse | Press either configured thumb button |
| Automatic stop | After about 10 seconds of detected silence |

The installer selects Dutch by default; Left Shift selects English for that
keyboard-started session only. The next unshifted keyboard start returns to the
configured default. There is **no default AltGr language toggle**. Mouse and
custom-combination starts use the worker's current language, so use the keyboard
when the language choice matters.

Open and focus the intended text field **before** starting. Text arrives in
blocks, not necessarily as soon as each word is spoken. Stop ends capture, but
queued transcription and corrections can still arrive afterwards. Corrections
are appended, never inserted into earlier text; their labels are currently
Dutch even in an English session.

## Before installing or sharing logs

`./install.sh` is not a preview: it writes/enables a service, starts or restarts
it and remaps mouse buttons. It may install packages into a new venv and download
a model on first service startup. Read the rebuild guide first.

**Local recognition does not mean no stored text.** The app creates no audio
archive, but CopyQ history and the systemd journal can contain the full dictation.
In particular, `DCT-COVERAGE` JSON includes raw/filtered text and corrections.
Review the [privacy notes](docs/privacy_testing.md#privacy-and-retention) before
using sensitive speech or sharing diagnostics.
