# Multilingual Dictation App based on Faster Whisper

This is based on the awesome work by https://github.com/guillaumekln/faster-whisper, https://github.com/foges/whisper-dictation and various PRs in the later repo

Multilingual dictation app based on the Faster Whisper to provide accurate and efficient speech-to-text conversion in any application. The app runs in the background and is triggered through a keyboard shortcut. It is also entirely offline, so no data will be shared. It allows users to set up their own keyboard combinations and choose from different Whisper models, and languages.

## Quick start
1. Run the app, switch to another application that accepts text input (editor, browser textarea, etc)
2. Use the key combination to toggle dictation.
Default to double-tapping right-cmd on macOS, right-super on Linux and Win+Z on Windows.
3. Start speaking. Text appears in real-time after each natural pause (~1 second of silence).
4. When you stop speaking for 10 seconds, the session ends automatically (double beep).
   Or tap the key again to stop manually (single beep).

## Prerequisites

### System packages (Linux)

```bash
sudo apt install portaudio19-dev xdotool xclip
```

- **portaudio19-dev** — required by PyAudio for microphone access
- **xdotool** — used to detect the active window (for Kitty terminal clipboard mode)
- **xclip** — used for clipboard-based text injection in Kitty terminal

### System packages (macOS)

```bash
brew install portaudio
```

### NVIDIA GPU (optional, recommended)

For real-time transcription with large models, a CUDA-capable GPU is strongly recommended.

```bash
# Check NVIDIA driver
nvidia-smi

# CUDA toolkit should be installed (CUDA 12.x recommended)
nvcc --version
```

The pip packages `torch` and `ctranslate2` with CUDA support are installed via requirements.
If you have issues, install PyTorch manually first:

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Permissions
The app requires accessibility permissions to register global hotkeys and permission to access your microphone for speech recognition.

For example, if you launch the app from MacOS terminal, you need to grant permission at Settings > Privacy & Security > Microphone
and Input Monitoring

![macos > settings > privacy security > microphone](macos-privacy-security-microphone.png)

## Installation
Clone the repository:

```bash
git clone https://github.com/doctorguile/faster-whisper-dictation.git
cd faster-whisper-dictation
```

Create a virtual environment:

```bash
python3 -m venv venv

source venv/bin/activate      # MacOS / Linux

venv\scripts\activate.bat     # Windows cmd.exe
```

Install the required packages:

```bash
pip3 install -r requirements.txt
```

## Usage

### Streaming mode (default)

The default mode uses **Silero VAD** (Voice Activity Detection) to split audio on natural
pauses. Each speech chunk is transcribed and typed immediately while recording continues.
This gives real-time feedback — you see text appear after each sentence.

```bash
# GPU with large model (recommended for accuracy)
python3 dictation.py -m large-v3 -v cuda -c float16 -l nl

# CPU with small model (no GPU required)
python3 dictation.py -m small -l en

# Custom silence threshold (split after 500ms pause instead of 1000ms)
python3 dictation.py -m large-v3 -v cuda -c float16 -l nl --silence-ms 500

# Disable auto-stop (keep recording until manual stop)
python3 dictation.py -m large-v3 -v cuda -c float16 -l nl --auto-stop-silence 0

# No time limit (auto-stop after silence still works)
python3 dictation.py -m large-v3 -v cuda -c float16 -l nl -t 0
```

**How it works:**

```
[Recording Thread] --> audio frames --> [VAD Thread] --> speech chunks
                                        (Silero VAD)
    --> transcription_queue --> [Transcription Thread] --> text
    --> typing_queue --> [Typing Thread] --> keyboard output
```

1. Double-tap Right Ctrl --> beep
2. Speak a sentence --> pause ~1 second --> text appears (~1.8s after pause)
3. Speak next sentence --> pause --> text appears
4. Stop speaking for 10 seconds --> double beep (auto-stop)
5. Or tap Right Ctrl --> single beep (manual stop)

### Batch mode (original)

Records all audio first, then transcribes everything at once after stopping.
Useful as a fallback if streaming mode has issues.

```bash
python3 dictation.py -m large-v3 -v cuda -c float16 -l nl --batch-mode
```

## Options

```
python3 dictation.py [-h] [-m MODEL_NAME] [-k KEY_COMBO] [-d DOUBLE_KEY]
                     [-t MAX_TIME] [-v DEVICE] [-c COMPUTE_TYPE]
                     [-l LANGUAGE] [--silence-ms MS] [--auto-stop-silence S]
                     [--batch-mode]

  -h, --help            show this help message and exit

  -m MODEL_NAME, --model-name MODEL_NAME
                        Size of the model to use (tiny, tiny.en, base, base.en,
                        small, small.en, medium, medium.en, large-v1, large-v2,
                        large-v3, or large). Default: base.

  -k KEY_COMBO, --key-combo KEY_COMBO
                        Key combination to toggle the app.
                        Examples: <cmd_l>+<alt>+x , <ctrl>+<alt>+a
                        Default: <win>+z on Windows.

  -d DOUBLE_KEY, --double-key DOUBLE_KEY
                        Key for double-tap activation (macOS/Linux).
                        Default: Right Cmd (macOS), Right Ctrl (Linux).

  -t MAX_TIME, --max-time MAX_TIME
                        Maximum recording time in seconds (safety limit).
                        Set to 0 to disable. Default: 30.

  -v DEVICE, --device DEVICE
                        Inference device: 'cpu', 'cuda', or 'auto'.
                        Default: cpu.

  -c COMPUTE_TYPE, --compute-type COMPUTE_TYPE
                        Compute type: 'int8', 'float16', or 'float32'.
                        Use float16 with CUDA for best speed/accuracy.
                        Default: int8.

  -l LANGUAGE, --language LANGUAGE
                        Force language (e.g., 'nl', 'en', 'fr', 'de').
                        Improves accuracy for short fragments.
                        Default: auto-detect.

  --silence-ms MS       Silence duration in ms before splitting a speech chunk
                        (streaming mode). Lower = faster feedback but may split
                        mid-sentence. Default: 1000.

  --auto-stop-silence S
                        Automatically stop after S seconds of silence
                        (streaming mode). Plays double beep to distinguish from
                        manual stop. Set to 0 to disable. Default: 10.

  --batch-mode          Use original batch mode instead of streaming mode.
```

## Autostart on login (Linux)

Create `~/.config/autostart/faster-whisper-dictation.desktop`:

```ini
[Desktop Entry]
Type=Application
Name=Faster Whisper Dictation
Comment=Start speech recognition on login
Exec=/bin/bash -c "cd /home/YOUR_USER/Repositories/faster-whisper-dictation && source venv/bin/activate && python3 dictation.py -m large-v3 -v cuda -c float16 -l nl -t 0"
Terminal=false
Hidden=false
X-GNOME-Autostart-enabled=true
StartupNotify=false
```

Replace `YOUR_USER` with your username and adjust the model/device/language options.

## Installation on a new machine (step by step)

Complete setup from scratch on Ubuntu with NVIDIA GPU:

```bash
# 1. System dependencies
sudo apt update
sudo apt install portaudio19-dev xdotool xclip git python3-venv

# 2. Clone and setup
cd ~/Repositories  # or wherever you keep repos
git clone https://github.com/doctorguile/faster-whisper-dictation.git
cd faster-whisper-dictation
python3 -m venv venv
source venv/bin/activate

# 3. Install Python packages
pip3 install -r requirements.txt

# 4. (GPU only) If torch doesn't detect CUDA, reinstall with CUDA support:
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 5. Test run
python3 dictation.py -m large-v3 -v cuda -c float16 -l nl -t 0

# 6. (Optional) Create autostart entry
mkdir -p ~/.config/autostart
cat > ~/.config/autostart/faster-whisper-dictation.desktop << 'EOF'
[Desktop Entry]
Type=Application
Name=Faster Whisper Dictation
Comment=Start speech recognition on login
Exec=/bin/bash -c "cd $HOME/Repositories/faster-whisper-dictation && source venv/bin/activate && python3 dictation.py -m large-v3 -v cuda -c float16 -l nl -t 0"
Terminal=false
Hidden=false
X-GNOME-Autostart-enabled=true
StartupNotify=false
EOF
```

The first run will download the Whisper model from Hugging Face (~3GB for large-v3).

## Kitty terminal support

When the active window is a Kitty terminal, text is injected via clipboard
(Ctrl+Shift+V) instead of character-by-character typing. This works around
a known Kitty bug with XTest keyboard simulation on AZERTY keyboards.

This is detected automatically — no configuration needed.

## Replace macOS default dictation trigger key
You can use this app to replace macOS built-in dictation. i.e. Double tap Right Cmd key to begin recording and stop recording with a single tap

To use this trigger, go to System Settings -> Keyboard, disable Dictation. If you double click Right Command key on any text field, macOS will ask whether you want to enable Dictation, so select Don't Ask Again.

## Setting the App as a Startup Item (macOS)

 1. Open System Preferences.
 2. Go to Users & Groups.
 3. Click on your username, then select the Login Items tab.
 4. Click the + button and add the `run.sh` script from the faster-whisper-dictation folder.
