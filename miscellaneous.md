# EchoFlag - System Dependencies and Environment Notes

This document tracks system-level dependencies, environment advice, and common errors.

## Required Python Packages

Defined in `requirements.txt`:
- streamlit==1.37.1
- openai==1.37.0
- python-dotenv==1.0.0
- gtts==2.5.0
- pydub==0.25.1
- SpeechRecognition==3.10.0
- pocketsphinx==0.1.15

Install with:
```bash
pip install -r requirements.txt
```

## System Packages (not in requirements.txt)

These are OS-level tools that some Python packages rely on. Install via your package manager (macOS uses Homebrew):

- swig — required to build `pocketsphinx` from source when wheels are not available
- ffmpeg — required by `pydub` for audio processing

macOS (Homebrew):
```bash
brew install swig ffmpeg
```

Note: Do NOT add these to `requirements.txt`. They are not Python packages.

## Recommended Python Version

- Preferred: Python 3.11 (stable wheels for `pocketsphinx` and audio stack)
- Possible but fragile: Python 3.13 (often triggers a source build of `pocketsphinx`, needing `swig` and toolchain)

Create a Python 3.11 virtual environment (macOS/Homebrew path may vary):
```bash
brew install python@3.11  # if not installed
/opt/homebrew/bin/python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Making PocketSphinx Optional (if offline STT not needed)

If you don't need offline speech-to-text on this machine:
- Remove `pocketsphinx` from `requirements.txt`
- Ensure `src/audio_to_transcript.py` falls back to online recognition when `pocketsphinx` is missing

## Common Errors

- `error: command 'swig' failed: No such file or directory`
  - Cause: `pocketsphinx` building from source but `swig` not installed
  - Fix: `brew install swig` (and prefer Python 3.11)

- `pydub` cannot find ffmpeg
  - Cause: `ffmpeg` not installed or not on PATH
  - Fix: `brew install ffmpeg`

## Quick Smoke Test

Once dependencies are installed:
```bash
# Example: run the LLM-to-audio pipeline
python src/llm_to_audio_conversation.py --severity YELLOW --violations 1 --output audio_output/sample.mp4

# Example: run the audio->transcript flow (ensure ffmpeg installed for pydub usage where relevant)
python src/audio_to_transcript.py --input some_audio_file.wav
```

Record any recurring errors and fixes here for quick reference.
