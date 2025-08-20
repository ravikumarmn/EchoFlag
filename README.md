# EchoFlag

A simple FastAPI backend + Streamlit client for audio transcription and violation flagging.

Notes:
- The Streamlit app in `src/app.py` uses the free Google Web Speech endpoint via `SpeechRecognition` by default (no Google Cloud project required) and OpenAI for violation analysis.
- If you wish to use Google Cloud Speech-to-Text, that is an optional future enhancement and is not required for this app to run.

## Project Overview

EchoFlag transcribes audio files with speaker differentiation and flags violations based on content severity levels:
- **Red**: High severity violations
- **Orange**: Medium severity violations
- **Yellow**: Low severity violations

## Setup Instructions

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   Note: If you are using Python 3.13+, the standard library module `audioop` was removed. This project includes the `audioop-lts` backport in `requirements.txt` to restore compatibility. Ensure your environment installs it successfully.

2. Ensure FFmpeg is installed on the system (required by `pydub`). On macOS:
   ```bash
   brew install ffmpeg
   ```

2. (Optional) **Google Cloud Setup (Step-by-Step)**:

   ### Create a Google Cloud Account
   1. Go to [cloud.google.com](https://cloud.google.com/)
   2. Click "Get Started for Free" or "Sign In" if you already have a Google account
   3. Complete the signup process (requires credit card for verification, but you get free credits)

   ### Create a New Project
   1. Go to the [Google Cloud Console](https://console.cloud.google.com/)
   2. Click on the project dropdown at the top of the page
   3. Click "New Project"
   4. Enter a project name (e.g., "EchoFlag")
   5. Click "Create"
   6. Wait for the project to be created, then select it from the dropdown

   ### Enable the Speech-to-Text API
   1. In the Google Cloud Console, go to the Navigation menu (hamburger icon)
   2. Select "APIs & Services" > "Library"
   3. Search for "Speech-to-Text API"
   4. Click on "Cloud Speech-to-Text API"
   5. Click "Enable"

   ### Create a Service Account
   1. In the Google Cloud Console, go to "IAM & Admin" > "Service Accounts"
   2. Click "Create Service Account"
   3. Enter a service account name (e.g., "echoflag-service")
   4. Add a description (optional)
   5. Click "Create and Continue"

   ### Assign Roles to the Service Account
   1. Click "Select a role"
   2. Search for "Speech-to-Text"
   3. Select "Cloud Speech-to-Text User" role
   4. Click "Continue"
   5. Click "Done"

   ### Create and Download Service Account Key
   1. Find your newly created service account in the list
   2. Click the three dots (actions menu) at the end of the row
   3. Select "Manage keys"
   4. Click "Add Key" > "Create new key"
   5. Select "JSON" as the key type
   6. Click "Create"
   7. The key file will automatically download to your computer
   8. Store this file securely - it grants access to your Google Cloud resources

3. **Environment Variables**:
   Required for the app to analyze violations via OpenAI:
   ```
   OPENAI_API_KEY=sk-...
   ```
   You can place this in a local `.env` for development. In hosted environments (e.g., Streamlit Cloud), set it as an environment variable in the app settings. 
   
   Optional (only if you choose to integrate Google Cloud STT later):
   ```
   GOOGLE_APPLICATION_CREDENTIALS=/path/to/your-service-account-key.json
   ```

## Usage

### Run the Backend (FastAPI)

Start the API server:

```bash
uvicorn src.api:app --reload --port 8000
```

Endpoints:

- `GET /health` → simple status
- `POST /transcribe` (multipart): fields: `file` (audio), `use_google` (bool)
- `POST /analyze` (multipart): fields: `file` (audio), `use_google` (bool), `model` (default `gpt-4`)

### Run the Frontend (Streamlit)

In another terminal:

```bash
streamlit run src/app.py
```

Set API base in the sidebar if different from default `http://127.0.0.1:8000`.

### CLI scripts (legacy)

```bash
python src/audio_to_transcript.py path/to/audio_file.wav
```

### Direct analysis from Python (programmatic)

```bash
python src/audio_to_violations.py --audio_file path/to/audio_file.wav --use-google --model gpt-4
```

## Project Structure

- `transcribe.py`: Audio transcription module with speaker differentiation
- `flagging.py`: Violation flagging system with severity levels
- `requirements.txt`: Project dependencies

## Deployment (Streamlit Community Cloud)

1. **Push to GitHub**: Ensure this repository (including `requirements.txt` and `packages.txt`) is pushed to GitHub.
2. **Create App**: Go to https://share.streamlit.io, click "New app", and select this repo/branch.
3. **Entry point**: Set the file to `src/app.py`.
4. **Secrets / Env Vars**: In the app settings, add environment variable:
   - `OPENAI_API_KEY`: your OpenAI API key
5. Backend note: Streamlit Cloud runs only the frontend. Host the FastAPI backend separately (e.g., on Render/Fly/EC2) and set `ECHOFLAG_API` env var to its base URL.
6. **System dependency**: The app needs FFmpeg for audio processing locally. On Streamlit Cloud, install via `packages.txt` if you also run audio conversions there.
6. **Run**: Deploy. The app will build and start automatically.

If you see build errors related to `pocketsphinx`, `swig`, or `ffmpeg` pip packages, remove them from `requirements.txt` (they are not needed). FFmpeg should be installed as a system package.

## Future Enhancements

- Real-time audio transcription
- Web interface for uploading and processing audio files
- Custom violation rules management interface
