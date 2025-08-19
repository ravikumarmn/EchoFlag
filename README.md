# EchoFlag

A system for audio transcription and violation flagging with a Streamlit UI.

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

### Transcribe an Audio File with Speaker Differentiation

```bash
python transcribe.py path/to/audio_file.wav [num_speakers]
```

### Flag Violations in a Transcript

```bash
python flagging.py path/to/transcript_file.json
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
5. **System dependency**: The app needs FFmpeg for audio processing. Streamlit Cloud installs it via the provided `packages.txt`.
6. **Run**: Deploy. The app will build and start automatically.

If you see build errors related to `pocketsphinx`, `swig`, or `ffmpeg` pip packages, remove them from `requirements.txt` (they are not needed). FFmpeg is installed as a system package via `packages.txt`.

## Future Enhancements

- Real-time audio transcription
- Web interface for uploading and processing audio files
- Custom violation rules management interface
