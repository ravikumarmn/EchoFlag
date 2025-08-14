# EchoFlag

A system for live audio transcription and violation flagging using Google Cloud Speech-to-Text with speaker differentiation.

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

2. **Google Cloud Setup (Step-by-Step)**:

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
   The `.env` file should contain:
   ```
   GOOGLE_APPLICATION_CREDENTIALS=/path/to/your-service-account-key.json
   ```
   
   Alternatively, set the environment variable in your terminal:
   ```bash
   # For macOS/Linux
   export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your-service-account-key.json"
   
   # For Windows PowerShell
   $env:GOOGLE_APPLICATION_CREDENTIALS="C:\path\to\your-service-account-key.json"
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

## Future Enhancements

- Real-time audio transcription
- Web interface for uploading and processing audio files
- Custom violation rules management interface
