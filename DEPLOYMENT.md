# EchoFlag - Streamlit Cloud Deployment Guide

## Prerequisites

1. **GitHub Repository**: Your code should be pushed to a GitHub repository
2. **Streamlit Cloud Account**: Sign up at [share.streamlit.io](https://share.streamlit.io)
3. **OpenAI API Key**: Required for LLM analysis functionality

## Deployment Steps

### 1. Prepare Your Repository

Ensure your repository has these files:
- `requirements.txt` - Python dependencies
- `packages.txt` - System packages for audio processing
- `src/app.py` - Main Streamlit application
- `secrets.toml.template` - Template for secrets configuration

### 2. Deploy to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "New app"
3. Connect your GitHub account if not already connected
4. Select your repository: `ravikumarmn/EchoFlag`
5. Set the main file path: `src/app.py`
6. Click "Deploy"

### 3. Configure Secrets

After deployment, configure your app secrets:

1. In your Streamlit Cloud dashboard, click on your app
2. Go to "Settings" → "Secrets"
3. Add the following secrets:

```toml
OPENAI_API_KEY = "sk-your-actual-openai-api-key-here"
```

### 4. App Configuration

The app is configured to:
- Use `st.secrets` for secure API key management
- Fall back to environment variables for local development
- Handle audio file uploads (mp3, mp4, wav)
- Process audio transcription and violation analysis

## Local Development

For local development:

1. Create `.streamlit/secrets.toml` from the template:
```bash
mkdir -p .streamlit
cp secrets.toml.template .streamlit/secrets.toml
```

2. Edit `.streamlit/secrets.toml` and add your API keys

3. Run the app:
```bash
streamlit run src/app.py
```

## Troubleshooting

### Common Issues

1. **Module Import Errors**: The app automatically handles Python path configuration
2. **Audio Processing**: System packages in `packages.txt` handle audio dependencies
3. **API Key Issues**: Check that secrets are properly configured in Streamlit Cloud

### System Dependencies

The `packages.txt` file includes necessary system packages:
- `ffmpeg` - Audio/video processing
- `python3-dev` - Python development headers
- `portaudio19-dev` - Audio I/O library
- `libsndfile1` - Sound file library

## Features

- **Audio Upload**: Support for MP3, MP4, and WAV files
- **Transcription**: Google Web Speech API or offline Sphinx
- **Violation Analysis**: OpenAI GPT-4 powered compliance checking
- **Results Export**: Download analysis results as JSON

## Security Notes

- Never commit actual API keys to version control
- Use Streamlit Cloud secrets for production deployment
- The `.streamlit/` directory is gitignored to prevent accidental commits
