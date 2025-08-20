#!/usr/bin/env python3
import streamlit as st
from dotenv import load_dotenv
load_dotenv()
# Must be the first Streamlit command on the page
st.set_page_config(page_title="EchoFlag - Audio Violations", layout="centered")

"""
EchoFlag Streamlit App (no external endpoints)

This UI processes audio locally by calling the internal analyzer directly and
uses st.secrets for configuration (e.g., OPENAI_API_KEY).

Run UI:
  streamlit run src/app.py
"""
import os
import tempfile
import json
from datetime import datetime

st.title("EchoFlag – Audio Violation Tester")
st.caption("Upload audio and run local transcription + LLM analysis. No HTTP calls.")

with st.sidebar:
    st.header("Settings")
    model = st.selectbox("LLM Model (analysis)", ["gpt-4"], index=0)
    use_google = st.toggle(
        "Use Google Web Speech (transcription)",
        value=True,
        help="If off, uses offline Sphinx (needs pocketsphinx installed)."
    )

# Configuration using st.secrets (for Streamlit Cloud deployment)
try:
    openai_key = st.secrets["OPENAI_API_KEY"]
    # Set environment variable for the analyzer module
    os.environ["OPENAI_API_KEY"] = openai_key
except KeyError:
    # Fallback to environment variable for local development
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        st.warning(
            "OPENAI_API_KEY not found in st.secrets or environment.\n"
            "For Streamlit Cloud: Add OPENAI_API_KEY to your app's secrets.\n"
            "For local development: Set it in environment or .env file.\n"
            "Examples:\n"
            "  export OPENAI_API_KEY=sk-...  # macOS/Linux\n"
            "  setx OPENAI_API_KEY sk-...    # Windows (new shell)\n"
            "Or create a .env file with: OPENAI_API_KEY=sk-..."
        )

# Import analyzer; it loads .env internally and reads OPENAI_API_KEY from env
import sys
import os
# Add the parent directory to sys.path to ensure imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    # When running via `streamlit run src/app.py`, Python often adds the script
    # directory (src/) to sys.path, so the module is `audio_to_violations`.
    from audio_to_violations import AudioToViolations
except Exception:
    try:
        # Fallback if project root is on sys.path and `src` is a package
        from src.audio_to_violations import AudioToViolations
    except Exception as e:
        st.error(f"Failed to load analyzer: {e}")
        AudioToViolations = None

st.subheader("1) Upload an audio file")
uploaded = st.file_uploader(
    "Choose an audio file (mp3/mp4/wav)",
    type=["mp3", "mp4", "wav"]
)

col1, col2 = st.columns(2)
with col1:
    transcribe_clicked = st.button("Transcribe Only", disabled=uploaded is None or AudioToViolations is None)
with col2:
    analyze_clicked = st.button("Analyze Violations", type="primary", disabled=uploaded is None or AudioToViolations is None)

if uploaded is not None:
    st.audio(uploaded)

def _save_to_temp(uploaded_file) -> str:
    suffix_map = {
        "audio/mpeg": ".mp3",
        "audio/wav": ".wav",
        "audio/x-wav": ".wav",
        "video/mp4": ".mp4",
        "audio/mp4": ".mp4",
    }
    mime = uploaded_file.type or "audio/mpeg"
    suffix = suffix_map.get(mime, os.path.splitext(uploaded_file.name)[1] or ".mp3")
    fd, path = tempfile.mkstemp(suffix=suffix, prefix="echoflag_")
    with os.fdopen(fd, "wb") as f:
        f.write(uploaded_file.getvalue())
    return path

def _make_processor():
    return AudioToViolations(output_dir="violations_output") if AudioToViolations else None

if transcribe_clicked and uploaded is not None and AudioToViolations is not None:
    temp_path = None
    try:
        temp_path = _save_to_temp(uploaded)
        st.info("Transcribing…")
        processor = _make_processor()
        res = processor.process_audio_file(temp_path, use_google=use_google)
        st.success("Transcription complete")
        st.json(res)
    except Exception as e:
        st.error(f"Transcription failed: {e}")
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

if analyze_clicked and uploaded is not None and AudioToViolations is not None:
    temp_path = None
    try:
        temp_path = _save_to_temp(uploaded)
        st.info("Analyzing…")
        processor = _make_processor()
        res = processor.process_and_analyze(temp_path, use_google=use_google, model=model)
        st.success("Analysis complete")
        st.download_button(
            "Download Analysis JSON",
            data=json.dumps(res, indent=2).encode("utf-8"),
            file_name=f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
        )
        st.json(res)
    except Exception as e:
        st.error(f"Analysis failed: {e}")
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
