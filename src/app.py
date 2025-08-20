#!/usr/bin/env python3
import streamlit as st
# Must be the first Streamlit command on the page
st.set_page_config(page_title="EchoFlag - Audio Violations", layout="centered")

"""
EchoFlag Streamlit Client

Minimal UI that sends audio to the FastAPI backend and displays results.

Run backend:
  uvicorn src.api:app --reload --port 8000

Run UI:
  streamlit run src/app.py
"""
import os
import requests
from datetime import datetime

API_BASE = os.environ.get("ECHOFLAG_API", "http://127.0.0.1:8080")

st.title("EchoFlag – Audio Violation Tester")
st.caption("Upload audio and analyze via the EchoFlag FastAPI backend.")

with st.sidebar:
    st.header("Settings")
    api_base = st.text_input("API Base URL", value=API_BASE)
    model = st.selectbox("LLM Model (analysis)", ["gpt-4"], index=0)
    use_google = st.toggle("Use Google Web Speech (transcription)", value=True, help="If off, backend may use offline Sphinx (less accurate)")

st.subheader("1) Upload an audio file")
uploaded = st.file_uploader("Choose an audio file (mp3/mp4/wav)", type=["mp3", "mp4", "wav", "wav"])   

col1, col2 = st.columns(2)
with col1:
    transcribe_clicked = st.button("Transcribe Only", disabled=uploaded is None)
with col2:
    analyze_clicked = st.button("Analyze Violations", type="primary", disabled=uploaded is None)

if uploaded is not None:
    st.audio(uploaded)

def _post_file(url: str, file, extra_form: dict):
    files = {"file": (file.name, file.getvalue(), file.type or "application/octet-stream")}
    data = extra_form or {}
    r = requests.post(url, files=files, data=data, timeout=120)
    r.raise_for_status()
    return r.json()

if transcribe_clicked and uploaded is not None:
    try:
        st.info("Transcribing…")
        res = _post_file(f"{api_base}/transcribe", uploaded, {"use_google": str(use_google).lower()})
        st.success("Transcription complete")
        st.json(res)
    except Exception as e:
        st.error(f"Transcription failed: {e}")

if analyze_clicked and uploaded is not None:
    try:
        st.info("Analyzing…")
        res = _post_file(f"{api_base}/analyze", uploaded, {"use_google": str(use_google).lower(), "model": model})
        st.success("Analysis complete")
        st.download_button("Download Analysis JSON", data=str(res).encode("utf-8"), file_name=f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", mime="application/json")
        st.json(res)
    except Exception as e:
        st.error(f"Analysis failed: {e}")
