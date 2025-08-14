#!/usr/bin/env python3
"""
EchoFlag Streamlit App

Features:
- Upload an audio file (mp3/mp4/wav) OR generate a dummy LLM-based conversation that violates mutual fund rules.
- Run violation analysis and display the JSON output.

Prereqs:
- .env with OPENAI_API_KEY
- ffmpeg installed (for pydub)
- Requirements: streamlit, pydub, gtts, SpeechRecognition, openai

Run:
  streamlit run src/app.py
"""
import os
import sys
import tempfile
from datetime import datetime
from typing import Optional

import streamlit as st
from dotenv import load_dotenv
from shutil import which as sh_which

# Ensure we can import project modules when running as a script
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
for p in (PROJECT_ROOT, SRC_DIR):
    if p not in sys.path:
        sys.path.append(p)

load_dotenv()

# Import local modules (non-fatal)
HAS_ANALYZER = True
ANALYZER_IMPORT_ERR = None
try:
    from audio_to_violations import AudioToViolations
except Exception as e:
    HAS_ANALYZER = False
    ANALYZER_IMPORT_ERR = str(e)

# LLM-based generator (single file conversation)
try:
    from llm_to_audio_conversation import LLMToAudioConversation
    HAS_LLM_GEN = True
except Exception as e:
    HAS_LLM_GEN = False

st.set_page_config(page_title="EchoFlag - Audio Violations", layout="centered")
st.title("EchoFlag – Audio Violation Tester")
st.caption("Upload audio or generate a dummy conversation that violates mutual fund distribution rules, then analyze and view JSON.")

# Detect ffmpeg availability without importing pydub
FFMPEG_AVAILABLE = sh_which("ffmpeg") is not None
if not FFMPEG_AVAILABLE:
    st.warning("ffmpeg not detected. Uploads are limited to WAV, and generated conversations will be analyzed as text (no audio synthesis).")

with st.sidebar:
    st.header("Settings")
    model = st.selectbox("LLM Model (analysis)", ["gpt-4"], index=0)
    use_google = st.toggle("Use Google Web Speech (transcription)", value=True, help="If off, uses offline Sphinx (less accurate)")

# Helper: write uploaded bytes to a temp file and return path

def _save_upload_to_temp(uploaded_file) -> str:
    suffix = os.path.splitext(uploaded_file.name)[1] or ".bin"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.write(uploaded_file.getbuffer())
    tmp.flush()
    tmp.close()
    return tmp.name

# Helper: convert arbitrary audio to wav for SpeechRecognition compatibility

def _to_wav_if_needed(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".wav"]:
        return path
    try:
        # Lazy-import pydub only when we actually need it
        from pydub import AudioSegment
        out_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        seg.export(out_path, format="wav")
        return out_path
    except Exception as e:
        st.error(f"Audio conversion failed: {e}")
        return path

# UI Tabs
upload_tab, generate_tab = st.tabs(["Upload Audio", "Generate Dummy Audio (LLM)"])

with upload_tab:
    st.subheader("1) Upload an audio file")
    allowed_types = ["wav"] if not FFMPEG_AVAILABLE else ["mp3", "mp4", "wav"]
    if not HAS_ANALYZER:
        st.info("Audio analyzer unavailable on this environment; upload analysis is disabled.")
    uploaded = st.file_uploader(f"Choose an audio file ({', '.join(allowed_types)})", type=allowed_types, disabled=not HAS_ANALYZER)
    analyze_clicked = st.button("Analyze Uploaded Audio", type="primary", disabled=(uploaded is None or not HAS_ANALYZER))

    if analyze_clicked and uploaded is not None:
        path = _save_upload_to_temp(uploaded)
        wav_path = path if (not FFMPEG_AVAILABLE) else _to_wav_if_needed(path)

        st.info("Transcribing and analyzing…")
        analyzer = AudioToViolations(output_dir="violations_output")
        try:
            result = analyzer.process_and_analyze(audio_file=wav_path, use_google=use_google, model=model)
        except Exception as e:
            st.error(f"Analysis failed: {e}")
            result = None
        if result:
            st.success("Analysis complete")
            st.download_button("Download Analysis JSON", data=str(result).encode("utf-8"), file_name=f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", mime="application/json")
            st.json(result)

with generate_tab:
    st.subheader("2) Generate dummy violation conversation and analyze")
    if not HAS_LLM_GEN:
        st.error("LLM generator not available. Ensure src/llm_to_audio_conversation.py is present and dependencies installed.")
    else:
        severity = st.selectbox("Severity", ["ALL", "RED", "ORANGE", "YELLOW"], index=0)
        n_viol = st.slider("Number of violation patterns", min_value=1, max_value=6, value=5)
        silence_ms = st.slider("Silence between turns (ms)", min_value=200, max_value=1500, value=700, step=50)
        generate_clicked = st.button("Generate Conversation + Analyze", type="primary")

        if generate_clicked:
            try:
                # Always generate dialogue via LLM
                conv_dialogue = []
                if HAS_LLM_GEN:
                    os.makedirs("audio_output", exist_ok=True)
                    base_out = os.path.join("audio_output", f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                    mp4_path = base_out + ".mp4"
                    conv = LLMToAudioConversation(output_path=mp4_path, silence_ms=silence_ms)
                    conv_dialogue = conv.generate_dialogue(severity, n_viol)
                if not conv_dialogue:
                    st.error("LLM did not return a dialogue. Try again.")
                    st.stop()

                analyzer = AudioToViolations(output_dir="violations_output") if HAS_ANALYZER else None

                if HAS_ANALYZER and FFMPEG_AVAILABLE:
                    # Audio synthesis path
                    audio_seg = conv.synthesize_and_combine(conv_dialogue)
                    # Export MP3 for playback and analysis
                    mp3_path = base_out + ".mp3"
                    audio_seg.export(mp3_path, format="mp3")
                    # Also keep MP4 if desired
                    conv.export_mp4(audio_seg)

                    st.audio(mp3_path)
                    st.info("Transcribing and analyzing generated audio…")
                    result = analyzer.process_and_analyze(audio_file=mp3_path, use_google=use_google, model=model)
                else:
                    # Text-only analysis path (no audio synthesis)
                    # Build paragraph from dialogue
                    paragraph = " \n".join([f"{spk}: {txt}" for spk, txt in conv_dialogue]).strip()
                    if HAS_ANALYZER:
                        sentence_spans = analyzer.split_sentences(paragraph)
                        analysis = analyzer.analyze_with_llm(paragraph, model)
                        result = analyzer.align_or_validate_spans(analysis, paragraph, sentence_spans)
                        # Prepare output consistent with audio flow
                        result = analyzer.build_output(
                            audio_file="LLM_generated_text_only",
                            paragraph=paragraph,
                            result=result,
                            sentence_spans=sentence_spans,
                        )
                    else:
                        # Minimal inline LLM fallback to produce JSON
                        try:
                            from openai import OpenAI
                            client = OpenAI()
                            system_prompt = (
                                "You are EchoFlag AI. Given a paragraph of dialogue (Salesperson/Client), "
                                "identify any compliance violations (mutual fund distribution). "
                                "Respond ONLY as JSON with keys: violations (array), summary (string), overall_risk (RED|ORANGE|YELLOW|NONE)."
                            )
                            user_content = f"Paragraph:\n{paragraph}\nReturn JSON."
                            resp = client.chat.completions.create(
                                model=model,
                                messages=[
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user", "content": user_content},
                                ],
                                temperature=0.2,
                            )
                            text = resp.choices[0].message.content
                            # Best-effort parse
                            import json as _json
                            result = _json.loads(text)
                        except Exception as e:
                            st.error(f"Fallback LLM analysis failed: {e}")
                            st.stop()

                st.success("Analysis complete")
                st.download_button(
                    "Download Analysis JSON",
                    data=str(result).encode("utf-8"),
                    file_name=f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                )
                st.json(result)
            except Exception as e:
                st.error(f"Generation or analysis failed: {e}")
