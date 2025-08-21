#!/usr/bin/env python3
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
# Must be the first Streamlit command on the page
st.set_page_config(page_title="EchoFlag - Audio Violations", layout="centered")


import os
import tempfile
import json
from datetime import datetime

st.title("EchoFlag – Audio Violation Tester")
st.caption("Upload audio and run local transcription + LLM analysis. No HTTP calls.")

with st.sidebar:
    st.header("Settings")
    model = "gpt-4o"
    
    # Speaker diarization toggle
    use_google_speech = st.checkbox(
        "Enable Speaker Diarization", 
        value=True,  # Default to accurate Google Speech
        help="Use Google Cloud Speech-to-Text for speaker identification (more accurate transcription)"
    )
    
    if use_google_speech:
        st.info("🎙️ **Transcription**: Google Cloud Speech (with speakers)\n📊 **Analysis**: OpenAI GPT-4")
    else:
        st.info("🎙️ **Transcription**: OpenAI Whisper\n📊 **Analysis**: OpenAI GPT-4")

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
    # Import from src package first (most reliable)
    from src.audio_to_violations import AudioToViolations
except Exception:
    try:
        # Fallback for when running from src directory
        from audio_to_violations import AudioToViolations
    except Exception as e:
        st.error(f"Failed to load analyzer: {e}")
        AudioToViolations = None

st.subheader("1) Upload an audio file")
uploaded = st.file_uploader(
    "Choose an audio file (mp3/mp4/wav)", type=["mp3", "mp4", "wav"]
)

col1, col2 = st.columns(2)
# with col1:
    # transcribe_clicked = st.button(
    #     "Transcribe Only", disabled=uploaded is None or AudioToViolations is None
    # )
with col1:
    analyze_clicked = st.button(
        "Analyze Violations",
        type="primary",
        disabled=uploaded is None or AudioToViolations is None,
    )

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
    return (
        AudioToViolations(output_dir="violations_output") if AudioToViolations else None
    )


# if transcribe_clicked and uploaded is not None and AudioToViolations is not None:
#     temp_path = None
#     try:
#         temp_path = _save_to_temp(uploaded)
#         st.info("Transcribing with OpenAI Whisper…")
#         processor = _make_processor()
#         res = processor.process_audio_file(temp_path)
#         st.success("Transcription complete")
#         st.json(res)
#     except Exception as e:
#         st.error(f"Transcription failed: {e}")
#     finally:
#         if temp_path and os.path.exists(temp_path):
#             try:
#                 os.remove(temp_path)
#             except Exception:
#                 pass

if analyze_clicked and uploaded is not None and AudioToViolations is not None:
    temp_path = None
    try:
        temp_path = _save_to_temp(uploaded)
        if use_google_speech:
            st.info("Analyzing with Google Cloud Speech + GPT-4…")
        else:
            st.info("Analyzing with OpenAI Whisper + GPT-4…")
        
        # Add progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        processor = _make_processor()
        
        # Update progress
        progress_bar.progress(20)
        status_text.text("Transcribing audio...")
        
        try:
            result = processor.process_and_analyze(temp_path, use_google=use_google_speech, model=model)
            progress_bar.progress(100)
            status_text.text("Analysis complete!")
            st.success("Analysis complete")
            st.download_button(
                "Download Analysis JSON",
                data=json.dumps(result, indent=2).encode("utf-8"),
                file_name=f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
            )
            
            # Display usage information if available
            if "usage" in result:
                with st.expander("📊 Token Usage and Cost"):
                    usage = result["usage"]
                    
                    # Create columns for transcription and analysis
                    col1, col2 = st.columns(2)
                    
                    # Transcription usage
                    with col1:
                        st.subheader("Transcription")
                        if usage["transcription"]["openai_whisper"]["audio_seconds"] > 0:
                            st.write(f"**OpenAI Whisper**")
                            st.write(f"Audio duration: {usage['transcription']['openai_whisper']['audio_seconds']:.2f} seconds")
                            if 'rate_per_minute' in usage['transcription']['openai_whisper']:
                                st.write(f"Rate: ${usage['transcription']['openai_whisper']['rate_per_minute']}/min")
                            st.write(f"Cost: ${usage['transcription']['openai_whisper']['estimated_cost']:.4f}")
                        
                        if usage["transcription"]["google_speech"]["audio_seconds"] > 0:
                            st.write(f"**Google Speech-to-Text (estimate)**")
                            st.write(f"Audio duration: {usage['transcription']['google_speech']['audio_seconds']:.2f} seconds")
                            if 'rate_per_minute' in usage['transcription']['google_speech']:
                                st.write(f"Rate: ${usage['transcription']['google_speech']['rate_per_minute']}/min")
                            if 'tier' in usage['transcription']['google_speech']:
                                st.write(f"Tier: {usage['transcription']['google_speech']['tier']}")
                            if 'note' in usage['transcription']['google_speech']:
                                st.caption(usage['transcription']['google_speech']['note'])
                            st.write(f"Cost: ${usage['transcription']['google_speech']['estimated_cost']:.4f}")
                    
                    # Analysis usage
                    with col2:
                        st.subheader("Analysis")
                        st.write(f"**Model**: {usage['analysis']['model']}")
                        st.write(f"Input tokens: {usage['analysis']['prompt_tokens']}")
                        if 'cached_tokens' in usage['analysis']:
                            st.write(f"Cached input tokens: {usage['analysis']['cached_tokens']}")
                        st.write(f"Output tokens: {usage['analysis']['completion_tokens']}")
                        st.write(f"Total tokens: {usage['analysis']['total_tokens']}")
                        if 'pricing' in usage['analysis']:
                            p = usage['analysis']['pricing']
                            st.write("Pricing (per 1M tokens):")
                            st.write(f"- Input: ${p.get('input_per_million', 0)}")
                            st.write(f"- Cached input: ${p.get('cached_input_per_million', p.get('input_per_million', 0))}")
                            st.write(f"- Output: ${p.get('output_per_million', 0)}")
                            if 'tier' in p:
                                st.write(f"Tier: {p['tier']}")
                        st.write(f"Cost: ${usage['analysis']['estimated_cost']:.4f}")
                    
                    # Total cost
                    st.subheader("Total Cost")
                    st.write(f"**Total estimated cost**: ${usage['total_estimated_cost']:.4f}")
            
            # Display the full result
            st.json(result)
        except Exception as analysis_error:
            st.error(f"Analysis failed: {analysis_error}")
            import traceback
            with st.expander("Debug Details"):
                st.text(traceback.format_exc())
            
    except Exception as e:
        st.error(f"Analysis failed: {e}")
        import traceback
        st.text(traceback.format_exc())
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
