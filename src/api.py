#!/usr/bin/env python3
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
from typing import Optional

# Ensure local imports work when started from project root
import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
for p in (PROJECT_ROOT, SRC_DIR):
    if p not in sys.path:
        sys.path.append(p)

from audio_to_violations import AudioToViolations

app = FastAPI(title="EchoFlag API", version="0.1.0")

# Allow Streamlit (localhost) to talk to the API easily
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}


def _save_upload(temp_suffix: str, upload: UploadFile) -> str:
    suffix = temp_suffix if temp_suffix.startswith(".") else f".{temp_suffix}"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    content = upload.file.read()
    tmp.write(content)
    tmp.flush()
    tmp.close()
    return tmp.name

@app.post("/transcribe")
def transcribe(
    file: UploadFile = File(...),
    use_google: bool = Form(True),
):
    try:
        ext = os.path.splitext(file.filename or "")[1] or ".wav"
        path = _save_upload(ext, file)
        processor = AudioToViolations(output_dir="violations_output")
        out = processor.process_audio_file(path, use_google=use_google)
        paragraph = processor.format_transcript_for_analysis(out["transcript"]) if isinstance(out, dict) else ""
        return {"transcript": out.get("transcript", out), "paragraph": paragraph}
    finally:
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass

@app.post("/analyze")
def analyze(
    file: UploadFile = File(...),
    use_google: bool = Form(True),
    model: str = Form("gpt-4"),
):
    try:
        ext = os.path.splitext(file.filename or "")[1] or ".wav"
        path = _save_upload(ext, file)
        processor = AudioToViolations(output_dir="violations_output")
        result = processor.process_and_analyze(audio_file=path, use_google=use_google, model=model)
        return JSONResponse(result)
    finally:
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass

# Run with: uvicorn src.api:app --reload --port 8000
