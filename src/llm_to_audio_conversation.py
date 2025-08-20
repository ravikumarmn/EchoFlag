#!/usr/bin/env python3

"""
Generate a Salesperson vs Client conversation with explicit mutual fund distribution violations
using an LLM, convert to audio per turn, and export a single combined .mp4 (audio-only).

Requirements:
- OPENAI_API_KEY in environment (.env supported)
- gTTS, pydub, ffmpeg installed and available in PATH (pydub uses ffmpeg)

Usage:
  python src/llm_to_audio_conversation.py \
    --severity ALL \
    --violations 5 \
    --output audio_output/mf_violations_conversation.mp4

This script aligns with EchoFlag's testing workflow by producing one audio file per conversation.
"""
import os
import re
import json
import argparse
import tempfile
from datetime import datetime
from typing import List, Tuple

from dotenv import load_dotenv
from openai import OpenAI
from gtts import gTTS
from pydub import AudioSegment

load_dotenv(dotenv_path=".env")

SALES_SPK = "Salesperson"
CLIENT_SPK = "Client"

SYSTEM_PROMPT = (
    "You are an expert in financial regulations creating realistic examples of non-compliant "
    "mutual fund sales conversations for internal testing."
)

PROMPT_TEMPLATE = (
    "Generate a realistic dialogue between a Salesperson and a Client about mutual funds.\n"
    "Include violations related to mutual fund distribution. Keep the conversation concise (6-10 turns total).\n"
    "Violations to include (paraphrase naturally, do NOT label them):\n{violations}\n\n"
    "Rules:\n"
    "- Alternate speakers strictly as 'Salesperson:' and 'Client:'.\n"
    "- Focus violations in Salesperson turns.\n"
    "- No meta commentary, no disclaimers, only the dialogue lines.\n"
)

VIOLATION_CATALOG = {
    "RED": [
        "Guaranteeing specific returns or profits",
        "Asserting there is no chance of loss",
        "Threatening legal action or implying unlawful practices",
        "Claims of regulatory endorsement when not applicable",
    ],
    "ORANGE": [
        "Marketing as risk-free / no risk",
        "False urgency (today only / must act now)",
        "Promises to double or triple money",
        "Overstating performance without basis",
    ],
    "YELLOW": [
        "Superlatives like 'best', 'perfect' investment",
        "Vague high-return claims",
        "'Exclusive offer' with no substantiation",
        "Unverifiable comparisons to other funds",
    ],
}

class LLMToAudioConversation:
    def __init__(self, output_path: str, silence_ms: int = 700):
        self.output_path = output_path
        self.silence_ms = silence_ms
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        out_dir = os.path.dirname(output_path) or "."
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

    def _build_prompt(self, severity: str, violations: int) -> str:
        import random
        pool: List[Tuple[str, str]] = []
        if severity == "ALL":
            for sev, lst in VIOLATION_CATALOG.items():
                pool.extend((sev, v) for v in lst)
        else:
            pool.extend((severity, v) for v in VIOLATION_CATALOG.get(severity, []))
        chosen = random.sample(pool, min(max(1, violations), len(pool)))
        violations_text = "\n".join([f"- {sev}: {v}" for sev, v in chosen])
        return PROMPT_TEMPLATE.format(violations=violations_text)

    def generate_dialogue(self, severity: str, violations: int) -> List[Tuple[str, str]]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": self._build_prompt(severity, violations)},
        ]
        resp = self.client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.7,
            max_tokens=900,
        )
        content = resp.choices[0].message.content.strip()
        lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
        dialogue: List[Tuple[str, str]] = []
        for ln in lines:
            if ln.startswith(f"{SALES_SPK}:"):
                dialogue.append((SALES_SPK, ln.split(":", 1)[1].strip()))
            elif ln.startswith(f"{CLIENT_SPK}:"):
                dialogue.append((CLIENT_SPK, ln.split(":", 1)[1].strip()))
        # Basic validation: enforce alternating speakers if possible
        normalized: List[Tuple[str, str]] = []
        last = None
        for spk, txt in dialogue:
            if spk == last:
                # flip if two in a row; default pattern Salesperson -> Client
                spk = CLIENT_SPK if spk == SALES_SPK else SALES_SPK
            normalized.append((spk, txt))
            last = spk
        return normalized

    def _tts_for_turn(self, speaker: str, text: str) -> AudioSegment:
        # gTTS single voice; vary accent via tld for slight differentiation
        tld = "com" if speaker == SALES_SPK else "co.uk"
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            tts = gTTS(text=text, lang="en", tld=tld, slow=False, lang_check=False)
            tts.save(tmp_path)
            seg = AudioSegment.from_file(tmp_path)
            return seg
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    def synthesize_and_combine(self, dialogue: List[Tuple[str, str]]) -> AudioSegment:
        combined = AudioSegment.silent(duration=500)
        gap = AudioSegment.silent(duration=self.silence_ms)
        for spk, text in dialogue:
            seg = self._tts_for_turn(spk, text)
            combined += seg + gap
        return combined

    def export_mp4(self, audio: AudioSegment) -> str:
        # pydub will use ffmpeg to export to mp4/m4a (AAC). Ensure ffmpeg is installed.
        target = self.output_path
        # Ensure extension
        if not target.lower().endswith(".mp4"):
            target += ".mp4"
        audio.export(target, format="mp4")
        return target


def main():
    parser = argparse.ArgumentParser(description="LLM->Audio: Salesperson vs Client violations conversation to a single MP4")
    parser.add_argument("--severity", choices=["RED", "ORANGE", "YELLOW", "ALL"], default="ALL")
    parser.add_argument("--violations", type=int, default=5)
    parser.add_argument("--output", default=os.path.join("audio_output", f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"))
    parser.add_argument("--silence-ms", type=int, default=700)
    args = parser.parse_args()

    conv = LLMToAudioConversation(output_path=args.output, silence_ms=args.silence_ms)

    dialogue = conv.generate_dialogue(args.severity, args.violations)
    if not dialogue:
        print("No dialogue generated.")
        return 1

    # Save dialogue as JSON for audit
    transcript_path = os.path.splitext(args.output)[0] + "_transcript.json"
    os.makedirs(os.path.dirname(transcript_path) or ".", exist_ok=True)
    with open(transcript_path, "w") as f:
        json.dump([{"speaker": spk, "text": txt} for spk, txt in dialogue], f, indent=2)
    print(f"Transcript saved to '{transcript_path}'")

    audio = conv.synthesize_and_combine(dialogue)
    out_path = conv.export_mp4(audio)
    print(f"Combined MP4 saved to '{out_path}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
