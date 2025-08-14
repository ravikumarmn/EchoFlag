#!/usr/bin/env python3
"""
LLM Span-based Violation Flagger for EchoFlag

Analyzes a paragraph or transcript JSON using an LLM and returns violations with:
- severity (RED/ORANGE/YELLOW)
- speaker (if provided via transcript JSON)
- sentence_index_start, sentence_index_end (0-based indices within the paragraph)
- char_start, char_end (0-based character indices within the full paragraph)
- exact text and brief explanation

Usage examples:
  python src/llm_span_flagger.py --input-text path/to/text.txt
  python src/llm_span_flagger.py --input-transcript path/to/transcript.json

Output: JSON saved under output_dir with detected violations and spans.
"""
import os
import re
import json
import argparse
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

import openai
from dotenv import load_dotenv

# Load env and set API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

CUSTOM_SYSTEM_PROMPT = (
    "You are EchoFlag AI, an expert compliance analyst for financial conversations. "
    "Given a paragraph (possibly concatenated from multiple speakers), identify compliance violations. "
    "Classify each as RED (high), ORANGE (medium), or YELLOW (low). "
    "Return precise spans: sentence_index_start, sentence_index_end (0-based, inclusive), and char_start, char_end (0-based, inclusive-exclusive) relative to the provided paragraph string. "
    "If a transcript JSON with speakers is used, attribute the violation to the most likely speaker; otherwise use 'Unknown'. "
    "Respond ONLY as strict JSON with keys: violations (array), summary (string), overall_risk (RED|ORANGE|YELLOW|NONE). "
    "Each violation object MUST have: severity, speaker, text, sentence_index_start, sentence_index_end, char_start, char_end, explanation."
)

USER_INSTRUCTIONS = (
    "Tasks:\n"
    "1) Read the paragraph.\n"
    "2) Split into sentences using normal English punctuation (. ? !)\n"
    "3) Detect all possible violations with categories:\n"
    "   - RED: illegal/fraud/scam/guarantees of returns, threats of legal action, criminal suggestions\n"
    "   - ORANGE: risk-free/no risk/false urgency/double your money\n"
    "   - YELLOW: best/perfect/high return/exclusive offer/etc.\n"
    "4) For each violation, provide the EXACT span indices in the full paragraph string: char_start and char_end (end exclusive).\n"
    "5) Also provide sentence_index_start and sentence_index_end that bound the violation.\n"
    "6) Output strict JSON only."
)


def read_paragraph_from_transcript(transcript_path: str) -> Tuple[str, Dict[str, str]]:
    """Load a transcript JSON {Speaker_X: text} and concatenate into a single paragraph.

    Returns:
        paragraph: concatenated text in deterministic order of keys sorted by name
        speakers_map: original dict for optional speaker attribution assistance
    """
    with open(transcript_path, "r") as f:
        data = json.load(f)
    # Deterministic order: sort keys
    parts = []
    for spk in sorted(data.keys()):
        parts.append(f"{spk}: {data[spk].strip()}")
    paragraph = " \n".join(parts).strip()
    return paragraph, data


def read_paragraph_from_text(path: str) -> str:
    with open(path, "r") as f:
        return f.read().strip()


def split_sentences(text: str) -> List[Tuple[int, int]]:
    """Naive sentence splitter returning list of (start, end) char spans for each sentence.
    Sentences are split on [.!?] keeping order; trims whitespace at boundaries.
    """
    spans = []
    start = 0
    i = 0
    while i < len(text):
        if text[i] in ".!?":
            end = i + 1
            # Extend to include trailing spaces
            while end < len(text) and text[end].isspace():
                end += 1
            # Trim leading/trailing whitespace for the span
            s = start
            e = end
            while s < e and text[s].isspace():
                s += 1
            while e > s and text[e - 1].isspace():
                e -= 1
            if s < e:
                spans.append((s, e))
            start = end
        i += 1
    # Last tail if any
    if start < len(text):
        s = start
        e = len(text)
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        if s < e:
            spans.append((s, e))
    return spans


def call_openai_span_flagger(paragraph: str, model: str = "gpt-4") -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": CUSTOM_SYSTEM_PROMPT},
        {"role": "user", "content": USER_INSTRUCTIONS + "\nParagraph:\n" + paragraph},
    ]
    resp = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0.1,
        max_tokens=2000,
    )
    content = resp.choices[0].message.content
    # Attempt to extract JSON
    try:
        if "```" in content:
            # try fenced JSON first
            m = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", content)
            if m:
                return json.loads(m.group(1))
        return json.loads(content)
    except Exception:
        return {"violations": [], "summary": "LLM returned non-JSON response.", "overall_risk": "NONE", "raw": content}


def align_or_validate_spans(result: Dict[str, Any], paragraph: str, sentence_spans: List[Tuple[int, int]]):
    """Validate and, if needed, adjust LLM-provided spans to the paragraph.
    - Ensures indices are within bounds
    - Optionally re-derives char spans by searching for the text if mismatch
    """
    n = len(paragraph)
    for v in result.get("violations", []):
        # Clamp sentence indices
        sis = max(0, min(len(sentence_spans) - 1, int(v.get("sentence_index_start", 0)))) if sentence_spans else 0
        sie = max(0, min(len(sentence_spans) - 1, int(v.get("sentence_index_end", sis)))) if sentence_spans else 0
        if sie < sis:
            sie = sis
        v["sentence_index_start"] = sis
        v["sentence_index_end"] = sie

        # Clamp char indices
        cs = int(v.get("char_start", 0))
        ce = int(v.get("char_end", cs))
        cs = max(0, min(n, cs))
        ce = max(0, min(n, ce))
        if ce < cs:
            ce = cs

        # If text not matching, attempt to realign using first occurrence
        text = str(v.get("text", "")).strip()
        if text:
            snippet = paragraph[cs:ce]
            if text.lower() not in snippet.lower():
                # Try to locate text in paragraph
                loc = paragraph.lower().find(text.lower())
                if loc != -1:
                    cs, ce = loc, loc + len(text)
        v["char_start"], v["char_end"] = cs, ce

        # Default speaker
        if not v.get("speaker"):
            v["speaker"] = "Unknown"

        # Ensure required keys exist
        v.setdefault("explanation", "")
        v.setdefault("severity", "YELLOW")


def build_output(paragraph: str, result: Dict[str, Any], sentence_spans: List[Tuple[int, int]]) -> Dict[str, Any]:
    out = {
        "paragraph": paragraph,
        "sentences": [
            {"index": i, "char_start": s, "char_end": e, "text": paragraph[s:e]}
            for i, (s, e) in enumerate(sentence_spans)
        ],
        "violations": result.get("violations", []),
        "summary": result.get("summary", ""),
        "overall_risk": result.get("overall_risk", "NONE"),
    }
    return out


def main():
    parser = argparse.ArgumentParser(description="LLM span-based violation flagger")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--input-text", help="Path to a raw text file (paragraph)")
    src.add_argument("--input-transcript", help="Path to transcript JSON {Speaker_X: text}")
    parser.add_argument("--model", default="gpt-4", help="OpenAI model (default: gpt-4)")
    parser.add_argument("--output-dir", default="llm_span_output", help="Directory to save results")

    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set in environment/.env")
        return 1

    os.makedirs(args.output_dir, exist_ok=True)

    # Read input into a paragraph string
    speakers_map = None
    if args.input_transcript:
        paragraph, speakers_map = read_paragraph_from_transcript(args.input_transcript)
        input_base = os.path.splitext(os.path.basename(args.input_transcript))[0]
    else:
        paragraph = read_paragraph_from_text(args.input_text)
        input_base = os.path.splitext(os.path.basename(args.input_text))[0]

    # Precompute sentence spans for later validation/reporting
    sentence_spans = split_sentences(paragraph)

    # Call LLM
    result = call_openai_span_flagger(paragraph, model=args.model)

    # Validate/align spans
    align_or_validate_spans(result, paragraph, sentence_spans)

    # Build final output
    output = build_output(paragraph, result, sentence_spans)

    # Save
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.output_dir, f"{input_base}_llm_spans_{ts}.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved analysis to: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
