#!/usr/bin/env python3
# Audio processing with OpenAI Whisper
from openai import OpenAI

"""
Audio to Violations Analyzer for EchoFlag

Combines audio transcription and LLM-based violation detection with span identification.
Takes an audio file, transcribes it, and analyzes for compliance violations.
"""
import os
import re
import json
import argparse
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
import tempfile

from openai import OpenAI
from pydub import AudioSegment
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class AudioToViolations:
    """Process audio files to detect compliance violations with span identification."""
    
    def __init__(self, output_dir="violations_output"):
        """
        Initialize the audio to violations analyzer.
        
        Args:
            output_dir (str): Directory to save generated analysis files
        """
        self.output_dir = output_dir
        self.transcripts_dir = os.path.join(output_dir, "transcripts")
        self.analysis_dir = os.path.join(output_dir, "analysis")
        
        # Create output directories if they don't exist
        for directory in [self.output_dir, self.transcripts_dir, self.analysis_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Custom prompts for LLM
        self.system_prompt = (
            "You are EchoFlag AI, an expert compliance analyst for financial conversations. "
            "Given a paragraph (possibly concatenated from multiple speakers), identify compliance violations. "
            "Classify each as RED (high), ORANGE (medium), or YELLOW (low). "
            "Return precise spans: sentence_index_start, sentence_index_end (0-based, inclusive), and char_start, char_end (0-based, inclusive-exclusive) relative to the provided paragraph string. "
            "If a transcript JSON with speakers is used, attribute the violation to the most likely speaker; otherwise use 'Unknown'. "
            "Respond ONLY as strict JSON with keys: violations (array), summary (string), overall_risk (RED|ORANGE|YELLOW|NONE). "
            "Each violation object MUST have: severity, speaker, text, sentence_index_start, sentence_index_end, char_start, char_end, explanation."
        )
        
        self.user_instructions = (
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
    
    def convert_to_wav(self, in_file: str) -> str:
        """
        Convert any supported audio format to WAV for SpeechRecognition.
        Uses pydub/ffmpeg to decode.

        Args:
            in_file: Path to input audio file (mp3/mp4/wav/..)

        Returns:
            Path to temporary WAV file
        """
        temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_wav_path = temp_wav.name
        temp_wav.close()

        # Let pydub figure out the format from extension/content
        audio = AudioSegment.from_file(in_file)
        audio.export(temp_wav_path, format="wav")

        return temp_wav_path
    
    def transcribe_audio(self, audio_file, use_google=True):
        """
        Transcribe audio file to text using OpenAI Whisper.
        
        Args:
            audio_file (str): Path to audio file
            use_google (bool): Ignored, kept for compatibility
            
        Returns:
            str: Transcribed text
        """
        try:
            # Open audio file and transcribe with Whisper
            with open(audio_file, "rb") as audio:
                transcript = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio
                )
            return transcript.text
        except Exception as e:
            return f"Transcription failed: {e}"
    
    def extract_speaker_from_filename(self, filename):
        """
        Extract speaker information from filename.
        
        Args:
            filename (str): Audio filename
            
        Returns:
            str: Speaker identifier or None
        """
        # Try to extract Speaker_X from filename
        basename = os.path.basename(filename)
        parts = basename.split('_')
        
        for i, part in enumerate(parts):
            if part.startswith('Speaker'):
                return f"{part}"
        
        return None
    
    def process_audio_file(self, audio_file, use_google=True):
        """
        Process audio file and save transcript.
        
        Args:
            audio_file (str): Path to audio file
            use_google (bool): Whether to use Google Web Speech API
            
        Returns:
            dict: Transcript data and file path
        """
        print(f"Processing audio file: {audio_file}")
        
        # Transcribe audio
        text = self.transcribe_audio(audio_file, use_google)
        
        # Extract speaker information from filename
        speaker = self.extract_speaker_from_filename(audio_file)
        if not speaker:
            speaker = "Unknown_Speaker"
        
        # Create transcript data
        transcript = {speaker: text}
        
        # Generate output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        basename = os.path.splitext(os.path.basename(audio_file))[0]
        output_file = os.path.join(self.transcripts_dir, f"{basename}_transcript_{timestamp}.json")
        
        # Save transcript
        with open(output_file, "w") as f:
            json.dump(transcript, f, indent=2)
        
        print(f"Transcript saved to '{output_file}'")
        
        return {"transcript": transcript, "file_path": output_file}
    
    def split_sentences(self, text):
        """
        Naive sentence splitter returning list of (start, end) char spans for each sentence.
        Sentences are split on [.!?] keeping order; trims whitespace at boundaries.
        
        Args:
            text (str): Text to split into sentences
            
        Returns:
            list: List of (start, end) character spans for each sentence
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
    
    def format_transcript_for_analysis(self, transcript):
        """
        Format transcript for LLM analysis.
        
        Args:
            transcript (dict): Transcript data with speaker labels
            
        Returns:
            str: Formatted paragraph text
        """
        # Concatenate text in deterministic order
        parts = []
        for speaker in sorted(transcript.keys()):
            parts.append(f"{speaker}: {transcript[speaker].strip()}")
        paragraph = " \n".join(parts).strip()
        return paragraph
    
    def analyze_with_llm(self, paragraph, model="gpt-4"):
        """
        Analyze paragraph for violations using LLM.
        
        Args:
            paragraph (str): Paragraph text to analyze
            model (str): LLM model to use
            
        Returns:
            dict: Analysis results
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_instructions + "\nParagraph:\n" + paragraph},
        ]
        
        try:
            # Call OpenAI API with updated client
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.1,
                max_tokens=2000
            )
            
            # Extract response content
            content = response.choices[0].message.content
            
            # Try to parse JSON from response
            try:
                if "```" in content:
                    # Try fenced JSON first
                    match = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", content)
                    if match:
                        return json.loads(match.group(1))
                return json.loads(content)
            except json.JSONDecodeError:
                print("Warning: Could not parse LLM response as JSON. Returning raw text.")
                return {
                    "violations": [],
                    "summary": "LLM returned non-JSON response.",
                    "overall_risk": "NONE",
                    "raw_response": content
                }
            
        except Exception as e:
            print(f"Error calling OpenAI API: {str(e)}")
            return {"error": str(e)}
    
    def align_or_validate_spans(self, result, paragraph, sentence_spans):
        """
        Validate and, if needed, adjust LLM-provided spans to the paragraph.
        
        Args:
            result (dict): LLM analysis results
            paragraph (str): Original paragraph text
            sentence_spans (list): List of sentence spans
            
        Returns:
            dict: Validated analysis results
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
            
        return result
    
    def build_output(self, audio_file, paragraph, result, sentence_spans):
        """
        Build final output with all required information.
        
        Args:
            audio_file (str): Path to audio file
            paragraph (str): Paragraph text
            result (dict): LLM analysis results
            sentence_spans (list): List of sentence spans
            
        Returns:
            dict: Complete output data
        """
        out = {
            "audio_file_name": os.path.basename(audio_file),
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
    
    def process_and_analyze(self, audio_file, model="gpt-4"):
        """
        Process audio file and analyze for violations using OpenAI Whisper + GPT.
        
        Args:
            audio_file (str): Path to audio file
            model (str): LLM model to use
            
        Returns:
            dict: Complete analysis results
        """
        print(f"Processing audio file: {audio_file}")
        
        # Transcribe audio
        text = self.transcribe_audio(audio_file)
        transcript = {"transcript": text}
        
        # Step 2: Format transcript for analysis
        paragraph = self.format_transcript_for_analysis(transcript)
        
        # Step 3: Split sentences
        sentence_spans = self.split_sentences(paragraph)
        
        # Step 4: Analyze with LLM
        analysis = self.analyze_with_llm(paragraph, model)
        
        # Step 5: Validate spans
        validated_analysis = self.align_or_validate_spans(analysis, paragraph, sentence_spans)
        
        # Step 6: Build output
        output = self.build_output(audio_file, paragraph, validated_analysis, sentence_spans)
        
        # Step 7: Save analysis
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        basename = os.path.splitext(os.path.basename(audio_file))[0]
        output_file = os.path.join(self.analysis_dir, f"{basename}_analysis_{timestamp}.json")
        
        with open(output_file, "w") as f:
            json.dump(output, f, indent=2)
        
        print(f"Analysis saved to '{output_file}'")
        
        return output

def main():
    """Main function to process audio and analyze for violations."""
    parser = argparse.ArgumentParser(description="Process audio and analyze for violations")
    parser.add_argument("--audio_file", help="Path to audio file")
    parser.add_argument("--output-dir", default="violations_output", help="Directory to save output files")
    parser.add_argument("--use-google", action="store_true", default=True, help="Use Google Web Speech API (requires internet)")
    parser.add_argument("--model", default="gpt-4", help="LLM model to use")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.audio_file):
        print(f"Error: File '{args.audio_file}' not found.")
        return 1
    
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please set it in your .env file or environment.")
        return 1
    
    # Create processor
    processor = AudioToViolations(args.output_dir)
    
    # Process and analyze
    output = processor.process_and_analyze(args.audio_file, args.use_google, args.model)
    
    # Print summary
    print("\nAnalysis Summary:")
    print("-" * 80)
    
    if "error" in output:
        print(f"Error: {output['error']}")
        return 1
    
    print(f"Audio file: {output['audio_file_name']}")
    print(f"Overall risk: {output['overall_risk']}")
    print(f"Summary: {output['summary']}")
    
    if output["violations"]:
        print(f"\nFound {len(output['violations'])} violations:")
        
        # Group by severity
        by_severity = {"RED": [], "ORANGE": [], "YELLOW": []}
        for violation in output["violations"]:
            severity = violation.get("severity", "UNKNOWN")
            if severity in by_severity:
                by_severity[severity].append(violation)
        
        # Print by severity
        for severity in ["RED", "ORANGE", "YELLOW"]:
            if by_severity[severity]:
                print(f"\n{severity} Violations ({len(by_severity[severity])}):")
                for i, v in enumerate(by_severity[severity], 1):
                    print(f"  {i}. {v['speaker']}: \"{v['text']}\"")
                    print(f"     Sentence: {v['sentence_index_start']}-{v['sentence_index_end']}, " +
                          f"Chars: {v['char_start']}-{v['char_end']}")
                    print(f"     Explanation: {v['explanation']}")
    else:
        print("\nNo violations found.")
    
    print("-" * 80)
    print("\nProcessing complete.")
    
    return 0

if __name__ == "__main__":
    exit(main())
