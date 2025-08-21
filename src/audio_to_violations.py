#!/usr/bin/env python3
# Audio processing with Google Cloud Speech-to-Text and OpenAI GPT
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
import time

from openai import OpenAI
try:
    import speech_recognition as sr
    GOOGLE_SPEECH_AVAILABLE = True
except ImportError:
    GOOGLE_SPEECH_AVAILABLE = False
    print("SpeechRecognition library not available - using OpenAI Whisper only")

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    print("pydub not available - audio conversion disabled")

from dotenv import load_dotenv

try:
    from src.usage_tracker import UsageTracker
except ImportError:
    try:
        from usage_tracker import UsageTracker
    except ImportError:
        print("UsageTracker not available - usage tracking disabled")
        UsageTracker = None

# Load environment variables
load_dotenv()

# Configure OpenAI client (removed global client to avoid conflicts)

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
        
        # Initialize Google Web Speech API client (SpeechRecognition library)
        self.recognizer = None
        self.google_available = False
        if GOOGLE_SPEECH_AVAILABLE:
            try:
                self.recognizer = sr.Recognizer()
                self.google_available = True
                print("Google Web Speech API initialized successfully")
            except Exception as e:
                print(f"Warning: Google Web Speech API not available: {e}")
                print("Falling back to OpenAI Whisper for transcription")
        else:
            print("SpeechRecognition library not installed - using OpenAI Whisper only")
            
        # Initialize usage tracker
        self.usage_tracker = UsageTracker() if UsageTracker else None
        
        # Custom prompts for LLM
        self.system_prompt = (
            "You are EchoFlag AI, an expert compliance analyst for financial conversations. "
            "Given a paragraph with speaker labels (e.g., 'Speaker_1: text', 'Speaker_2: text'), identify compliance violations. "
            "Classify each as RED (high), ORANGE (medium), or YELLOW (low). "
            "Return precise spans: sentence_index_start, sentence_index_end (0-based, inclusive), and char_start, char_end (0-based, inclusive-exclusive) relative to the provided paragraph string. "
            "IMPORTANT: For each violation, identify which speaker said it by looking at the speaker labels in the text (Speaker_1, Speaker_2, etc.). "
            "If no clear speaker can be identified, use 'Unknown'. "
            "Respond ONLY as strict JSON with keys: violations (array), summary (string), overall_risk (RED|ORANGE|YELLOW|NONE). "
            "Each violation object MUST have: severity, speaker, text, sentence_index_start, sentence_index_end, char_start, char_end, explanation."
        )
        
        self.user_instructions = (
            "Tasks:\n"
            "1) Read the paragraph with speaker labels (e.g., 'Speaker_1: guaranteed returns').\n"
            "2) Detect all possible violations with categories:\n"
            "   - RED: illegal/fraud/scam/guarantees of returns, threats of legal action, criminal suggestions\n"
            "   - ORANGE: risk-free/no risk/false urgency/double your money\n"
            "   - YELLOW: best/perfect/high return/exclusive offer/etc.\n"
            "3) For each violation, identify the speaker who said it (Speaker_1, Speaker_2, etc.).\n"
            "4) Provide the EXACT span indices in the full paragraph string: char_start and char_end (end exclusive).\n"
            "5) Also provide sentence_index_start and sentence_index_end that bound the violation.\n"
            "6) Output strict JSON only."
        )
    
    def convert_to_wav(self, in_file: str) -> str:
        """
        Convert any supported audio format to WAV for Google Speech API.
        Uses pydub/ffmpeg to decode with specific requirements for diarization.

        Args:
            in_file: Path to input audio file (mp3/mp4/wav/..)

        Returns:
            Path to temporary WAV file (or original file if conversion fails)
        """
        if not PYDUB_AVAILABLE:
            print("Warning: pydub not available, using original file")
            return in_file
            
        try:
            temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            temp_wav_path = temp_wav.name
            temp_wav.close()

            # Load audio and convert to mono 16kHz for optimal diarization
            audio = AudioSegment.from_file(in_file)
            
            # Convert to mono (required for speaker diarization)
            audio = audio.set_channels(1)
            
            # Set sample rate to 16kHz (optimal for Google Speech)
            audio = audio.set_frame_rate(16000)
            
            # Export as WAV
            audio.export(temp_wav_path, format="wav")
            
            return temp_wav_path
        except Exception as e:
            print(f"Warning: Audio conversion failed: {e}")
            print("Using original file")
            return in_file
    
    def transcribe_audio(self, audio_file, use_google=True):
        """
        Transcribe audio file with speaker diarization.
        
        Args:
            audio_file (str): Path to audio file
            use_google (bool): If True, use Google Cloud Speech with diarization
            
        Returns:
            dict or str: If use_google=True, returns dict with speaker info.
                        If use_google=False, returns plain text from Whisper.
        """
        if use_google and self.google_available:
            return self._transcribe_with_google_diarization(audio_file)
        else:
            if use_google and not self.google_available:
                print("Google Cloud Speech not available, falling back to Whisper")
            return self._transcribe_with_whisper(audio_file)
    
    def _transcribe_with_whisper(self, audio_file):
        """
        Transcribe audio file using OpenAI Whisper with basic speaker detection.
        
        Args:
            audio_file (str): Path to audio file
            
        Returns:
            dict: Transcribed text with basic speaker separation
        """
        try:
            # Track audio duration for usage calculation
            if self.usage_tracker:
                audio_duration = self.usage_tracker.get_audio_duration(audio_file)
                self.usage_tracker.track_whisper_usage(audio_duration)
            
            start_time = time.time()
            with open(audio_file, "rb") as audio:
                transcript = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio
                )
            end_time = time.time()
            
            # Basic speaker separation based on sentence patterns and pauses
            text = transcript.text
            speakers_dict = self._detect_speakers_from_text(text)
            
            return {
                "transcript": speakers_dict,
                "full_text": text,
                "method": "whisper_with_basic_detection"
            }
            
        except Exception as e:
            return f"Whisper transcription failed: {e}"
    
    def _transcribe_with_google_diarization(self, audio_file):
        """
        Transcribe audio file using Google Web Speech API.
        Note: This public API doesn't support speaker diarization,
        so we'll use basic speaker detection heuristics.
        
        Args:
            audio_file (str): Path to audio file
            
        Returns:
            dict: Transcription with basic speaker separation
        """
        try:
            # Convert audio to WAV format for better compatibility
            wav_file = self.convert_to_wav(audio_file)
            
            # Use Google Web Speech API
            return self._transcribe_with_google_web_api(wav_file)
            
        except Exception as e:
            return {"error": f"Google Web Speech transcription failed: {e}"}
    
    def _transcribe_with_google_web_api(self, wav_file):
        """Transcribe audio using Google Web Speech API via SpeechRecognition library"""
        try:
            # Track audio duration for usage calculation
            if self.usage_tracker:
                audio_duration = self.usage_tracker.get_audio_duration(wav_file)
                self.usage_tracker.track_google_speech_usage(audio_duration)
            
            # Load audio file
            with sr.AudioFile(wav_file) as source:
                # Adjust for ambient noise and record audio
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio_data = self.recognizer.record(source)
            
            print("Transcribing with Google Web Speech API...")
            
            start_time = time.time()
            # Use Google Web Speech API for transcription
            text = self.recognizer.recognize_google(audio_data, language="en-US")
            end_time = time.time()
            
            print(f"Transcription completed: {text[:100]}...")
            
            # Apply basic speaker detection since Web Speech API doesn't support diarization
            speakers_dict = self._detect_speakers_from_text(text)
            
            # Clean up temporary WAV file
            if wav_file != os.path.basename(wav_file) and os.path.exists(wav_file):
                os.unlink(wav_file)
            
            return {
                "transcript": speakers_dict,
                "full_text": text,
                "method": "google_web_api_with_detection"
            }
            
        except sr.UnknownValueError:
            return {"error": "Google Web Speech API could not understand the audio"}
        except sr.RequestError as e:
            return {"error": f"Google Web Speech API request failed: {e}"}
        except Exception as e:
            # Clean up on error
            if wav_file != os.path.basename(wav_file) and os.path.exists(wav_file):
                os.unlink(wav_file)
            return {"error": f"Google Web Speech transcription failed: {e}"}
    
    
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
            # Call OpenAI API with instance client with timeout
            print(f"Calling OpenAI API with model: {model}")
            print(f"Message length: {len(str(messages))}")
            
            import time
            start_time = time.time()
            
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.1,
                max_tokens=2000
            )
            
            end_time = time.time()
            print(f"OpenAI API call took {end_time - start_time:.2f} seconds")
            
            # Track token usage
            if self.usage_tracker:
                self.usage_tracker.track_openai_chat_usage(response, model)
            
            # Extract response content
            content = response.choices[0].message.content
            print(f"OpenAI response: {content}")
            
            # Try to parse JSON from response
            try:
                if "```" in content:
                    # Try fenced JSON first
                    match = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", content)
                    if match:
                        result = json.loads(match.group(1))
                        print(f"Parsed JSON from fenced block: {result}")
                        return result
                result = json.loads(content)
                print(f"Parsed JSON directly: {result}")
                return result
            except json.JSONDecodeError as je:
                print(f"JSON decode error: {je}")
                print(f"Raw content: {content}")
                return {
                    "violations": [],
                    "summary": "LLM returned non-JSON response.",
                    "overall_risk": "NONE",
                    "raw_response": content
                }
            
        except Exception as e:
            print(f"Error calling OpenAI API: {str(e)}")
            return {"error": str(e)}
    
    def align_or_validate_spans(self, result, paragraph, sentence_spans, transcript=None):
        """
        Validate and, if needed, adjust LLM-provided spans to the paragraph.
        
        Args:
            result (dict): LLM analysis results
            paragraph (str): Original paragraph text
            sentence_spans (list): List of sentence spans
            transcript (dict): Original transcript with speaker info
            
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

            # Improve speaker identification
            speaker = v.get("speaker", "")
            if not speaker or speaker == "Unknown":
                # Try to identify speaker from the violation text location in paragraph
                violation_text = paragraph[cs:ce]
                speaker = self._identify_speaker_from_context(paragraph, cs, transcript)
                v["speaker"] = speaker

            # Ensure required keys exist
            v.setdefault("explanation", "")
            v.setdefault("severity", "YELLOW")
            
        return result
    
    def _detect_speakers_from_text(self, text):
        """
        Basic speaker detection from transcribed text using patterns and heuristics.
        
        Args:
            text (str): Transcribed text
            
        Returns:
            dict: Speaker-separated text
        """
        # Split text into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) <= 1:
            return {"Speaker_1": text}
        
        # Simple heuristic: alternate speakers for different sentences
        # This is a basic approach - in reality, you'd use more sophisticated methods
        speakers = {}
        current_speaker = 1
        speaker_segments = []
        
        # Group sentences by potential speaker changes
        # Look for patterns that might indicate speaker changes
        for i, sentence in enumerate(sentences):
            speaker_key = f"Speaker_{current_speaker}"
            
            # Simple heuristic: change speaker every few sentences or on certain patterns
            if i > 0 and (
                i % 2 == 0 or  # Change every 2 sentences
                any(word in sentence.lower() for word in ['yes', 'no', 'okay', 'right', 'sure', 'well']) or
                sentence.startswith(('I ', 'You ', 'We ', 'They '))
            ):
                # Potentially switch speaker
                if current_speaker == 1:
                    current_speaker = 2
                else:
                    current_speaker = 1
                speaker_key = f"Speaker_{current_speaker}"
            
            if speaker_key not in speakers:
                speakers[speaker_key] = []
            speakers[speaker_key].append(sentence)
        
        # Combine sentences for each speaker
        final_speakers = {}
        for speaker, sentences_list in speakers.items():
            final_speakers[speaker] = '. '.join(sentences_list) + '.'
        
        # Ensure we have at least Speaker_1
        if not final_speakers:
            final_speakers["Speaker_1"] = text
        
        return final_speakers
    
    def _identify_speaker_from_context(self, paragraph, char_position, transcript=None):
        """
        Identify speaker based on character position in formatted paragraph.
        
        Args:
            paragraph (str): Formatted paragraph with speaker labels
            char_position (int): Character position of violation
            transcript (dict): Original transcript with speaker info
            
        Returns:
            str: Speaker identifier
        """
        # Find the speaker label that precedes this character position
        text_before = paragraph[:char_position]
        
        # Look for speaker labels in reverse order
        speaker_positions = []
        for speaker in (transcript.keys() if transcript else []):
            label = f"{speaker}:"
            pos = text_before.rfind(label)
            if pos != -1:
                speaker_positions.append((pos, speaker))
        
        # Return the speaker with the latest position before the violation
        if speaker_positions:
            speaker_positions.sort(reverse=True)
            return speaker_positions[0][1]
        
        return "Unknown"
    
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
            # "sentences": [
            #     {"index": i, "char_start": s, "char_end": e, "text": paragraph[s:e]}
            #     for i, (s, e) in enumerate(sentence_spans)
            # ],
            "violations": result.get("violations", []),
            "summary": result.get("summary", ""),
            "overall_risk": result.get("overall_risk", "NONE"),
        }
        
        # Add usage information if available
        if self.usage_tracker:
            out["usage"] = self.usage_tracker.get_usage_summary()
            
        return out
    
    def process_and_analyze(self, audio_file, use_google=True, model="gpt-4o"):
        """
        Process audio file and analyze for violations using Google Speech + GPT.
        
        Args:
            audio_file (str): Path to audio file
            use_google (bool): Whether to use Google Speech with diarization
            model (str): LLM model to use
            
        Returns:
            dict: Complete analysis results
        """
        # Reset usage tracker for new analysis
        if self.usage_tracker:
            self.usage_tracker.reset()
            
        # Step 1: Transcribe audio
        print("Step 1: Transcribing audio...")
        try:
            transcription_result = self.transcribe_audio(audio_file, use_google)
            print(f"Transcription result type: {type(transcription_result)}")
            print(f"Transcription result: {str(transcription_result)[:200]}...")
        except Exception as e:
            print(f"Transcription failed: {e}")
            return {"error": f"Transcription failed: {str(e)}"}
        
        # Handle different transcription result formats
        if isinstance(transcription_result, dict):
            if "transcript" in transcription_result:
                transcript = transcription_result["transcript"]
            else:
                # Assume it's already a speaker-segmented transcript
                transcript = transcription_result
        else:
            # String result - convert to speaker format
            if use_google:
                # This shouldn't happen with Google Speech, but fallback
                transcript = {"Speaker_1": str(transcription_result)}
            else:
                transcript = {"Speaker_1": str(transcription_result)}
        
        # Step 2: Format transcript for analysis
        paragraph = self.format_transcript_for_analysis(transcript)
        
        # Step 3: Split sentences
        sentence_spans = self.split_sentences(paragraph)
        
        # Step 4: Analyze with LLM
        print(f"Analyzing paragraph: {paragraph[:200]}...")
        
        # Keep full paragraph for accurate analysis with Google Speech
        print(f"Analyzing full paragraph ({len(paragraph)} chars) for accurate results")
        
        analysis = self.analyze_with_llm(paragraph, model)
        print(f"LLM analysis result: {analysis}")
        
        # Step 5: Validate spans
        validated_analysis = self.align_or_validate_spans(analysis, paragraph, sentence_spans, transcript)
        
        # Step 6: Build output
        output = self.build_output(audio_file, paragraph, validated_analysis, sentence_spans)
        
        # Add speaker information if available
        if use_google and isinstance(transcription_result, dict):
            output["speakers_detected"] = list(transcript.keys())
            if "words_info" in transcription_result:
                output["word_timestamps"] = transcription_result["words_info"]
        
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
