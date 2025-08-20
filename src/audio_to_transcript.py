#!/usr/bin/env python3
"""
Convert audio files to transcript text using OpenAI Whisper.
This script takes an audio file and generates a transcript.
"""
import os
import json
import argparse
from openai import OpenAI
from pydub import AudioSegment
import tempfile
from datetime import datetime

class AudioToTranscript:
    """Convert audio files to transcript text."""
    
    def __init__(self, output_dir="transcripts_output"):
        """
        Initialize the audio to transcript converter.
        
        Args:
            output_dir (str): Directory to save generated transcript files
        """
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def transcribe_audio(self, audio_file):
        """
        Transcribe audio file to text using OpenAI Whisper.
        
        Args:
            audio_file (str): Path to audio file
            
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
            print(f"Error transcribing audio: {e}")
            return ""
    
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
    
    def process_audio_file(self, audio_file):
        """
        Process audio file and save transcript using OpenAI Whisper.
        
        Args:
            audio_file (str): Path to audio file
            
        Returns:
            dict: Transcript data
        """
        print(f"Processing audio file: {audio_file}")
        
        # Transcribe audio
        text = self.transcribe_audio(audio_file)
        
        # Extract speaker information from filename
        speaker = self.extract_speaker_from_filename(audio_file)
        if not speaker:
            speaker = "Unknown_Speaker"
        
        # Create transcript data
        transcript = {speaker: text}
        
        # Generate output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        basename = os.path.splitext(os.path.basename(audio_file))[0]
        output_file = os.path.join(self.output_dir, f"{basename}_transcript_{timestamp}.json")
        
        # Save transcript
        with open(output_file, "w") as f:
            json.dump(transcript, f, indent=2)
        
        print(f"Transcript saved to '{output_file}'")
        
        return transcript

def main():
    """Main function to convert audio to transcript."""
    parser = argparse.ArgumentParser(description="Convert audio files to transcript text using OpenAI Whisper")
    parser.add_argument("audio_file", help="Path to audio file")
    parser.add_argument("--output-dir", default="transcripts_output", help="Directory to save transcript files")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.audio_file):
        print(f"Error: File '{args.audio_file}' not found.")
        return 1
    
    # Create converter
    converter = AudioToTranscript(args.output_dir)
    
    # Process audio file
    transcript = converter.process_audio_file(args.audio_file)
    
    # Print transcript
    print("\nTranscript:")
    for speaker, text in transcript.items():
        print(f"{speaker}: {text}")
    
    print("\nAudio to transcript conversion complete.")
    
    return 0

if __name__ == "__main__":
    exit(main())
