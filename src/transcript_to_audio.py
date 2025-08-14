#!/usr/bin/env python3
"""
Convert transcript JSON files to audio files using Google Text-to-Speech (gTTS).
This script takes a transcript JSON file and generates audio files for each speaker.
"""
import os
import json
import argparse
from gtts import gTTS
from pydub import AudioSegment
from datetime import datetime
import time

class TranscriptToAudio:
    """Convert transcript JSON to audio files using gTTS."""
    
    def __init__(self, output_dir="audio_output"):
        """
        Initialize the transcript to audio converter.
        
        Args:
            output_dir (str): Directory to save generated audio files
        """
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def get_voice_properties(self, speaker_id):
        """
        Get voice properties based on speaker ID.
        
        Args:
            speaker_id (str): Speaker identifier (e.g., "Speaker_1")
            
        Returns:
            dict: Voice properties for gTTS
        """
        # Set different voice properties for different speakers
        if speaker_id == "Speaker_1":  # Agent
            return {
                "tld": "com",  # Top-level domain for the Google TTS service
                "lang": "en",  # Language
                "slow": False,  # Speed
                "lang_check": False  # Skip language check for faster processing
            }
        else:  # Client or other speakers
            return {
                "tld": "co.uk",  # Different accent for client
                "lang": "en",
                "slow": False,
                "lang_check": False
            }
    
    def convert_transcript_to_audio(self, transcript_file):
        """
        Convert transcript JSON to audio files.
        
        Args:
            transcript_file (str): Path to transcript JSON file
            
        Returns:
            dict: Paths to generated audio files by speaker
        """
        # Load transcript
        with open(transcript_file, "r") as f:
            transcript = json.load(f)
        
        # Generate timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate audio for each speaker
        audio_files = {}
        for speaker, text in transcript.items():
            # Get voice properties for this speaker
            voice_props = self.get_voice_properties(speaker)
            
            # Generate filename
            base_name = os.path.splitext(os.path.basename(transcript_file))[0]
            filename = f"{base_name}_{speaker}_{timestamp}.mp3"
            output_file = os.path.join(self.output_dir, filename)
            
            print(f"Generating audio for {speaker}...")
            print(f"Text: {text[:50]}...")
            
            try:
                # Create gTTS object
                tts = gTTS(text=text, **voice_props)
                
                # Save audio to file
                tts.save(output_file)
                
                # Verify file was created and has content
                if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                    audio_files[speaker] = output_file
                    print(f"Audio saved to '{output_file}'")
                else:
                    print(f"Warning: Audio file '{output_file}' was not created or is empty.")
                
                # Add a small delay to avoid rate limiting
                time.sleep(1)
                
            except Exception as e:
                print(f"Error generating audio for {speaker}: {str(e)}")
        
        return audio_files
    
    def combine_audio_files(self, audio_files, output_file=None):
        """
        Combine audio files from different speakers into a conversation.
        
        Args:
            audio_files (dict): Paths to audio files by speaker
            output_file (str): Path to save the combined audio file
            
        Returns:
            str: Path to combined audio file
        """
        if not audio_files:
            print("No audio files to combine.")
            return None
        
        if not output_file:
            # Generate output filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(self.output_dir, f"combined_{timestamp}.mp3")
        
        try:
            # Create a new empty audio segment
            combined = AudioSegment.silent(duration=500)  # Start with 0.5s silence
            
            # Add each speaker's audio with a small gap between them
            for speaker, file_path in sorted(audio_files.items()):
                if os.path.exists(file_path):
                    audio = AudioSegment.from_file(file_path)
                    combined += audio + AudioSegment.silent(duration=1000)  # Add 1s silence between speakers
            
            # Export the combined audio
            combined.export(output_file, format="mp3")
            print(f"Combined audio saved to '{output_file}'")
            return output_file
            
        except Exception as e:
            print(f"Error combining audio files: {str(e)}")
            print("Note: Audio files are still available separately for each speaker.")
            return None

def main():
    """Main function to convert transcript to audio."""
    parser = argparse.ArgumentParser(description="Convert transcript JSON to audio files")
    parser.add_argument("transcript_file", help="Path to transcript JSON file")
    parser.add_argument("--output-dir", default="audio_output", help="Directory to save audio files")
    parser.add_argument("--combine", action="store_true", help="Combine audio files into a conversation")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.transcript_file):
        print(f"Error: File '{args.transcript_file}' not found.")
        return 1
    
    # Create converter
    converter = TranscriptToAudio(args.output_dir)
    
    # Convert transcript to audio
    audio_files = converter.convert_transcript_to_audio(args.transcript_file)
    
    # Combine audio files if requested
    if args.combine and audio_files:
        converter.combine_audio_files(audio_files)
    
    print("\nAudio generation complete.")
    print(f"Audio files saved to '{args.output_dir}'")
    
    return 0

if __name__ == "__main__":
    exit(main())
