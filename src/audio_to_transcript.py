#!/usr/bin/env python3
"""
Convert audio files to transcript text using speech recognition.
This script takes an audio file and generates a transcript.
"""
import os
import json
import argparse
import speech_recognition as sr
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
        
        # Initialize recognizer
        self.recognizer = sr.Recognizer()
    
    def convert_mp3_to_wav(self, mp3_file):
        """
        Convert MP3 file to WAV format for speech recognition.
        
        Args:
            mp3_file (str): Path to MP3 file
            
        Returns:
            str: Path to temporary WAV file
        """
        # Create a temporary file
        temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_wav_path = temp_wav.name
        temp_wav.close()
        
        # Convert MP3 to WAV
        audio = AudioSegment.from_mp3(mp3_file)
        audio.export(temp_wav_path, format="wav")
        
        return temp_wav_path
    
    def transcribe_audio(self, audio_file, use_google=False):
        """
        Transcribe audio file to text.
        
        Args:
            audio_file (str): Path to audio file
            use_google (bool): Whether to use Google Web Speech API
            
        Returns:
            str: Transcribed text
        """
        # Check if file is MP3 and convert if needed
        if audio_file.lower().endswith('.mp3'):
            wav_file = self.convert_mp3_to_wav(audio_file)
            is_temp = True
        else:
            wav_file = audio_file
            is_temp = False
        
        try:
            # Load audio file
            with sr.AudioFile(wav_file) as source:
                # Adjust for ambient noise and record
                self.recognizer.adjust_for_ambient_noise(source)
                audio_data = self.recognizer.record(source)
                
                # Transcribe audio
                if use_google:
                    # Use Google Web Speech API (requires internet)
                    text = self.recognizer.recognize_google(audio_data)
                else:
                    # Use Sphinx (offline, but less accurate)
                    text = self.recognizer.recognize_sphinx(audio_data)
                
                return text
        except sr.UnknownValueError:
            return "Speech Recognition could not understand audio"
        except sr.RequestError as e:
            return f"Could not request results; {e}"
        finally:
            # Clean up temporary file if created
            if is_temp and os.path.exists(wav_file):
                os.remove(wav_file)
    
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
    
    def process_audio_file(self, audio_file, use_google=False):
        """
        Process audio file and save transcript.
        
        Args:
            audio_file (str): Path to audio file
            use_google (bool): Whether to use Google Web Speech API
            
        Returns:
            dict: Transcript data
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
        output_file = os.path.join(self.output_dir, f"{basename}_transcript_{timestamp}.json")
        
        # Save transcript
        with open(output_file, "w") as f:
            json.dump(transcript, f, indent=2)
        
        print(f"Transcript saved to '{output_file}'")
        
        return transcript

def main():
    """Main function to convert audio to transcript."""
    parser = argparse.ArgumentParser(description="Convert audio files to transcript text")
    parser.add_argument("audio_file", help="Path to audio file")
    parser.add_argument("--output-dir", default="transcripts_output", help="Directory to save transcript files")
    parser.add_argument("--use-google", action="store_true", help="Use Google Web Speech API (requires internet)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.audio_file):
        print(f"Error: File '{args.audio_file}' not found.")
        return 1
    
    # Create converter
    converter = AudioToTranscript(args.output_dir)
    
    # Process audio file
    transcript = converter.process_audio_file(args.audio_file, args.use_google)
    
    # Print transcript
    print("\nTranscript:")
    for speaker, text in transcript.items():
        print(f"{speaker}: {text}")
    
    print("\nAudio to transcript conversion complete.")
    
    return 0

if __name__ == "__main__":
    exit(main())
