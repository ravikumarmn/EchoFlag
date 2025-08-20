#!/usr/bin/env python3
"""
Generate violation text using OpenAI's API for testing the EchoFlag system.
This script generates realistic conversations containing violations of mutual funds distribution rules.
"""
import os
import json
import argparse
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path=".env")

class ViolationTextGenerator:
    """Generate financial conversation text with compliance violations using LLM."""
    
    def __init__(self, output_dir="test_transcripts"):
        """
        Initialize the violation text generator.
        
        Args:
            output_dir (str): Directory to save generated transcript files
        """
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Define violation categories
        self.violation_categories = {
            "RED": [
                "Guaranteeing specific returns or profits",
                "Making fraudulent or deceptive claims",
                "Threatening legal action against clients",
                "Claiming investments are approved by regulatory bodies when they aren't",
                "Pressuring clients to make immediate decisions with threats"
            ],
            "ORANGE": [
                "Claiming investments are 'risk-free' or have 'no risk'",
                "Creating false urgency with limited time offers",
                "Promising to double or triple investments",
                "Misrepresenting fund performance history",
                "Omitting material information about fees or risks"
            ],
            "YELLOW": [
                "Using superlatives like 'best' or 'perfect' investment",
                "Exaggerating potential returns",
                "Offering 'exclusive' opportunities without justification",
                "Making unsubstantiated comparisons to other investments",
                "Overstating the advisor's expertise or track record"
            ]
        }
    
    def generate_violation_text(self, severity="ALL", num_violations=3):
        """
        Generate conversation text with violations using LLM.
        
        Args:
            severity (str): Severity level of violations (RED, ORANGE, YELLOW, or ALL)
            num_violations (int): Number of violations to include
            
        Returns:
            dict: Generated transcript with speaker labels
        """
        # Determine which violation categories to include
        if severity == "ALL":
            # Include violations from all severity levels
            violation_types = []
            for sev, violations in self.violation_categories.items():
                violation_types.extend([(sev, v) for v in violations])
        else:
            # Include violations only from the specified severity level
            violation_types = [(severity, v) for v in self.violation_categories.get(severity, [])]
        
        # Select random violations
        import random
        selected_violations = random.sample(violation_types, min(num_violations, len(violation_types)))
        
        # Create prompt for the LLM
        prompt = self._create_prompt(selected_violations)
        
        # Generate conversation using OpenAI
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",  # Use appropriate model
                messages=[
                    {"role": "system", "content": "You are an expert in financial regulations who can generate realistic examples of non-compliant financial advisor conversations."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            # Extract response content
            content = response.choices[0].message.content
            
            # Parse the response to get the conversation
            return self._parse_response(content)
            
        except Exception as e:
            print(f"Error calling OpenAI API: {str(e)}")
            # Return a simple fallback conversation if API call fails
            return {
                "Speaker_1": "Hello, I'm your financial advisor. I recommend this mutual fund.",
                "Speaker_2": "Can you tell me more about it?"
            }
    
    def _create_prompt(self, violations):
        """Create a prompt for the LLM based on selected violations."""
        violations_text = "\n".join([f"- {sev} violation: {desc}" for sev, desc in violations])
        
        prompt = f"""
        Generate a realistic conversation between a financial advisor (Speaker_1) and a client (Speaker_2) about mutual funds.
        
        The conversation should include the following compliance violations:
        {violations_text}
        
        Format the conversation as a dialogue with 2-3 exchanges between the advisor and client.
        The advisor should make most of the violations.
        Make the violations obvious but still sound natural in conversation.
        
        DO NOT include any disclaimers or explanations outside the conversation.
        DO NOT label the violations in the output.
        """
        
        return prompt
    
    def _parse_response(self, content):
        """Parse the LLM response into a transcript format."""
        lines = content.strip().split('\n')
        transcript = {"Speaker_1": "", "Speaker_2": ""}
        current_speaker = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check for speaker indicators
            if line.startswith("Speaker_1:") or line.startswith("Financial Advisor:") or line.startswith("Advisor:"):
                current_speaker = "Speaker_1"
                line = line.split(":", 1)[1].strip()
            elif line.startswith("Speaker_2:") or line.startswith("Client:"):
                current_speaker = "Speaker_2"
                line = line.split(":", 1)[1].strip()
                
            # Add line to current speaker's text if speaker is identified
            if current_speaker:
                if transcript[current_speaker]:
                    transcript[current_speaker] += " " + line
                else:
                    transcript[current_speaker] = line
        
        return transcript
    
    def save_transcript(self, transcript, severity="mixed"):
        """
        Save the generated transcript to a JSON file.
        
        Args:
            transcript (dict): Generated transcript
            severity (str): Severity level indicator for filename
            
        Returns:
            str: Path to the saved transcript file
        """
        # Create filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"llm_generated_{severity.lower()}_{timestamp}.json"
        
        # Save to file
        output_file = os.path.join(self.output_dir, filename)
        with open(output_file, "w") as f:
            json.dump(transcript, f, indent=2)
        
        print(f"Transcript saved to '{output_file}'")
        return output_file

def main():
    """Main function to generate violation text using LLM."""
    parser = argparse.ArgumentParser(description="Generate violation text using LLM")
    parser.add_argument("--severity", choices=["RED", "ORANGE", "YELLOW", "ALL"], 
                        default="ALL", help="Severity level of violations")
    parser.add_argument("--violations", type=int, default=3,
                        help="Number of violations to include")
    parser.add_argument("--output-dir", default="test_transcripts",
                        help="Directory to save generated transcripts")
    
    args = parser.parse_args()
    
    # Create generator
    generator = ViolationTextGenerator(args.output_dir)
    
    # Generate transcript with violations
    transcript = generator.generate_violation_text(args.severity, args.violations)
    
    # Save transcript
    output_file = generator.save_transcript(transcript, args.severity)
    
    print("\nViolation text generation complete.")
    print("You can now convert this transcript to audio:")
    print(f"python src/transcript_to_audio.py {output_file} --combine")
    
    return 0

if __name__ == "__main__":
    exit(main())
