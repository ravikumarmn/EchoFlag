#!/usr/bin/env python3
"""
Quick test to isolate the OpenAI API issue
"""
import os
import sys
from dotenv import load_dotenv
import openai

# Load environment variables
load_dotenv()

def test_openai_api():
    """Test OpenAI API directly with a simple call"""
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not found")
        return False
    
    try:
        # Initialize client
        client = openai.OpenAI(api_key=api_key)
        
        # Simple test message
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say 'Hello World' in JSON format like {\"message\": \"Hello World\"}"}
        ]
        
        print("Testing OpenAI API...")
        import time
        start_time = time.time()
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.1,
            max_tokens=100,
            timeout=15
        )
        
        end_time = time.time()
        print(f"API call completed in {end_time - start_time:.2f} seconds")
        
        content = response.choices[0].message.content
        print(f"Response: {content}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_openai_api()
    print(f"Test {'PASSED' if success else 'FAILED'}")
