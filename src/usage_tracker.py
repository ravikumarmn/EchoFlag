#!/usr/bin/env python3
"""
Usage Tracker for EchoFlag

Tracks token usage and calculates pricing for OpenAI API calls and Google Speech-to-Text.
"""
from typing import Dict, Any, Optional, Union
import time

class UsageTracker:
    """Tracks API usage and calculates costs for EchoFlag."""
    
    # OpenAI pricing per 1,000,000 tokens (2025 tables you provided)
    # Only include models we actually use now; extend easily if needed.
    OPENAI_TEXT_PRICING_PER_MILLION = {
        # gpt-4o family
        "gpt-4o": {
            "input": 2.50,
            "cached_input": 1.25,
            "output": 10.00,
            "tier": "standard",
        },
        "gpt-4o-mini": {
            "input": 0.15,
            "cached_input": 0.075,
            "output": 0.60,
            "tier": "standard",
        },
        # Legacy fallback if a different model name is passed
        "gpt-4": {
            # Keep a sensible fallback using older public rates (per 1M)
            "input": 30.00,
            "cached_input": 15.00,
            "output": 60.00,
            "tier": "standard",
        },
    }
    
    # OpenAI Whisper pricing (per minute)
    OPENAI_TRANSCRIBE_PRICING_PER_MIN = {
        "whisper-1": 0.006,
    }
    
    # Google Speech-to-Text pricing (choose default estimator: V2 Standard $0.016/min)
    GOOGLE_SPEECH_PRICING_PER_MIN = {
        "v2_standard": 0.016,
        # Other options you may enable later:
        # "v2_logged": 0.012,
        # "v2_dynamic_batch": 0.003,
        # "v1_no_logging": 0.024,
    }
    
    def __init__(self):
        """Initialize usage tracker with empty usage data."""
        self.reset()
    
    def reset(self):
        """Reset all usage data."""
        self.transcription_usage = {
            "openai_whisper": {
                "audio_seconds": 0,
                "estimated_cost": 0.0,
                "rate_per_minute": self.OPENAI_TRANSCRIBE_PRICING_PER_MIN["whisper-1"],
            },
            "google_speech": {
                "audio_seconds": 0,
                "estimated_cost": 0.0,
                "rate_per_minute": self.GOOGLE_SPEECH_PRICING_PER_MIN["v2_standard"],
                "tier": "v2_standard",
                "note": "Estimated using Google Cloud STT v2 Standard rate; actual engine is Google Web Speech API",
            }
        }
        
        self.analysis_usage = {
            "prompt_tokens": 0,
            "cached_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "model": "",
            "estimated_cost": 0.0,
            "pricing": {
                "tier": "standard",
                "input_per_million": 0.0,
                "cached_input_per_million": 0.0,
                "output_per_million": 0.0,
            },
        }
        
        self.total_estimated_cost = 0.0
    
    def track_openai_chat_usage(self, response: Any, model: str) -> None:
        """
        Track token usage from an OpenAI chat completion response.
        
        Args:
            response: OpenAI API response object
            model: Model name used for the request
        """
        if not hasattr(response, "usage"):
            return
        
        usage = response.usage
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        total_tokens = getattr(usage, "total_tokens", prompt_tokens + completion_tokens) or 0

        # Attempt to read cached tokens if provided by SDK
        cached_tokens = 0
        details = getattr(usage, "prompt_tokens_details", None)
        if details is not None:
            cached_tokens = getattr(details, "cached_tokens", 0) or 0

        non_cached_prompt = max(prompt_tokens - cached_tokens, 0)

        # Pick pricing for selected model (fallback to gpt-4o)
        pricing = self.OPENAI_TEXT_PRICING_PER_MILLION.get(model, self.OPENAI_TEXT_PRICING_PER_MILLION["gpt-4o"])

        input_rate = pricing["input"]
        cached_rate = pricing.get("cached_input", input_rate)
        output_rate = pricing["output"]
        tier = pricing.get("tier", "standard")

        # Convert tokens to cost using per-1M denomination
        input_cost = (non_cached_prompt / 1_000_000) * input_rate
        cached_input_cost = (cached_tokens / 1_000_000) * cached_rate
        output_cost = (completion_tokens / 1_000_000) * output_rate
        total_cost = input_cost + cached_input_cost + output_cost

        # Accumulate usage
        self.analysis_usage["prompt_tokens"] += prompt_tokens
        self.analysis_usage["cached_tokens"] += cached_tokens
        self.analysis_usage["completion_tokens"] += completion_tokens
        self.analysis_usage["total_tokens"] += total_tokens
        self.analysis_usage["model"] = model
        self.analysis_usage["estimated_cost"] += total_cost
        self.analysis_usage["pricing"] = {
            "tier": tier,
            "input_per_million": input_rate,
            "cached_input_per_million": cached_rate,
            "output_per_million": output_rate,
        }
        self.total_estimated_cost += total_cost
    
    def track_whisper_usage(self, audio_duration_seconds: float) -> None:
        """
        Track Whisper API usage based on audio duration.
        
        Args:
            audio_duration_seconds: Duration of audio in seconds
        """
        # Convert to minutes for pricing calculation
        minutes = audio_duration_seconds / 60
        cost = minutes * self.OPENAI_TRANSCRIBE_PRICING_PER_MIN["whisper-1"]
        
        self.transcription_usage["openai_whisper"]["audio_seconds"] += audio_duration_seconds
        self.transcription_usage["openai_whisper"]["estimated_cost"] += cost
        self.total_estimated_cost += cost
    
    def track_google_speech_usage(self, audio_duration_seconds: float, enhanced: bool = False) -> None:
        """
        Track Google Speech-to-Text API usage based on audio duration.
        
        Args:
            audio_duration_seconds: Duration of audio in seconds
            enhanced: Whether enhanced model was used
        """
        # Convert to minutes for pricing calculation
        minutes = audio_duration_seconds / 60
        # Default to v2 standard estimator
        model_type = "v2_standard"
        rate = self.GOOGLE_SPEECH_PRICING_PER_MIN[model_type]
        cost = minutes * rate
        
        self.transcription_usage["google_speech"]["audio_seconds"] += audio_duration_seconds
        self.transcription_usage["google_speech"]["estimated_cost"] += cost
        self.transcription_usage["google_speech"]["rate_per_minute"] = rate
        self.transcription_usage["google_speech"]["tier"] = model_type
        self.total_estimated_cost += cost
    
    def get_audio_duration(self, audio_file: str) -> float:
        """
        Get audio duration in seconds.
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            Duration in seconds
        """
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(audio_file)
            return len(audio) / 1000.0  # pydub duration is in milliseconds
        except Exception:
            # If pydub fails, return a default duration
            return 0.0
    
    def get_usage_summary(self) -> Dict[str, Any]:
        """
        Get complete usage summary.
        
        Returns:
            Dictionary with usage data
        """
        return {
            "transcription": {
                "openai_whisper": self.transcription_usage["openai_whisper"],
                "google_speech": self.transcription_usage["google_speech"]
            },
            "analysis": self.analysis_usage,
            "total_estimated_cost": self.total_estimated_cost
        }
