#!/usr/bin/env python3
"""
Streamlit Cloud entry point for EchoFlag app.
This file is required for Streamlit Cloud deployment.
"""
import sys
import os

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import and run the main app
from src.app import *
