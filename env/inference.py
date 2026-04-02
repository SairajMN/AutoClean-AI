"""
Inference module for OpenEnv Data Cleaning Environment.
Provides the main entry point for Hugging Face Spaces deployment.
"""

import os
import sys

# Add the current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app

# Export the FastAPI app for Hugging Face Spaces
__all__ = ["app"]