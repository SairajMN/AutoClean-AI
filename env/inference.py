"""
Inference module for OpenEnv Data Cleaning Environment.
Provides the main entry point for Hugging Face Spaces deployment.
"""

from server.app import app

# Export the FastAPI app for Hugging Face Spaces
__all__ = ["app"]