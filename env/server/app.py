"""
Server entry point for OpenEnv Data Cleaner.
Imports and re-exports the main app for multi-mode deployment.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app


def main():
    """Entry point for the server script."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()