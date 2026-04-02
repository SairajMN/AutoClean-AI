
#!/usr/bin/env python3
"""
Hugging Face Spaces Deployment Script
Automates the entire deployment process for OpenEnv Data Cleaner.

Usage:
    python deploy_hf.py --username YOUR_USERNAME --token YOUR_HF_TOKEN [--repo-name openenv-datacleaner]
"""

import os
import sys
import argparse
import subprocess
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("hf-deploy")


def check_prerequisites():
    """Check that all required tools are installed."""
    required_tools = ["git", "pip"]
    
    for tool in required_tools:
        try:
            subprocess.run(
                [tool, "--version"],
                capture_output=True,
                check=True
            )
            logger.info(f"✓ {tool} is installed")
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error(f"✗ {tool} is not installed or not in PATH")
            sys.exit(1)


def install_dependencies():
    """Install required Python packages."""
    logger.info("Installing huggingface_hub...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--upgrade", "huggingface_hub"],
        check=True
    )
    logger.info("✓ huggingface_hub installed")


def create_hf_space(username: str, repo_name: str, token: str) -> str:
    """Create Hugging Face Space if it doesn't exist."""
    from huggingface_hub import HfApi, HfFolder
    
    api = HfApi()
    
    # Store token
    HfFolder.save_token(token)
    
    repo_id = f"{username}/{repo_name}"
    
    try:
        # Check if repo exists
        repo_info = api.repo_info(repo_id=repo_id, repo_type="space")
        logger.info(f"✓ Space already exists: {repo_id}")
        return repo_id
    except Exception:
        # Create the space
        logger.info(f"Creating Hugging Face Space: {repo_id}")
        api.create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="docker",
            exist_ok=True
        )
        logger.info(f"✓ Space created: {repo_id}")
        return repo_id


def setup_git_repo(repo_url: str):
    """Initialize and configure git repository."""
    logger.info("Setting up git repository...")
    
    # Initialize git if not already initialized
    if not Path(".git").exists():
        subprocess.run(["git", "init"], check=True)
        logger.info("✓ Git repository initialized")
    
    # Install git LFS
    subprocess.run(["git", "lfs", "install"], check=True)
    logger.info("✓ Git LFS installed")
    
    # Add or update remote
    try:
        subprocess.run(
            ["git", "remote", "set-url", "origin", repo_url],
            check=True
        )
        logger.info("✓ Remote URL updated")
    except subprocess.CalledProcessError:
        subprocess.run(
            ["git", "remote", "add", "origin", repo_url],
            check=True
        )
        logger.info("✓ Remote added")


def commit_and_push(message: str = "Deploy to Hugging Face Spaces"):
    """Commit all changes and push to Hugging Face."""
    logger.info("Committing changes...")
    
    # Add all files
    subprocess.run(["git", "add", "."], check=True)
    
    # Check if there are changes to commit
    result = subprocess.run(
        ["git", "diff", "--cached", "--quiet"],
        capture_output=True
    )
    
    if result.returncode != 0:
        # There are changes to commit
        subprocess.run(["git", "commit", "-m", message], check=True)
        logger.info("✓ Changes committed")
    else:
        logger.info("No changes to commit")
    
    # Push to remote
    logger.info("Pushing to Hugging Face...")
    subprocess.run(
        ["git", "push", "-u", "origin", "main", "--force"],
        check=True
    )
    logger.info("✓ Pushed to Hugging Face Spaces")


def verify_deployment(username: str, repo_name: str):
    """Print verification information."""
    space_url = f"https://huggingface.co/spaces/{username}/{repo_name}"
    api_url = f"https://{username}-{repo_name}.hf.space"
    
    logger.info("=" * 60)
    logger.info("DEPLOYMENT COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Space URL: {space_url}")
    logger.info(f"API URL: {api_url}")
    logger.info("")
    logger.info("Verification endpoints:")
    logger.info(f"  GET  {api_url}/health")
    logger.info(f"  POST {api_url}/reset")
    logger.info(f"  POST {api_url}/step")
    logger.info(f"  GET  {api_url}/tasks")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Deploy OpenEnv Data Cleaner to Hugging Face Spaces"
    )
    parser.add_argument(
        "--username",
        required=True,
        help="Your Hugging Face username"
    )
    parser.add_argument(
        "--token",
        required=True,
        help="Your Hugging Face access token (get from https://huggingface.co/settings/tokens)"
    )
    parser.add_argument(
        "--repo-name",
        default="openenv-datacleaner",
        help="Name of the Hugging Face Space (default: openenv-datacleaner)"
    )
    parser.add_argument(
        "--commit-message",
        default="Deploy OpenEnv Data Cleaner",
        help="Git commit message"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("OpenEnv Data Cleaner - Hugging Face Deployment")
    logger.info("=" * 60)
    
    # Step 1: Check prerequisites
    logger.info("Step 1: Checking prerequisites...")
    check_prerequisites()
    
    # Step 2: Install dependencies
    logger.info("Step 2: Installing dependencies...")
    # Step 2: Install dependencies
    logger.info("=" * 60)
    
        default="openenv-datacleaner",
        "--repo-name",
        required=True,
