#!/bin/bash
# ============================================================
# OpenEnv Data Cleaner - Hugging Face Spaces Deployment Script
# Automated CLI deployment using huggingface_hub
# ============================================================

set -e  # Exit on error

# Configuration
REPO_NAME="openenv-datacleaner"
SPACE_TITLE="OpenEnv Data Cleaner"
SPACE_DESCRIPTION="OpenEnv-compliant AI-powered data cleaning environment"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}OpenEnv Data Cleaner - Hugging Face Spaces Deployment${NC}"
echo -e "${GREEN}============================================================${NC}"

# Step 1: Check prerequisites
echo -e "\n${YELLOW}[1/6] Checking prerequisites...${NC}"

if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: python3 is required but not installed.${NC}"
    exit 1
fi

if ! command -v git &> /dev/null; then
    echo -e "${RED}Error: git is required but not installed.${NC}"
    exit 1
fi

if ! command -v docker &> /dev/null; then
    echo -e "${YELLOW}Warning: docker is not installed. Local testing will not be available.${NC}"
fi

echo -e "${GREEN}Prerequisites check passed.${NC}"

# Step 2: Install dependencies
echo -e "\n${YELLOW}[2/6] Installing Python dependencies...${NC}"

pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
pip install --quiet huggingface_hub

echo -e "${GREEN}Dependencies installed.${NC}"

# Step 3: Hugging Face authentication
echo -e "\n${YELLOW}[3/6] Hugging Face authentication...${NC}"

if [ -z "$HF_TOKEN" ]; then
    echo -e "${YELLOW}HF_TOKEN environment variable not set.${NC}"
    echo -e "Please enter your Hugging Face token (get one at https://huggingface.co/settings/tokens):"
    read -s HF_TOKEN
    export HF_TOKEN
fi

# Login to Hugging Face
huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential 2>/dev/null || true

echo -e "${GREEN}Authentication complete.${NC}"

# Step 4: Get username and create repo
echo -e "\n${YELLOW}[4/6] Setting up Hugging Face Space repository...${NC}"

# Get username from token
HF_USERNAME=$(huggingface-cli whoami 2>/dev/null | head -1 | tr -d '[:space:]')

if [ -z "$HF_USERNAME" ]; then
    echo -e "${RED}Error: Could not determine Hugging Face username.${NC}"
    echo -e "Please ensure you're logged in: huggingface-cli login"
    exit 1
fi

REPO_ID="${HF_USERNAME}/${REPO_NAME}"
echo -e "Repository: ${GREEN}${REPO_ID}${NC}"

# Create the space if it doesn't exist
python3 -c "
from huggingface_hub import HfApi
api = HfApi()
try:
    api.create_repo(
        repo_id='${REPO_ID}',
        repo_type='space',
        space_sdk='docker',
        exist_ok=True
    )
    print('Space repository ready.')
except Exception as e:
    print(f'Note: {e}')
"

echo -e "${GREEN}Space repository setup complete.${NC}"

# Step 5: Prepare deployment files
echo -e "\n${YELLOW}[5/6] Preparing deployment files...${NC}"

# Create .dockerignore if not exists
if [ ! -f ".dockerignore" ]; then
    cat > .dockerignore << 'EOF'
.git
.gitignore
__pycache__
*.pyc
.env
.venv
node_modules
*.md
deploy.sh
test.txt
EOF
fi

# Create README for the space
cat > README.md << EOF
---
title: ${SPACE_TITLE}
emoji: 🧹
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
---

# ${SPACE_TITLE}

${SPACE_DESCRIPTION}

## OpenEnv-Compliant Data Cleaning Environment

This space runs an OpenEnv-native data cleaning environment that allows AI agents to:

- **Clean datasets** through a structured action system
- **Execute data cleaning operations** via the OpenEnv lifecycle
- **Get graded** on cleaning quality with deterministic scoring

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| \`/health\` | GET | Health check |
| \`/reset\` | POST | Initialize a new task |
| \`/step\` | POST | Execute a cleaning action |
| \`/tasks\` | GET | List available tasks |
| \`/state\` | GET | Get current environment state |
| \`/submit\` | POST | Submit solution for grading |
| \`/revert\` | POST | Revert last action |

## Available Tasks

- **easy_001**: Basic data cleaning (drop nulls, remove duplicates)
- **medium_001**: Intermediate cleaning (handle nulls, validate emails, remove outliers)
- **hard_001**: Advanced cleaning (full pipeline with type conversion and normalization)

## Usage Example

\`\`\`python
import requests

BASE_URL = "https://${REPO_ID}.hf.space"

# Reset with a task
response = requests.post(f"{BASE_URL}/reset", json={"task_id": "easy_001"})
print(response.json())

# Execute cleaning steps
requests.post(f"{BASE_URL}/step", json={"action_type": "drop_nulls", "params": {}})
requests.post(f"{BASE_URL}/step", json={"action_type": "remove_duplicates", "params": {}})

# Submit for grading
response = requests.post(f"{BASE_URL}/submit")
print(response.json())
\`\`\`

## OpenEnv Compliance

This environment implements the full OpenEnv lifecycle:
- \`reset(task_id, session_id)\` - Initialize environment
- \`step(action)\` - Execute actions with (observation, reward, done, info)
- \`state()\` - Get current environment state

Built with \`openenv-core\` for full compatibility.
EOF

echo -e "${GREEN}Deployment files prepared.${NC}"

# Step 6: Deploy via git
echo -e "\n${YELLOW}[6/6] Deploying to Hugging Face Spaces...${NC}"

# Initialize git if needed
if [ ! -d ".git" ]; then
    git init
    git lfs install 2>/dev/null || true
fi

# Set up remote
git remote remove hf 2>/dev/null || true
git remote add hf "https://huggingface.co/spaces/${REPO_ID}"

# Stage and commit
git add -A
git commit -m "Deploy OpenEnv Data Cleaner v1.0.0

- OpenEnv-compliant environment using openenv-core
- FastAPI wrapper with /health, /reset, /step, /tasks endpoints
- Task definitions: easy_001, medium_001, hard_001
- Structured reward system with quality/progress/penalty
- Grading system with deterministic scoring
- Action engine with validation and rollback support
- Docker deployment for Hugging Face Spaces" || true

# Push to Hugging Face
echo -e "${YELLOW}Pushing to Hugging Face Spaces...${NC}"
git push hf main --force

echo -e "\n${GREEN}============================================================${NC}"
echo -e "${GREEN}Deployment Complete!${NC}"
echo -e "${GREEN}============================================================${NC}"
echo -e ""
echo -e "Space URL: ${GREEN}https://huggingface.co/spaces/${REPO_ID}${NC}"
echo -e "API Base:  ${GREEN}https://${REPO_ID}.hf.space${NC}"
echo -e ""
echo -e "${YELLOW}Note: It may take a few minutes for the space to build and start.${NC}"
echo -e "${YELLOW}Check the space URL for deployment status.${NC}"
echo -e ""

# Verify deployment
echo -e "${YELLOW}Verifying deployment...${NC}"
sleep 10

for i in {1..30}; do
    STATUS=$(curl -s -o /dev/null -w "%{http_code}" "https://${REPO_ID}.hf.space/health" 2>/dev/null || echo "000")
    if [ "$STATUS" = "200" ]; then
        echo -e "${GREEN}Space is running! Health check passed.${NC}"
        curl -s "https://${REPO_ID}.hf.space/health" | python3 -m json.tool 2>/dev/null || true
        break
    fi
    echo -e "${YELLOW}Waiting for space to start... (attempt $i/30, status: $STATUS)${NC}"
    sleep 10
done

echo -e "\n${GREEN}Done!${NC}"