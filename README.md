---
title: OpenEnv Data Cleaner
emoji: 🧹
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
---

# OpenEnv Data Cleaner

OpenEnv-compliant AI-powered data cleaning environment for Hugging Face Spaces.

## Overview

This space runs an OpenEnv-native data cleaning environment that allows AI agents to:

- **Clean datasets** through a structured action system
- **Execute data cleaning operations** via the OpenEnv lifecycle
- **Get graded** on cleaning quality with deterministic scoring

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/reset` | POST | Initialize a new task |
| `/step` | POST | Execute a cleaning action |
| `/tasks` | GET | List available tasks |
| `/state` | GET | Get current environment state |
| `/submit` | POST | Submit solution for grading |
| `/revert` | POST | Revert last action |

## Available Tasks

- **easy_001**: Basic data cleaning (drop nulls, remove duplicates)
- **medium_001**: Intermediate cleaning (handle nulls, validate emails, remove outliers)
- **hard_001**: Advanced cleaning (full pipeline with type conversion and normalization)

## Usage Example

```python
import requests

BASE_URL = "https://sairaj2-openenv-datacleaner.hf.space"

# Reset with a task
response = requests.post(f"{BASE_URL}/reset", json={"task_id": "easy_001"})
print(response.json())

# Execute cleaning steps
requests.post(f"{BASE_URL}/step", json={"action_type": "drop_nulls", "params": {}})
requests.post(f"{BASE_URL}/step", json={"action_type": "remove_duplicates", "params": {}})

# Submit for grading
response = requests.post(f"{BASE_URL}/submit")
print(response.json())
```

## OpenEnv Compliance

This environment implements the full OpenEnv lifecycle:
- `reset(task_id, session_id)` - Initialize environment
- `step(action)` - Execute actions with (observation, reward, done, info)
- `state()` - Get current environment state

Built with `openenv-core` for full compatibility.

## Architecture

- **env/datacleaner_env.py**: OpenEnv-compliant environment extending BaseEnv
- **env/action_engine.py**: Deterministic action execution with validation and rollback
- **env/grader.py**: Task-specific grading with deterministic scoring
- **env/reward.py**: Structured reward system (quality + progress - penalty)
- **env/tasks.py**: Task definitions with dataset configurations
