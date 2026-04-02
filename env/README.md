---
title: OpenEnv Data Cleaner
emoji: 🧹
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
---

# OpenEnv Data Cleaner

An OpenEnv-compliant AI-powered data cleaning environment built on `openenv-core`.

## Features

- **OpenEnv-native**: Built using `openenv-core` base classes
- **Data Cleaning Actions**: Drop nulls, fill nulls, remove duplicates, filter rows, drop columns, convert types, validate emails, outlier removal, normalization
- **Task-based Learning**: Three difficulty levels (easy, medium, hard)
- **Grading System**: Deterministic scoring based on data quality improvements
- **Reward System**: Structured rewards with quality, progress, and penalty components
- **Web Interface**: Interactive UI for manual data cleaning
- **Docker Ready**: Deployable to Hugging Face Spaces

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
python app.py
```

## API Endpoints

- `GET /` - Web interface
- `GET /health` - Health check
- `POST /reset` - Initialize a new task
- `POST /step` - Execute a cleaning action
- `POST /submit` - Submit solution for grading
- `POST /revert` - Revert last action
- `GET /tasks` - List available tasks
- `GET /state` - Get current environment state
- `GET /dataset` - Get dataset information
- `GET /history` - Get action history

## Tasks

| Task ID | Difficulty | Description |
|---------|------------|-------------|
| easy_001 | Easy | Basic cleaning: drop nulls and remove duplicates |
| medium_001 | Medium | Intermediate: handle nulls, validate emails, remove outliers |
| hard_001 | Hard | Advanced: full pipeline with type conversion and normalization |

## Deployment

Deploy to Hugging Face Spaces:

```bash
openenv push ./env
```
