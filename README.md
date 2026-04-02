# OpenEnv Data Cleaner

An OpenEnv-compliant AI-powered data cleaning environment for training and evaluating AI agents on real-world data cleaning tasks.

## Environment Description

Data cleaning is a critical step in any data science or machine learning pipeline. Real-world datasets often contain missing values, duplicates, inconsistent formats, outliers, and other quality issues that can significantly impact downstream analysis and model performance.

This environment simulates a realistic data cleaning workflow where AI agents must identify and fix data quality issues through a series of targeted actions. The environment provides:

- **Realistic datasets** with common data quality problems
- **10 data cleaning actions** covering the most common cleaning operations
- **3 difficulty levels** from basic to advanced cleaning pipelines
- **Deterministic grading** with scores from 0.0 to 1.0
- **Shaped rewards** providing partial progress signals throughout the episode

## Action Space

The environment supports the following actions:

| Action | Parameters | Description |
|--------|------------|-------------|
| `drop_nulls` | `column` (optional) | Remove rows with null values |
| `fill_nulls` | `column`, `strategy` (mean/median/mode/forward_fill/backward_fill) | Fill null values |
| `remove_duplicates` | `columns` (optional) | Remove duplicate rows |
| `filter_rows` | `column`, `operator`, `value` | Filter rows based on condition |
| `drop_columns` | `columns` | Remove specified columns |
| `convert_types` | `column`, `dtype` (str/int/float/datetime) | Convert column data types |
| `validate_email` | `column`, `drop_invalid` (bool) | Validate email format |
| `outlier_removal` | `column`, `multiplier` (float) | Remove outliers using IQR method |
| `normalize` | `column`, `method` (minmax/zscore) | Normalize numeric columns |
| `submit` | none | Submit solution for grading |
| `revert` | none | Revert last action |

## Observation Space

Each observation contains:
- `dataset_info`: Current dataset metadata (shape, columns, null counts, dtypes)
- `available_actions`: List of valid actions
- `step_count`: Number of steps taken
- `task_id`: Current task identifier
- `message`: Status message
- `done`: Whether the episode is complete

## Task Descriptions

| Task ID | Difficulty | Description | Expected Actions |
|---------|------------|-------------|------------------|
| `easy_001` | Easy | Basic cleaning: drop nulls and remove duplicates from a 100-row dataset | drop_nulls, remove_duplicates |
| `medium_001` | Medium | Intermediate: handle nulls, validate emails, remove outliers from a 200-row dataset | fill_nulls, validate_email, outlier_removal |
| `hard_001` | Hard | Advanced: full pipeline with type conversion and normalization on a 500-row dataset | drop_nulls, fill_nulls, remove_duplicates, validate_email, convert_types, outlier_removal, normalize |

## Grading Criteria

Each task is graded on multiple criteria with weights:

- **easy_001**: null_handling (40%), duplicate_handling (40%), efficiency (20%)
- **medium_001**: null_handling (25%), email_validation (30%), outlier_handling (25%), efficiency (20%)
- **hard_001**: null_handling (15%), duplicate_handling (10%), email_validation (15%), type_conversion (20%), outlier_handling (20%), normalization (10%), efficiency (10%)

## Setup and Usage

### Prerequisites

- Python 3.10+
- Docker (for containerized deployment)

### Local Setup

```bash
# Navigate to the env directory
cd env

# Install dependencies
pip install -r requirements.txt

# Run the server
python app.py
```

The server will start on `http://localhost:7860`.

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web interface |
| `/health` | GET | Health check |
| `/reset` | POST | Initialize a new task |
| `/step` | POST | Execute a cleaning action |
| `/submit` | POST | Submit solution for grading |
| `/revert` | POST | Revert last action |
| `/tasks` | GET | List available tasks |
| `/state` | GET | Get current environment state |
| `/dataset` | GET | Get dataset information |
| `/history` | GET | Get action history |

### Running the Inference Script

```bash
# Set environment variables
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="your-api-key"
export SPACE_URL="http://localhost:7860"

# Run inference
python inference.py
```

### Docker Deployment

```bash
# Build the Docker image
cd env
docker build -t openenv-datacleaner .

# Run the container
docker run -p 7860:7860 openenv-datacleaner
```

### Hugging Face Spaces Deployment

```bash
# Install openenv-core
pip install openenv-core

# Deploy
openenv push ./env
```

## Baseline Scores

| Task | Score | Status |
|------|-------|--------|
| easy_001 | TBD | - |
| medium_001 | TBD | - |
| hard_001 | TBD | - |
| **Average** | **TBD** | - |

*Run `python inference.py` to generate baseline scores.*

## Project Structure

```
AutoClean-AI/
├── inference.py          # Baseline inference script
├── openenv.yaml          # OpenEnv configuration
├── README.md             # This file
├── .dockerignore         # Docker ignore patterns
└── env/                  # Environment package
    ├── __init__.py
    ├── app.py            # FastAPI server
    ├── client.py         # OpenEnv client
    ├── datacleaner_env.py # Main environment
    ├── Dockerfile        # Docker configuration
    ├── grader.py         # Grading system
    ├── inference.py      # HF Spaces entry point
    ├── models.py         # Data models
    ├── openenv.yaml      # OpenEnv config
    ├── pyproject.toml    # Dependencies
    ├── README.md         # Environment docs
    ├── requirements.txt  # Pip requirements
    ├── reward.py         # Reward system
    ├── tasks.py          # Task definitions
    └── static/
        └── index.html    # Web interface
```

## License

MIT License