"""
OpenEnv Data Cleaning Environment - Task Definitions
Defines tasks with dataset configurations and grading criteria.
"""

from typing import List, Dict, Any

from models import TaskConfig


def get_tasks() -> List[str]:
    """
    Get list of all available task IDs.
    OpenEnv-compatible task registration.
    """
    return [
        "easy_001",
        "medium_001",
        "hard_001",
        "employee_demo"
    ]


def get_task_config(task_id: str) -> TaskConfig:
    """
    Get configuration for a specific task.
    """
    task_configs = {
        "easy_001": TaskConfig(
            task_id="easy_001",
            difficulty="easy",
            description="Basic data cleaning: drop nulls and remove duplicates",
            dataset_config={
                "rows": 100,
                "null_percentage": 0.1,
                "duplicate_count": 5,
                "columns": ["id", "name", "age", "email", "salary"]
            },
            expected_actions=["drop_nulls", "remove_duplicates"],
            grading_criteria={
                "null_handling": 0.4,
                "duplicate_handling": 0.4,
                "efficiency": 0.2
            }
        ),
        "medium_001": TaskConfig(
            task_id="medium_001",
            difficulty="medium",
            description="Intermediate cleaning: handle nulls, validate emails, remove outliers",
            dataset_config={
                "rows": 200,
                "null_percentage": 0.15,
                "duplicate_count": 10,
                "invalid_email_count": 20,
                "outlier_count": 8,
                "columns": ["id", "name", "age", "email", "salary", "department"]
            },
            expected_actions=["fill_nulls", "validate_email", "outlier_removal"],
            grading_criteria={
                "null_handling": 0.25,
                "email_validation": 0.3,
                "outlier_handling": 0.25,
                "efficiency": 0.2
            }
        ),
        "hard_001": TaskConfig(
            task_id="hard_001",
            difficulty="hard",
            description="Advanced cleaning: full pipeline with type conversion and normalization",
            dataset_config={
                "rows": 500,
                "null_percentage": 0.2,
                "duplicate_count": 25,
                "invalid_email_count": 40,
                "outlier_count": 20,
                "type_issues": True,
                "columns": ["id", "name", "age", "email", "salary", "department", "join_date", "score"]
            },
            expected_actions=[
                "drop_nulls",
                "fill_nulls",
                "remove_duplicates",
                "validate_email",
                "convert_types",
                "outlier_removal",
                "normalize"
            ],
            grading_criteria={
                "null_handling": 0.15,
                "duplicate_handling": 0.1,
                "email_validation": 0.15,
                "type_conversion": 0.2,
                "outlier_handling": 0.2,
                "normalization": 0.1,
                "efficiency": 0.1
            }
        ),
        "employee_demo": TaskConfig(
            task_id="employee_demo",
            difficulty="demo",
            description="Real employee dataset. Practice cleaning real HR data with missing values, formatting issues and duplicates.",
            dataset_config={
                "source": "employee",
                "columns": ["Education", "JoiningYear", "City", "PaymentTier", "Age", "Gender", "EverBenched", "ExperienceInCurrentDomain", "LeaveOrNot"]
            },
            expected_actions=[
                "fill_nulls",
                "remove_duplicates",
                "standardize_format",
                "detect_outliers"
            ],
            grading_criteria={
                "null_handling": 0.25,
                "duplicate_handling": 0.25,
                "format_standardization": 0.25,
                "outlier_handling": 0.25
            }
        )
    }

    if task_id not in task_configs:
        raise ValueError(f"Unknown task: {task_id}. Available: {get_tasks()}")

    return task_configs[task_id]


def get_all_task_configs() -> Dict[str, TaskConfig]:
    """Get all task configurations."""
    return {task_id: get_task_config(task_id) for task_id in get_tasks()}