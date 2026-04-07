"""
Employee Dataset Demo Task
Real HR Employee Attrition Dataset
"""

import pandas as pd
import numpy as np
from typing import Dict, Any

TASK_ID = "employee_demo"
DIFFICULTY = "demo"
DESCRIPTION = "Real employee dataset. Practice cleaning real HR data with missing values, formatting issues and duplicates."


def generate_dataset() -> pd.DataFrame:
    """Load and return the employee dataset with intentional errors added for cleaning practice"""
    # Load original dataset
    df = pd.read_csv("/Users/sahil/Downloads/Employee.csv")
    
    # Add realistic cleaning challenges
    df = df.copy()
    
    # 1. Add 5% missing values randomly
    mask = np.random.random(df.shape) < 0.05
    df[mask] = np.nan
    
    # 2. Add duplicate rows
    duplicates = df.sample(n=25).copy()
    df = pd.concat([df, duplicates], ignore_index=True)
    
    # 3. Mess up case in Education column
    df['Education'] = df['Education'].apply(
        lambda x: x.lower() if isinstance(x, str) and np.random.random() < 0.3 else x
    )
    
    # 4. Add some invalid ages
    invalid_idx = np.random.choice(df.index, 12)
    df.loc[invalid_idx, 'Age'] = np.random.randint(100, 150, size=12)
    
    # 5. Mess up gender values
    df['Gender'] = df['Gender'].apply(
        lambda x: x.upper() if isinstance(x, str) and np.random.random() < 0.25 else x
    )
    
    return df


def get_task_config() -> Dict[str, Any]:
    return {
        "task_id": TASK_ID,
        "difficulty": DIFFICULTY,
        "description": DESCRIPTION,
        "expected_actions": [
            "fill_missing",
            "drop_duplicates",
            "standardize_format",
            "detect_outliers"
        ],
        "max_steps": 12,
        "target_score": 0.85
    }