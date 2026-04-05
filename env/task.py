import pandas as pd
import numpy as np
from typing import Dict, Any


def generate_task(dataset_size: int = 1000, dirt_level: float = 0.3) -> pd.DataFrame:
    """
    Generate a dirty dataset for the AutoClean AI task
    Contains: missing values, duplicates, inconsistent types, outliers, messy text
    """
    np.random.seed(42)
    
    data = {
        'id': np.arange(dataset_size),
        'age': np.random.normal(35, 12, dataset_size).astype(int),
        'income': np.random.lognormal(10, 1, dataset_size).astype(int),
        'gender': np.random.choice(['Male', 'Female', 'male', 'female', 'M', 'F', None], dataset_size, 
                                  p=[0.3, 0.3, 0.1, 0.1, 0.05, 0.05, 0.1]),
        'join_date': pd.date_range('2020-01-01', periods=dataset_size).tolist(),
        'score': np.random.normal(50, 15, dataset_size),
        'comments': np.random.choice(['Good', 'Excellent', 'Bad', 'Average', ' ', None, '  '], dataset_size),
        'category': np.random.choice(['A', 'B', 'C', 'D', None], dataset_size, p=[0.25, 0.25, 0.25, 0.2, 0.05])
    }
    
    df = pd.DataFrame(data)
    
    # Add missing values
    mask = np.random.choice([True, False], size=df.shape, p=[dirt_level * 0.4, 1 - dirt_level * 0.4])
    df = df.mask(mask)
    
    # Add duplicates
    duplicates = df.sample(frac=dirt_level * 0.25, random_state=42)
    df = pd.concat([df, duplicates], ignore_index=True)
    
    # Add outliers
    numeric_cols = ['age', 'income', 'score']
    for col in numeric_cols:
        outliers_idx = np.random.choice(df.index, size=int(dataset_size * dirt_level * 0.1), replace=False)
        df.loc[outliers_idx, col] = df[col].mean() * 10
    
    # Mess up data types
    df['age'] = df['age'].apply(lambda x: str(x) if np.random.random() < 0.1 else x)
    df['income'] = df['income'].apply(lambda x: f"${x}" if np.random.random() < 0.15 else x)
    
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


def get_task_description() -> Dict[str, Any]:
    return {
        "name": "AutoClean AI Data Cleaning Challenge",
        "goal": "Maximize the dataset cleanliness score by applying optimal cleaning operations",
        "success_threshold": 0.95,
        "max_steps": 50,
        "allowed_actions": [
            "fill_missing",
            "remove_duplicates", 
            "normalize",
            "fix_types",
            "remove_outliers",
            "drop_column",
            "encode_categorical",
            "handle_text"
        ]
    }