"""
Test script for the OpenEnv Data Cleaning Grader
"""
import pandas as pd
import numpy as np
from grader import Grader
from tasks import get_task_config

def create_test_dataset(task_id: str) -> pd.DataFrame:
    """Create a test dataset for the given task"""
    config = get_task_config(task_id)
    
    if task_id == "easy_001":
        # Create dataset with some nulls and duplicates
        data = {
            "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "name": ["Alice", "Bob", "Charlie", "David", "Eve", "Frank", "Grace", "Henry", "Ivy", "Jack"],
            "age": [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
            "email": ["a@example.com", "b@example.com", "c@example.com", "d@example.com", "e@example.com",
                     "f@example.com", "g@example.com", "h@example.com", "i@example.com", "j@example.com"],
            "salary": [50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000, 130000, 140000]
        }
        df = pd.DataFrame(data)
        # Add some nulls
        df.loc[2, 'age'] = np.nan
        df.loc[5, 'email'] = np.nan
        # Add a duplicate
        df = pd.concat([df, df.iloc[[0]]]).reset_index(drop=True)
        return df
    
    elif task_id == "medium_001":
        # Create dataset with various issues
        data = {
            "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "name": ["Alice", "Bob", "Charlie", "David", "Eve", "Frank", "Grace", "Henry", "Ivy", "Jack"],
            "age": [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
            "email": ["a@example.com", "b@example.com", "c@example.com", "d@example.com", "e@example.com",
                     "f@example.com", "g@example.com", "h@example.com", "i@example.com", "j@example.com"],
            "salary": [50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000, 130000, 140000],
            "department": ["HR", "IT", "Finance", "IT", "HR", "Finance", "IT", "HR", "Finance", "IT"]
        }
        df = pd.DataFrame(data)
        # Add some nulls
        df.loc[3, 'age'] = np.nan
        df.loc[7, 'email'] = np.nan
        # Add invalid emails
        df.loc[1, 'email'] = "invalid-email"
        df.loc[4, 'email'] = "another-invalid"
        # Add outliers
        df.loc[9, 'salary'] = 500000
        return df
    
    elif task_id == "hard_001":
        # Create dataset with advanced issues
        data = {
            "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "name": ["Alice", "Bob", "Charlie", "David", "Eve", "Frank", "Grace", "Henry", "Ivy", "Jack"],
            "age": [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
            "email": ["a@example.com", "b@example.com", "c@example.com", "d@example.com", "e@example.com",
                     "f@example.com", "g@example.com", "h@example.com", "i@example.com", "j@example.com"],
            "salary": [50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000, 130000, 140000],
            "department": ["HR", "IT", "Finance", "IT", "HR", "Finance", "IT", "HR", "Finance", "IT"],
            "join_date": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01", "2020-05-01",
                                        "2020-06-01", "2020-07-01", "2020-08-01", "2020-09-01", "2020-10-01"]),
            "score": [85, 90, 78, 92, 88, 76, 95, 89, 84, 91]
        }
        df = pd.DataFrame(data)
        # Add various issues
        df.loc[2, 'age'] = np.nan
        df.loc[5, 'email'] = np.nan
        df.loc[8, 'salary'] = np.nan
        # Add invalid emails
        df.loc[1, 'email'] = "invalid-email"
        # Add outliers
        df.loc[9, 'salary'] = 500000
        # Add type issues
        df.loc[3, 'id'] = "three"
        df.loc[6, 'age'] = "fifty-five"
        return df
    
    elif task_id == "employee_demo":
        # Create employee dataset with various issues
        data = {
            "Education": ["Bachelor", "Master", "PhD", "Bachelor", "Master", "PhD", "Bachelor", "Master", "PhD", "Bachelor"],
            "JoiningYear": [2018, 2019, 2020, 2018, 2019, 2020, 2018, 2019, 2020, 2018],
            "City": ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia", "San Antonio", "San Diego", "Dallas", "San Jose"],
            "PaymentTier": ["Gold", "Silver", "Bronze", "Gold", "Silver", "Bronze", "Gold", "Silver", "Bronze", "Gold"],
            "Age": [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
            "Gender": ["Male", "Female", "Male", "Female", "Male", "Female", "Male", "Female", "Male", "Female"],
            "EverBenched": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            "ExperienceInCurrentDomain": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            "LeaveOrNot": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        }
        df = pd.DataFrame(data)
        # Add some nulls
        df.loc[2, 'Age'] = np.nan
        df.loc[5, 'Gender'] = np.nan
        # Add duplicates
        df = pd.concat([df, df.iloc[[0]]]).reset_index(drop=True)
        # Add formatting issues
        df.loc[1, 'Education'] = 'bachelor'
        df.loc[4, 'Gender'] = 'female'
        df.loc[7, 'City'] = 'los angeles'
        # Add outliers
        df.loc[9, 'ExperienceInCurrentDomain'] = 50
        return df
    
    else:
        raise ValueError(f"Unknown task: {task_id}")

def test_grader():
    """Test the grader with all tasks"""
    grader = Grader()
    tasks = ["easy_001", "medium_001", "hard_001", "employee_demo"]
    
    for task_id in tasks:
        print(f"\n{'='*60}")
        print(f"Testing task: {task_id}")
        print(f"{'='*60}")
        
        # Create test dataset
        original_dataset = create_test_dataset(task_id)
        current_dataset = original_dataset.copy()
        
        # Simulate some cleaning actions
        action_history = []
        
        # Grade the solution
        grader.setup(task_id, original_dataset, current_dataset, action_history)
        result = grader.grade()
        
        # Print results
        print(f"\nTask: {task_id}")
        print(f"Final Score: {result.final_score}")
        print(f"\nScore Breakdown:")
        for criterion, score in result.breakdown.items():
            print(f"  {criterion}: {score}")
        print(f"\nFeedback:")
        print(result.feedback)
        
        # Verify score is strictly between 0 and 1
