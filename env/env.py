import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import json
from datetime import datetime
from utils.cleaners import DataCleaners
from utils.validators import DataValidators
from utils.transformers import DataTransformers


class AutoCleanEnv:
    def __init__(self):
        self.state = None
        self.raw_dataset = None
        self.history = []
        self.current_step = 0
        self.max_steps = 50
        self.reward = 0.0
        self.versions = {}
        self.schema = None
        self.dirty_metrics = None
        self.cleaners = DataCleaners()
        self.validators = DataValidators()
        self.transformers = DataTransformers()

    def reset(self, dataset: pd.DataFrame = None) -> Dict[str, Any]:
        """Reset environment to initial state"""
        self.current_step = 0
        self.history = []
        self.reward = 0.0
        self.versions = {}
        
        if dataset is not None:
            self.raw_dataset = dataset.copy()
            self.state = dataset.copy()
        
        self.versions['v0_raw'] = self.state.copy()
        self.schema = self._detect_schema(self.state)
        self.dirty_metrics = self._calculate_metrics(self.state)
        
        return self._get_observation()

    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Execute an action and advance environment state"""
        self.current_step += 1
        before_state = self.state.copy()
        
        action_type = action.get('type')
        params = action.get('params', {})
        
        try:
            self.state = self._execute_action(action_type, params, self.state)
            success = True
            explanation = self._get_action_explanation(action_type, params)
        except Exception as e:
            success = False
            explanation = f"Action failed: {str(e)}"
            
        metrics_after = self._calculate_metrics(self.state)
        self.reward = metrics_after['total_score']
        
        self.versions[f'v{self.current_step}'] = self.state.copy()
        
        diff = self._generate_diff(before_state, self.state)
        
        history_entry = {
            'step': self.current_step,
            'action': action,
            'explanation': explanation,
            'success': success,
            'metrics_before': self.history[-1]['metrics_after'] if self.history else self.dirty_metrics,
            'metrics_after': metrics_after,
            'diff': diff,
            'timestamp': datetime.now().isoformat()
        }
        self.history.append(history_entry)
        
        done = self.reward >= 0.95 or self.current_step >= self.max_steps
        
        return self._get_observation(), self.reward, done, history_entry

    def _execute_action(self, action_type: str, params: Dict, df: pd.DataFrame) -> pd.DataFrame:
        """Execute specified cleaning action"""
        actions = {
            'fill_missing': self.cleaners.fill_missing_values,
            'remove_duplicates': self.cleaners.remove_duplicates,
            'normalize': self.transformers.normalize_column,
            'fix_types': self.transformers.fix_data_types,
            'remove_outliers': self.cleaners.remove_outliers,
            'drop_column': self.cleaners.drop_column,
            'encode_categorical': self.transformers.encode_categorical,
            'handle_text': self.cleaners.clean_text_column
        }
        
        if action_type not in actions:
            raise ValueError(f"Unknown action type: {action_type}")
            
        return actions[action_type](df, **params)

    def _calculate_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate all cleanliness metrics"""
        missing_ratio = self.validators.missing_values_ratio(df)
        duplicate_ratio = self.validators.duplicate_rows_ratio(df)
        type_consistency = self.validators.data_type_consistency(df)
        outlier_ratio = self.validators.outlier_ratio(df)
        schema_validity = self.validators.schema_validity(df, self.schema)
        
        weights = {
            'missing': 0.35,
            'duplicates': 0.25,
            'types': 0.20,
            'outliers': 0.15,
            'schema': 0.05
        }
        
        total_score = (
            (1 - missing_ratio) * weights['missing'] +
            (1 - duplicate_ratio) * weights['duplicates'] +
            type_consistency * weights['types'] +
            (1 - outlier_ratio) * weights['outliers'] +
            schema_validity * weights['schema']
        )
        
        return {
            'total_score': round(float(total_score), 4),
            'missing_ratio': round(float(missing_ratio), 4),
            'duplicate_ratio': round(float(duplicate_ratio), 4),
            'type_consistency': round(float(type_consistency), 4),
            'outlier_ratio': round(float(outlier_ratio), 4),
            'schema_validity': round(float(schema_validity), 4),
            'rows': len(df),
            'columns': len(df.columns)
        }

    def _detect_schema(self, df: pd.DataFrame) -> Dict[str, str]:
        """Auto-detect column types and schema"""
        schema = {}
        for col in df.columns:
            dtype = str(df[col].dtype)
            if 'int' in dtype or 'float' in dtype:
                schema[col] = 'numeric'
            elif 'datetime' in dtype:
                schema[col] = 'datetime'
            elif dtype == 'object':
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio < 0.1:
                    schema[col] = 'categorical'
                else:
                    schema[col] = 'text'
            else:
                schema[col] = 'unknown'
        return schema

    def _generate_diff(self, before: pd.DataFrame, after: pd.DataFrame) -> Dict[str, Any]:
        """Generate difference report between dataset versions - OPTIMIZED"""
        rows_changed = len(after) - len(before)
        columns_removed = list(set(before.columns) - set(after.columns))
        columns_added = list(set(after.columns) - set(before.columns))
        
        values_modified = 0
        # Fast path: only compute values_modified if shapes match exactly
        if before.shape == after.shape:
            try:
                values_modified = int((before.values != after.values).sum())
            except:
                values_modified = 0
        
        return {
            'rows_changed': rows_changed,
            'values_modified': values_modified,
            'columns_removed': columns_removed,
            'columns_added': columns_added
        }
