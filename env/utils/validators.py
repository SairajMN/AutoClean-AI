import pandas as pd
import numpy as np
from typing import Dict, Any
from sklearn.ensemble import IsolationForest


class DataValidators:
    def missing_values_ratio(self, df: pd.DataFrame) -> float:
        """Calculate ratio of missing values in dataset"""
        return float(df.isna().sum().sum() / (df.shape[0] * df.shape[1]))

    def duplicate_rows_ratio(self, df: pd.DataFrame) -> float:
        """Calculate ratio of duplicate rows"""
        return float(df.duplicated().sum() / len(df))

    def data_type_consistency(self, df: pd.DataFrame) -> float:
        """Score how consistent data types are per column"""
        consistency_score = 0.0
        for col in df.columns:
            non_null = df[col].dropna()
            if len(non_null) == 0:
                continue
            types = non_null.apply(type).value_counts(normalize=True)
            consistency_score += types.iloc[0]
        return consistency_score / len(df.columns)

    def outlier_ratio(self, df: pd.DataFrame) -> float:
        """Calculate ratio of outliers across all numeric columns"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return 0.0
            
        total_outliers = 0
        total_values = 0
        
        for col in numeric_cols:
            data = df[col].dropna().values.reshape(-1, 1)
            if len(data) < 10:
                continue
            try:
                iso = IsolationForest(contamination=0.1, random_state=42)
                outliers = iso.fit_predict(data)
                total_outliers += sum(outliers == -1)
                total_values += len(data)
            except:
                continue
                
        return total_outliers / total_values if total_values > 0 else 0.0

    def schema_validity(self, df: pd.DataFrame, schema: Dict[str, str]) -> float:
        """Validate dataset against expected schema"""
        valid = 0
        for col, expected_type in schema.items():
            if col not in df.columns:
                continue
            current_type = self._get_column_type(df[col])
            if current_type == expected_type:
                valid += 1
        return valid / len(schema) if schema else 1.0

    def _get_column_type(self, series: pd.Series) -> str:
        dtype = str(series.dtype)
        if 'int' in dtype or 'float' in dtype:
            return 'numeric'
        elif 'datetime' in dtype:
            return 'datetime'
        elif dtype == 'object':
            unique_ratio = series.nunique() / len(series.dropna())
            if unique_ratio < 0.1:
                return 'categorical'
            else:
                return 'text'
        return 'unknown'