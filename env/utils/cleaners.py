import pandas as pd
import numpy as np
import re
from typing import Optional, Union


class DataCleaners:
    def fill_missing_values(self, df: pd.DataFrame, column: str = None, strategy: str = 'auto') -> pd.DataFrame:
        """Fill missing values using specified strategy"""
        df = df.copy()
        
        if column is None:
            columns = df.columns
        else:
            columns = [column]
            
        for col in columns:
            if df[col].isna().sum() == 0:
                continue
                
            if strategy == 'auto':
                if pd.api.types.is_numeric_dtype(df[col]):
                    strategy = 'median'
                elif pd.api.types.is_datetime64_any_dtype(df[col]):
                    strategy = 'ffill'
                else:
                    strategy = 'mode'
            
            if strategy == 'mean':
                df[col] = df[col].fillna(df[col].mean())
            elif strategy == 'median':
                df[col] = df[col].fillna(df[col].median())
            elif strategy == 'mode':
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else '')
            elif strategy == 'ffill':
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
            elif strategy == 'zero':
                df[col] = df[col].fillna(0)
            elif strategy == 'empty':
                df[col] = df[col].fillna('')
                
        return df

    def remove_duplicates(self, df: pd.DataFrame, subset: list = None) -> pd.DataFrame:
        """Remove duplicate rows"""
        return df.drop_duplicates(subset=subset, keep='first').reset_index(drop=True)

    def remove_outliers(self, df: pd.DataFrame, column: str, method: str = 'IQR', threshold: float = 1.5) -> pd.DataFrame:
        """Remove outliers from numeric column"""
        df = df.copy()
        
        if method == 'IQR':
            q1 = df[column].quantile(0.25)
            q3 = df[column].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
            
        elif method == 'zscore':
            from scipy import stats
            z_scores = np.abs(stats.zscore(df[column].dropna()))
            df = df[z_scores < threshold]
            
        return df.reset_index(drop=True)

    def drop_column(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Drop specified column"""
        return df.drop(columns=[column], errors='ignore')

    def clean_text_column(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Clean text values: trim, remove special chars, standardize case"""
        df = df.copy()
        
        def clean_text(text):
            if pd.isna(text):
                return text
            text = str(text).strip()
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'[^\w\s\-.,@]', '', text)
            return text
            
        df[column] = df[column].apply(clean_text)
        return df