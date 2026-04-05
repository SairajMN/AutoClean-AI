import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder


class DataTransformers:
    def normalize_column(self, df: pd.DataFrame, column: str, method: str = 'min-max') -> pd.DataFrame:
        """Normalize numeric column using specified method"""
        df = df.copy()
        
        if method == 'min-max':
            scaler = MinMaxScaler()
            df[column] = scaler.fit_transform(df[[column]])
        elif method == 'standard':
            scaler = StandardScaler()
            df[column] = scaler.fit_transform(df[[column]])
            
        return df

    def fix_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Automatically detect and fix incorrect data types"""
        df = df.copy()
        
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_numeric(df[col])
                    continue
                except:
                    pass
                try:
                    df[col] = pd.to_datetime(df[col])
                    continue
                except:
                    pass
        return df

    def encode_categorical(self, df: pd.DataFrame, column: str, method: str = 'label') -> pd.DataFrame:
        """Encode categorical columns"""
        df = df.copy()
        
        if method == 'label':
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column].astype(str))
        elif method == 'onehot':
            dummies = pd.get_dummies(df[column], prefix=column, drop_first=True)
            df = pd.concat([df.drop(columns=[column]), dummies], axis=1)
            
        return df