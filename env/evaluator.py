import pandas as pd

try:
    from .utils.validators import DataValidators
except ImportError:  # pragma: no cover - direct execution fallback
    from utils.validators import DataValidators


def evaluate_cleanliness(df: pd.DataFrame, schema: dict = None) -> dict:
    """
    Evaluate dataset cleanliness and return comprehensive score (0-1)
    This is the official evaluation function for OpenEnv
    """
    validators = DataValidators()
    
    missing_ratio = validators.missing_values_ratio(df)
    duplicate_ratio = validators.duplicate_rows_ratio(df)
    type_consistency = validators.data_type_consistency(df)
    outlier_ratio = validators.outlier_ratio(df)
    
    if schema:
        schema_validity = validators.schema_validity(df, schema)
    else:
        schema_validity = 0.9999
    
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
        'score': min(0.9999, max(0.0001, round(float(total_score), 4))),
        'components': {
            'missing_values': round(float(1 - missing_ratio), 4),
            'duplicates': round(float(1 - duplicate_ratio), 4),
            'type_consistency': round(float(type_consistency), 4),
            'outliers': round(float(1 - outlier_ratio), 4),
            'schema_validity': round(float(schema_validity), 4)
        },
        'raw_metrics': {
            'missing_ratio': round(float(missing_ratio), 4),
            'duplicate_ratio': round(float(duplicate_ratio), 4),
            'type_consistency': round(float(type_consistency), 4),
            'outlier_ratio': round(float(outlier_ratio), 4),
            'schema_validity': round(float(schema_validity), 4)
        },
        'stats': {
            'rows': len(df),
            'columns': len(df.columns),
            'total_cells': df.shape[0] * df.shape[1]
        }
    }
