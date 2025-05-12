"""
Data validation utilities for ShelfScale datasets
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def validate_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate data quality and generate report
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with validation results
    """
    report = {
        'row_count': len(df),
        'column_count': len(df.columns),
        'missing_values': df.isna().sum().to_dict(),
        'duplicate_count': len(df) - len(df.drop_duplicates()),
        'column_types': {col: str(dtype) for col, dtype in df.dtypes.items()},
        'numeric_stats': {},
        'categorical_stats': {},
        'warnings': []
    }
    
    # Check numeric columns
    for col in df.select_dtypes(include=[np.number]).columns:
        try:
            report['numeric_stats'][col] = {
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'mean': float(df[col].mean()),
                'median': float(df[col].median()),
                'std': float(df[col].std()),
                'missing_pct': float((df[col].isna().sum() / len(df)) * 100)
            }
            
            # Add warnings for outliers
            q1 = float(df[col].quantile(0.25))
            q3 = float(df[col].quantile(0.75))
            iqr = q3 - q1
            lower_bound = q1 - (1.5 * iqr)
            upper_bound = q3 + (1.5 * iqr)
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
            outlier_pct = (len(outliers) / len(df)) * 100
            
            if outlier_pct > 5:
                report['warnings'].append(f"Column '{col}' has {outlier_pct:.2f}% outlier values")
        except Exception as e:
            report['warnings'].append(f"Error analyzing column '{col}': {str(e)}")
    
    # Check categorical columns
    for col in df.select_dtypes(include=['object']).columns:
        try:
            value_counts = df[col].value_counts()
            report['categorical_stats'][col] = {
                'unique_count': len(value_counts),
                'top_values': value_counts.head(5).to_dict(),
                'missing_pct': float((df[col].isna().sum() / len(df)) * 100)
            }
            
            # Add warnings for highly imbalanced categories
            if len(value_counts) > 1:
                most_common = value_counts.iloc[0]
                total = len(df)
                if (most_common / total) > 0.9:
                    report['warnings'].append(
                        f"Column '{col}' has imbalanced values: {value_counts.index[0]} appears in {(most_common/total)*100:.2f}% of rows"
                    )
        except Exception as e:
            report['warnings'].append(f"Error analyzing column '{col}': {str(e)}")
    
    return report