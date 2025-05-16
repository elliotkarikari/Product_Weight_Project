"""
Helper functions for working with food data
"""

import pandas as pd
import numpy as np
import os
import re
import json
from typing import Dict, List, Optional, Union, Any, Tuple


class NumPyJSONEncoder(json.JSONEncoder):
    """JSON encoder that can handle NumPy data types"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int_, np.intc, np.intp, np.int8,
                          np.int16, np.int32, np.int64, np.uint8,
                          np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        return super(NumPyJSONEncoder, self).default(obj)


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from various file formats based on extension
    
    Args:
        file_path: Path to the data file
        
    Returns:
        DataFrame with loaded data
    """
    # Get file extension
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    
    # Load based on extension
    if ext == '.csv':
        return pd.read_csv(file_path)
    elif ext in ['.xlsx', '.xls']:
        return pd.read_excel(file_path)
    elif ext == '.json':
        return pd.read_json(file_path)
    elif ext == '.pkl':
        return pd.read_pickle(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")


def save_data(df: pd.DataFrame, file_path: str, index: bool = False) -> None:
    """
    Save DataFrame to various file formats based on extension
    
    Args:
        df: DataFrame to save
        file_path: Path to save the data to
        index: Whether to include index in output
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    
    # Get file extension
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    
    # Save based on extension
    if ext == '.csv':
        df.to_csv(file_path, index=index)
    elif ext == '.xlsx':
        df.to_excel(file_path, index=index)
    elif ext == '.json':
        with open(file_path, 'w') as f:
            json.dump(df.to_dict(orient='records'), f, cls=NumPyJSONEncoder, indent=2)
    elif ext == '.pkl':
        df.to_pickle(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")


def extract_numeric_value(text: str) -> Optional[float]:
    """
    Extract numeric value from a string
    
    Args:
        text: Input string
        
    Returns:
        Extracted numeric value or None if not found
    """
    if pd.isna(text) or not isinstance(text, str):
        return None
    
    # Find numeric patterns in string
    matches = re.findall(r'[-+]?\d*\.\d+|\d+', text)
    
    if matches:
        try:
            # Return the first match as float
            return float(matches[0])
        except (ValueError, TypeError):
            return None
    
    return None


def extract_unit(text: str) -> Optional[str]:
    """
    Extract unit from a string
    
    Args:
        text: Input string
        
    Returns:
        Extracted unit or None if not found
    """
    if pd.isna(text) or not isinstance(text, str):
        return None
    
    # Common units
    units = ['g', 'kg', 'mg', 'l', 'ml', 'oz', 'lb']
    
    # Find unit in string
    for unit in units:
        # Look for unit with space or no space before it
        pattern = r'(\d+[\s]*)' + re.escape(unit) + r'\b'
        if re.search(pattern, text.lower()):
            return unit
    
    return None


def convert_weight(value: float, from_unit: str, to_unit: str) -> float:
    """
    Convert weight from one unit to another
    
    Args:
        value: Weight value to convert
        from_unit: Source unit
        to_unit: Target unit
        
    Returns:
        Converted weight value
    """
    # Conversion factors to grams
    to_gram = {
        'g': 1,
        'kg': 1000,
        'mg': 0.001,
        'oz': 28.35,
        'lb': 453.592
    }
    
    # Conversion factors to milliliters
    to_ml = {
        'ml': 1,
        'l': 1000
    }
    
    # Check if units are valid
    if from_unit not in to_gram and from_unit not in to_ml:
        raise ValueError(f"Unsupported source unit: {from_unit}")
    
    if to_unit not in to_gram and to_unit not in to_ml:
        raise ValueError(f"Unsupported target unit: {to_unit}")
    
    # Check if conversion is within same system (mass or volume)
    if (from_unit in to_gram and to_unit in to_ml) or (from_unit in to_ml and to_unit in to_gram):
        raise ValueError(f"Cannot convert between mass ({from_unit}) and volume ({to_unit})")
    
    # Convert to standard unit (g or ml) then to target unit
    if from_unit in to_gram:
        standard_value = value * to_gram[from_unit]
        return standard_value / to_gram[to_unit]
    else:  # from_unit in to_ml
        standard_value = value * to_ml[from_unit]
        return standard_value / to_ml[to_unit]


def get_unique_values(df: pd.DataFrame, column: str) -> List[Any]:
    """
    Get sorted unique values from a DataFrame column
    
    Args:
        df: Input DataFrame
        column: Column name
        
    Returns:
        List of unique values
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    return sorted(df[column].unique().tolist())


def get_path(file_path: str) -> str:
    """
    Get absolute path for a file, handling both absolute and relative paths
    
    Args:
        file_path: Path to the file (absolute or relative)
        
    Returns:
        Absolute path to the file
    """
    if os.path.isabs(file_path):
        return file_path
    else:
        # If relative, make it relative to the current working directory
        return os.path.abspath(file_path)


def split_data_by_group(df: pd.DataFrame, 
                       group_col: str = 'Food Group') -> Dict[str, pd.DataFrame]:
    """
    Split DataFrame into separate DataFrames by group
    
    Args:
        df: Input DataFrame
        group_col: Column to group by
        
    Returns:
        Dictionary of DataFrames with group names as keys
    """
    if group_col not in df.columns:
        raise ValueError(f"Column '{group_col}' not found in DataFrame")
    
    # Get unique groups
    groups = df[group_col].unique()
    
    # Create dictionary of DataFrames
    result = {}
    for group in groups:
        result[group] = df[df[group_col] == group].copy()
    
    return result 