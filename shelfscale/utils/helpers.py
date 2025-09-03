"""
Helper functions for working with food data
"""

import pandas as pd
import numpy as np
import os
import re
import json
from typing import Dict, List, Optional, Union, Any, Tuple

from shelfscale.data_processing.weight_extraction import WeightExtractor # Import WeightExtractor

# Default WeightExtractor instances for helper functions
# target_unit 'g' is the most common default for weight-related helpers
DEFAULT_WEIGHT_EXTRACTOR = WeightExtractor(target_unit='g')
# For volume conversions, if needed separately, though convert_weight will handle specific target_unit
# DEFAULT_VOL_EXTRACTOR = WeightExtractor(target_unit='ml')


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
    Extracts a numeric value from a string using weight extraction logic.
    
    Returns the numeric value if present; otherwise, returns None. Returns None if the input is not a non-empty string.
    """
    if pd.isna(text) or not isinstance(text, str) or not text.strip(): # Added strip check
        return None
    
    # Use WeightExtractor to extract the numeric value.
    # Note: This is now context-dependent on weight extraction logic.
    # If a general number is needed, this might be too specific.
    # However, in this project, numeric extraction is usually for weights.
    numeric_value, _ = DEFAULT_WEIGHT_EXTRACTOR.extract(text) # extract returns (value, unit)
    
    return numeric_value


def extract_unit(text: str) -> Optional[str]:
    """
    Extracts the standardized unit from a string using the default WeightExtractor.
    
    Returns:
        The extracted unit as a string, standardized to the extractor's target unit if conversion occurred, or None if no unit is found or input is invalid.
    """
    if pd.isna(text) or not isinstance(text, str) or not text.strip():
        return None
        
    # Use WeightExtractor to extract the unit.
    _, unit = DEFAULT_WEIGHT_EXTRACTOR.extract(text) # extract returns (value, unit)
    
    # The unit returned by WeightExtractor.extract is already standardized to the extractor's target_unit
    # if a conversion happened, or it's the original unit if no conversion rule applied or it matched target.
    # The original extract_unit helper just returned the found unit string.
    # If the goal is to identify the *original* unit in the string, this behavior changes.
    # However, WeightExtractor.extract's unit part is the *result* of its processing, which is usually what's desired.
    # For now, we return the unit found by WeightExtractor.
    return unit


def convert_weight(value: float, from_unit: str, to_unit: str) -> Optional[float]:
    """
    Converts a weight value from one unit to another using WeightExtractor.
    
    Attempts to convert the given value from the source unit to the target unit. Returns the converted value if successful. Raises ValueError if the units are unsupported or if conversion between mass and volume is attempted. Returns None if conversion is not possible but the units are otherwise compatible.
    
    Args:
        value: The numeric value to convert.
        from_unit: The unit of the input value.
        to_unit: The desired target unit.
    
    Returns:
        The converted value in the target unit, or None if conversion is not possible.
    
    Raises:
        ValueError: If the source or target unit is unsupported, or if conversion between mass and volume is attempted.
    """
    if value is None or from_unit is None or to_unit is None:
        return None # Or raise error, original raised ValueError for unsupported units

    # Instantiate WeightExtractor with the desired target_unit
    # Ensure to_unit is lowercase as WeightExtractor's internal keys are lowercase.
    try:
        # WeightExtractor's constructor doesn't validate target_unit against its known units,
        # but its standardize_value_and_unit method will handle it.
        extractor = WeightExtractor(target_unit=to_unit.lower())
    except Exception as e: # Catch potential issues if target_unit itself is problematic for WE init
        # This is unlikely as WE init is simple, but good for robustness
        # print(f"Error initializing WeightExtractor for target unit '{to_unit.lower()}': {e}")
        # This function should ideally raise ValueError for unsupported target units like original
        # For now, let standardize_value_and_unit handle it.
        # Fallback or raise: For now, let's see what standardize_value_and_unit does.
        # It will log a warning and return original value/unit if conversion not possible.
        pass


    converted_value, standardized_unit = extractor.standardize_value_and_unit(value, from_unit)

    if standardized_unit == extractor.target_unit: # Check if conversion was to the intended target
        return converted_value
    else:
        # This means conversion was not possible (e.g., incompatible units, unknown from_unit)
        # Original function raised ValueError for unsupported units or incompatible systems.
        # WeightExtractor.standardize_value_and_unit logs a warning and returns original value/unit.
        # To match original behavior more closely for unsupported/incompatible, we might raise error here.
        # For now, returning None to indicate conversion failure to the *specific to_unit*.
        # print(f"Conversion from '{from_unit}' to '{to_unit}' failed or was not applicable. "
        # f"Standardized to: {standardized_unit} with value {converted_value}")
        # Raise error if from_unit or to_unit is not in extractor's known systems or if systems incompatible
        from_unit_lower = from_unit.lower()
        to_unit_lower = to_unit.lower()

        from_is_weight = any(fu_key for fu_key in extractor.weight_units if from_unit_lower.startswith(fu_key))
        from_is_volume = any(fu_key for fu_key in extractor.volume_units if from_unit_lower.startswith(fu_key))
        to_is_weight = any(tu_key for tu_key in extractor.weight_units if to_unit_lower.startswith(tu_key))
        to_is_volume = any(tu_key for tu_key in extractor.volume_units if to_unit_lower.startswith(tu_key))

        if not (from_is_weight or from_is_volume):
            raise ValueError(f"Unsupported source unit: {from_unit}")
        if not (to_is_weight or to_is_volume):
            raise ValueError(f"Unsupported target unit: {to_unit}")
        if (from_is_weight and to_is_volume) or (from_is_volume and to_is_weight):
            raise ValueError(f"Cannot convert between mass ({from_unit}) and volume ({to_unit})")
        
        # If units were compatible but standardize_value_and_unit didn't convert to target_unit,
        # it implies from_unit was not recognized by the specific target_unit based WE instance.
        # This case should ideally be covered by the checks above or means from_unit is truly unknown.
        return None # Fallback for "conversion failed"


def get_unique_values(df: pd.DataFrame, column: str) -> List[Any]:
    """
    Returns a sorted list of unique values from a specified DataFrame column.
    
    Raises:
        ValueError: If the specified column does not exist in the DataFrame.
    
    Returns:
        A sorted list containing the unique values from the given column.
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