"""
Data transformation utilities for ShelfScale datasets
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union


def create_food_group_summary(df: pd.DataFrame, 
                             group_col: str = 'Food Group',
                             weight_col: str = 'Weight_Value') -> pd.DataFrame:
    """
                             Generates a summary DataFrame with statistical measures of weights grouped by a specified food group.
                             
                             Args:
                                 df: Input DataFrame containing food group and weight data.
                                 group_col: Column name representing the food group.
                                 weight_col: Column name representing the weight values.
                             
                             Returns:
                                 A DataFrame with count, mean, median, standard deviation, minimum, and maximum of weights for each food group.
                             
                             Raises:
                                 ValueError: If the specified group or weight column is not present in the DataFrame.
                             """
    logger = logging.getLogger(__name__) # Ensure logger is available

    if group_col not in df.columns:
        msg = f"Group column '{group_col}' not found in DataFrame for food group summary."
        logger.error(msg)
        raise ValueError(msg)
        
    if weight_col not in df.columns:
        msg = f"Weight column '{weight_col}' not found in DataFrame for food group summary."
        logger.error(msg)
        raise ValueError(msg)
    
    # Remove rows with missing values in key columns
    filtered_df = df.dropna(subset=[group_col, weight_col])

    if filtered_df.empty:
        logger.warning(f"DataFrame is empty after dropping NaNs in '{group_col}' or '{weight_col}'. Returning empty summary.")
        # Return empty DataFrame with expected columns for consistency
        return pd.DataFrame(columns=[group_col, 'count', 'mean', 'median', 'std', 'min', 'max'])
    
    logger.info(f"Creating food group summary for column '{group_col}' using weight column '{weight_col}'.")
    # Group by food group and calculate statistics
    summary = filtered_df.groupby(group_col)[weight_col].agg([
        ('count', 'count'),
        ('mean', 'mean'),
        ('median', 'median'),
        ('std', 'std'),
        ('min', 'min'),
        ('max', 'max')
    ]).reset_index()
    
    return summary


def normalize_weights(df: pd.DataFrame, 
                     weight_col: str = 'Weight_Value', 
                     unit_col: str = 'Weight_Unit',
                     target_unit: str = 'g') -> pd.DataFrame:
    """
                     Normalizes weight values in a DataFrame to a specified target unit.
                     
                     Each row's weight and unit are converted to the target unit (default 'g'). If conversion is successful, the normalized value and unit are stored in 'Normalized_Weight' and 'Normalized_Unit' columns; otherwise, NaN and the original unit are recorded. Returns a DataFrame with the added or updated normalized columns.
                     """
    if weight_col not in df.columns:
        raise ValueError(f"Weight column '{weight_col}' not found in DataFrame.")
    if unit_col not in df.columns:
        raise ValueError(f"Unit column '{unit_col}' not found in DataFrame.")

    from shelfscale.data_processing.weight_extraction import WeightExtractor # Local import

    extractor = WeightExtractor(target_unit=target_unit)
    
    # Create a copy to avoid modifying the original DataFrame
    normalized_df = df.copy()

    # Initialize new columns if they don't exist
    if 'Normalized_Weight' not in normalized_df.columns:
        normalized_df['Normalized_Weight'] = np.nan
    if 'Normalized_Unit' not in normalized_df.columns:
        normalized_df['Normalized_Unit'] = None

    # Iterate over rows and apply normalization
    for index, row in normalized_df.iterrows():
        original_value = row[weight_col]
        original_unit = row[unit_col]

        if pd.isna(original_value) or pd.isna(original_unit):
            normalized_df.loc[index, 'Normalized_Weight'] = np.nan
            normalized_df.loc[index, 'Normalized_Unit'] = None # Or keep original unit if preferred for NaNs
            continue

        converted_value, standardized_unit = extractor.standardize_value_and_unit(original_value, str(original_unit))
        
        if standardized_unit == extractor.target_unit and converted_value is not None:
            normalized_df.loc[index, 'Normalized_Weight'] = converted_value
            normalized_df.loc[index, 'Normalized_Unit'] = standardized_unit
        else:
            # If conversion was not possible or unit is not the target unit, store NaN or original value.
            # Storing NaN for weight and None for unit indicates failed/inapplicable normalization.
            normalized_df.loc[index, 'Normalized_Weight'] = np.nan 
            normalized_df.loc[index, 'Normalized_Unit'] = original_unit # Keep original unit to show what it was
            # logger.warning(f"Could not normalize weight for row {index}: value {original_value} {original_unit} to target {target_unit}. "
            # f"Got {converted_value} {standardized_unit}")


    return normalized_df


def pivot_food_groups(df: pd.DataFrame,
                     group_col: str = 'Food Group',
                     weight_col: str = 'Normalized_Weight') -> pd.DataFrame:
    """
                     Creates a pivot table summarizing total weights for each food group.
                     
                     Args:
                         df: DataFrame containing food group and weight data.
                         group_col: Column name representing food groups.
                         weight_col: Column name representing weights to aggregate.
                     
                     Returns:
                         A DataFrame where each column corresponds to a food group and the values are the sum of weights for that group. Returns an empty DataFrame if no valid data is available.
                     """
    logger = logging.getLogger(__name__) # Ensure logger is available

    if group_col not in df.columns:
        msg = f"Group column '{group_col}' not found in DataFrame for pivot table."
        logger.error(msg)
        raise ValueError(msg)
        
    if weight_col not in df.columns:
        msg = f"Weight column '{weight_col}' not found in DataFrame for pivot table."
        logger.error(msg)
        raise ValueError(msg)
    
    # Remove rows with missing values in key columns
    filtered_df = df.dropna(subset=[group_col, weight_col])

    if filtered_df.empty:
        logger.warning(f"DataFrame is empty after dropping NaNs in '{group_col}' or '{weight_col}'. Returning empty pivot table.")
        return pd.DataFrame() # Return empty DataFrame

    logger.info(f"Creating pivot table for food groups in column '{group_col}' using weight column '{weight_col}'.")
    # Create pivot table with sum of weights by food group
    pivot = pd.pivot_table(
        filtered_df,
        values=weight_col,
        index=None,
        columns=group_col,
        aggfunc='sum'
    ).fillna(0)
    
    return pivot 