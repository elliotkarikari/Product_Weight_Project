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
    Create a summary DataFrame with statistics by food group
    
    Args:
        df: Input DataFrame
        group_col: Name of the food group column
        weight_col: Name of the weight column
        
    Returns:
        DataFrame with food group statistics
    """
    if group_col not in df.columns:
        raise ValueError(f"Column '{group_col}' not found in DataFrame")
        
    if weight_col not in df.columns:
        raise ValueError(f"Column '{weight_col}' not found in DataFrame")
    
    # Remove rows with missing values in key columns
    filtered_df = df.dropna(subset=[group_col, weight_col])
    
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
    Normalize weights to a standard unit
    
    Args:
        df: Input DataFrame
        weight_col: Name of the weight column
        unit_col: Name of the unit column
        target_unit: Target unit for normalization ('g' or 'ml')
        
    Returns:
        DataFrame with normalized weights
    """
    if weight_col not in df.columns:
        raise ValueError(f"Column '{weight_col}' not found in DataFrame")
        
    if unit_col not in df.columns:
        raise ValueError(f"Column '{unit_col}' not found in DataFrame")
    
    # Create a copy to avoid modifying the original
    normalized_df = df.copy()
    
    # Define conversion factors
    conversion_factors = {
        'kg': 1000,  # 1 kg = 1000 g
        'g': 1,      # 1 g = 1 g
        'l': 1000,   # 1 l = 1000 ml
        'ml': 1      # 1 ml = 1 ml
    }
    
    # Determine the target system (mass or volume)
    is_mass_target = target_unit in ['g', 'kg']
    
    # Function to convert and normalize weights
    def normalize_weight(row):
        if pd.isna(row[weight_col]) or pd.isna(row[unit_col]):
            return np.nan
        
        unit = row[unit_col].lower()
        
        # Skip if units are incompatible (mass vs. volume)
        if (is_mass_target and unit in ['ml', 'l']) or (not is_mass_target and unit in ['g', 'kg']):
            return np.nan
            
        # Convert to target unit
        if unit in conversion_factors:
            return row[weight_col] * conversion_factors[unit]
        
        return np.nan
    
    # Apply normalization
    normalized_df['Normalized_Weight'] = normalized_df.apply(normalize_weight, axis=1)
    normalized_df['Normalized_Unit'] = target_unit
    
    return normalized_df


def pivot_food_groups(df: pd.DataFrame,
                     group_col: str = 'Food Group',
                     weight_col: str = 'Normalized_Weight') -> pd.DataFrame:
    """
    Create a pivot table with food groups and their total weights
    
    Args:
        df: Input DataFrame
        group_col: Name of the food group column
        weight_col: Name of the weight column
        
    Returns:
        Pivot table with food group weights
    """
    if group_col not in df.columns:
        raise ValueError(f"Column '{group_col}' not found in DataFrame")
        
    if weight_col not in df.columns:
        raise ValueError(f"Column '{weight_col}' not found in DataFrame")
    
    # Remove rows with missing values in key columns
    filtered_df = df.dropna(subset=[group_col, weight_col])
    
    # Create pivot table with sum of weights by food group
    pivot = pd.pivot_table(
        filtered_df,
        values=weight_col,
        index=None,
        columns=group_col,
        aggfunc='sum'
    ).fillna(0)
    
    return pivot 