"""
Data cleaning utilities for ShelfScale datasets
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Optional, Union


def clean_weight_column(df: pd.DataFrame, weight_col: str = 'Weight') -> pd.DataFrame:
    """
    Clean and standardize weight column in a DataFrame
    
    Args:
        df: Input DataFrame
        weight_col: Name of the weight column
        
    Returns:
        DataFrame with cleaned weight column
    """
    # Create a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    if weight_col not in cleaned_df.columns:
        raise ValueError(f"Column '{weight_col}' not found in DataFrame")
    
    # Function to extract numeric value and unit
    def extract_weight_and_unit(weight_str):
        if pd.isna(weight_str) or not isinstance(weight_str, str):
            return np.nan, np.nan
            
        # Remove any non-alphanumeric/space/punctuation characters
        weight_str = re.sub(r'[^\w\s\.\,\-]', '', weight_str.lower())
        
        # Common pattern: number followed by unit (g, kg, ml, l)
        match = re.search(r'(\d+(?:\.\d+)?)\s*(g|kg|ml|l)', weight_str)
        
        if match:
            value = float(match.group(1))
            unit = match.group(2)
            
            # Convert to grams/ml for standardization
            if unit == 'kg':
                value = value * 1000
                unit = 'g'
            elif unit == 'l':
                value = value * 1000
                unit = 'ml'
                
            return value, unit
            
        return np.nan, np.nan
    
    # Apply the extraction function
    weights_and_units = cleaned_df[weight_col].apply(extract_weight_and_unit)
    cleaned_df['Weight_Value'] = weights_and_units.apply(lambda x: x[0])
    cleaned_df['Weight_Unit'] = weights_and_units.apply(lambda x: x[1])
    
    return cleaned_df


def clean_food_groups(df: pd.DataFrame, group_col: str = 'Food Group') -> pd.DataFrame:
    """
    Standardize food group names
    
    Args:
        df: Input DataFrame
        group_col: Name of the food group column
        
    Returns:
        DataFrame with standardized food group names
    """
    # Create a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    if group_col not in cleaned_df.columns:
        raise ValueError(f"Column '{group_col}' not found in DataFrame")
    
    # Mapping of variations to standard names
    group_mapping = {
        'vegetable': 'Vegetables',
        'vegetables': 'Vegetables',
        'fruit': 'Fruit',
        'fruits': 'Fruit',
        'meat': 'Meat and meat products',
        'meat product': 'Meat and meat products',
        'meat products': 'Meat and meat products',
        'cereal': 'Cereals',
        'cereals': 'Cereals',
        'dairy': 'Milk and milk products',
        'milk': 'Milk and milk products',
        'milk product': 'Milk and milk products',
        'milk products': 'Milk and milk products',
        'fish': 'Fish and fish products',
        'fish product': 'Fish and fish products',
        'fish products': 'Fish and fish products',
        'alcoholic': 'Alcoholic beverages',
        'alcoholic beverage': 'Alcoholic beverages',
        'alcohol': 'Alcoholic beverages'
    }
    
    # Apply standardization (case insensitive)
    def standardize_group(group):
        if pd.isna(group) or not isinstance(group, str):
            return group
            
        lower_group = group.lower()
        for key, value in group_mapping.items():
            if key in lower_group:
                return value
        return group
        
    cleaned_df[group_col] = cleaned_df[group_col].apply(standardize_group)
    
    return cleaned_df


def remove_duplicates(df: pd.DataFrame, cols: List[str] = None) -> pd.DataFrame:
    """
    Remove duplicate entries based on specified columns
    
    Args:
        df: Input DataFrame
        cols: Columns to check for duplicates, default is all columns
        
    Returns:
        DataFrame with duplicates removed
    """
    return df.drop_duplicates(subset=cols)


def handle_missing_values(df: pd.DataFrame, strategy: str = 'drop') -> pd.DataFrame:
    """
    Handle missing values in DataFrame
    
    Args:
        df: Input DataFrame
        strategy: Strategy for handling missing values ('drop', 'fill_mean', 'fill_median', 'fill_mode')
        
    Returns:
        DataFrame with missing values handled
    """
    if strategy == 'drop':
        return df.dropna()
        
    elif strategy == 'fill_mean':
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].mean())
        return df
        
    elif strategy == 'fill_median':
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
        return df
        
    elif strategy == 'fill_mode':
        for col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else None)
        return df
        
    else:
        raise ValueError(f"Invalid strategy: {strategy}. Valid options are 'drop', 'fill_mean', 'fill_median', 'fill_mode'") 