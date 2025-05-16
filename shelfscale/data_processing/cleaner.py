"""
Comprehensive data cleaning pipeline for ShelfScale datasets
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, Any, List, Optional, Union
import logging

# Import from other modules
from .weight_extraction import WeightExtractor
from .categorization import FoodCategorizer
from .validation import validate_data

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataCleaner:
    """Data cleaning and preprocessing for ShelfScale datasets"""
    
    def __init__(self, 
                standardize_columns: bool = True,
                handle_duplicates: bool = True,
                missing_strategy: str = 'drop',
                categorize_foods: bool = True,
                categories_path: str = None):
        """
        Initialize data cleaner
        
        Args:
            standardize_columns: Whether to standardize column names
            handle_duplicates: Whether to remove duplicate rows
            missing_strategy: How to handle missing values ('drop', 'fill_zeros', 'fill_mean')
            categorize_foods: Whether to categorize food items
            categories_path: Path to food categories file
        """
        self.standardize_columns = standardize_columns
        self.handle_duplicates = handle_duplicates
        self.missing_strategy = missing_strategy
        self.categorize_foods = categorize_foods
        
        # Initialize food categorizer
        self.categorizer = FoodCategorizer(
            food_categories_path=categories_path
        )
        
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess a DataFrame
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        if df is None or len(df) == 0:
            logger.warning("Empty DataFrame provided for cleaning")
            return pd.DataFrame()
        
        logger.info(f"Starting cleaning pipeline on DataFrame with {len(df)} rows")
        
        # Make a copy to avoid modifying the original
        cleaned_df = df.copy()
        
        # Track changes
        changes = {}
        
        # Standardize column names
        if self.standardize_columns:
            cleaned_df = self._standardize_column_names(cleaned_df)
            
        logger.info(f"Columns after standardization: {cleaned_df.columns.tolist()}")
        
        # Remove duplicate rows
        if self.handle_duplicates:
            logger.info("Removing duplicate rows")
            initial_rows = len(cleaned_df)
            cleaned_df = cleaned_df.drop_duplicates()
            changes['duplicates_removed'] = initial_rows - len(cleaned_df)
        
        # Handle missing values
        logger.info(f"Handling missing values with strategy: {self.missing_strategy}")
        changes['missing_handled'] = self._handle_missing_values(cleaned_df)
        
        # Categorize food names if requested
        if self.categorize_foods:
            food_name_col = self._find_food_name_column(cleaned_df)
            if food_name_col:
                food_category_col = 'Food_Category'
                super_category_col = 'Super_Category'
                
                cleaned_df = self.categorizer.clean_food_categories(
                    cleaned_df, 
                    food_name_col, 
                    food_category_col, 
                    super_category_col
                )
                
                changes['categorized'] = len(cleaned_df)
        
        logger.info(f"Cleaning complete. Changes made: {changes}")
        logger.info(f"Rows after cleaning: {len(cleaned_df)}")
        
        return cleaned_df
    
    def _standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names for consistency
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with standardized column names
        """
        # Create a copy
        standardized_df = df.copy()
        
        # Define column name mappings
        mapping = {}
        for col in standardized_df.columns:
            col_lower = col.lower()
            
            # Food name
            if "food" in col_lower and "name" in col_lower:
                mapping[col] = "Food_Name"
            elif "product" in col_lower and "name" in col_lower:
                mapping[col] = "Food_Name"
            elif "name" in col_lower and not any(x in col_lower for x in ["brand", "company", "manufacturer"]):
                mapping[col] = "Food_Name"
                
            # Weight
            elif "weight" in col_lower and "g" in col_lower:
                mapping[col] = "Weight_g"
            elif "weight" in col_lower and not any(x in col_lower for x in ["volume", "height"]):
                mapping[col] = "Weight_g"
                
            # Food categories
            elif "category" in col_lower:
                if "super" in col_lower or "parent" in col_lower:
                    mapping[col] = "Super_Category"
                else:
                    mapping[col] = "Food_Category"
                    
            # Food groups
            elif "group" in col_lower and not "super" in col_lower:
                mapping[col] = "Food_Group"
        
        # Apply mappings
        if mapping:
            standardized_df = standardized_df.rename(columns=mapping)
            
        return standardized_df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> float:
        """
        Handle missing values based on strategy
        
        Args:
            df: Input DataFrame (modified in-place)
            
        Returns:
            Number of missing values handled
        """
        # Count missing values
        missing_count = df.isna().sum().sum()
        
        # Apply strategy
        if self.missing_strategy == 'drop':
            # Drop rows with any missing values
            df.dropna(inplace=True)
            
        elif self.missing_strategy == 'fill_zeros':
            # Fill numeric columns with zeros, non-numeric with empty string
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col].fillna(0, inplace=True)
                else:
                    df[col].fillna("", inplace=True)
                    
        elif self.missing_strategy == 'fill_mean':
            # Fill numeric columns with mean, non-numeric with most frequent value
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col].fillna(df[col].mean(), inplace=True)
                else:
                    most_frequent = df[col].mode()[0] if not df[col].mode().empty else ""
                    df[col].fillna(most_frequent, inplace=True)
        
        return missing_count
    
    def _find_food_name_column(self, df: pd.DataFrame) -> Optional[str]:
        """
        Find the best column to use for food names
        
        Args:
            df: Input DataFrame
            
        Returns:
            Best column name or None if not found
        """
        # Check for standardized column
        if "Food_Name" in df.columns:
            return "Food_Name"
        
        # Check for common variations
        common_names = [
            "Food Name", "FoodName", "Name", "Product Name", 
            "ProductName", "Item", "Description"
        ]
        
        for name in common_names:
            if name in df.columns:
                return name
        
        # Look for columns with "food" and "name" in their name
        for col in df.columns:
            col_lower = col.lower()
            if "food" in col_lower and "name" in col_lower:
                return col
            if "name" in col_lower and not any(x in col_lower for x in ["brand", "manufacturer"]):
                return col
                
        return None

    def validate_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate data quality and generate report
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with validation results
        """
        return validate_data(df)