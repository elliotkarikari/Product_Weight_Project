"""
Comprehensive data cleaning pipeline for ShelfScale datasets
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, Any
import logging

# Import from other modules
from .weight_extraction import WeightExtractor
from .categorization import FoodCategorizer
from .validation import validate_data

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataCleaner:
    """Configurable data cleaning pipeline for food datasets"""
    
    def __init__(self, config=None):
        """
        Initialize with configuration
        
        Args:
            config: Dictionary with cleaning configuration
        """
        self.config = config or {}
        self.default_config = {
            'weight': {
                'enabled': True,
                'column': 'Weight',
                'target_unit': 'g',
                'keep_original': True
            },
            'categories': {
                'enabled': True,
                'column': 'Product Name',
                'new_column': 'Food_Category',
                'super_category_column': 'Super_Category',
                'mapping_file': None
            },
            'text': {
                'enabled': True,
                'columns': ['Product Name'],
                'remove_special_chars': True,
                'lowercase': True
            },
            'duplicates': {
                'enabled': True,
                'columns': None
            },
            'missing_values': {
                'enabled': True,
                'strategy': 'drop'
            }
        }
        
        # Merge with default config
        for section, defaults in self.default_config.items():
            if section not in self.config:
                self.config[section] = defaults
            else:
                for key, value in defaults.items():
                    if key not in self.config[section]:
                        self.config[section][key] = value
                        
        # Initialize components
        self.weight_extractor = WeightExtractor(
            target_unit=self.config['weight']['target_unit']
        )
        
        self.categorizer = FoodCategorizer(
            mapping_file=self.config['categories'].get('mapping_file')
        )
        
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the cleaning pipeline to a DataFrame
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        logger.info(f"Starting cleaning pipeline on DataFrame with {len(df)} rows")
        
        # Make a copy of the input DataFrame
        cleaned_df = df.copy()
        
        # Standardize column names for common variations
        cleaned_df = self._standardize_column_names(cleaned_df)
        logger.info(f"Columns after standardization: {cleaned_df.columns.tolist()}")
        
        # Track changes
        changes = []
        
        # 1. Clean text columns
        if self.config['text']['enabled']:
            for col in self.config['text']['columns']:
                if col in cleaned_df.columns:
                    logger.info(f"Cleaning text in column: {col}")
                    
                    # Apply text cleaning
                    cleaned_df[col] = cleaned_df[col].apply(self._clean_text)
                    changes.append(f"Cleaned text in '{col}'")
        
        # 2. Clean weight column
        if self.config['weight']['enabled'] and self.config['weight']['column'] in cleaned_df.columns:
            logger.info(f"Cleaning weights in column: {self.config['weight']['column']}")
            
            # Extract weights
            extracted = cleaned_df[self.config['weight']['column']].apply(self.weight_extractor.extract)
            cleaned_df['Weight_Value'] = extracted.apply(lambda x: x[0])
            cleaned_df['Weight_Unit'] = extracted.apply(lambda x: x[1])
            
            # Calculate success rate
            success_count = cleaned_df['Weight_Value'].notna().sum()
            total_count = len(cleaned_df)
            success_rate = (success_count / total_count) * 100 if total_count > 0 else 0
            
            changes.append(f"Extracted weights with {success_rate:.2f}% success rate")
            
            # Remove original if configured
            if not self.config['weight']['keep_original']:
                cleaned_df.drop(columns=[self.config['weight']['column']], inplace=True)
                changes.append(f"Removed original weight column '{self.config['weight']['column']}'")
        
        # 3. Clean food categories
        if self.config['categories']['enabled'] and self.config['categories']['column'] in cleaned_df.columns:
            logger.info(f"Cleaning food categories from column: {self.config['categories']['column']}")
            
            # Apply categorization
            cleaned_df[self.config['categories']['new_column']] = cleaned_df[self.config['categories']['column']].apply(
                self.categorizer.categorize
            )
            
            # Add super categories
            cleaned_df[self.config['categories']['super_category_column']] = cleaned_df[self.config['categories']['new_column']].apply(
                self.categorizer.get_super_category
            )
            
            changes.append(f"Added food categories in '{self.config['categories']['new_column']}'")
            changes.append(f"Added super categories in '{self.config['categories']['super_category_column']}'")
        
        # 4. Remove duplicates
        if self.config['duplicates']['enabled']:
            logger.info("Removing duplicate rows")
            
            # Count before
            rows_before = len(cleaned_df)
            
            # Remove duplicates
            cleaned_df = cleaned_df.drop_duplicates(subset=self.config['duplicates']['columns'])
            
            # Count after
            rows_after = len(cleaned_df)
            rows_removed = rows_before - rows_after
            
            if rows_removed > 0:
                changes.append(f"Removed {rows_removed} duplicate rows")
        
        # 5. Handle missing values
        if self.config['missing_values']['enabled']:
            logger.info(f"Handling missing values with strategy: {self.config['missing_values']['strategy']}")
            
            # Count missing values before
            missing_before = cleaned_df.isna().sum().sum()
            
            # Apply strategy
            if self.config['missing_values']['strategy'] == 'drop':
                cleaned_df = cleaned_df.dropna()
            elif self.config['missing_values']['strategy'] == 'fill_mean':
                numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
            elif self.config['missing_values']['strategy'] == 'fill_median':
                numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
            elif self.config['missing_values']['strategy'] == 'fill_mode':
                for col in cleaned_df.columns:
                    if cleaned_df[col].isna().any():
                        mode_val = cleaned_df[col].mode()
                        if not mode_val.empty:
                            cleaned_df[col] = cleaned_df[col].fillna(mode_val[0])
                            
            # Count missing values after
            missing_after = cleaned_df.isna().sum().sum()
            
            changes.append(f"Handled {missing_before - missing_after} missing values")
        
        # Log summary
        logger.info(f"Cleaning complete. Changes made: {', '.join(changes)}")
        logger.info(f"Rows after cleaning: {len(cleaned_df)}")
        
        return cleaned_df
    
    def _clean_text(self, text):
        """Clean and standardize text"""
        if pd.isna(text) or not isinstance(text, str):
            return text
            
        # Apply configured transformations
        if self.config['text']['lowercase']:
            text = text.lower()
            
        if self.config['text']['remove_special_chars']:
            # Keep alphanumeric, spaces, and some punctuation
            text = re.sub(r'[^\w\s\.,;-]', '', text)
            
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text

    def validate_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate data quality and generate report
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with validation results
        """
        return validate_data(df)

    def _standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names to handle common variations
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with standardized column names
        """
        # Dictionary of common variations and their standard forms
        name_variants = {
            'FoodName': 'Food Name',
            'Food_Name': 'Food Name',
            'foodname': 'Food Name',
            'food_name': 'Food Name',
            'food name': 'Food Name',
            
            'FoodCode': 'Food Code',
            'Food_Code': 'Food Code',
            'foodcode': 'Food Code',
            'food_code': 'Food Code',
            'food code': 'Food Code',
            
            'FoodGroup': 'Food Group',
            'Food_Group': 'Food Group',
            'foodgroup': 'Food Group',
            'food_group': 'Food Group',
            'food group': 'Food Group',
            
            'PackSize': 'Pack Size',
            'Pack_Size': 'Pack Size',
            'packsize': 'Pack Size',
            'pack_size': 'Pack Size',
            'pack size': 'Pack Size',
            
            'Weight': 'Weight_g',
            'Weight_g': 'Weight_g',
            'weight_g': 'Weight_g',
            'weight': 'Weight_g',
            'WeightGrams': 'Weight_g',
            'WeightG': 'Weight_g'
        }
        
        # Create a copy of the dataframe to avoid modifying the original during iteration
        result_df = df.copy()
        
        # Standardize column names
        rename_map = {}
        for col in df.columns:
            std_name = name_variants.get(col)
            if std_name and std_name not in df.columns:
                rename_map[col] = std_name
        
        # Only rename if needed
        if rename_map:
            logger.info(f"Standardizing column names: {rename_map}")
            result_df = result_df.rename(columns=rename_map)
        
        return result_df