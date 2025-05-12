"""
Weight extraction utilities for ShelfScale datasets
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Optional, Union, Tuple, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class WeightExtractor:
    """Class for extracting weight information from text strings"""
    
    def __init__(self, target_unit='g'):
        """
        Initialize the weight extractor
        
        Args:
            target_unit: Unit to standardize to ('g' or 'ml')
        """
        self.target_unit = target_unit
        self.conversion_factors = {
            'kg': 1000,    # to g
            'g': 1,        # to g
            'mg': 0.001,   # to g
            'l': 1000,     # to ml
            'ml': 1,       # to ml
            'oz': 28.35,   # to g
            'lb': 453.592  # to g
        }
        
        # Compile regex patterns for performance
        self.patterns = [
            # Simple pattern: "100g" or "100 g"
            re.compile(r'(\d+(?:\.\d+)?)\s*(g|kg|ml|l|mg|oz|lb)\b', re.IGNORECASE),
            
            # Multipack pattern: "3 x 100g"
            re.compile(r'(\d+)\s*x\s*(\d+(?:\.\d+)?)\s*(g|kg|ml|l|mg|oz|lb)\b', re.IGNORECASE),
            
            # Pack pattern: "6pk x 25g"
            re.compile(r'(\d+)\s*(?:pk|pack)s?\s*(?:x\s*)?(\d+(?:\.\d+)?)\s*(g|kg|ml|l|mg|oz|lb)\b', re.IGNORECASE),
            
            # Range pattern: "100-150g" (take average)
            re.compile(r'(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)\s*(g|kg|ml|l|mg|oz|lb)\b', re.IGNORECASE)
        ]
    
    def extract(self, text: str) -> Tuple[Optional[float], Optional[str]]:
        """
        Extract weight and unit from text
        
        Args:
            text: Input text containing weight information
            
        Returns:
            Tuple of (weight value, unit)
        """
        if pd.isna(text) or not isinstance(text, str):
            return None, None
            
        # Clean text for parsing
        clean_text = text.lower().strip()
        clean_text = re.sub(r'[^\w\s\.\,\-x()]', '', clean_text)
        
        # Try each pattern in order
        for pattern in self.patterns:
            match = pattern.search(clean_text)
            if match:
                return self._process_match(match)
                
        # No patterns matched
        logger.debug(f"No weight pattern found in: '{text}'")
        return None, None
    
    def _process_match(self, match) -> Tuple[Optional[float], Optional[str]]:
        """Process a regex match based on the pattern type"""
        groups = match.groups()
        
        # Simple pattern: single value with unit
        if len(groups) == 2:
            value, unit = float(groups[0]), groups[1].lower()
        
        # Multipack or pack pattern: quantity * weight with unit
        elif len(groups) == 3 and match.re.pattern.find(r'x\s*') > 0:
            quantity, weight, unit = float(groups[0]), float(groups[1]), groups[2].lower()
            value = quantity * weight
            
        # Range pattern: take average of min and max
        elif len(groups) == 3 and match.re.pattern.find(r'[-–]') > 0:
            min_val, max_val, unit = float(groups[0]), float(groups[1]), groups[2].lower()
            value = (min_val + max_val) / 2
            
        else:
            logger.warning(f"Unexpected match format: {match.groups()}")
            return None, None
            
        # Standardize unit
        if unit in self.conversion_factors:
            # For mass-to-volume conversion, need density info which we don't have
            is_mass_unit = unit in ['g', 'kg', 'mg', 'oz', 'lb']
            is_volume_unit = unit in ['ml', 'l']
            
            target_is_mass = self.target_unit in ['g', 'kg', 'mg']
            
            # Cannot convert between mass and volume
            if (is_mass_unit and not target_is_mass) or (is_volume_unit and target_is_mass):
                logger.warning(f"Cannot convert between mass ({unit}) and volume ({self.target_unit})")
                return value, unit
                
            # Convert to target unit
            converted_value = value * self.conversion_factors[unit]
            return converted_value, 'g' if target_is_mass else 'ml'
            
        return value, unit


def clean_weights(df: pd.DataFrame, 
                  weight_col: str = 'Weight', 
                  target_unit: str = 'g',
                  keep_original: bool = True) -> pd.DataFrame:
    """
    Enhanced weight cleaning function with better pattern recognition
    
    Args:
        df: Input DataFrame
        weight_col: Column containing weight information
        target_unit: Unit to standardize to ('g' or 'ml')
        keep_original: Whether to keep the original weight column
        
    Returns:
        DataFrame with standardized weight information
    """
    if weight_col not in df.columns:
        raise ValueError(f"Column '{weight_col}' not found in DataFrame")
        
    # Create a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # Create extractor
    extractor = WeightExtractor(target_unit=target_unit)
    
    # Extract weights and units
    logger.info(f"Extracting weights from '{weight_col}' column")
    extracted = cleaned_df[weight_col].apply(extractor.extract)
    
    # Add new columns
    cleaned_df['Weight_Value'] = extracted.apply(lambda x: x[0])
    cleaned_df['Weight_Unit'] = extracted.apply(lambda x: x[1])
    
    # Calculate success rate
    success_count = cleaned_df['Weight_Value'].notna().sum()
    total_count = len(cleaned_df)
    success_rate = (success_count / total_count) * 100 if total_count > 0 else 0
    
    logger.info(f"Successfully extracted {success_count} weights out of {total_count} entries ({success_rate:.2f}%)")
    
    # Remove original column if not keeping it
    if not keep_original:
        cleaned_df.drop(columns=[weight_col], inplace=True)
        
    return cleaned_df