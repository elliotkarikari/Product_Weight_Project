"""
Enhanced data cleaning utilities for ShelfScale datasets
"""

import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import and re-export all components
from .weight_extraction import WeightExtractor, clean_weights
from .categorization import FoodCategorizer, clean_food_categories
from .validation import validate_data
from .cleaner import DataCleaner

# Define public API
__all__ = [
    'WeightExtractor',
    'clean_weights',
    'FoodCategorizer',
    'clean_food_categories',
    'validate_data',
    'DataCleaner'
]