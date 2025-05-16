"""
ShelfScale - A Standardised Data Product for understanding nutrition and sustainability metrics at the basket level
"""

__version__ = '0.2.0'

# Import key components for easy access
from shelfscale.data_processing import (
    WeightExtractor,
    clean_weights,
    predict_missing_weights,
    DataCleaner
)

# Define what's available when using from shelfscale import *
__all__ = [
    'WeightExtractor',
    'clean_weights',
    'predict_missing_weights',
    'DataCleaner'
] 