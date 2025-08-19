"""
ShelfScale - LLM-Enhanced Food Product Analysis Platform
A standardized data product for understanding nutrition and sustainability metrics at the basket level
with advanced LLM-powered semantic matching capabilities.
"""

__version__ = '3.0.0'

# Import key components for easy access
from shelfscale.data_processing import (
    WeightExtractor,
    clean_weights,
    predict_missing_weights,
    DataCleaner
)

# Import LLM-enhanced matching system
try:
    from shelfscale.matching import FoodMatcher
    MATCHING_AVAILABLE = True
except ImportError:
    MATCHING_AVAILABLE = False

# Import scoring systems
try:
    from shelfscale.scoring import score_traffic_lights, score_nutri
    SCORING_AVAILABLE = True
except ImportError:
    SCORING_AVAILABLE = False

# Define what's available when using from shelfscale import *
__all__ = [
    'WeightExtractor',
    'clean_weights', 
    'predict_missing_weights',
    'DataCleaner'
]

if MATCHING_AVAILABLE:
    __all__.append('FoodMatcher')
    
if SCORING_AVAILABLE:
    __all__.extend(['score_traffic_lights', 'score_nutri']) 