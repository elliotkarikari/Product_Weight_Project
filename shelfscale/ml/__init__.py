"""
Machine Learning module for ShelfScale
Contains AI-enhanced matching algorithms and prediction models
"""

# Import with graceful fallback for missing dependencies
try:
    from .text_preprocessor import EnhancedTextPreprocessor
except ImportError as e:
    print(f"Warning: Could not import EnhancedTextPreprocessor: {e}")
    EnhancedTextPreprocessor = None

try:
    from .matching_engine import AIMatchingEngine
except ImportError as e:
    print(f"Warning: Could not import AIMatchingEngine: {e}")
    AIMatchingEngine = None

try:
    from .feature_extractor import FeatureExtractor
except ImportError as e:
    print(f"Warning: Could not import FeatureExtractor: {e}")
    FeatureExtractor = None

try:
    from .model_trainer import ModelTrainer
except ImportError as e:
    print(f"Warning: Could not import ModelTrainer: {e}")
    ModelTrainer = None

# Only export successfully imported modules
__all__ = []
if EnhancedTextPreprocessor:
    __all__.append('EnhancedTextPreprocessor')
if AIMatchingEngine:
    __all__.append('AIMatchingEngine')
if FeatureExtractor:
    __all__.append('FeatureExtractor')
if ModelTrainer:
    __all__.append('ModelTrainer')