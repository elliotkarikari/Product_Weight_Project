"""
ShelfScale Matching Module - LLM-Enhanced Food Product Matching

This module provides advanced food product matching capabilities using:
- LLM-Enhanced Semantic Matching (Primary)
- Enhanced AI ML Ensemble Methods (Fallback)
- Traditional String Matching (Final Fallback)

Main Interface:
- FoodMatcher: Primary matching class with intelligent system selection
- ShelfScaleMatcher: Direct access to the LLM-enhanced system

The system automatically uses the best available matching method and gracefully
degrades to fallback systems when needed.
"""

# Import the main interfaces
try:
    from .algorithm import FoodMatcher, match_foods, preprocess_text, hybrid_fuzzy_matching
    MAIN_ALGORITHM_AVAILABLE = True
except ImportError:
    MAIN_ALGORITHM_AVAILABLE = False

# Import LLM-enhanced system directly
try:
    from .llm_algorithm import ShelfScaleMatcher
    LLM_SYSTEM_AVAILABLE = True
except ImportError:
    LLM_SYSTEM_AVAILABLE = False

# Import enhanced AI fallback
try:
    from .algorithm_ai_enhanced import EnhancedFoodMatcher
    AI_ENHANCED_AVAILABLE = True
except ImportError:
    AI_ENHANCED_AVAILABLE = False

# Original algorithm removed - functionality merged into enhanced AI

# Define exports based on what's available
__all__ = []

if MAIN_ALGORITHM_AVAILABLE:
    __all__.extend(['FoodMatcher', 'match_foods', 'preprocess_text', 'hybrid_fuzzy_matching'])

if LLM_SYSTEM_AVAILABLE:
    __all__.append('ShelfScaleMatcher')

if AI_ENHANCED_AVAILABLE:
    __all__.append('EnhancedFoodMatcher')

# Original algorithm removed

# Log system status
import logging
logger = logging.getLogger(__name__)

logger.info(f"ShelfScale Matching Module loaded:")
logger.info(f"  ✅ Main Algorithm: {MAIN_ALGORITHM_AVAILABLE}")
logger.info(f"  🤖 LLM System: {LLM_SYSTEM_AVAILABLE}")
logger.info(f"  🧠 Enhanced AI: {AI_ENHANCED_AVAILABLE}")

if LLM_SYSTEM_AVAILABLE:
    logger.info("🚀 LLM-Enhanced Matching System is PRIMARY")
elif AI_ENHANCED_AVAILABLE:
    logger.info("⚠️  Enhanced AI Matching is PRIMARY (LLM unavailable)")
else:
    logger.error("❌ No matching systems available!") 