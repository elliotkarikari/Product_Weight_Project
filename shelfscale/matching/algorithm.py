"""
ShelfScale Food Matching Algorithm - Primary Interface

This module provides the main interface for food product matching in ShelfScale.
As of the latest update, this system prioritizes LLM-Enhanced Matching for
superior semantic understanding of food products.

System Architecture (Priority Order):
1. LLM-Enhanced Matching (Primary) - Semantic food understanding, brand-aware
2. Enhanced AI Matching (Fallback) - ML ensemble with feature engineering  
3. Original Matching (Final Fallback) - Traditional string similarity

For new development, consider using llm_algorithm.ShelfScaleMatcher directly.
This module maintains backward compatibility while upgrading the underlying system.
"""

import os
import logging
import pandas as pd
import numpy as np
import asyncio
from typing import Dict, List, Optional, Union, Tuple, Any

# Import the new LLM-enhanced system as primary
try:
    from .llm_algorithm import ShelfScaleMatcher
    LLM_SYSTEM_AVAILABLE = True
except ImportError as e:
    LLM_SYSTEM_AVAILABLE = False

# Import legacy enhanced AI as fallback
try:
    from .algorithm_ai_enhanced import EnhancedFoodMatcher, preprocess_text as enhanced_preprocess_text
    AI_ENHANCED_AVAILABLE = True
except ImportError:
    AI_ENHANCED_AVAILABLE = False
    def enhanced_preprocess_text(text):
        return str(text).lower().strip()

import shelfscale.config as config

# Configure logging
logger = logging.getLogger(__name__)

# Log the system status
if LLM_SYSTEM_AVAILABLE:
    logger.info("✅ Using LLM-Enhanced Matching System v3.0 (PRIMARY)")
elif AI_ENHANCED_AVAILABLE:
    logger.info("⚠️ Using Enhanced AI Matching v2.0 (LLM unavailable)")
else:
    logger.warning("❌ Using basic fallback matching (Limited functionality)")


def preprocess_text(text: str) -> str:
    """
    Text preprocessing using the best available method
    
    Args:
        text: Input text
        
    Returns:
        Preprocessed text
    """
    if AI_ENHANCED_AVAILABLE:
        return enhanced_preprocess_text(text)
    else:
        # Basic fallback preprocessing
        return str(text).lower().strip()


class FoodMatcher:
    """
    Primary Food Matching Interface for ShelfScale
    
    This class provides backward-compatible access to the advanced matching
    capabilities while internally using the most sophisticated system available.
    
    Features:
    - LLM-powered semantic matching (when available)
    - Intelligent food category understanding
    - Separate brand analysis
    - Production-ready async/sync interfaces
    - Comprehensive fallback systems
    """
    
    def __init__(self, 
                 similarity_threshold: float = 0.7,
                 model_path: Optional[str] = None,
                 features_path: Optional[str] = None,
                 learning_enabled: bool = True,
                 use_llm: bool = True):
        """
        Initialize the food matcher with the best available system
        
        Args:
            similarity_threshold: Minimum confidence for matches
            model_path: Path for model persistence
            features_path: Legacy parameter (maintained for compatibility)
            learning_enabled: Whether to use ML capabilities
            use_llm: Whether to prioritize LLM matching
        """
        logger.info("Initializing ShelfScale Food Matcher")
        
        self.similarity_threshold = similarity_threshold
        self.model_path = model_path or config.FOOD_MATCHER_MODEL_PATH
        self.features_path = features_path or config.FOOD_MATCHER_FEATURES_PATH
        self.learning_enabled = learning_enabled
        
        # Initialize the best available system
        if LLM_SYSTEM_AVAILABLE and use_llm:
            logger.info("🤖 Initializing with LLM-Enhanced System")
            self.matcher = ShelfScaleMatcher(
                confidence_threshold=similarity_threshold,
                use_llm=True,
                use_enhanced_ai=True,
                model_path=self.model_path
            )
            self.system_type = "llm_enhanced"
            
        elif AI_ENHANCED_AVAILABLE:
            logger.info("🧠 Initializing with Enhanced AI System")
            self.matcher = EnhancedFoodMatcher(
                similarity_threshold=similarity_threshold,
                model_path=self.model_path,
                confidence_threshold=similarity_threshold,
                use_ensemble=learning_enabled
            )
            self.system_type = "enhanced_ai"
            
        else:
            logger.warning("⚠️ Using basic fallback system")
            self.matcher = None
            self.system_type = "basic_fallback"
        
        logger.info(f"Food Matcher initialized with {self.system_type} system")
    
    def fit(self, df1: pd.DataFrame, df2: pd.DataFrame,
            text_col1: str, text_col2: str,
            category_col1: str = None, category_col2: str = None) -> Dict[str, Any]:
        """
        Train the matching system (where applicable)
        
        Args:
            df1: First training dataframe
            df2: Second training dataframe
            text_col1: Text column in df1
            text_col2: Text column in df2
            category_col1: Category column in df1
            category_col2: Category column in df2
            
        Returns:
            Training results
        """
        if self.system_type == "enhanced_ai" and hasattr(self.matcher, 'fit'):
            return self.matcher.fit(df1, df2, text_col1, text_col2, category_col1, category_col2)
        else:
            logger.info("Training not applicable for current system")
            return {"message": "Training not required for current matching system"}
    
    async def match_foods(self, df1: pd.DataFrame, df2: pd.DataFrame,
                         text_col1: str, text_col2: str,
                         max_matches: int = 3,
                         category_col1: str = None,
                         category_col2: str = None) -> pd.DataFrame:
        """
        Match foods between dataframes (async interface)
        
        Args:
            df1: Source dataframe
            df2: Target dataframe
            text_col1: Text column in df1
            text_col2: Text column in df2
            max_matches: Maximum matches per item
            category_col1: Optional category column in df1
            category_col2: Optional category column in df2
            
        Returns:
            DataFrame with matched results
        """
        logger.info(f"Matching {len(df1)} items against {len(df2)} targets using {self.system_type}")
        
        if self.system_type == "llm_enhanced":
            return await self.matcher.match_products(
                df1, df2, text_col1, text_col2, max_matches, category_col1, category_col2
            )
            
        elif self.system_type == "enhanced_ai":
            return self.matcher.match_items(
                df1, df2, text_col1, text_col2, category_col1, category_col2, max_matches
            )
            
        else:
            # Basic fallback - minimal matching
            logger.warning("Using basic fallback matching")
            return self._basic_fallback_matching(df1, df2, text_col1, text_col2, max_matches)
    
    def match_foods_sync(self, df1: pd.DataFrame, df2: pd.DataFrame,
                        text_col1: str, text_col2: str,
                        max_matches: int = 3,
                        category_col1: str = None,
                        category_col2: str = None) -> pd.DataFrame:
        """
        Match foods between dataframes (synchronous interface)
        
        This is the main interface for backward compatibility.
        """
        if self.system_type == "llm_enhanced":
            return self.matcher.match_products_sync(
                df1, df2, text_col1, text_col2, max_matches, category_col1, category_col2
            )
        else:
            # For non-async systems, run directly
            return asyncio.run(self.match_foods(
                df1, df2, text_col1, text_col2, max_matches, category_col1, category_col2
            ))
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        if self.system_type == "llm_enhanced":
            # Use LLM for single product matching
            try:
                result = asyncio.run(self.matcher.match_single_product(text1, text2))
                return result.get('confidence', 0.0)
            except:
                pass
        
        if self.system_type == "enhanced_ai" and hasattr(self.matcher, 'calculate_similarity'):
            return self.matcher.calculate_similarity(text1, text2)
        
        # Basic fallback
        return self._basic_similarity(text1, text2)
    
    def _basic_fallback_matching(self, df1: pd.DataFrame, df2: pd.DataFrame,
                               text_col1: str, text_col2: str, max_matches: int) -> pd.DataFrame:
        """Basic fallback matching when no advanced systems available"""
        matches = []
        
        for idx1, row1 in df1.iterrows():
            text1 = str(row1[text_col1]).lower()
            
            item_matches = []
            for idx2, row2 in df2.iterrows():
                text2 = str(row2[text_col2]).lower()
                similarity = self._basic_similarity(text1, text2)
                
                if similarity >= self.similarity_threshold:
                    item_matches.append({
                        'source_index': idx1,
                        'target_index': idx2,
                        'source_text': text1,
                        'target_text': text2,
                        'llm_confidence': similarity,
                        'matching_method': 'basic_fallback',
                        'system_confidence': 'low'
                    })
            
            # Sort by similarity and take top matches
            item_matches.sort(key=lambda x: x['llm_confidence'], reverse=True)
            matches.extend(item_matches[:max_matches])
        
        return pd.DataFrame(matches)
    
    def _basic_similarity(self, text1: str, text2: str) -> float:
        """Basic word overlap similarity"""
        words1 = set(str(text1).lower().split())
        words2 = set(str(text2).lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union) if union else 0.0
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status and capabilities"""
        base_status = {
            'system_type': self.system_type,
            'llm_available': LLM_SYSTEM_AVAILABLE,
            'enhanced_ai_available': AI_ENHANCED_AVAILABLE,
            'confidence_threshold': self.similarity_threshold,
            'capabilities': self._get_capabilities()
        }
        
        if hasattr(self.matcher, 'get_system_status'):
            # Merge with underlying system status
            matcher_status = self.matcher.get_system_status()
            base_status.update(matcher_status)
        
        return base_status
    
    def _get_capabilities(self) -> List[str]:
        """Get list of system capabilities"""
        capabilities = []
        
        if self.system_type == "llm_enhanced":
            capabilities.extend([
                "semantic_food_matching",
                "brand_aware_analysis", 
                "contextual_reasoning",
                "multi_language_support",
                "hybrid_scoring"
            ])
        
        if self.system_type in ["llm_enhanced", "enhanced_ai"]:
            capabilities.extend([
                "ml_ensemble_matching",
                "feature_engineering",
                "continuous_learning"
            ])
        
        capabilities.extend([
            "fuzzy_string_matching",
            "basic_preprocessing",
            "similarity_scoring"
        ])
        
        return capabilities
    
    def clear_cache(self):
        """Clear all system caches"""
        if hasattr(self.matcher, 'clear_cache'):
            self.matcher.clear_cache()
        logger.info("System caches cleared")
    
    def save_models(self):
        """Save all trained models"""
        if hasattr(self.matcher, 'save_models'):
            self.matcher.save_models()
        logger.info("Models saved")


# Legacy function for backward compatibility
def match_foods(df1: pd.DataFrame, df2: pd.DataFrame, 
               food_col1: str, food_col2: str,
               similarity_threshold: float = 0.7) -> pd.DataFrame:
    """
    Legacy function for backward compatibility
    
    DEPRECATED: Use FoodMatcher class instead for better control and features
    """
    logger.warning("Using deprecated match_foods function. Consider upgrading to FoodMatcher class.")
    
    matcher = FoodMatcher(similarity_threshold=similarity_threshold)
    return matcher.match_foods_sync(df1, df2, food_col1, food_col2, max_matches=1)


# Hybrid function from enhanced AI (maintained for compatibility)
def hybrid_fuzzy_matching(text1: str, text2: str, weights: Dict[str, float] = None) -> float:
    """Legacy hybrid fuzzy matching function"""
    if AI_ENHANCED_AVAILABLE:
        from .algorithm_ai_enhanced import hybrid_fuzzy_matching as enhanced_hybrid
        return enhanced_hybrid(text1, text2, weights)
    else:
        # Basic fallback
        matcher = FoodMatcher()
        return matcher._basic_similarity(text1, text2)


# Export main interfaces
__all__ = [
    'FoodMatcher',
    'match_foods',
    'preprocess_text', 
    'hybrid_fuzzy_matching'
]