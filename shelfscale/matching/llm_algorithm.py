"""
LLM-Enhanced Food Matching Algorithm - Primary Matching System

This is the main food matching algorithm for ShelfScale, using advanced LLM
capabilities for intelligent product matching with comprehensive fallback systems.

Features:
- LLM-powered semantic matching for food products
- Separate brand and food analysis
- Intelligent category detection and reasoning
- Hybrid scoring with traditional ML fallbacks
- Production-ready async/sync interfaces
- Comprehensive error handling and logging
"""

import logging
import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any

# Import LLM components as primary
try:
    from ..ml.llm_matcher import LLMMatcher
    LLM_AVAILABLE = True
except ImportError:
    logging.warning("LLM matcher not available")
    LLM_AVAILABLE = False

# Import enhanced AI as fallback
try:
    from .algorithm_ai_enhanced import EnhancedFoodMatcher
    AI_ENHANCED_AVAILABLE = True
except ImportError:
    logging.warning("Enhanced AI matcher not available")
    AI_ENHANCED_AVAILABLE = False

# Import original as final fallback
try:
    from .algorithm_original import FoodMatcher as OriginalFoodMatcher
    ORIGINAL_AVAILABLE = True
except ImportError:
    logging.warning("Original matcher not available")
    ORIGINAL_AVAILABLE = False

logger = logging.getLogger(__name__)


class ShelfScaleMatcher:
    """
    Primary food matching system for ShelfScale using LLM-enhanced algorithms
    
    This class provides the main interface for all food product matching in ShelfScale,
    with intelligent fallback systems for maximum reliability.
    
    Architecture:
    1. LLM-Enhanced Matching (Primary) - Semantic understanding, food-focused
    2. Enhanced AI Matching (Fallback) - ML ensemble with feature engineering
    3. Original Matching (Final Fallback) - Traditional string similarity
    """
    
    def __init__(self, 
                 confidence_threshold: float = 0.7,
                 use_llm: bool = True,
                 use_enhanced_ai: bool = True,
                 model_path: Optional[str] = None):
        """
        Initialize the ShelfScale matcher
        
        Args:
            confidence_threshold: Minimum confidence for accepting matches
            use_llm: Whether to use LLM matching as primary
            use_enhanced_ai: Whether to use enhanced AI as fallback
            model_path: Path for saving/loading models
        """
        logger.info("Initializing ShelfScale Primary Matching System")
        
        self.confidence_threshold = confidence_threshold
        self.model_path = model_path
        
        # Initialize matching systems in priority order
        self._init_llm_matcher(use_llm)
        self._init_enhanced_ai_matcher(use_enhanced_ai)
        self._init_original_matcher()
        
        # Statistics tracking
        self.stats = {
            'total_matches': 0,
            'llm_matches': 0,
            'ai_matches': 0,
            'original_matches': 0,
            'failed_matches': 0
        }
        
        logger.info(f"ShelfScale Matcher initialized - LLM: {self.llm_available}, "
                   f"Enhanced AI: {self.ai_enhanced_available}, "
                   f"Original: {self.original_available}")
    
    def _init_llm_matcher(self, use_llm: bool):
        """Initialize LLM matcher as primary system"""
        self.llm_available = use_llm and LLM_AVAILABLE
        if self.llm_available:
            try:
                self.llm_matcher = LLMMatcher(
                    confidence_threshold=self.confidence_threshold,
                    use_hybrid_scoring=True
                )
                logger.info("✅ LLM matcher initialized as PRIMARY system")
            except Exception as e:
                logger.error(f"Failed to initialize LLM matcher: {e}")
                self.llm_available = False
        else:
            self.llm_matcher = None
            logger.info("❌ LLM matcher not available")
    
    def _init_enhanced_ai_matcher(self, use_enhanced_ai: bool):
        """Initialize enhanced AI matcher as fallback"""
        self.ai_enhanced_available = use_enhanced_ai and AI_ENHANCED_AVAILABLE
        if self.ai_enhanced_available:
            try:
                self.enhanced_ai_matcher = EnhancedFoodMatcher(
                    similarity_threshold=self.confidence_threshold,
                    confidence_threshold=self.confidence_threshold,
                    use_ensemble=True
                )
                logger.info("✅ Enhanced AI matcher initialized as FALLBACK system")
            except Exception as e:
                logger.error(f"Failed to initialize enhanced AI matcher: {e}")
                self.ai_enhanced_available = False
        else:
            self.enhanced_ai_matcher = None
            logger.info("❌ Enhanced AI matcher not available")
    
    def _init_original_matcher(self):
        """Initialize original matcher as final fallback"""
        self.original_available = ORIGINAL_AVAILABLE
        if self.original_available:
            try:
                self.original_matcher = OriginalFoodMatcher(
                    similarity_threshold=max(0.6, self.confidence_threshold - 0.1)
                )
                logger.info("✅ Original matcher initialized as FINAL FALLBACK")
            except Exception as e:
                logger.error(f"Failed to initialize original matcher: {e}")
                self.original_available = False
        else:
            self.original_matcher = None
            logger.info("❌ Original matcher not available")
    
    async def match_products(self, 
                            df1: pd.DataFrame, 
                            df2: pd.DataFrame,
                            text_col1: str, 
                            text_col2: str,
                            max_matches_per_item: int = 3,
                            category_col1: str = None,
                            category_col2: str = None) -> pd.DataFrame:
        """
        Match products between two dataframes using the best available method
        
        Args:
            df1: First dataframe (source)
            df2: Second dataframe (target) 
            text_col1: Text column in df1
            text_col2: Text column in df2
            max_matches_per_item: Maximum matches per source item
            category_col1: Optional category column in df1
            category_col2: Optional category column in df2
            
        Returns:
            DataFrame with matched results including confidence scores and reasoning
        """
        logger.info(f"Matching {len(df1)} source items against {len(df2)} target items")
        self.stats['total_matches'] += 1
        
        # Try LLM matching first (primary)
        if self.llm_available:
            try:
                logger.info("🤖 Using LLM-Enhanced Matching (PRIMARY)")
                matches = await self.llm_matcher.match_dataframes_llm(
                    df1, df2, text_col1, text_col2,
                    max_matches_per_item=max_matches_per_item,
                    pre_filter_similarity=0.3
                )
                
                if not matches.empty:
                    self.stats['llm_matches'] += 1
                    matches['matching_method'] = 'llm_enhanced'
                    matches['system_confidence'] = 'high'
                    logger.info(f"✅ LLM matching successful: {len(matches)} matches found")
                    return matches
                else:
                    logger.info("🔄 LLM found no high-confidence matches, trying fallbacks...")
                    
            except Exception as e:
                logger.warning(f"⚠️ LLM matching failed: {e}, trying fallbacks...")
        
        # Try Enhanced AI matching (fallback)
        if self.ai_enhanced_available:
            try:
                logger.info("🧠 Using Enhanced AI Matching (FALLBACK)")
                matches = self.enhanced_ai_matcher.match_items(
                    df1, df2, text_col1, text_col2,
                    category_col1, category_col2,
                    max_matches_per_item=max_matches_per_item
                )
                
                if not matches.empty:
                    self.stats['ai_matches'] += 1
                    matches['matching_method'] = 'enhanced_ai'
                    matches['system_confidence'] = 'medium'
                    logger.info(f"✅ Enhanced AI matching successful: {len(matches)} matches found")
                    return matches
                else:
                    logger.info("🔄 Enhanced AI found no matches, trying final fallback...")
                    
            except Exception as e:
                logger.warning(f"⚠️ Enhanced AI matching failed: {e}, trying final fallback...")
        
        # Try Original matching (final fallback)
        if self.original_available:
            try:
                logger.info("🔧 Using Original Matching (FINAL FALLBACK)")
                matches = self.original_matcher.match_datasets(
                    df1, df2, text_col1, text_col2
                )
                
                if not matches.empty:
                    self.stats['original_matches'] += 1
                    # Standardize column names
                    if 'Source_Index' in matches.columns:
                        matches = matches.rename(columns={
                            'Source_Index': 'source_index',
                            'Target_Index': 'target_index',
                            'Similarity_Score': 'llm_confidence'
                        })
                    matches['matching_method'] = 'original'
                    matches['system_confidence'] = 'low'
                    logger.info(f"✅ Original matching successful: {len(matches)} matches found")
                    return matches
                else:
                    logger.warning("❌ All matching methods found no results")
                    
            except Exception as e:
                logger.error(f"❌ Original matching failed: {e}")
        
        # All methods failed
        self.stats['failed_matches'] += 1
        logger.error("❌ All matching methods failed or unavailable")
        return pd.DataFrame()
    
    def match_products_sync(self, 
                           df1: pd.DataFrame, 
                           df2: pd.DataFrame,
                           text_col1: str, 
                           text_col2: str,
                           max_matches_per_item: int = 3,
                           category_col1: str = None,
                           category_col2: str = None) -> pd.DataFrame:
        """
        Synchronous wrapper for match_products for backward compatibility
        """
        try:
            return asyncio.run(self.match_products(
                df1, df2, text_col1, text_col2, max_matches_per_item,
                category_col1, category_col2
            ))
        except RuntimeError as e:
            if "asyncio.run() cannot be called from a running event loop" in str(e):
                # We're already in an async context, create a new task
                loop = asyncio.get_event_loop()
                task = loop.create_task(self.match_products(
                    df1, df2, text_col1, text_col2, max_matches_per_item,
                    category_col1, category_col2
                ))
                return loop.run_until_complete(task)
            else:
                raise
    
    async def match_single_product(self, 
                                  product1: str, 
                                  product2: str,
                                  context: Dict = None) -> Dict[str, Any]:
        """
        Match two individual product strings
        
        Args:
            product1: First product description
            product2: Second product description
            context: Optional context information
            
        Returns:
            Dictionary with match result and reasoning
        """
        if self.llm_available:
            try:
                return await self.llm_matcher.match_products_llm(product1, product2, context)
            except Exception as e:
                logger.warning(f"LLM single product matching failed: {e}")
        
        # Fallback to traditional similarity
        if self.original_available:
            return self.original_matcher.calculate_similarity(product1, product2)
        
        return {'confidence': 0.0, 'match': False, 'reasoning': 'No matching methods available'}
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status and statistics"""
        return {
            'systems_available': {
                'llm_enhanced': self.llm_available,
                'enhanced_ai': self.ai_enhanced_available,
                'original': self.original_available
            },
            'primary_system': 'llm_enhanced' if self.llm_available else 
                            'enhanced_ai' if self.ai_enhanced_available else 
                            'original' if self.original_available else 'none',
            'matching_statistics': self.stats.copy(),
            'confidence_threshold': self.confidence_threshold,
            'recommendations': self._get_system_recommendations()
        }
    
    def _get_system_recommendations(self) -> List[str]:
        """Get system optimization recommendations"""
        recommendations = []
        
        if not self.llm_available:
            recommendations.append("Install LLM dependencies for best matching accuracy")
        
        if not self.ai_enhanced_available:
            recommendations.append("Enable Enhanced AI matching for better fallback performance")
            
        if self.stats['failed_matches'] > 0:
            recommendations.append(f"Consider lowering confidence threshold (current: {self.confidence_threshold})")
            
        if self.stats['total_matches'] > 10:
            success_rate = 1 - (self.stats['failed_matches'] / self.stats['total_matches'])
            if success_rate < 0.8:
                recommendations.append("Low success rate detected - review input data quality")
        
        return recommendations
    
    def clear_cache(self):
        """Clear all caches across matching systems"""
        if self.llm_available:
            self.llm_matcher.clear_cache()
        logger.info("All caches cleared")
    
    def save_models(self):
        """Save all trained models"""
        if self.llm_available:
            self.llm_matcher.export_cache(f"{self.model_path}_llm_cache.json")
        
        if self.ai_enhanced_available:
            self.enhanced_ai_matcher.save_model()
            
        logger.info("All models saved")


# Backward compatibility aliases
FoodMatcher = ShelfScaleMatcher
LLMFoodMatcher = ShelfScaleMatcher

# Export the main interface
__all__ = ['ShelfScaleMatcher', 'FoodMatcher', 'LLMFoodMatcher']