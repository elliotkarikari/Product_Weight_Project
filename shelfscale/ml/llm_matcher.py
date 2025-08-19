"""
LLM-Enhanced Product Matching System

This module provides advanced LLM-powered matching capabilities that complement
the existing ML-based matching system with reasoning and contextual understanding.
"""

import logging
import json
import re
import sqlite3
import statistics
from typing import Dict, List, Optional, Tuple, Union, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
from dataclasses import dataclass, asdict
import openai
import os
from pathlib import Path
from collections import defaultdict

# Import with graceful fallback
try:
    from ..utils.logging_config import get_logger, monitor_performance
except ImportError:
    import logging
    def get_logger(name):
        return logging.getLogger(name)
    def monitor_performance(name):
        def decorator(func):
            return func
        return decorator

logger = get_logger(__name__)


@dataclass
class MatchCandidate:
    """Represents a potential match between two products"""
    source_text: str
    target_text: str
    confidence: float
    reasoning: str
    features: Dict[str, Any]
    llm_score: float
    hybrid_score: float


@dataclass
class FeedbackEntry:
    """Represents user feedback for learning"""
    timestamp: str
    product1: str
    product2: str
    llm_confidence: float
    llm_prediction: bool
    user_feedback: bool
    simplified_product_a: str = ""
    simplified_product_b: str = ""
    correction_reasoning: str = ""
    session_id: str = "default"


class LLMMatcher:
    """
    LLM-Enhanced Product Matcher
    
    Uses Large Language Models to provide intelligent, context-aware matching
    that understands food semantics, brand relationships, and product variants.
    """
    
    def __init__(self, 
                 model_name: str = "gpt-4o-mini",
                 confidence_threshold: float = 0.8,
                 use_hybrid_scoring: bool = True,
                 max_batch_size: int = 10,
                 api_key: Optional[str] = None,
                 enable_learning: bool = True,
                 learning_db_path: str = "llm_feedback.db"):
        """
        Initialize LLM matcher with learning capabilities
        
        Args:
            model_name: LLM model to use (default: gpt-4o-mini for cost efficiency)
            confidence_threshold: Minimum confidence for matches
            use_hybrid_scoring: Combine LLM with traditional ML scores
            max_batch_size: Maximum items to process in one batch
            api_key: OpenAI API key (if None, will use OPENAI_API_KEY env var)
            enable_learning: Whether to enable learning from feedback
            learning_db_path: Path to SQLite database for storing feedback
        """
        self.model_name = model_name
        self.initial_threshold = confidence_threshold  # Store original threshold
        self.confidence_threshold = confidence_threshold
        self.use_hybrid_scoring = use_hybrid_scoring
        self.max_batch_size = max_batch_size
        self.enable_learning = enable_learning
        self.learning_db_path = Path(learning_db_path)
        
        # Initialize OpenAI client
        api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        self.client = openai.OpenAI(api_key=api_key)
        
        # Initialize components
        self.system_prompt = self._create_system_prompt()
        self.matching_cache = {}
        
        # Learning components
        if self.enable_learning:
            self._init_learning_database()
            self._load_learned_threshold()
            self.learned_patterns = self._load_learned_patterns()
        else:
            self.learned_patterns = {}
        
        # Statistics
        self.stats = {
            'total_llm_calls': 0,
            'successful_matches': 0,
            'cache_hits': 0,
            'processing_time': 0.0,
            'token_usage': 0,
            'feedback_entries': 0,
            'learning_updates': 0
        }
        
        if self.enable_learning:
            logger.info(f"LLM Matcher with Learning initialized - Threshold: {self.confidence_threshold} (adaptive)")
        else:
            logger.info(f"LLM Matcher initialized - Threshold: {self.confidence_threshold} (static)")
        
    def _create_system_prompt(self) -> str:
        """Create the system prompt for food product matching"""
        return """You are an expert food product matching system. Your task is to determine if two food product descriptions refer to the same or equivalent products.

Key considerations for matching:
1. CORE PRODUCT: Same base food item (e.g., "chicken breast" vs "chicken breast fillet")
2. BRAND VARIATIONS: Different brands of same product can match if core product is identical
3. SIZE/WEIGHT: Different package sizes of same product should match
4. PREPARATION: Different preparations (raw/cooked/frozen) are typically different products
5. FLAVORS/VARIANTS: Different flavors are usually different products unless explicitly requested
6. NUTRITIONAL EQUIVALENCE: Products with similar nutritional profiles and ingredients

Matching Rules:
- EXACT MATCH (0.95-1.0): Nearly identical products, minor wording differences
- HIGH MATCH (0.8-0.94): Same core product, different brands/packaging
- MEDIUM MATCH (0.6-0.79): Related products, some differences in preparation/form
- LOW MATCH (0.3-0.59): Different but related food category
- NO MATCH (0.0-0.29): Completely different products

Always provide:
1. Match confidence score (0.0-1.0)
2. Clear reasoning for the decision
3. Key factors that influenced the score
4. Simplified product names for both products (e.g., "cheese stick", "chicken breast", "whole milk")

For simplified_product names, use the most specific but still general form:
- "Cheddar Cheese" → "cheddar cheese" (not just "cheese" or "dairy")
- "Chicken Breast Fillets" → "chicken breast" (specific cut)
- "Apple Juice" → "apple juice" (specific type)
- "White Bread" → "white bread" (specific type)
- "Cheese Sticks" → "cheese stick" (specific form)

Respond in JSON format:
{
    "confidence": 0.85,
    "match": true,
    "reasoning": "Both are chicken breast products. Product A is fresh while Product B is frozen, but they are nutritionally equivalent base products.",
    "key_factors": ["same_core_product", "different_preparation_state", "brand_difference"],
    "category_match": true,
    "nutritional_similarity": "high",
    "simplified_product_a": "chicken breast",
    "simplified_product_b": "chicken breast"
}"""

    @monitor_performance("llm_matching")
    async def match_products_llm(self, 
                                product1: str, 
                                product2: str,
                                context: Dict = None) -> Dict[str, Any]:
        """
        Use LLM to match two products
        
        Args:
            product1: First product description
            product2: Second product description  
            context: Additional context (brand, category, etc.)
            
        Returns:
            Match result with confidence and reasoning
        """
        # Check cache first
        cache_key = f"{product1}||{product2}"
        if cache_key in self.matching_cache:
            self.stats['cache_hits'] += 1
            return self.matching_cache[cache_key]
            
        # Prepare the matching prompt
        prompt = self._create_matching_prompt(product1, product2, context)
        
        try:
            # Make LLM call
            self.stats['total_llm_calls'] += 1
            result = await self._call_llm(prompt)
            
            # Parse and validate response
            parsed_result = self._parse_llm_response(result)
            
            # Cache the result
            self.matching_cache[cache_key] = parsed_result
            
            if parsed_result.get('match', False):
                self.stats['successful_matches'] += 1
                
            return parsed_result
            
        except Exception as e:
            logger.error(f"LLM matching error: {e}")
            return self._fallback_match(product1, product2)
            
    def _create_matching_prompt(self, 
                               product1: str, 
                               product2: str,
                               context: Dict = None) -> str:
        """Create the specific matching prompt for two products"""
        
        context_info = ""
        if context:
            context_info = f"\nAdditional context:\n"
            for key, value in context.items():
                context_info += f"- {key}: {value}\n"
                
        prompt = f"""Compare these two food products and determine if they match:

Product A: "{product1}"
Product B: "{product2}"
{context_info}

Analyze these products considering:
1. Core food item identity
2. Brand differences (if any)
3. Package size/weight differences
4. Preparation method differences
5. Flavor/variant differences
6. Nutritional equivalence

Provide your analysis in the specified JSON format."""

        return prompt
        
    async def _call_llm(self, prompt: str) -> str:
        """
        Make API call to LLM service
        
        Args:
            prompt: The prompt to send
            
        Returns:
            LLM response text
        """
        # This is a placeholder for actual LLM API integration
        # In practice, you would integrate with OpenAI, Anthropic, or local models
        
        # Make real API call to OpenAI GPT-4o-mini  
        # Use sync version in async context - OpenAI client handles this internally
        return self._call_openai_api(prompt)
        
    def _call_openai_api(self, prompt: str) -> str:
        """Make real API call to OpenAI GPT-4o-mini"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Low temperature for consistent results
                max_tokens=500,   # Reasonable limit for cost control
                response_format={"type": "json_object"}  # Ensure JSON response
            )
            
            # Update statistics
            self.stats['total_llm_calls'] += 1
            self.stats['token_usage'] += response.usage.total_tokens
            
            content = response.choices[0].message.content
            
            # Log successful API call
            logger.debug(f"OpenAI API call successful. Tokens used: {response.usage.total_tokens}")
            
            return content
            
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            # Return a structured error response
            return json.dumps({
                "confidence": 0.0,
                "match": False,
                "reasoning": f"API call failed: {str(e)}",
                "brand_analysis": "Unable to analyze due to API error",
                "matched_food_categories": [],
                "key_factors": ["api_error"],
                "category_match": False,
                "nutritional_similarity": "unknown",
                "simplified_product_a": "",
                "simplified_product_b": ""
            })
        
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse and validate LLM response"""
        try:
            parsed = json.loads(response)
            
            # Validate required fields
            required_fields = ['confidence', 'match', 'reasoning']
            for field in required_fields:
                if field not in parsed:
                    raise ValueError(f"Missing required field: {field}")
                    
            # Validate confidence range
            confidence = parsed['confidence']
            if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
                raise ValueError(f"Invalid confidence value: {confidence}")
                
            return parsed
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Error parsing LLM response: {e}")
            return self._create_fallback_response()
            
    def _create_fallback_response(self) -> Dict[str, Any]:
        """Create a fallback response when LLM fails"""
        return {
            "confidence": 0.0,
            "match": False,
            "reasoning": "LLM processing failed, using fallback",
            "key_factors": ["processing_error"],
            "category_match": False,
            "nutritional_similarity": "unknown",
            "simplified_product_a": "",
            "simplified_product_b": ""
        }
        
    def _fallback_match(self, product1: str, product2: str) -> Dict[str, Any]:
        """Fallback matching when API fails - minimal functionality"""
        logger.warning("Using fallback matching due to API failure")
        
        # Basic word overlap similarity
        words1 = set(product1.lower().split())
        words2 = set(product2.lower().split())
        if words1 and words2:
            similarity = len(words1 & words2) / len(words1 | words2)
        else:
            similarity = 0.0
            
        return {
            "confidence": similarity,
            "match": similarity >= 0.5,
            "reasoning": f"Fallback word overlap matching due to API failure (similarity: {similarity:.3f})",
            "brand_analysis": "Unable to analyze brands in fallback mode",
            "matched_food_categories": [],
            "key_factors": ["api_fallback"],
            "category_match": False,
            "nutritional_similarity": "unknown",
            "simplified_product_a": "",
            "simplified_product_b": ""
        }
            
    async def batch_match_products(self, 
                                  product_pairs: List[Tuple[str, str]],
                                  contexts: List[Dict] = None) -> List[Dict[str, Any]]:
        """
        Match multiple product pairs efficiently
        
        Args:
            product_pairs: List of (product1, product2) tuples
            contexts: Optional list of context dicts for each pair
            
        Returns:
            List of match results
        """
        logger.info(f"Batch matching {len(product_pairs)} product pairs")
        
        results = []
        contexts = contexts or [None] * len(product_pairs)
        
        # Process in batches to avoid overwhelming the LLM
        for i in range(0, len(product_pairs), self.max_batch_size):
            batch_pairs = product_pairs[i:i + self.max_batch_size]
            batch_contexts = contexts[i:i + self.max_batch_size]
            
            # Create async tasks for this batch
            tasks = []
            for (prod1, prod2), context in zip(batch_pairs, batch_contexts):
                task = self.match_products_llm(prod1, prod2, context)
                tasks.append(task)
                
            # Execute batch
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle any exceptions
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Error in batch matching: {result}")
                    pair = batch_pairs[j]
                    result = self._fallback_match(pair[0], pair[1])
                    
                results.append(result)
                
        logger.info(f"Completed batch matching with {self.stats['successful_matches']} successful matches")
        return results
        
    async def match_dataframes_llm(self, 
                            df1: pd.DataFrame, 
                            df2: pd.DataFrame,
                            text_col1: str, 
                            text_col2: str,
                            max_matches_per_item: int = 3,
                            pre_filter_similarity: float = 0.4) -> pd.DataFrame:
        """
        Match items between dataframes using LLM
        
        Args:
            df1: First dataframe
            df2: Second dataframe
            text_col1: Text column in df1
            text_col2: Text column in df2
            max_matches_per_item: Maximum matches per item
            pre_filter_similarity: Pre-filter threshold to reduce LLM calls
            
        Returns:
            DataFrame with LLM-enhanced matches
        """
        logger.info(f"LLM matching {len(df1)} x {len(df2)} items")
        
        # Pre-filter using fuzzy matching to reduce LLM calls
        try:
            candidate_pairs = self._pre_filter_candidates(
                df1, df2, text_col1, text_col2, pre_filter_similarity
            )
        except ImportError:
            logger.warning("fuzzywuzzy not available, using all pairs")
            candidate_pairs = []
            for idx1, row1 in df1.iterrows():
                for idx2, row2 in df2.iterrows():
                    candidate_pairs.append({
                        'idx1': idx1,
                        'idx2': idx2,
                        'text1': str(row1[text_col1]),
                        'text2': str(row2[text_col2]),
                        'similarity': 0.5  # Default similarity
                    })
        
        logger.info(f"Pre-filtered to {len(candidate_pairs)} candidate pairs")
        
        # Prepare pairs for LLM matching
        product_pairs = [(pair['text1'], pair['text2']) for pair in candidate_pairs]
        contexts = [
            {
                'source_index': pair['idx1'],
                'target_index': pair['idx2'],
                'initial_similarity': pair['similarity']
            }
            for pair in candidate_pairs
        ]
        
        # Run LLM matching
        llm_results = await self.batch_match_products(product_pairs, contexts)
        
        # Combine results
        matches = []
        for candidate, llm_result in zip(candidate_pairs, llm_results):
            if llm_result['match'] and llm_result['confidence'] >= self.confidence_threshold:
                match_data = {
                    'source_index': candidate['idx1'],
                    'target_index': candidate['idx2'],
                    'source_text': candidate['text1'],
                    'target_text': candidate['text2'],
                    'llm_confidence': llm_result['confidence'],
                    'llm_reasoning': llm_result['reasoning'],
                    'brand_analysis': llm_result.get('brand_analysis', ''),
                    'matched_food_categories': llm_result.get('matched_food_categories', []),
                    'initial_similarity': candidate['similarity'],
                    'hybrid_score': self._calculate_hybrid_score(
                        candidate['similarity'], 
                        llm_result['confidence']
                    ),
                    'key_factors': llm_result.get('key_factors', []),
                    'category_match': llm_result.get('category_match', False),
                    'nutritional_similarity': llm_result.get('nutritional_similarity', 'unknown'),
                    'simplified_product_a': llm_result.get('simplified_product_a', ''),
                    'simplified_product_b': llm_result.get('simplified_product_b', '')
                }
                matches.append(match_data)
                
        # Convert to DataFrame and sort by hybrid score
        matches_df = pd.DataFrame(matches)
        if not matches_df.empty:
            matches_df = matches_df.sort_values('hybrid_score', ascending=False)
            
            # Limit matches per source item
            matches_df = (matches_df.groupby('source_index')
                         .head(max_matches_per_item)
                         .reset_index(drop=True))
                         
        logger.info(f"Found {len(matches_df)} high-confidence LLM matches")
        return matches_df
        
    def _pre_filter_candidates(self, 
                              df1: pd.DataFrame, 
                              df2: pd.DataFrame,
                              text_col1: str, 
                              text_col2: str,
                              threshold: float) -> List[Dict]:
        """Pre-filter candidate pairs using fast similarity"""
        try:
            from fuzzywuzzy import fuzz
            use_fuzzy = True
        except ImportError:
            logger.warning("fuzzywuzzy not available, using basic string matching")
            use_fuzzy = False
            
        candidates = []
        
        for idx1, row1 in df1.iterrows():
            text1 = str(row1[text_col1])
            
            for idx2, row2 in df2.iterrows():
                text2 = str(row2[text_col2])
                
                if use_fuzzy:
                    # Use fuzzywuzzy for better similarity
                    similarity = fuzz.token_set_ratio(text1, text2) / 100.0
                else:
                    # Fallback to basic word overlap
                    words1 = set(text1.lower().split())
                    words2 = set(text2.lower().split())
                    if words1 and words2:
                        similarity = len(words1 & words2) / len(words1 | words2)
                    else:
                        similarity = 0.0
                
                if similarity >= threshold:
                    candidates.append({
                        'idx1': idx1,
                        'idx2': idx2,
                        'text1': text1,
                        'text2': text2,
                        'similarity': similarity
                    })
                    
        return candidates
        
    def _calculate_hybrid_score(self, 
                               similarity_score: float, 
                               llm_confidence: float) -> float:
        """Calculate hybrid score combining similarity and LLM confidence"""
        if not self.use_hybrid_scoring:
            return llm_confidence
            
        # Weighted combination: LLM gets higher weight
        hybrid_score = (0.3 * similarity_score) + (0.7 * llm_confidence)
        return hybrid_score
        
    def learn_from_feedback(self, 
                           product1: str, 
                           product2: str,
                           is_correct_match: bool,
                           llm_result: Dict = None):
        """
        Learn from user feedback to improve matching
        
        Args:
            product1: First product
            product2: Second product
            is_correct_match: Whether the match was correct
            llm_result: Original LLM result for this pair
        """
        # Store feedback for model improvement
        feedback_entry = {
            'timestamp': datetime.now().isoformat(),
            'product1': product1,
            'product2': product2,
            'is_correct': is_correct_match,
            'llm_result': llm_result,
            'confidence_correct': (
                is_correct_match == (llm_result.get('confidence', 0) >= self.confidence_threshold)
                if llm_result else False
            )
        }
        
        # In a production system, this would update model weights or training data
        logger.info(f"Feedback recorded: {product1} <-> {product2} = {is_correct_match}")
        
    def get_matching_stats(self) -> Dict[str, Any]:
        """Get comprehensive matching statistics"""
        stats = self.stats.copy()
        
        # Calculate derived metrics
        if stats['total_llm_calls'] > 0:
            stats['success_rate'] = stats['successful_matches'] / stats['total_llm_calls']
            stats['cache_hit_rate'] = stats['cache_hits'] / (stats['total_llm_calls'] + stats['cache_hits'])
        else:
            stats['success_rate'] = 0.0
            stats['cache_hit_rate'] = 0.0
            
        stats['cache_size'] = len(self.matching_cache)
        
        return stats
        
    def clear_cache(self):
        """Clear the matching cache"""
        self.matching_cache.clear()
        logger.info("Matching cache cleared")
        
    def export_cache(self, filepath: str):
        """Export matching cache for analysis"""
        with open(filepath, 'w') as f:
            json.dump(self.matching_cache, f, indent=2)
        logger.info(f"Cache exported to {filepath}")
    
    # ==================== LEARNING SYSTEM METHODS ====================
    
    def _init_learning_database(self):
        """Initialize SQLite database for learning feedback"""
        with sqlite3.connect(self.learning_db_path) as conn:
            cursor = conn.cursor()
            
            # Feedback table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    product1 TEXT NOT NULL,
                    product2 TEXT NOT NULL,
                    llm_confidence REAL NOT NULL,
                    llm_prediction BOOLEAN NOT NULL,
                    user_feedback BOOLEAN NOT NULL,
                    simplified_product_a TEXT,
                    simplified_product_b TEXT,
                    correction_reasoning TEXT
                )
            """)
            
            # Learning state table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS learning_state (
                    id INTEGER PRIMARY KEY,
                    confidence_threshold REAL NOT NULL,
                    learned_patterns TEXT,
                    last_updated TEXT NOT NULL,
                    total_feedback_count INTEGER DEFAULT 0
                )
            """)
            
            conn.commit()
    
    def _load_learned_threshold(self):
        """Load learned confidence threshold from database"""
        if not self.learning_db_path.exists():
            return
            
        with sqlite3.connect(self.learning_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT confidence_threshold FROM learning_state WHERE id = 1")
            result = cursor.fetchone()
            
            if result:
                learned_threshold = result[0]
                logger.info(f"Loaded learned threshold: {learned_threshold} (was {self.confidence_threshold})")
                self.confidence_threshold = learned_threshold
    
    def _load_learned_patterns(self) -> Dict[str, Any]:
        """Load learned patterns from database"""
        if not self.learning_db_path.exists():
            return {}
            
        with sqlite3.connect(self.learning_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT learned_patterns FROM learning_state WHERE id = 1")
            result = cursor.fetchone()
            
            if result and result[0]:
                try:
                    return json.loads(result[0])
                except json.JSONDecodeError:
                    logger.warning("Could not parse learned patterns from database")
                    
        return {}
    
    def _save_learning_state(self):
        """Save current learning state to database"""
        with sqlite3.connect(self.learning_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO learning_state 
                (id, confidence_threshold, learned_patterns, last_updated, total_feedback_count)
                VALUES (1, ?, ?, ?, ?)
            """, (
                self.confidence_threshold,
                json.dumps(self.learned_patterns),
                datetime.now().isoformat(),
                self.stats.get('feedback_entries', 0)
            ))
            conn.commit()
    
    def add_feedback(self, feedback: FeedbackEntry) -> Dict[str, Any]:
        """
        Add user feedback to improve the model
        
        Args:
            feedback: FeedbackEntry with user correction
            
        Returns:
            Learning insights and system adjustments
        """
        if not self.enable_learning:
            return {"status": "learning_disabled", "message": "Learning is not enabled for this matcher"}
        
        logger.info(f"Adding feedback: {feedback.product1} <-> {feedback.product2} = {feedback.user_feedback}")
        
        # Store feedback in database
        with sqlite3.connect(self.learning_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO feedback (
                    timestamp, session_id, product1, product2, llm_confidence,
                    llm_prediction, user_feedback, simplified_product_a, 
                    simplified_product_b, correction_reasoning
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                feedback.timestamp, feedback.session_id, feedback.product1, feedback.product2,
                feedback.llm_confidence, feedback.llm_prediction, feedback.user_feedback,
                feedback.simplified_product_a, feedback.simplified_product_b, 
                feedback.correction_reasoning
            ))
            conn.commit()
        
        # Update statistics
        self.stats['feedback_entries'] += 1
        
        # Analyze feedback and learn
        insights = self._analyze_feedback(feedback)
        
        # Check if we should update learning
        if self._should_update_learning():
            learning_updates = self._update_learning()
            insights.update(learning_updates)
            self.stats['learning_updates'] += 1
        
        return insights
    
    def _analyze_feedback(self, feedback: FeedbackEntry) -> Dict[str, Any]:
        """Analyze individual feedback for immediate insights"""
        insights = {
            "feedback_type": "correction" if feedback.llm_prediction != feedback.user_feedback else "confirmation",
            "confidence_level": "high" if feedback.llm_confidence > 0.8 else "medium" if feedback.llm_confidence > 0.6 else "low",
            "patterns_updated": []
        }
        
        # Update learned patterns
        if feedback.simplified_product_a and feedback.simplified_product_b:
            pattern_key = f"{feedback.simplified_product_a}||{feedback.simplified_product_b}"
            
            if pattern_key not in self.learned_patterns:
                self.learned_patterns[pattern_key] = {
                    "correct_predictions": 0,
                    "incorrect_predictions": 0,
                    "avg_confidence_when_correct": [],
                    "avg_confidence_when_wrong": []
                }
            
            pattern = self.learned_patterns[pattern_key]
            
            if feedback.llm_prediction == feedback.user_feedback:
                # Correct prediction
                pattern["correct_predictions"] += 1
                pattern["avg_confidence_when_correct"].append(feedback.llm_confidence)
                insights["patterns_updated"].append(f"Reinforced pattern: {pattern_key}")
            else:
                # Incorrect prediction
                pattern["incorrect_predictions"] += 1
                pattern["avg_confidence_when_wrong"].append(feedback.llm_confidence)
                insights["patterns_updated"].append(f"Corrected pattern: {pattern_key}")
        
        return insights
    
    def _should_update_learning(self) -> bool:
        """Check if we should trigger a learning update"""
        # Update learning every 10 feedback entries
        return self.stats['feedback_entries'] % 10 == 0 and self.stats['feedback_entries'] > 0
    
    def _update_learning(self) -> Dict[str, Any]:
        """Update learning based on accumulated feedback"""
        logger.info("Updating learning based on recent feedback")
        
        # Calculate new optimal threshold
        old_threshold = self.confidence_threshold
        new_threshold = self._calculate_optimal_threshold()
        
        updates = {
            "threshold_updated": False,
            "old_threshold": old_threshold,
            "new_threshold": new_threshold,
            "performance_metrics": self._calculate_recent_performance()
        }
        
        # Update threshold if significantly different
        if abs(new_threshold - old_threshold) > 0.05:
            self.confidence_threshold = new_threshold
            updates["threshold_updated"] = True
            logger.info(f"Threshold updated: {old_threshold:.3f} -> {new_threshold:.3f}")
        
        # Save learning state
        self._save_learning_state()
        
        return updates
    
    def _calculate_optimal_threshold(self) -> float:
        """Calculate optimal confidence threshold based on recent feedback"""
        with sqlite3.connect(self.learning_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT llm_confidence, llm_prediction, user_feedback 
                FROM feedback 
                ORDER BY timestamp DESC 
                LIMIT 50
            """)
            recent_feedback = cursor.fetchall()
        
        if len(recent_feedback) < 10:
            return self.confidence_threshold
        
        # Try different thresholds and find the one with best F1 score
        best_threshold = self.confidence_threshold
        best_f1 = 0
        
        for threshold in np.arange(0.5, 0.95, 0.05):
            tp = fp = tn = fn = 0
            
            for confidence, llm_pred, user_feedback in recent_feedback:
                predicted_match = confidence >= threshold
                actual_match = user_feedback
                
                if predicted_match and actual_match:
                    tp += 1
                elif predicted_match and not actual_match:
                    fp += 1
                elif not predicted_match and actual_match:
                    fn += 1
                else:
                    tn += 1
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        # Apply smoothing to avoid drastic changes
        learning_rate = 0.2
        adjusted_threshold = self.confidence_threshold + learning_rate * (best_threshold - self.confidence_threshold)
        return round(adjusted_threshold, 3)
    
    def _calculate_recent_performance(self) -> Dict[str, float]:
        """Calculate performance metrics for recent feedback"""
        with sqlite3.connect(self.learning_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT llm_prediction, user_feedback, llm_confidence 
                FROM feedback 
                ORDER BY timestamp DESC 
                LIMIT 30
            """)
            results = cursor.fetchall()
        
        if not results:
            return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0}
        
        # Calculate metrics
        correct = sum(1 for llm_pred, user_fb, _ in results if llm_pred == user_fb)
        total = len(results)
        accuracy = correct / total
        
        # Calculate precision, recall, F1
        tp = sum(1 for llm_pred, user_fb, _ in results if llm_pred and user_fb)
        fp = sum(1 for llm_pred, user_fb, _ in results if llm_pred and not user_fb)
        fn = sum(1 for llm_pred, user_fb, _ in results if not llm_pred and user_fb)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "accuracy": round(accuracy, 3),
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1_score": round(f1_score, 3),
            "total_samples": total
        }
    
    def get_learning_insights(self) -> Dict[str, Any]:
        """Get comprehensive learning insights and statistics"""
        if not self.enable_learning:
            return {"status": "learning_disabled"}
        
        insights = {
            "learning_enabled": True,
            "current_threshold": self.confidence_threshold,
            "initial_threshold": self.initial_threshold,
            "threshold_adjustment": self.confidence_threshold - self.initial_threshold,
            "total_feedback": self.stats.get('feedback_entries', 0),
            "learning_updates": self.stats.get('learning_updates', 0),
            "learned_patterns_count": len(self.learned_patterns),
            "recent_performance": self._calculate_recent_performance()
        }
        
        # Add pattern analysis
        if self.learned_patterns:
            problem_patterns = []
            reliable_patterns = []
            
            for pattern_key, pattern_data in self.learned_patterns.items():
                total_predictions = pattern_data["correct_predictions"] + pattern_data["incorrect_predictions"]
                if total_predictions >= 3:  # Only analyze patterns with sufficient data
                    accuracy = pattern_data["correct_predictions"] / total_predictions
                    
                    if accuracy < 0.6:
                        problem_patterns.append({
                            "pattern": pattern_key,
                            "accuracy": round(accuracy, 3),
                            "total_predictions": total_predictions
                        })
                    elif accuracy > 0.8:
                        reliable_patterns.append({
                            "pattern": pattern_key,
                            "accuracy": round(accuracy, 3),
                            "total_predictions": total_predictions
                        })
            
            insights["problem_patterns"] = sorted(problem_patterns, key=lambda x: x["accuracy"])[:5]
            insights["reliable_patterns"] = sorted(reliable_patterns, key=lambda x: x["accuracy"], reverse=True)[:5]
        
        return insights
    
    def simulate_learning_session(self, test_cases: List[Tuple[str, str, bool, str, str]]) -> Dict[str, Any]:
        """
        Simulate a learning session for testing
        
        Args:
            test_cases: List of (product1, product2, correct_match, simplified_a, simplified_b) tuples
            
        Returns:
            Simulation results
        """
        if not self.enable_learning:
            return {"status": "learning_disabled"}
        
        session_id = f"simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        results = []
        
        initial_threshold = self.confidence_threshold
        
        for i, (product1, product2, correct_match, simplified_a, simplified_b) in enumerate(test_cases):
            # Simulate LLM prediction (in real use, this would come from actual LLM call)
            simulated_confidence = np.random.uniform(0.5, 0.95)
            simulated_prediction = simulated_confidence >= self.confidence_threshold
            
            feedback = FeedbackEntry(
                timestamp=datetime.now().isoformat(),
                session_id=session_id,
                product1=product1,
                product2=product2,
                llm_confidence=simulated_confidence,
                llm_prediction=simulated_prediction,
                user_feedback=correct_match,
                simplified_product_a=simplified_a,
                simplified_product_b=simplified_b,
                correction_reasoning=f"Test case {i+1}"
            )
            
            insights = self.add_feedback(feedback)
            results.append({
                "case": i + 1,
                "products": f"{product1} <-> {product2}",
                "llm_prediction": simulated_prediction,
                "user_feedback": correct_match,
                "confidence": round(simulated_confidence, 3),
                "insights": insights
            })
        
        final_insights = self.get_learning_insights()
        
        return {
            "session_id": session_id,
            "total_cases": len(test_cases),
            "initial_threshold": initial_threshold,
            "final_threshold": self.confidence_threshold,
            "threshold_changed": abs(self.confidence_threshold - initial_threshold) > 0.01,
            "results": results,
            "learning_summary": final_insights
        }