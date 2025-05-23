"""
Food matching algorithms to match products across different datasets
with machine learning capabilities for continuous improvement
"""

import os
import re
import logging
import pandas as pd
import numpy as np
import pickle
from typing import Dict, List, Optional, Union, Tuple, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from fuzzywuzzy import fuzz
from sklearn.model_selection import StratifiedKFold
from Levenshtein import distance as lev_distance
import joblib

from shelfscale.config_manager import get_config

# Configure logging
logger = logging.getLogger(__name__)

# Get configuration
config = get_config()


def preprocess_text(text: str) -> str:
    """
    Preprocess text for matching by lowercasing and removing punctuation
    
    Args:
        text: Input text
        
    Returns:
        Preprocessed text
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    # Convert to lowercase and remove extra whitespace
    processed = text.lower().strip()
    
    # Replace punctuation with spaces
    for char in ",.;:!?/\\()[]{}<>\"'":
        processed = processed.replace(char, ' ')
    
    # Replace multiple spaces with a single space
    processed = ' '.join(processed.split())
    
    return processed


class FoodMatcher:
    """Food matching algorithm with machine learning capabilities to find matches between food datasets"""
    
    def __init__(self, similarity_threshold: float = 0.6, 
                 model_path: Optional[str] = None,
                 features_path: Optional[str] = None,
                 learning_enabled: bool = True):
        """
        Initialize the food matcher
        
        Args:
            similarity_threshold: Threshold for considering a match
            model_path: Path to save/load the machine learning model. Defaults to config.FOOD_MATCHER_MODEL_PATH.
            features_path: Path to save/load feature importance information. Defaults to config.FOOD_MATCHER_FEATURES_PATH.
            learning_enabled: Whether to use machine learning capabilities
        """
        self.similarity_threshold = similarity_threshold
        self.vectorizer = None
        self.model_path = model_path if model_path is not None else config.FOOD_MATCHER_MODEL_PATH
        self.features_path = features_path if features_path is not None else config.FOOD_MATCHER_FEATURES_PATH
        self.learning_enabled = learning_enabled
        self.model = None
        self.feature_importance = {}
        
        # Model directory is created by config.py
        # os.makedirs(os.path.dirname(self.model_path), exist_ok=True) 
        
        # Try to load existing model and features
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the trained machine learning model and feature importance if available"""
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                logger.info(f"Loaded existing matching model from {self.model_path}")
            
            if os.path.exists(self.features_path):
                with open(self.features_path, 'rb') as f:
                    self.feature_importance = pickle.load(f)
                logger.info(f"Loaded feature importance data from {self.features_path}")
        except Exception as e:
            logger.warning(f"Could not load model or features: {e}")
            self.model = None
            self.feature_importance = {}
    
    def _save_model(self) -> None:
        """Save the trained machine learning model and feature importance"""
        try:
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.model, f)
            
            with open(self.features_path, 'wb') as f:
                pickle.dump(self.feature_importance, f)
            
            logger.info(f"Saved matching model to {self.model_path}")
        except Exception as e:
            logger.warning(f"Could not save model or features: {e}")
    
    def _compute_tfidf_similarity(self, source_df: pd.DataFrame, 
                                 target_df: pd.DataFrame,
                                 source_col: str,
                                 target_col: str) -> pd.DataFrame:
        """
        Compute TF-IDF based cosine similarity between two datasets
        
        Args:
            source_df: Source DataFrame
            target_df: Target DataFrame
            source_col: Column name in source DataFrame for matching
            target_col: Column name in target DataFrame for matching
            
        Returns:
            DataFrame with match results
        """
        # Validate columns exist
        if source_col not in source_df.columns:
            logger.error(f"Source column '{source_col}' not found. Available columns: {source_df.columns.tolist()}")
            # Try to find a similar column name
            possible_cols = [col for col in source_df.columns if source_col.lower().replace('_', ' ') in col.lower().replace('_', ' ')]
            if possible_cols:
                source_col = possible_cols[0]
                logger.info(f"Using '{source_col}' instead")
            else:
                logger.error(f"No suitable replacement for '{source_col}' found")
                # Return empty similarity matrix
                return np.zeros((len(source_df), len(target_df)))
        
        if target_col not in target_df.columns:
            logger.error(f"Target column '{target_col}' not found. Available columns: {target_df.columns.tolist()}")
            # Try to find a similar column name
            possible_cols = [col for col in target_df.columns if target_col.lower().replace('_', ' ') in col.lower().replace('_', ' ')]
            if possible_cols:
                target_col = possible_cols[0]
                logger.info(f"Using '{target_col}' instead")
            else:
                logger.error(f"No suitable replacement for '{target_col}' found")
                # Return empty similarity matrix
                return np.zeros((len(source_df), len(target_df)))
                
        # Get text columns and preprocess
        source_texts = source_df[source_col].apply(preprocess_text)
        target_texts = target_df[target_col].apply(preprocess_text)
        
        # Fit and transform the vectorizer on combined texts
        all_texts = pd.concat([source_texts, target_texts])
        self.vectorizer = TfidfVectorizer(analyzer='word', 
                                         ngram_range=(1, 2),
                                         min_df=2,
                                         stop_words='english')
        
        # Create TF-IDF matrix
        tfidf_matrix = self.vectorizer.fit_transform(all_texts)
        
        # Split matrices back to source and target
        source_tfidf = tfidf_matrix[:len(source_texts)]
        target_tfidf = tfidf_matrix[len(source_texts):]
        
        # Compute cosine similarity
        similarity_matrix = cosine_similarity(source_tfidf, target_tfidf)
        
        return similarity_matrix
    
    def _calculate_additional_features(self, source_item: pd.Series, target_item: pd.Series,
                                      source_col: str, target_col: str,
                                      additional_match_cols: List[Tuple[str, str]] = None) -> Dict[str, float]:
        """
        Calculate additional features for matching beyond TF-IDF
        
        Args:
            source_item: Source item series
            target_item: Target item series
            source_col: Primary column in source for matching
            target_col: Primary column in target for matching
            additional_match_cols: Additional columns to use for matching
            
        Returns:
            Dictionary of features
        """
        features = {}
        
        # Source and target text - safely handle missing columns
        source_text = str(source_item.get(source_col, "")) if source_col in source_item else ""
        target_text = str(target_item.get(target_col, "")) if target_col in target_item else ""
        
        # Log warning if columns are missing
        if source_col not in source_item:
            logger.warning(f"Source column '{source_col}' not found in item. Available: {source_item.index.tolist()}")
            
        if target_col not in target_item:
            logger.warning(f"Target column '{target_col}' not found in item. Available: {target_item.index.tolist()}")
        
        # 1. Calculate fuzz similarity ratios
        # Use Python-Levenshtein library for faster, more accurate calculations
        features['ratio'] = fuzz.ratio(source_text.lower(), target_text.lower()) / 100.0
        features['partial_ratio'] = fuzz.partial_ratio(source_text.lower(), target_text.lower()) / 100.0
        features['token_sort_ratio'] = fuzz.token_sort_ratio(source_text.lower(), target_text.lower()) / 100.0
        features['token_set_ratio'] = fuzz.token_set_ratio(source_text.lower(), target_text.lower()) / 100.0
        
        # 2. Levenshtein edit distance - normalized to a similarity score
        max_len = max(len(source_text), len(target_text))
        if max_len > 0:
            try:
                # Handle import differences
                try:
                    # New way using imported lev_distance
                    edit_distance = lev_distance(source_text.lower(), target_text.lower())
                except NameError:
                    # Fallback to fuzzywuzzy's implementation
                    from Levenshtein import distance as levenshtein_distance
                    edit_distance = levenshtein_distance(source_text.lower(), target_text.lower())
                features['levenshtein_similarity'] = 1 - (edit_distance / max_len)
            except Exception as e:
                logger.warning(f"Error calculating Levenshtein distance: {e}")
                features['levenshtein_similarity'] = 0
        else:
            features['levenshtein_similarity'] = 0
        
        # Food domain-specific features
        
        # 1. Food group similarity - if food groups are available
        food_group_match = 0.0
        source_group = source_item.get('Food Group', '') or source_item.get('Food_Group', '') or source_item.get('Super Group', '') or source_item.get('Super_Group', '')
        target_group = target_item.get('Food Group', '') or target_item.get('Food_Group', '') or target_item.get('Super Group', '') or target_item.get('Super_Group', '')
        
        if source_group and target_group:
            source_group = str(source_group).lower()
            target_group = str(target_group).lower()
            # Exact match gets full score
            if source_group == target_group:
                food_group_match = 1.0
            # Partial match gets partial score
            elif source_group in target_group or target_group in source_group:
                food_group_match = 0.7
            # Word-level similarity
            else:
                food_group_match = fuzz.token_sort_ratio(source_group, target_group) / 100.0
        
        features["food_group_match"] = food_group_match
        
        # 2. Food code pattern matching (e.g., McCance & Widdowson codes)
        food_code_match = 0.0
        source_code = source_item.get('Food Code', '') or source_item.get('Food_Code', '')
        target_code = target_item.get('Food Code', '') or target_item.get('Food_Code', '')
        
        if source_code and target_code:
            source_code = str(source_code)
            target_code = str(target_code)
            
            # Exact code match
            if source_code == target_code:
                food_code_match = 1.0
            # Code prefix match (same food category)
            elif source_code.split('-')[0] == target_code.split('-')[0]:
                food_code_match = 0.8
            # Similar codes (Levenstein distance)
            else:
                food_code_match = max(0, 1 - (lev_distance(source_code, target_code) / max(len(source_code), len(target_code))))
        
        features["food_code_match"] = food_code_match
        
        # 3. Detect food preparation state (raw, cooked, frozen, etc.)
        source_states = self._detect_food_state(source_text)
        target_states = self._detect_food_state(target_text)
        
        # Calculate state similarity score
        if source_states and target_states:
            common_states = set(source_states).intersection(set(target_states))
            all_states = set(source_states).union(set(target_states))
            state_match = len(common_states) / len(all_states) if all_states else 0
        else:
            state_match = 0.5  # Neutral if we can't determine states
        
        features["food_state_match"] = state_match
        
        # 4. Detect and compare unit measures
        source_unit, source_amount = self._extract_unit_measure(source_text)
        target_unit, target_amount = self._extract_unit_measure(target_text)
        
        unit_match = 0.5  # Default neutral
        
        if source_unit and target_unit:
            if source_unit == target_unit:
                unit_match = 1.0
            elif self._are_compatible_units(source_unit, target_unit):
                unit_match = 0.8
            else:
                unit_match = 0.2
        
        features["unit_measure_match"] = unit_match
        
        # 5. Check for ingredient keywords
        common_ingredient_words = self._count_common_ingredient_words(source_text, target_text)
        features["common_ingredient_count"] = min(common_ingredient_words / 3, 1.0)  # Normalize
        
        # Process additional columns if provided
        if additional_match_cols:
            for source_col_name, target_col_name in additional_match_cols:
                if source_col_name in source_item and target_col_name in target_item:
                    source_val = str(source_item[source_col_name]) if not pd.isna(source_item[source_col_name]) else ""
                    target_val = str(target_item[target_col_name]) if not pd.isna(target_item[target_col_name]) else ""
                    
                    if source_val and target_val:
                        # Calculate similarity for this column pair
                        col_similarity = fuzz.token_sort_ratio(source_val.lower(), target_val.lower()) / 100.0
                        features[f"{source_col_name}_{target_col_name}_match"] = col_similarity
        
        return features
    
    def _detect_food_state(self, text: str) -> List[str]:
        """
        Detect food state keywords in text
        
        Args:
            text: Food item text
            
        Returns:
            List of detected states
        """
        states = []
        text = text.lower()
        
        # Common food states
        state_keywords = {
            'raw': ['raw', 'fresh', 'uncooked'],
            'cooked': ['cooked', 'boiled', 'fried', 'roasted', 'baked', 'grilled', 'steamed', 'sautéed', 'sauteed'],
            'frozen': ['frozen', 'freeze', 'freezing'],
            'dried': ['dried', 'dry', 'dehydrated'],
            'canned': ['canned', 'tinned', 'in water', 'in brine', 'in oil', 'in syrup'],
            'processed': ['processed', 'preserved'],
            'whole': ['whole', 'intact', 'unpeeled'],
            'peeled': ['peeled', 'skinless', 'skin removed'],
            'sliced': ['sliced', 'diced', 'chopped', 'cubed', 'pieces']
        }
        
        for state, keywords in state_keywords.items():
            if any(keyword in text for keyword in keywords):
                states.append(state)
        
        return states
    
    def _extract_unit_measure(self, text: str) -> Tuple[str, float]:
        """
        Extract unit and measurement from text
        
        Args:
            text: Food item text
            
        Returns:
            Tuple of (unit, amount)
        """
        text = text.lower()
        
        # Common units and their variations
        units = {
            'g': ['g', 'gram', 'grams'],
            'kg': ['kg', 'kilo', 'kilos', 'kilogram', 'kilograms'],
            'mg': ['mg', 'milligram', 'milligrams'],
            'ml': ['ml', 'milliliter', 'milliliters', 'millilitre', 'millilitres'],
            'l': ['l', 'liter', 'liters', 'litre', 'litres'],
            'oz': ['oz', 'ounce', 'ounces'],
            'lb': ['lb', 'lbs', 'pound', 'pounds'],
            'cup': ['cup', 'cups'],
            'tbsp': ['tbsp', 'tablespoon', 'tablespoons'],
            'tsp': ['tsp', 'teaspoon', 'teaspoons'],
            'piece': ['piece', 'pieces', 'pc', 'pcs']
        }
        
        # Try to match unit and amount
        amount = None
        found_unit = None
        
        # Look for number + unit patterns
        measure_pattern = r'(\d+\.?\d*)\s*([a-zA-Z]+)'
        matches = re.findall(measure_pattern, text)
        
        for match in matches:
            value, unit_text = match
            try:
                value = float(value)
                
                # Find the standardized unit
                for std_unit, variations in units.items():
                    if any(unit_text == var or unit_text.startswith(var) for var in variations):
                        found_unit = std_unit
                        amount = value
                        break
                
                if found_unit:
                    break
            except ValueError:
                continue
        
        return found_unit, amount
    
    def _are_compatible_units(self, unit1: str, unit2: str) -> bool:
        """
        Check if two units are compatible
        
        Args:
            unit1: First unit
            unit2: Second unit
            
        Returns:
            True if units are compatible
        """
        # Define compatible unit groups
        compatible_groups = [
            {'g', 'kg', 'mg'},                # Weight
            {'ml', 'l'},                      # Volume (metric)
            {'oz', 'lb'},                     # Weight (imperial)
            {'cup', 'tbsp', 'tsp'},           # Volume (cooking)
            {'piece'}                         # Count
        ]
        
        for group in compatible_groups:
            if unit1 in group and unit2 in group:
                return True
        
        return False
    
    def _count_common_ingredient_words(self, text1: str, text2: str) -> int:
        """
        Count common ingredient words between texts
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Count of common ingredient words
        """
        # Common ingredient words that might indicate matching food items
        ingredient_keywords = {
            'apple', 'banana', 'beef', 'bread', 'broccoli', 'butter', 'carrot', 'cheese', 'chicken', 
            'chocolate', 'cinnamon', 'cod', 'corn', 'cucumber', 'egg', 'fish', 'flour', 'garlic', 
            'ginger', 'grapes', 'honey', 'lamb', 'lemon', 'lettuce', 'milk', 'mushroom', 'mustard', 
            'oats', 'oil', 'olive', 'onion', 'orange', 'pasta', 'peanut', 'pepper', 'pork', 
            'potato', 'rice', 'salmon', 'salt', 'spinach', 'strawberry', 'sugar', 'tomato', 
            'tuna', 'vanilla', 'vinegar', 'wheat', 'yogurt'
        }
        
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        # Find ingredients in both texts
        common_ingredients = sum(1 for word in words1.intersection(words2) if word in ingredient_keywords)
        
        return common_ingredients
    
    def _predict_match_probability(self, features: Dict[str, float], similarity_score: float) -> float:
        """
        Predict match probability using the trained model or by combining features
        
        Args:
            features: Additional calculated features
            similarity_score: TF-IDF similarity score
            
        Returns:
            Match probability between 0 and 1
        """
        # If we have a trained model and learning is enabled, use it
        if self.model is not None and self.learning_enabled:
            # Prepare features for prediction
            feature_vector = [similarity_score] + list(features.values())
            
            # Make prediction
            try:
                match_probability = self.model.predict_proba([feature_vector])[0, 1]
                return match_probability
            except Exception as e:
                logger.warning(f"Error predicting with model: {e}. Falling back to weighted average.")
        
        # Fall back to weighted average of features
        # Start with the baseline similarity score
        weighted_score = similarity_score * 0.5
        
        # Add weighted contributions from other features
        if 'fuzz_ratio' in features:
            weighted_score += features['fuzz_ratio'] * 0.2
        if 'fuzz_partial_ratio' in features:
            weighted_score += features['fuzz_partial_ratio'] * 0.15
        if 'fuzz_token_sort_ratio' in features:
            weighted_score += features['fuzz_token_sort_ratio'] * 0.15
            
        # Apply feature importance weights if available
        if self.feature_importance:
            weighted_score = 0
            all_features = {'tf_idf': similarity_score, **features}
            total_weight = 0
            
            for feat, value in all_features.items():
                if feat in self.feature_importance:
                    weight = self.feature_importance[feat]
                    weighted_score += value * weight
                    total_weight += weight
            
            if total_weight > 0:
                weighted_score /= total_weight
            else:
                weighted_score = similarity_score  # Default to similarity score
        
        return weighted_score
    
    def match_datasets(self, 
                      source_df: pd.DataFrame, 
                      target_df: pd.DataFrame,
                      source_col: str,
                      target_col: str,
                      additional_match_cols: List[Tuple[str, str]] = None) -> pd.DataFrame:
        """
        Match items between two datasets based on text similarity
        
        Args:
            source_df: Source DataFrame
            target_df: Target DataFrame
            source_col: Column in source DataFrame to use for matching
            target_col: Column in target DataFrame to use for matching
            additional_match_cols: Additional column pairs for matching
            
        Returns:
            DataFrame with match results
        """
        # Log dataset info
        logger.info(f"Source DataFrame columns: {source_df.columns.tolist()}")
        logger.info(f"Target DataFrame columns: {target_df.columns.tolist()}")
        logger.info(f"Looking for source column: '{source_col}' and target column: '{target_col}'")
        
        # Make copies to avoid modifying the originals
        source_df_copy = source_df.copy()
        target_df_copy = target_df.copy()
        
        # Handle empty DataFrames
        if len(source_df_copy) == 0:
            logger.warning("Source DataFrame is empty. Cannot match.")
            return pd.DataFrame(columns=['Source_Index', 'Target_Index', 'Similarity_Score'])
        
        if len(target_df_copy) == 0:
            logger.warning("Target DataFrame is empty. Cannot match.")
            return pd.DataFrame(columns=['Source_Index', 'Target_Index', 'Similarity_Score'])
        
        # Check if the requested columns exist, if not try to find similar ones
        source_col_actual = self._find_similar_column(source_df_copy, source_col)
        if source_col_actual != source_col:
            logger.info(f"Using '{source_col_actual}' instead of '{source_col}' in source DataFrame")
            source_col = source_col_actual
            
        target_col_actual = self._find_similar_column(target_df_copy, target_col)
        if target_col_actual != target_col:
            logger.info(f"Using '{target_col_actual}' instead of '{target_col}' in target DataFrame")
            target_col = target_col_actual
        
        # Check again if columns exist after trying to find similar ones
        if source_col not in source_df_copy.columns:
            logger.error(f"Source column '{source_col}' not found in source DataFrame")
            # Create a dummy column to avoid errors
            source_df_copy[source_col] = ""
            
        if target_col not in target_df_copy.columns:
            logger.error(f"Target column '{target_col}' not found in target DataFrame")
            # Create a dummy column to avoid errors
            target_df_copy[target_col] = ""
        
        # Fill NaN values in matching columns to avoid errors
        source_df_copy[source_col] = source_df_copy[source_col].fillna("").astype(str)
        target_df_copy[target_col] = target_df_copy[target_col].fillna("").astype(str)
        
        # Compute text similarity
        similarity_matrix = self._compute_tfidf_similarity(
            source_df_copy, 
            target_df_copy,
            source_col,
            target_col
        )
        
        # Find potential matches that exceed the threshold
        matches = []
        
        for source_idx, similarities in enumerate(similarity_matrix):
            # Get target indices of potential matches
            target_indices = np.where(similarities >= self.similarity_threshold)[0]
            
            if len(target_indices) == 0:
                continue
                
            # Get the source item
            source_item = source_df_copy.iloc[source_idx]
            
            for target_idx in target_indices:
                # Get the target item
                target_item = target_df_copy.iloc[target_idx]
                
                # Get the similarity score
                similarity = similarities[target_idx]
            
                # Calculate additional features for machine learning scoring
                features = self._calculate_additional_features(
                    source_item, 
                    target_item,
                    source_col,
                    target_col,
                    additional_match_cols
                )
                
                # Use ML model if available for final score
                match_probability = self._predict_match_probability(features, similarity)
                
                # Override with similarity if no ML used
                if not self.learning_enabled or self.model is None:
                    match_probability = similarity
                
                # Store match information
                match_info = {
                    'Source_Index': source_idx,
                    'Target_Index': target_idx,
                    'Similarity_Score': similarity,
                    'Match_Probability': match_probability,
                    'features': features  # Store features for later analysis
                }
                
                matches.append(match_info)
        
        # Create a DataFrame from matches
        if not matches:
            logger.warning("No matches found exceeding the similarity threshold")
            return pd.DataFrame(columns=['Source_Index', 'Target_Index', 'Similarity_Score', 'Match_Probability', 'features'])
        
        match_df = pd.DataFrame(matches)
        
        # Sort by similarity score
        match_df = match_df.sort_values('Similarity_Score', ascending=False)
        
        # Keep only the best match for each source item
        if self.learning_enabled and self.model is not None:
            best_matches = match_df.sort_values('Match_Probability', ascending=False).drop_duplicates(subset=['Source_Index'])
        else:
            best_matches = match_df.sort_values('Similarity_Score', ascending=False).drop_duplicates(subset=['Source_Index'])
        
        logger.info(f"Found {len(best_matches)} matches with similarity > {self.similarity_threshold:.2f}")
        
        return best_matches
        
    def _find_similar_column(self, df: pd.DataFrame, column: str) -> str:
        """
        Find a similar column name if the exact column doesn't exist
        
        Args:
            df: DataFrame to search in
            column: Column name to find
            
        Returns:
            The found column name or the original if no similar column is found
        """
        if column in df.columns:
            return column
            
        # Try different variations
        column_lower = column.lower().replace('_', ' ')
        
        # Check for direct similarity
        for col in df.columns:
            col_lower = col.lower().replace('_', ' ')
            
            # Exact match with different casing/spacing
            if col_lower == column_lower:
                return col
            
            # One is contained in the other
            if col_lower in column_lower or column_lower in col_lower:
                return col
                
        # Handle specific common variations
        common_pairs = {
            'food name': ['food_name', 'foodname', 'name', 'description', 'product', 'product_name', 'product name', 'item', 'item_name'],
            'food_name': ['food name', 'foodname', 'name', 'description', 'product', 'product_name', 'product name', 'item', 'item_name'],
            'name': ['food_name', 'food name', 'foodname', 'description', 'product', 'product_name', 'product name', 'item', 'item_name'],
            'description': ['food_name', 'food name', 'foodname', 'name', 'product', 'product_name', 'product name', 'item', 'item_name'],
            'sample_name': ['sample name', 'name', 'food_name', 'food name', 'product', 'description'],
            'sample name': ['sample_name', 'name', 'food_name', 'food name', 'product', 'description'],
            'weight': ['weight_g', 'weight(g)', 'pack_size', 'portion_size', 'size'],
            'weight_g': ['weight', 'weight(g)', 'pack_size', 'portion_size', 'size'],
            'pack_size': ['weight', 'weight_g', 'portion_size', 'size', 'weight(g)'],
            'portion_size': ['weight', 'weight_g', 'pack_size', 'size', 'weight(g)']
        }
        
        # Check if our column is in the common pairs
        column_key = column.lower().replace('_', ' ')
        if column_key in common_pairs:
            alternatives = common_pairs[column_key]
            for alt in alternatives:
                # Check for each alternative
                for col in df.columns:
                    col_lower = col.lower().replace('_', ' ')
                    if col_lower == alt:
                        return col
        
        logger.warning(f"Column '{column}' not found and no similar column found in DataFrame")
        return column  # Return original if no match found
    
    def merge_matched_datasets(self, 
                              source_df: pd.DataFrame, 
                              target_df: pd.DataFrame,
                              match_df: pd.DataFrame,
                              source_cols: List[str] = None,
                              target_cols: List[str] = None,
                              merged_cols: Dict[str, str] = None) -> pd.DataFrame:
        """
        Merge two datasets based on matches
        
        Args:
            source_df: Source DataFrame
            target_df: Target DataFrame
            match_df: DataFrame with match results
            source_cols: Columns to include from source (None=all)
            target_cols: Columns to include from target (None=all)
            merged_cols: Dictionary mapping target columns to new merged names
            
        Returns:
            Merged DataFrame
        """
        if len(match_df) == 0:
            logger.warning("No matches found to merge. Returning source dataframe")
            return source_df.copy()
            
        if len(source_df) == 0:
            logger.warning("Source dataframe is empty. Nothing to merge")
            return pd.DataFrame()
        
        # Handle empty dfs or match_df
        if len(target_df) == 0:
            logger.warning("Target dataframe is empty. Returning source dataframe")
            return source_df.copy()
        
        # Create a copy of the source dataframe
        merged_df = source_df.copy()
        
        # Add a temporary source_index column for matching
        merged_df['__temp_source_index'] = merged_df.index
        
        # Process the merged columns mapping
        if merged_cols is None:
            merged_cols = {}
        
        # Keep track of columns added from target
        added_cols = []
        
        try:
            # For each match, add target columns to source row
            for _, match in match_df.iterrows():
                source_idx = match['Source_Index']
                target_idx = match['Target_Index']
                
                # Skip if indices are out of bounds
                if source_idx >= len(source_df) or target_idx >= len(target_df):
                    continue
                
                # Get the target row
                target_row = target_df.iloc[target_idx]
                
                # Determine which columns to include
                if target_cols is None:
                    target_cols_to_add = target_row.index.tolist()
                else:
                    target_cols_to_add = [col for col in target_cols if col in target_row.index]
                
                # Add each target column to the source row
                for col in target_cols_to_add:
                    # Skip if column is empty
                    if pd.isna(target_row[col]):
                        continue
                        
                    # Determine the new column name
                    new_col = merged_cols.get(col, f"{col}_target")
                    
                    # Create the column if it doesn't exist
                    if new_col not in merged_df.columns:
                        merged_df[new_col] = np.nan
                        added_cols.append(new_col)
                    
                    # Add the value to the correct source row
                    source_matches = merged_df['__temp_source_index'] == source_idx
                    if any(source_matches):
                        merged_df.loc[source_matches, new_col] = target_row[col]
        except Exception as e:
            logger.error(f"Error merging datasets: {e}")
        
        # Remove the temporary index column
        merged_df = merged_df.drop(columns=['__temp_source_index'])
        
        # Log the merging results
        logger.info(f"Merged dataframe has {len(merged_df)} rows and {len(merged_df.columns)} columns")
        logger.info(f"Added {len(added_cols)} columns from target: {', '.join(added_cols)}")
        
        # If we added no columns, something went wrong
        if len(added_cols) == 0:
            logger.warning("No target columns were added during merge. This may indicate a problem with the matching or column selection.")
        
        return merged_df
    
    def train_from_matches(self, 
                          matches_df: pd.DataFrame, 
                          is_correct_match: List[bool] = None) -> Dict[str, float]:
        """
        Train the matcher model from existing match data with feedback
        
        Args:
            matches_df: DataFrame of potential matches
            is_correct_match: List of booleans indicating if each match is correct
            
        Returns:
            Dictionary of feature importance scores
        """
        # Prepare the features and labels for training
        X = []  # Features
        y = []  # Labels (1 = correct match, 0 = incorrect match)
        
        logger.info(f"Training from matches dataframe with {len(matches_df)} rows")
        
        # If matches_df includes is_correct_match column, use that
        if is_correct_match is None and 'is_correct_match' in matches_df.columns:
            is_correct_match = matches_df['is_correct_match'].astype(bool).tolist()
            logger.info(f"Using is_correct_match column from DataFrame with {sum(is_correct_match)} positive labels")
        
        # Use similarity score as a fallback if no labels provided
        if is_correct_match is None:
            # Use similarity as proxy for correctness (this is a weak heuristic)
            if 'similarity_score' in matches_df.columns:
                is_correct_match = (matches_df['similarity_score'] >= self.similarity_threshold).tolist()
                logger.info(f"Generated labels using similarity threshold: {sum(is_correct_match)} positive labels")
            else:
                logger.warning("No labels provided and no similarity_score column found. Cannot train.")
                return {}
        
        # Sanity check on labels
        if len(is_correct_match) != len(matches_df):
            logger.error(f"Length of labels ({len(is_correct_match)}) doesn't match DataFrame rows ({len(matches_df)})")
            return {}
            
        # Extract features for each match
        for i, (_, match) in enumerate(matches_df.iterrows()):
            # Skip rows with missing feature data
            if pd.isna(match.get('features')) or not isinstance(match.get('features'), dict):
                continue
                
            # Get features and convert to vector
            features = match.get('features', {})
            
            # If the features are stored as string (e.g., when loaded from CSV)
            if isinstance(features, str):
                try:
                    import json
                    features = json.loads(features)
                except Exception as e:
                    logger.warning(f"Could not parse features from string: {e}")
                    continue
            
            # Skip if no valid features
            if not features:
                continue
                
            X.append(features)
            y.append(1 if is_correct_match[i] else 0)
        
        # Check if we have enough data to train
        if len(X) < 10 or len(set(y)) < 2:
            logger.warning("Not enough training data or all labels are the same. Cannot train.")
            return {}
        
        logger.info(f"Training with {len(X)} examples ({sum(y)} positive, {len(y) - sum(y)} negative)")
        
        # Convert to DataFrame for easier handling
        X_df = pd.DataFrame(X)
        
        # Filter out non-numeric columns that would cause issues with RandomForestClassifier
        numeric_cols = X_df.select_dtypes(include=['number']).columns.tolist()
        if len(numeric_cols) < len(X_df.columns):
            logger.warning(f"Filtered out {len(X_df.columns) - len(numeric_cols)} non-numeric feature columns")
            X_df = X_df[numeric_cols]
        
        if X_df.empty:
            logger.error("No numeric features available for training")
            return {}
        
        # Check for NaN values and replace them
        if X_df.isna().any().any():
            logger.warning("Replacing NaN values with 0 in feature matrix")
            X_df = X_df.fillna(0)
        
        # Split the data for training and testing
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_df, y, test_size=0.3, random_state=42, stratify=y if len(set(y)) > 1 else None
            )
            
            # Create and train the model
            clf = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight='balanced' if sum(y) / len(y) < 0.3 else None  # Handle class imbalance
            )
            
            # Try to train the model
            try:
                clf.fit(X_train, y_train)
                
                # Evaluate the model
                y_pred = clf.predict(X_test)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                
                logger.info(f"Model performance: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
                
                # Update the model and feature importance
                self.model = clf
                
                # Extract feature importance
                feature_importance = {}
                for i, feature in enumerate(X_df.columns):
                    feature_importance[feature] = float(clf.feature_importances_[i])  # Convert numpy types to native Python types
                
                # Save the feature importance
                self.feature_importance = feature_importance
                self._save_model()
                
                return feature_importance
                
            except Exception as e:
                logger.error(f"Error training model: {e}")
                return {}
                
        except Exception as e:
            logger.error(f"Error in train-test split: {e}")
            return {}
    
    def learn_from_feedback(self, 
                           source_item: str, 
                           target_item: str, 
                           is_correct_match: bool, 
                           features: Dict[str, float]) -> None:
        """
        Learn from user feedback about match correctness
        
        Args:
            source_item: Source item text
            target_item: Target item text
            is_correct_match: Whether the match is correct
            features: Features used for the match
        """
        if not self.learning_enabled:
            logger.info("Learning is disabled. Skipping feedback incorporation.")
            return
        
        if not features:
            logger.warning("No features provided for learning. Skipping feedback incorporation.")
            return
        
        # Log the feedback
        logger.info(f"Received feedback: '{source_item}' and '{target_item}' match is {is_correct_match}")
        
        # Create feature data for model update
        X = [list(features.values())]
        y = [1 if is_correct_match else 0]
        
        # If we don't have a model yet, create one
        if self.model is None:
            # Initialize a basic model with the feedback
            self.model = RandomForestClassifier(n_estimators=10, random_state=42)
            self.model.fit(X, y)
            
            # Set equal feature importance to start
            self.feature_importance = {feat: 1.0/len(features) for feat in features.keys()}
        else:
            # Partial fit is not available for RandomForest, so we'd need to keep a running dataset
            # or use a different approach. For now, we'll just update feature importance
            
            # Adjust feature importance based on feedback
            importance_shift = 0.1  # Small shift to avoid drastic changes
            for feat in features.keys():
                if feat in self.feature_importance:
                    # Increase importance for correct features, decrease for incorrect
                    if is_correct_match:
                        # If match is correct, increase importance of high-value features
                        if features[feat] > 0.5:
                            self.feature_importance[feat] += importance_shift
                    else:
                        # If match is incorrect, decrease importance of high-value features
                        if features[feat] > 0.5:
                            self.feature_importance[feat] -= importance_shift
            
            # Normalize feature importance
            total = sum(self.feature_importance.values())
            if total > 0:
                self.feature_importance = {k: v/total for k, v in self.feature_importance.items()}
        
        # Save the updated model and feature importance
        self._save_model()
        
    def evaluate_matching_performance(self, 
                                    source_df: pd.DataFrame, 
                                    target_df: pd.DataFrame,
                                    source_col: str,
                                    target_col: str,
                                    ground_truth: List[Tuple[int, int]]) -> Dict[str, float]:
        """
        Evaluate the performance of the matching algorithm using ground truth data
        
        Args:
            source_df: Source DataFrame
            target_df: Target DataFrame
            source_col: Primary column name in source for matching
            target_col: Primary column name in target for matching
            ground_truth: List of (source_index, target_index) tuples representing correct matches
            
        Returns:
            Dictionary of performance metrics
        """
        # Perform matching
        matches = self.match_datasets(source_df, target_df, source_col, target_col)
        
        if matches.empty:
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'match_count': 0,
                'ground_truth_count': len(ground_truth)
            }
        
        # Convert ground truth to set for easy comparison
        ground_truth_set = set((src, tgt) for src, tgt in ground_truth)
        
        # Get matched pairs
        matched_pairs = set(
            (int(row['Source_Index']), int(row['Target_Index'])) 
            for _, row in matches.iterrows()
        )
        
        # Calculate metrics
        true_positives = len(matched_pairs.intersection(ground_truth_set))
        false_positives = len(matched_pairs - ground_truth_set)
        false_negatives = len(ground_truth_set - matched_pairs)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'match_count': len(matched_pairs),
            'ground_truth_count': len(ground_truth_set)
        }
    
    def get_verified_matches(self, 
                            match_df: pd.DataFrame, 
                            verification_threshold: float = 0.85,
                            apply_ml: bool = True) -> pd.DataFrame:
        """
        Get verified matches based on high confidence scores
        
        Args:
            match_df: DataFrame with match results
            verification_threshold: Threshold for considering a match as verified
            apply_ml: Whether to use ML model for verification
            
        Returns:
            DataFrame with verified matches
        """
        # Fix the circular logic problem by not using similarity alone for verification
        if not apply_ml or self.model is None:
            # Without ML, use multiple criteria for verification
            logger.info(f"Using rule-based verification with threshold {verification_threshold}")
            
            # Create a copy to avoid modifying the original
            verified_matches = match_df.copy()
            
            # Use multiple criteria for verification (not just similarity)
            verified = (
                (verified_matches['Similarity_Score'] > verification_threshold) |
                (verified_matches['fuzz_token_sort_ratio'] > 0.92) |
                (
                    (verified_matches['fuzz_partial_ratio'] > 0.95) & 
                    (verified_matches['food_group_match'] > 0.8)
                )
            )
            
            verified_matches['verified'] = verified
            return verified_matches[verified_matches['verified']]
        else:
            # Apply ML model for verification
            logger.info("Using ML model for match verification")
            
            # Convert to features
            feature_cols = [col for col in match_df.columns if col not in ['Source_Index', 'Target_Index', 'verified']]
            features = match_df[feature_cols].to_dict('records')
            
            # Predict each match
            verified_probs = []
            for feature_set in features:
                prob = self._predict_match_probability(feature_set, feature_set.get('Similarity_Score', 0))
                verified_probs.append(prob)
            
            match_df['verification_probability'] = verified_probs
            match_df['verified'] = match_df['verification_probability'] > verification_threshold
            
            # Return verified matches
            return match_df[match_df['verified']] 