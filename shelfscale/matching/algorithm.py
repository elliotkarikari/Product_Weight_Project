"""
Food matching algorithms to match products across different datasets
with machine learning capabilities for continuous improvement
"""

import pandas as pd
import numpy as np
import os
import pickle
import logging
import re
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
                 model_path: str = "models/food_matcher_model.pkl",
                 features_path: str = "models/food_matcher_features.pkl",
                 learning_enabled: bool = True):
        """
        Initialize the food matcher
        
        Args:
            similarity_threshold: Threshold for considering a match
            model_path: Path to save/load the machine learning model
            features_path: Path to save/load feature importance information
            learning_enabled: Whether to use machine learning capabilities
        """
        self.similarity_threshold = similarity_threshold
        self.vectorizer = None
        self.model_path = model_path
        self.features_path = features_path
        self.learning_enabled = learning_enabled
        self.model = None
        self.feature_importance = {}
        
        # Create model directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
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
        Match items between two datasets
        
        Args:
            source_df: Source DataFrame
            target_df: Target DataFrame
            source_col: Primary column name in source DataFrame for matching
            target_col: Primary column name in target DataFrame for matching
            additional_match_cols: Additional column pairs for enhancing matches [(source_col, target_col), ...]
            
        Returns:
            DataFrame with match results
        """
        # Log available columns for debugging
        logger.info(f"Source DataFrame columns: {source_df.columns.tolist()}")
        logger.info(f"Target DataFrame columns: {target_df.columns.tolist()}")
        logger.info(f"Looking for source column: '{source_col}' and target column: '{target_col}'")
        
        # Validate column existence
        if source_col not in source_df.columns or target_col not in target_df.columns:
            logger.warning("Missing columns detected. Will attempt to use similar column names in _compute_tfidf_similarity.")
        
        # Calculate primary similarity
        similarity_matrix = self._compute_tfidf_similarity(
            source_df, target_df, source_col, target_col
        )
        
        # Initialize results DataFrame
        matches = []
        
        # For each source item, find the best match
        for i, source_item in source_df.iterrows():
            # Get top match index and score
            best_match_idx = np.argmax(similarity_matrix[i])
            tf_idf_score = similarity_matrix[i, best_match_idx]
            
            # Get target item
            target_item = target_df.iloc[best_match_idx]
            
            # Calculate additional features
            features = self._calculate_additional_features(
                source_item, target_item, source_col, target_col, additional_match_cols
            )
            
            # Predict match probability
            match_probability = self._predict_match_probability(features, tf_idf_score)
            
            # Only include matches above threshold
            if match_probability >= self.similarity_threshold:
                match = {
                    'Source_Index': i,
                    'Target_Index': best_match_idx,
                    'Source_Item': source_item[source_col],
                    'Target_Item': target_item[target_col],
                    'TF_IDF_Score': tf_idf_score,
                    'Similarity_Score': match_probability,
                    **features
                }
                matches.append(match)
                
        # Create results DataFrame
        match_df = pd.DataFrame(matches)
        
        # Sort by similarity score in descending order
        if not match_df.empty:
            match_df = match_df.sort_values('Similarity_Score', ascending=False)
        
        return match_df
    
    def merge_matched_datasets(self, 
                              source_df: pd.DataFrame, 
                              target_df: pd.DataFrame,
                              match_df: pd.DataFrame,
                              source_cols: List[str] = None,
                              target_cols: List[str] = None,
                              merged_cols: Dict[str, str] = None) -> pd.DataFrame:
        """
        Merge source and target DataFrames based on matching results
        
        Args:
            source_df: Source DataFrame
            target_df: Target DataFrame
            match_df: Matching results DataFrame from match_datasets
            source_cols: Source columns to include in result (None=all)
            target_cols: Target columns to include in result (None=all)
            merged_cols: Dictionary to rename columns {'old_name': 'new_name'}
            
        Returns:
            Merged DataFrame with data from both sources
        """
        if match_df.empty:
            return pd.DataFrame()
        
        # Ensure indices are valid
        valid_source_indices = [idx for idx in match_df['Source_Index'] if idx < len(source_df)]
        valid_target_indices = [idx for idx in match_df['Target_Index'] if idx < len(target_df)]
        
        # Create a new match_df with only valid indices
        valid_matches = match_df[
            match_df['Source_Index'].isin(valid_source_indices) & 
            match_df['Target_Index'].isin(valid_target_indices)
        ].copy()
        
        if valid_matches.empty:
            logger.warning("No valid matches found after filtering invalid indices")
            return pd.DataFrame()
        
        # Use iloc for integer-based indexing to select rows
        source_rows = [source_df.iloc[int(idx)] for idx in valid_matches['Source_Index']]
        target_rows = [target_df.iloc[int(idx)] for idx in valid_matches['Target_Index']]
        
        # Convert to DataFrames
        source_subset = pd.DataFrame(source_rows)
        target_subset = pd.DataFrame(target_rows)
        
        # Select specific columns if requested
        if source_cols:
            source_subset = source_subset[[col for col in source_cols if col in source_subset.columns]]
        
        if target_cols:
            target_subset = target_subset[[col for col in target_cols if col in target_subset.columns]]
            
        # Add _source and _target suffixes to avoid column name conflicts
        source_subset = source_subset.add_suffix('_source')
        target_subset = target_subset.add_suffix('_target')
        
        # Get similarity scores for valid matches
        similarity_scores = valid_matches[['Similarity_Score']].reset_index(drop=True)
        
        # Merge DataFrames
        merged_df = pd.concat([source_subset.reset_index(drop=True), 
                              target_subset.reset_index(drop=True), 
                              similarity_scores], axis=1)
        
        # Rename columns if specified
        if merged_cols:
            merged_df = merged_df.rename(columns=merged_cols)
        
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