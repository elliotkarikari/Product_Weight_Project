"""
Food matching algorithms to match products across different datasets
with machine learning capabilities for continuous improvement
"""

import pandas as pd
import numpy as np
import os
import pickle
import logging
from typing import Dict, List, Optional, Union, Tuple, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from fuzzywuzzy import fuzz

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
        
        # Source and target text
        source_text = str(source_item[source_col]) if not pd.isna(source_item[source_col]) else ""
        target_text = str(target_item[target_col]) if not pd.isna(target_item[target_col]) else ""
        
        # Basic features
        features["text_length_ratio"] = min(len(source_text), len(target_text)) / max(len(source_text), len(target_text)) if max(len(source_text), len(target_text)) > 0 else 0
        features["word_count_ratio"] = min(len(source_text.split()), len(target_text.split())) / max(len(source_text.split()), len(target_text.split())) if max(len(source_text.split()), len(target_text.split())) > 0 else 0
        features["fuzz_ratio"] = fuzz.ratio(source_text.lower(), target_text.lower()) / 100.0
        features["fuzz_partial_ratio"] = fuzz.partial_ratio(source_text.lower(), target_text.lower()) / 100.0
        features["fuzz_token_sort_ratio"] = fuzz.token_sort_ratio(source_text.lower(), target_text.lower()) / 100.0
        
        # Additional column features
        if additional_match_cols:
            for i, (src_col, tgt_col) in enumerate(additional_match_cols):
                if src_col in source_item.index and tgt_col in target_item.index:
                    src_val = str(source_item[src_col]) if not pd.isna(source_item[src_col]) else ""
                    tgt_val = str(target_item[tgt_col]) if not pd.isna(target_item[tgt_col]) else ""
                    
                    if src_val and tgt_val:
                        features[f"additional_col_{i}_ratio"] = fuzz.ratio(src_val.lower(), tgt_val.lower()) / 100.0
                        features[f"additional_col_{i}_partial"] = fuzz.partial_ratio(src_val.lower(), tgt_val.lower()) / 100.0
        
        return features
    
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
        Match items between source and target DataFrames
        
        Args:
            source_df: Source DataFrame
            target_df: Target DataFrame
            source_col: Primary column name in source DataFrame for matching
            target_col: Primary column name in target DataFrame for matching
            additional_match_cols: Additional column pairs for enhancing matches [(source_col, target_col), ...]
            
        Returns:
            DataFrame with match results
        """
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
                          is_correct_match: List[bool] = None) -> None:
        """
        Train the matching model from existing matches
        
        Args:
            matches_df: DataFrame with match results and features
            is_correct_match: List of boolean values indicating if each match is correct
        """
        if not self.learning_enabled:
            logger.info("Learning is disabled. Skipping training.")
            return
        
        if matches_df.empty:
            logger.warning("No matches to learn from. Skipping training.")
            return
        
        # Check if we have the necessary columns
        required_features = ['TF_IDF_Score', 'fuzz_ratio', 'fuzz_partial_ratio', 'fuzz_token_sort_ratio']
        if not all(col in matches_df.columns for col in required_features):
            logger.warning("Missing required feature columns for training. Skipping training.")
            return
        
        # If is_correct_match is provided, use it as labels
        if is_correct_match is not None:
            if len(is_correct_match) != len(matches_df):
                logger.warning("Length of is_correct_match doesn't match the number of matches. Skipping training.")
                return
            
            y = np.array(is_correct_match)
        else:
            # Otherwise, assume all high-similarity matches are correct
            y = (matches_df['Similarity_Score'] >= 0.8).astype(int).values
        
        # Extract features
        feature_cols = [col for col in matches_df.columns 
                      if col not in ['Source_Index', 'Target_Index', 'Source_Item', 'Target_Item', 'Similarity_Score']]
        
        X = matches_df[feature_cols].values
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train RandomForest model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        logger.info(f"Model trained with metrics - Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
        
        # Update feature importance
        self.feature_importance = dict(zip(['tf_idf'] + feature_cols[1:], model.feature_importances_))
        
        # Update model
        self.model = model
        
        # Save model and feature importance
        self._save_model()
    
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