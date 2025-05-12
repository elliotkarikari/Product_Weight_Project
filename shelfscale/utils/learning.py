"""
Utilities for training and improving the food matching algorithm over time
"""

import os
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any

from shelfscale.matching.algorithm import FoodMatcher
from shelfscale.utils.helpers import load_data, save_data

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_existing_matches() -> Dict[str, pd.DataFrame]:
    """
    Load existing matches from CSV files in the output directory
    
    Returns:
        Dictionary with match dataframes
    """
    matches = {}
    output_dir = "output"
    
    # Define match files to look for
    match_files = {
        "mw_fps_matches": "mw_fps_matches.csv",
        "mw_fvs_matches": "mw_fvs_matches.csv"
    }
    
    # Try to load each match file
    for match_type, filename in match_files.items():
        file_path = os.path.join(output_dir, filename)
        if os.path.exists(file_path):
            try:
                matches[match_type] = pd.read_csv(file_path)
                logger.info(f"Loaded {len(matches[match_type])} matches from {filename}")
            except Exception as e:
                logger.warning(f"Failed to load matches from {filename}: {e}")
    
    return matches


def extract_features_from_matches(matches_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract and format features from matches dataframe
    
    Args:
        matches_df: DataFrame with match results
        
    Returns:
        DataFrame with extracted features
    """
    # Check if this is already a feature-rich dataframe from new algorithm
    if 'TF_IDF_Score' in matches_df.columns:
        return matches_df
    
    # For old matches, we need to derive features
    # Get source and target items
    source_items = matches_df.filter(regex='_source$')
    target_items = matches_df.filter(regex='_target$')
    
    # Extract base feature columns
    feature_df = pd.DataFrame()
    feature_df['Similarity_Score'] = matches_df['Similarity_Score']
    
    # For older matches, we'll simulate basic features
    # This is a rough approximation since we don't have the original feature calculations
    
    # We'll look for source and target text columns
    source_text_col = None
    target_text_col = None
    
    # Try to find name/description columns
    for col in source_items.columns:
        if any(term in col.lower() for term in ['name', 'description', 'item']):
            source_text_col = col
            break
    
    for col in target_items.columns:
        if any(term in col.lower() for term in ['name', 'description', 'item']):
            target_text_col = col
            break
    
    # If we found text columns, calculate basic text features
    if source_text_col and target_text_col:
        from fuzzywuzzy import fuzz
        
        # Add simulated features
        feature_df['TF_IDF_Score'] = matches_df['Similarity_Score'] * 0.7  # Approximate TF-IDF score
        
        # Calculate text similarity features
        feature_df['fuzz_ratio'] = [
            fuzz.ratio(str(s), str(t)) / 100.0 
            for s, t in zip(matches_df[source_text_col], matches_df[target_text_col])
        ]
        
        feature_df['fuzz_partial_ratio'] = [
            fuzz.partial_ratio(str(s), str(t)) / 100.0 
            for s, t in zip(matches_df[source_text_col], matches_df[target_text_col])
        ]
        
        feature_df['fuzz_token_sort_ratio'] = [
            fuzz.token_sort_ratio(str(s), str(t)) / 100.0 
            for s, t in zip(matches_df[source_text_col], matches_df[target_text_col])
        ]
        
        # Calculate length-based features
        feature_df['text_length_ratio'] = [
            min(len(str(s)), len(str(t))) / max(len(str(s)), len(str(t))) if max(len(str(s)), len(str(t))) > 0 else 0
            for s, t in zip(matches_df[source_text_col], matches_df[target_text_col])
        ]
        
        feature_df['word_count_ratio'] = [
            min(len(str(s).split()), len(str(t).split())) / max(len(str(s).split()), len(str(t).split())) if max(len(str(s).split()), len(str(t).split())) > 0 else 0
            for s, t in zip(matches_df[source_text_col], matches_df[target_text_col])
        ]
        
    else:
        # If we don't have text columns, just create dummy features
        feature_df['TF_IDF_Score'] = matches_df['Similarity_Score'] * 0.7
        feature_df['fuzz_ratio'] = matches_df['Similarity_Score'] * 0.8
        feature_df['fuzz_partial_ratio'] = matches_df['Similarity_Score'] * 0.85
        feature_df['fuzz_token_sort_ratio'] = matches_df['Similarity_Score'] * 0.9
        feature_df['text_length_ratio'] = 0.8  # Default value
        feature_df['word_count_ratio'] = 0.75  # Default value
    
    # Keep source and target indices
    feature_df['Source_Index'] = matches_df['Source_Index'] if 'Source_Index' in matches_df else -1
    feature_df['Target_Index'] = matches_df['Target_Index'] if 'Target_Index' in matches_df else -1
    
    # Add source and target items
    if source_text_col and target_text_col:
        feature_df['Source_Item'] = matches_df[source_text_col]
        feature_df['Target_Item'] = matches_df[target_text_col]
    else:
        feature_df['Source_Item'] = "Unknown"
        feature_df['Target_Item'] = "Unknown"
    
    return feature_df


def get_verified_matches(matches_df: pd.DataFrame, threshold: float = 0.9) -> List[bool]:
    """
    Get a list of verified matches based on high similarity scores
    
    Args:
        matches_df: DataFrame with match results
        threshold: Similarity threshold for considering a match verified
        
    Returns:
        List of booleans indicating whether each match is verified
    """
    return (matches_df['Similarity_Score'] >= threshold).tolist()


def train_matcher_from_existing_data() -> FoodMatcher:
    """
    Train a FoodMatcher using existing match data
    
    Returns:
        Trained FoodMatcher instance
    """
    # Load existing matches
    matches_dict = load_existing_matches()
    if not matches_dict:
        logger.warning("No existing matches found")
        return FoodMatcher()
    
    # Create and train matcher
    matcher = FoodMatcher(similarity_threshold=0.6)
    
    # Train on each match dataset
    for match_type, matches_df in matches_dict.items():
        if not matches_df.empty:
            logger.info(f"Training matcher with {len(matches_df)} matches from {match_type}")
            
            # Extract features
            features_df = extract_features_from_matches(matches_df)
            
            # Get verified matches (high similarity assumed to be correct)
            verified_matches = get_verified_matches(features_df)
            
            # Train the matcher
            matcher.train_from_matches(features_df, verified_matches)
    
    return matcher


def generate_training_data_from_matches(matcher: FoodMatcher, threshold: float = 0.8) -> None:
    """
    Generate training data from existing matches for improving the algorithm
    
    Args:
        matcher: FoodMatcher instance
        threshold: Similarity threshold for considering a match as training data
    """
    # Load existing matches
    matches_dict = load_existing_matches()
    if not matches_dict:
        logger.warning("No existing matches found")
        return
    
    # Combine all matches
    all_matches = pd.concat(matches_dict.values(), ignore_index=True)
    
    # Extract features
    features_df = extract_features_from_matches(all_matches)
    
    # Filter high-quality matches
    high_quality = features_df[features_df['Similarity_Score'] >= threshold]
    
    # Create training data CSV
    output_dir = "output"
    training_file = os.path.join(output_dir, "training_data.csv")
    
    # Save training data
    high_quality.to_csv(training_file, index=False)
    logger.info(f"Generated training data with {len(high_quality)} high-quality matches")


def evaluate_matcher_performance() -> Dict[str, float]:
    """
    Evaluate the performance of the matching algorithm on existing data
    
    Returns:
        Dictionary with performance metrics
    """
    # Load existing matches
    matches_dict = load_existing_matches()
    if not matches_dict:
        logger.warning("No existing matches found")
        return {}
    
    # Create matcher
    matcher = train_matcher_from_existing_data()
    
    # Metrics
    metrics = {
        'total_matches': 0,
        'high_quality_matches': 0,
        'precision_estimate': 0.0,
        'average_similarity': 0.0
    }
    
    # Calculate metrics
    for match_type, matches_df in matches_dict.items():
        if not matches_df.empty:
            # Count matches
            metrics['total_matches'] += len(matches_df)
            
            # Count high quality matches
            high_quality = sum(matches_df['Similarity_Score'] >= 0.8)
            metrics['high_quality_matches'] += high_quality
            
            # Average similarity
            metrics['average_similarity'] += matches_df['Similarity_Score'].mean() * len(matches_df)
    
    # Calculate averages
    if metrics['total_matches'] > 0:
        metrics['average_similarity'] /= metrics['total_matches']
        metrics['precision_estimate'] = metrics['high_quality_matches'] / metrics['total_matches']
    
    return metrics


def apply_feedback_to_matches(source_item: str, target_item: str, is_correct: bool) -> None:
    """
    Apply feedback to improve the matcher for specific items
    
    Args:
        source_item: Source item text
        target_item: Target item text
        is_correct: Whether the match is correct
    """
    # Load matcher
    matcher = train_matcher_from_existing_data()
    
    # Simulate features
    features = {
        'text_length_ratio': min(len(source_item), len(target_item)) / max(len(source_item), len(target_item)) if max(len(source_item), len(target_item)) > 0 else 0,
        'word_count_ratio': min(len(source_item.split()), len(target_item.split())) / max(len(source_item.split()), len(target_item.split())) if max(len(source_item.split()), len(target_item.split())) > 0 else 0,
        'fuzz_ratio': fuzz.ratio(source_item.lower(), target_item.lower()) / 100.0,
        'fuzz_partial_ratio': fuzz.partial_ratio(source_item.lower(), target_item.lower()) / 100.0,
        'fuzz_token_sort_ratio': fuzz.token_sort_ratio(source_item.lower(), target_item.lower()) / 100.0
    }
    
    # Apply feedback
    matcher.learn_from_feedback(source_item, target_item, is_correct, features)
    
    # Log feedback
    logger.info(f"Applied feedback: '{source_item}' and '{target_item}' match is {is_correct}")


def create_weight_predictions(df: pd.DataFrame, product_name_col: str, 
                            existing_weight_col: Optional[str] = None) -> pd.DataFrame:
    """
    Create weight predictions for products based on existing matches
    
    Args:
        df: DataFrame with products
        product_name_col: Column name containing product names
        existing_weight_col: Optional column with existing weights
        
    Returns:
        DataFrame with predicted weights
    """
    # Load matcher
    matcher = train_matcher_from_existing_data()
    
    # Load existing weight data
    consolidated_weights = load_data("output/consolidated_weights.csv")
    
    # Create a small dataset from consolidated weights
    weight_df = consolidated_weights[['Food Name', 'Standardized_Weight']].dropna()
    weight_df = weight_df[weight_df['Standardized_Weight'] > 0]
    
    # Match the input products to weight_df
    if weight_df.empty:
        logger.warning("No weight data available for predictions")
        return df
    
    result_df = df.copy()
    
    # Create a new column for predicted weights
    result_df['Predicted_Weight'] = None
    result_df['Prediction_Confidence'] = 0.0
    
    # Match products to weight database
    matches = matcher.match_datasets(
        result_df, 
        weight_df, 
        product_name_col, 
        'Food Name',
        similarity_threshold=0.5  # Lower threshold to get more predictions
    )
    
    # If we got matches, update predictions
    if not matches.empty:
        # Create dictionary of matches
        match_dict = {}
        for _, row in matches.iterrows():
            source_idx = int(row['Source_Index'])
            target_idx = int(row['Target_Index'])
            confidence = row['Similarity_Score']
            
            # Only keep the highest confidence match for each product
            if source_idx not in match_dict or confidence > match_dict[source_idx][1]:
                match_dict[source_idx] = (target_idx, confidence)
        
        # Update predictions
        for source_idx, (target_idx, confidence) in match_dict.items():
            if source_idx < len(result_df):
                weight = weight_df.iloc[target_idx]['Standardized_Weight']
                result_df.loc[source_idx, 'Predicted_Weight'] = weight
                result_df.loc[source_idx, 'Prediction_Confidence'] = confidence
    
    # If existing weights are available, use them preferentially
    if existing_weight_col and existing_weight_col in result_df.columns:
        mask = result_df[existing_weight_col].notna()
        result_df.loc[mask, 'Predicted_Weight'] = result_df.loc[mask, existing_weight_col]
        result_df.loc[mask, 'Prediction_Confidence'] = 1.0
    
    return result_df


if __name__ == "__main__":
    # If run directly, train a matcher and generate training data
    from fuzzywuzzy import fuzz
    matcher = train_matcher_from_existing_data()
    generate_training_data_from_matches(matcher)
    metrics = evaluate_matcher_performance()
    print("Matcher performance metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}") 