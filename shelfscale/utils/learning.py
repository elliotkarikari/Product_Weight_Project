"""
Utilities for training and improving the food matching algorithm over time
"""

import os
import pandas as pd
import numpy as np
import logging
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import shelfscale.config as config
from shelfscale.matching.algorithm import FoodMatcher
from shelfscale.utils.helpers import load_data, save_data # Assuming load_data and save_data don't need config paths directly

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# HISTORY_FILE is now config.MODEL_PERFORMANCE_HISTORY_PATH
# DATA_PATHS dictionary is removed, functions will use specific config paths.

def load_performance_history() -> List[Dict]:
    """
    Loads the model performance history from the configured JSON file.
    
    Returns:
        A list of performance history records, or an empty list if the file does not exist or cannot be read.
    """
    history_file_path = config.MODEL_PERFORMANCE_HISTORY_PATH
    if not os.path.exists(history_file_path):
        return []
    
    try:
        with open(history_file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load performance history: {e}")
        return []

def save_performance_history(history: List[Dict]) -> None:
    """
    Saves the performance history to the configured file as JSON.
    
    Args:
    	history: A list of dictionaries representing performance records.
    """
    history_file_path = config.MODEL_PERFORMANCE_HISTORY_PATH
    try:
        # Model directory (dirname of history_file_path) is created by config.py
        # os.makedirs(os.path.dirname(history_file_path), exist_ok=True)
        
        with open(history_file_path, 'w') as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save performance history: {e}")

def record_performance(metrics: Dict[str, float], dataset_info: Dict[str, Any]) -> None:
    """
    Record model performance metrics to the history file
    
    Args:
        metrics: Performance metrics
        dataset_info: Information about the dataset used
    """
    # Load existing history
    history = load_performance_history()
    
    # Create new performance record
    # convert numpy types → Python built-ins
    def _py(v):
        return float(v) if isinstance(v, (np.floating, np.float_)) else \
               int(v)   if isinstance(v, (np.integer,)) else v
    
    record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": {k: _py(v) for k, v in metrics.items()},
        "dataset_info": dataset_info
    }
    
    # Add to history
    history.append(record)
    
    # Save updated history
    save_performance_history(history)
    
    logger.info(f"Recorded performance metrics: {metrics}")

def get_performance_trend() -> Dict[str, List[float]]:
    """
    Get performance trend data
    
    Returns:
        Dictionary with performance metrics over time
    """
    history = load_performance_history()
    
    # Extract metrics over time
    trend = {
        "timestamps": [],
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "match_count": []
    }
    
    for record in history:
        trend["timestamps"].append(record["timestamp"])
        metrics = record["metrics"]
        
        trend["accuracy"].append(metrics.get("accuracy", 0))
        trend["precision"].append(metrics.get("precision", 0))
        trend["recall"].append(metrics.get("recall", 0))
        trend["f1"].append(metrics.get("f1", 0))
        trend["match_count"].append(metrics.get("total_matches", 0))
    
    return trend

def load_existing_matches() -> Dict[str, pd.DataFrame]:
    """
    Loads existing match data from configured CSV files.
    
    Returns:
        A dictionary mapping match types to their corresponding DataFrames. If a file is missing or cannot be loaded, its entry is omitted from the result.
    """
    matches = {}
    # Use paths from config.EXISTING_MATCH_FILES_TO_LOAD
    
    for match_type, file_path in config.EXISTING_MATCH_FILES_TO_LOAD.items():
        if os.path.exists(file_path):
            try:
                matches[match_type] = pd.read_csv(file_path)
                logger.info(f"Loaded {len(matches[match_type])} matches from {os.path.basename(file_path)}")
            except Exception as e:
                logger.warning(f"Failed to load matches from {os.path.basename(file_path)} at {file_path}: {e}")
        else:
            logger.info(f"Match file not found for {match_type}: {file_path}")
    
    return matches

# New functions for enhanced data loading and integration

def load_mccance_widdowson_data() -> pd.DataFrame:
    """
    Loads the McCance & Widdowson food composition dataset from the configured Excel file.
    
    Returns:
        A DataFrame containing food composition data, or an empty DataFrame if loading fails.
    """
    # Assuming this should load the main McCance & Widdowson Excel file
    file_path = config.MCCANCE_WIDDOWSON_PATH 
    
    try:
        # load_data can handle .xlsx, but McCance Widdowson is usually Excel.
        # For consistency with how raw_processor loads it, or if specific sheets are needed,
        # direct pandas.read_excel might be better.
        # However, current load_data in helpers.py supports .xlsx.
        df = load_data(file_path) 
        logger.info(f"Loaded {len(df)} items from McCance & Widdowson dataset at {file_path}")
        return df
    except Exception as e:
        logger.error(f"Failed to load McCance & Widdowson data: {e}")
        return pd.DataFrame()

def load_food_portion_data() -> pd.DataFrame:
    """
    Loads processed Food Portion Sizes data from a configured CSV file or directory.
    
    Attempts to load the specific processed file if available; otherwise, loads and combines all CSV files in the designated directory. Returns an empty DataFrame if no data is found or an error occurs.
    
    Returns:
        pd.DataFrame: Combined food portion size data, or empty DataFrame on failure.
    """
    # This function expects processed food portion data.
    # config.PROCESSED_FPS_PATH points to the specific processed CSV file.
    # config.FPS_SUBDIR points to the directory.
    # The original code iterated through CSVs in DATA_PATHS["food_portion_sizes"] (which was "Data/Processed/FoodPortionSized")
    # We'll adapt to load the specific processed file, or all CSVs in that dir if multiple are expected.
    
    processed_fps_dir = os.path.join(config.PROCESSED_DATA_DIR, config.FPS_SUBDIR)
    
    try:
        if os.path.exists(config.PROCESSED_FPS_PATH): # Prefer the specific configured processed file
             df_list = [load_data(config.PROCESSED_FPS_PATH)]
        elif os.path.isdir(processed_fps_dir): # Fallback to loading all CSVs in the directory
            logger.info(f"Specific processed FPS file not found at {config.PROCESSED_FPS_PATH}. Trying all CSVs in {processed_fps_dir}")
            csv_files = [f for f in os.listdir(processed_fps_dir) if f.endswith('.csv')]
            if not csv_files:
                logger.warning(f"No CSV files found in {processed_fps_dir}")
                return pd.DataFrame()
            df_list = [load_data(os.path.join(processed_fps_dir, f)) for f in csv_files]
        else:
            logger.warning(f"Processed food portion data path not found: {config.PROCESSED_FPS_PATH} or directory {processed_fps_dir}")
            return pd.DataFrame()

        combined_df = pd.concat(df_list, ignore_index=True)
        logger.info(f"Loaded {len(combined_df)} items from Food Portion Sizes data from {processed_fps_dir}")
        return combined_df
    except Exception as e:
        logger.error(f"Failed to load Food Portion Sizes data: {e}")
        return pd.DataFrame()

def load_fruit_veg_survey_data() -> Dict[str, pd.DataFrame]:
    """
    Loads the Fruit & Vegetable Survey data as a dictionary of DataFrames.
    
    Attempts to load a combined processed CSV file; if unavailable, loads year-specific data from subfolders. Returns an empty dictionary if no data is found.
    
    Returns:
        A dictionary mapping "combined" or year strings to their corresponding DataFrames.
    """
    # This function expects processed FVS data.
    # config.PROCESSED_FVS_TABLES_PATH points to the combined CSV in the "tables" subdir.
    # config.FVS_TABLES_SUBDIR points to "Fruit and Veg Sample reports/tables"
    
    fvs_tables_dir = os.path.join(config.PROCESSED_DATA_DIR, config.FVS_TABLES_SUBDIR)
    result = {}

    try:
        # Attempt to load the primary combined file first
        if os.path.exists(config.PROCESSED_FVS_TABLES_PATH):
            df_combined = load_data(config.PROCESSED_FVS_TABLES_PATH)
            # If the combined file itself implies multiple years or sources, it should be handled here.
            # For now, assume it's the main dataset.
            result["combined"] = df_combined
            logger.info(f"Loaded {len(df_combined)} items from combined Fruit & Veg Survey at {config.PROCESSED_FVS_TABLES_PATH}")
            return result # Return early if combined file is loaded

        # Fallback: Original logic to check for year folders if the main combined file isn't there
        # This part might be deprecated if raw_processor always creates the combined file.
        if os.path.isdir(fvs_tables_dir):
            logger.info(f"Combined FVS file not found. Checking for year folders in {fvs_tables_dir}")
            year_folders = [d for d in os.listdir(fvs_tables_dir) if os.path.isdir(os.path.join(fvs_tables_dir, d))]
            
            for year in year_folders:
                year_path = os.path.join(fvs_tables_dir, year)
                csv_files = [f for f in os.listdir(year_path) if f.endswith('.csv')]
                
                if not csv_files: continue
                    
                year_dfs = [load_data(os.path.join(year_path, f)) for f in csv_files]
                if year_dfs:
                    combined_df_year = pd.concat(year_dfs, ignore_index=True)
                    result[year] = combined_df_year
                    logger.info(f"Loaded {len(combined_df_year)} items from Fruit & Veg Survey {year}")
            if result: return result # Return if year-specific data was found

        logger.warning(f"No Fruit & Veg survey data found at {config.PROCESSED_FVS_TABLES_PATH} or in year subfolders of {fvs_tables_dir}")
        return {} # Return empty if no data found
            
    except Exception as e:
        logger.error(f"Failed to load Fruit & Veg Survey data: {e}")
        return {}

def load_mw_data_reduction() -> Dict[str, pd.DataFrame]:
    """
    Loads McCance & Widdowson data reduction files from configured directories.
    
    Returns:
        A dictionary containing DataFrames for individual tables, super group, super group cleaned, and total reduction types. If loading fails for any type, the corresponding DataFrame will be empty.
    """
    # Base path for MW Data Reduction in PROCESSED_DATA_DIR
    mw_reduction_base_dir = os.path.join(config.PROCESSED_DATA_DIR, config.MW_DATA_REDUCTION_SUBDIR)
    
    result = {
        "individual_tables": pd.DataFrame(),
        "super_group": pd.DataFrame(),
        "super_group_cleaned": pd.DataFrame(),
        "total": pd.DataFrame()
    }
    
    try:
        # Load Individual Tables
        indiv_path = os.path.join(config.PROCESSED_DATA_DIR, config.MW_REDUCED_INDIVIDUAL_TABLES_SUBDIR)
        if os.path.exists(indiv_path) and os.path.isdir(indiv_path):
            indiv_files = [f for f in os.listdir(indiv_path) if f.endswith('.csv')]
            indiv_dfs = [load_data(os.path.join(indiv_path, f)) for f in indiv_files]
            if indiv_dfs:
                result["individual_tables"] = pd.concat(indiv_dfs, ignore_index=True)
                logger.info(f"Loaded {len(result['individual_tables'])} items from MW Individual Tables")
        
        # Load Super Group (this is config.MW_PROCESSED_SUPER_GROUP_PATH)
        sg_path = config.MW_PROCESSED_SUPER_GROUP_PATH
        if os.path.exists(sg_path) and os.path.isdir(sg_path):
            sg_files = [f for f in os.listdir(sg_path) if f.endswith('.csv') and os.path.isfile(os.path.join(sg_path, f))]
            sg_dfs = []
            for file in sg_files:
                df = load_data(os.path.join(sg_path, file))
                group_name = os.path.splitext(file)[0]
                df['Super_Group'] = group_name
                sg_dfs.append(df)
            if sg_dfs:
                result["super_group"] = pd.concat(sg_dfs, ignore_index=True)
                logger.info(f"Loaded {len(result['super_group'])} items from MW Super Group")
        
        # Load Super Group Cleaned
        sg_cleaned_path = os.path.join(config.PROCESSED_DATA_DIR, config.MW_REDUCED_SUPER_GROUP_CLEANED_SUBDIR)
        if os.path.exists(sg_cleaned_path) and os.path.isdir(sg_cleaned_path):
            sg_cleaned_files = [f for f in os.listdir(sg_cleaned_path) if f.endswith('.csv')]
            sg_cleaned_dfs = [load_data(os.path.join(sg_cleaned_path, f)) for f in sg_cleaned_files]
            if sg_cleaned_dfs:
                result["super_group_cleaned"] = pd.concat(sg_cleaned_dfs, ignore_index=True)
                logger.info(f"Loaded {len(result['super_group_cleaned'])} items from MW Super Group Cleaned")
        
        # Load Total (this is config.MW_PROCESSED_FULL_FILE)
        total_file_path = config.MW_PROCESSED_FULL_FILE
        if os.path.exists(total_file_path):
            result["total"] = load_data(total_file_path)
            logger.info(f"Loaded {len(result['total'])} items from MW Reduced Total ({total_file_path})")
        
        return result
    except Exception as e:
        logger.error(f"Failed to load MW Data Reduction: {e}")
        return result # Return partially loaded data if error occurs mid-way

def load_reduced_with_weights() -> pd.DataFrame:
    """
    Loads the processed labelling dataset ("ReducedwithWeights") containing combined weight data.
    
    Returns:
        pd.DataFrame: The loaded dataset, or an empty DataFrame if the file is missing or cannot be loaded.
    """
    # This should point to config.PROCESSED_LABELLING_PATH
    file_path = config.PROCESSED_LABELLING_PATH
    
    try:
        if os.path.exists(file_path):
            df = load_data(file_path)
            logger.info(f"Loaded {len(df)} items from ReducedwithWeights (Processed Labelling) from {file_path}")
            return df
        else:
            logger.warning(f"Processed Labelling Data (ReducedwithWeights) file not found at {file_path}")
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"Failed to load ReducedwithWeights data from {file_path}: {e}")
        return df
    except Exception as e:
        logger.error(f"Failed to load ReducedwithWeights data: {e}")
        return pd.DataFrame()

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
        
        # Add domain-specific features
        
        # 1. Food group similarity
        feature_df['food_group_match'] = 0.5  # Default neutral value
        
        # 2. Check for food codes
        source_code_col = next((col for col in source_items.columns if 'code' in col.lower()), None)
        target_code_col = next((col for col in target_items.columns if 'code' in col.lower()), None)
        
        if source_code_col and target_code_col:
            feature_df['food_code_match'] = [
                1.0 if str(s) == str(t) else 0.0
                for s, t in zip(matches_df[source_code_col], matches_df[target_code_col])
            ]
        else:
            feature_df['food_code_match'] = 0.0
        
        # 3. Detect food preparation state
        feature_df['food_state_match'] = 0.5  # Default neutral value
        
        # 4. Unit measure match
        feature_df['unit_measure_match'] = 0.5  # Default neutral value
        
        # 5. Common ingredient count
        feature_df['common_ingredient_count'] = 0.0  # Default value
        
    else:
        # If we don't have text columns, just create dummy features
        feature_df['TF_IDF_Score'] = matches_df['Similarity_Score'] * 0.7
        feature_df['fuzz_ratio'] = matches_df['Similarity_Score'] * 0.8
        feature_df['fuzz_partial_ratio'] = matches_df['Similarity_Score'] * 0.85
        feature_df['fuzz_token_sort_ratio'] = matches_df['Similarity_Score'] * 0.9
        feature_df['text_length_ratio'] = 0.8  # Default value
        feature_df['word_count_ratio'] = 0.75  # Default value
        feature_df['food_group_match'] = 0.5  # Default value
        feature_df['food_code_match'] = 0.0  # Default value
        feature_df['food_state_match'] = 0.5  # Default value
        feature_df['unit_measure_match'] = 0.5  # Default value
        feature_df['common_ingredient_count'] = 0.0  # Default value
    
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

def get_verified_matches(matches_df: pd.DataFrame, threshold: float = 0.9, 
                       add_noise: bool = False, random_seed: int = 42) -> List[bool]:
    """
    Get a list of verified matches based on high similarity scores
    
    Args:
        matches_df: DataFrame with match results
        threshold: Similarity threshold for considering a match verified
        add_noise: Whether to simulate human disagreement by adding noise
        random_seed: Seed for reproducible randomization when add_noise=True
        
    Returns:
        List of booleans indicating whether each match is verified
    """
    # The current approach leads to circular logic where we're training on the same metric
    # we're trying to predict. Let's create a more nuanced approach using different features
    
    verified = []
    
    # Use different metrics for verification than just Similarity_Score
    for _, row in matches_df.iterrows():
        # Only consider high fuzzy ratio matches as verified
        if 'fuzz_ratio' in row and 'fuzz_token_sort_ratio' in row:
            # Using a different combination of features than what's in the main similarity score
            word_similarity = row.get('fuzz_token_sort_ratio', 0)
            exact_similarity = row.get('fuzz_ratio', 0)
            
            # Try to find cases where the words match well even if exact match isn't perfect
            is_good_match = (word_similarity > 0.85) or (exact_similarity > 0.7 and word_similarity > 0.6)
            
            # Use text length to reduce noise
            if 'text_length_ratio' in row:
                length_ratio = row['text_length_ratio'] 
                # Penalize matches where length is very different
                if length_ratio < 0.3:  # Very different lengths
                    is_good_match = is_good_match and (word_similarity > 0.9)
            
            verified.append(is_good_match)
        else:
            # Fallback to original method if features aren't available
            verified.append(row['Similarity_Score'] >= threshold)
    
    # Add some noise to avoid perfect scores and make evaluation more realistic
    # But only if explicitly requested, and with a fixed seed for reproducibility
    if add_noise:
        import random
        rng = random.Random(random_seed)  # Create seeded RNG for reproducibility
        for i in range(len(verified)):
            if rng.random() < 0.05:  # 5% chance to flip
                verified[i] = not verified[i]
            
    return verified

# New function to create weight predictions based on food categories and learned matches
def create_weight_predictions(matcher: FoodMatcher, dataset: pd.DataFrame, 
                             weight_col: str = 'Weight_Value',
                             food_group_col: str = 'Food_Group') -> pd.DataFrame:
    """
    Create weight predictions for items missing weights based on similar items
    
    Args:
        matcher: Trained FoodMatcher instance
        dataset: DataFrame with food items
        weight_col: Name of weight column
        food_group_col: Name of food group column
        
    Returns:
        DataFrame with predicted weights added
    """
    # Create a copy of the dataset
    result_df = dataset.copy()
    
    # Identify items missing weights
    missing_weights = result_df[result_df[weight_col].isna()]
    
    if len(missing_weights) == 0:
        logger.info("No missing weights to predict")
        return result_df
    
    logger.info(f"Predicting weights for {len(missing_weights)} items")
    
    # Items with weights (potential donors)
    donors = result_df[result_df[weight_col].notna()]
    
    if len(donors) == 0:
        logger.warning("No donor items with weights available for prediction")
        return result_df
    
    # Group items by food group if available
    if food_group_col in result_df.columns:
        # For each food group, predict weights
        for group in missing_weights[food_group_col].unique():
            if pd.isna(group):
                continue
                
            # Get missing items in this group
            group_missing = missing_weights[missing_weights[food_group_col] == group]
            
            # Get donors in this group
            group_donors = donors[donors[food_group_col] == group]
            
            if len(group_donors) == 0:
                # No donors in this group, skip
                continue
                
            # For each missing item, find the best match among donors
            for idx, missing_item in group_missing.iterrows():
                # Find best match
                best_match = None
                best_score = 0
                
                # Use the matcher to find best match
                for _, donor in group_donors.iterrows():
                    # TODO: Use public API methods instead of private methods
                    # The FoodMatcher class should provide:
                    # 1. compute_similarity(item1, item2, col1, col2) -> returns features
                    # 2. predict_match_probability(features) -> returns match probability
                    
                    # Current approach relies on private methods (avoid in production)
                    features = matcher._calculate_additional_features(
                        missing_item, donor, 'Food_Name', 'Food_Name'
                    )
                    
                    score = matcher._predict_match_probability(features, features.get('fuzz_token_sort_ratio', 0))
                    
                    # Recommended approach using public API (to be implemented in FoodMatcher)
                    # features = matcher.compute_similarity(missing_item, donor, 'Food_Name', 'Food_Name')
                    # score = matcher.predict_match_probability(features)
                    
                    if score > best_score:
                        best_score = score
                        best_match = donor
                
                # If we found a good match, use its weight
                if best_match is not None and best_score >= 0.7:
                    result_df.loc[idx, weight_col] = best_match[weight_col]
                    result_df.loc[idx, 'Weight_Source'] = 'Predicted'
                    result_df.loc[idx, 'Prediction_Score'] = best_score
    else:
        # No food group column, match across all donors
        for idx, missing_item in missing_weights.iterrows():
            # Use text columns for matching
            text_cols = [col for col in missing_item.index if any(term in col.lower() for term in ['name', 'description', 'item'])]
            
            if not text_cols:
                continue
                
            text_col = text_cols[0]
            
            # Find best match
            best_match = None
            best_score = 0
            
            # Use the matcher to find best match
            for _, donor in donors.iterrows():
                # TODO: Use public API methods instead of private methods
                # The FoodMatcher class should provide:
                # 1. compute_similarity(item1, item2, col1, col2) -> returns features
                # 2. predict_match_probability(features) -> returns match probability
                
                # Current approach relies on private methods (avoid in production)
                features = matcher._calculate_additional_features(
                    missing_item, donor, text_col, text_col
                )
                
                score = matcher._predict_match_probability(features, features.get('fuzz_token_sort_ratio', 0))
                
                # Recommended approach using public API (to be implemented in FoodMatcher)
                # features = matcher.compute_similarity(missing_item, donor, text_col, text_col)
                # score = matcher.predict_match_probability(features)
                
                if score > best_score:
                    best_score = score
                    best_match = donor
            
            # If we found a good match, use its weight
            if best_match is not None and best_score >= 0.7:
                result_df.loc[idx, weight_col] = best_match[weight_col]
                result_df.loc[idx, 'Weight_Source'] = 'Predicted'
                result_df.loc[idx, 'Prediction_Score'] = best_score
    
    # Count successful predictions
    prediction_count = (result_df['Weight_Source'] == 'Predicted').sum()
    logger.info(f"Successfully predicted weights for {prediction_count} items")
    
    return result_df

def load_all_data_sources() -> Dict[str, Any]:
    """
    Load all available data sources
    
    Returns:
        Dictionary with all loaded data sources
    """
    data_sources = {}
    
    # Load McCance & Widdowson data
    mw_data = load_mccance_widdowson_data()
    if not mw_data.empty:
        data_sources["mw_data"] = mw_data
    
    # Load Food Portion data
    portion_data = load_food_portion_data()
    if not portion_data.empty:
        data_sources["portion_data"] = portion_data
    
    # Load Fruit & Veg Survey data
    fruit_veg_data = load_fruit_veg_survey_data()
    if fruit_veg_data:
        for year, df in fruit_veg_data.items():
            data_sources[f"fruit_veg_{year}"] = df
    
    # Load MW Data Reduction
    mw_reduction = load_mw_data_reduction()
    for key, df in mw_reduction.items():
        if not df.empty:
            data_sources[f"mw_reduction_{key}"] = df
    
    # Load ReducedwithWeights
    reduced_with_weights = load_reduced_with_weights()
    if not reduced_with_weights.empty:
        data_sources["reduced_with_weights"] = reduced_with_weights
    
    return data_sources

def extract_gold_standard_matches_from_notebooks() -> pd.DataFrame:
    """
    Attempts to extract manually verified gold standard matches from Jupyter notebooks.
    
    Scans the configured Data Product notebooks directory for code cells that may contain match results. Currently, this function only logs the presence of potential match data and returns an empty DataFrame as a placeholder.
    """
    # Path to notebooks directory from config
    base_notebooks_dir = config.NOTEBOOKS_DIR
    data_product_subdir_name = config.DATA_PRODUCT_NOTEBOOKS_SUBDIR # "Data_Product"
    
    gold_matches = []
    
    try:
        # Check Data_Product directory for notebooks with matching results
        data_product_dir_path = os.path.join(base_notebooks_dir, data_product_subdir_name)
        if os.path.exists(data_product_dir_path) and os.path.isdir(data_product_dir_path):
            notebook_files = [f for f in os.listdir(data_product_dir_path) if f.endswith('.ipynb')]
            
            for notebook_file_name in notebook_files:
                try:
                    # Load notebook as JSON
                    # import json # Already imported at top
                    notebook_full_path = os.path.join(data_product_dir_path, notebook_file_name)
                    
                    with open(notebook_full_path, 'r', encoding='utf-8') as f:
                        nb_content = json.load(f)
                    
                    # Look for cells with matching results
                    for cell in nb_content.get('cells', []):
                        if cell.get('cell_type') == 'code':
                            source = ''.join(cell.get('source', []))
                            
                            # Look for dataframe assignments with matches
                            if any(term in source for term in ['matches', 'matched', 'match_results']):
                                # This cell might contain matching results
                                # We'll extract this information in a real implementation
                                # For now, just log it
                                logger.info(f"Found potential match data in notebook: {notebook_file_name}")
                except Exception as e:
                    logger.warning(f"Error processing notebook {notebook_file_name}: {e}")
        else:
            logger.warning(f"Notebooks subdirectory for Data Product not found: {data_product_dir_path}")

        # If we found any matches, convert to DataFrame
        if gold_matches:
            gold_df = pd.DataFrame(gold_matches)
            logger.info(f"Extracted {len(gold_df)} gold standard matches from notebooks")
            return gold_df
        else:
            logger.warning("No gold standard matches found in notebooks")
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error extracting gold standard matches: {e}")
        return pd.DataFrame()

# Enhanced version of train_matcher_from_existing_data
def train_matcher_from_existing_data(include_gold_standard: bool = True) -> FoodMatcher:
    """
    Train a FoodMatcher using existing match data
    
    Args:
        include_gold_standard: Whether to include gold standard matches from notebooks
        
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
    
    # Track metrics for all datasets
    all_metrics = {}
    
    # Train on each match dataset
    for match_type, matches_df in matches_dict.items():
        if not matches_df.empty:
            logger.info(f"Training matcher with {len(matches_df)} matches from {match_type}")
            
            # Extract features
            features_df = extract_features_from_matches(matches_df)
            
            # Get verified matches (high similarity assumed to be correct)
            verified_matches = get_verified_matches(features_df)
            
            # Train the matcher
            metrics = matcher.train_from_matches(features_df, verified_matches)
            
            # Store metrics
            if metrics:
                all_metrics[match_type] = metrics
    
    # Include gold standard matches if requested
    if include_gold_standard:
        gold_matches = extract_gold_standard_matches_from_notebooks()
        if not gold_matches.empty:
            logger.info(f"Training matcher with {len(gold_matches)} gold standard matches")
            
            # Extract features
            gold_features = extract_features_from_matches(gold_matches)
            
            # All gold standard matches are verified
            gold_verified = [True] * len(gold_features)
            
            # Train the matcher with higher weight for gold standard
            gold_metrics = matcher.train_from_matches(gold_features, gold_verified, sample_weight=2.0)
            
            # Store metrics
            if gold_metrics:
                all_metrics["gold_standard"] = gold_metrics
    
    # Store combined metrics in matcher for later reference
    if all_metrics:
        # Average metrics across datasets
        combined_metrics = {}
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            values = [m.get(metric, 0) for m in all_metrics.values()]
            if values:
                combined_metrics[metric] = sum(values) / len(values)
        
        matcher.last_train_metrics = combined_metrics
    
    return matcher

def generate_training_data_from_matches(matcher: FoodMatcher, threshold: float = 0.8) -> None:
    """
    Generates and saves training data from existing matches that meet a similarity threshold.
    
    Combines all existing match datasets, extracts features, filters for high-quality matches based on the specified similarity threshold, and saves the resulting training data to a configured file path for use in improving the matching algorithm.
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
    
    # Create training data CSV using configured path
    training_file_path = config.TRAINING_DATA_PATH
    
    # Save training data
    # Ensure output directory exists (though config.py should handle OUTPUT_DIR creation)
    os.makedirs(os.path.dirname(training_file_path), exist_ok=True) 
    high_quality.to_csv(training_file_path, index=False)
    logger.info(f"Generated training data with {len(high_quality)} high-quality matches to {training_file_path}")

def evaluate_matcher_performance() -> Dict[str, float]:
    """
    Evaluates the performance of the food matching algorithm on existing match datasets.
    
    Loads all available match data, computes aggregate metrics such as total matches, high-quality matches, average similarity, and precision estimate, and incorporates model cross-validation metrics if available. Records the evaluation results in the performance history.
    
    Returns:
        A dictionary containing performance metrics including total matches, high-quality matches, average similarity, precision estimate, and standard classification metrics if available.
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
        'average_similarity': 0.0,
        'accuracy': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'f1': 0.0
    }
    
    # Dataset info for history tracking
    dataset_info = {
        'sources': list(matches_dict.keys()),
        'source_counts': {k: len(v) for k, v in matches_dict.items()}
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
    
    # Get cross-validation metrics from model if available
    if hasattr(matcher, 'last_train_metrics') and matcher.last_train_metrics:
        metrics.update(matcher.last_train_metrics)
    
    # Record performance in history
    record_performance(metrics, dataset_info)
    
    return metrics

def apply_feedback_to_matches(matcher, feedback_data: List[Dict], feedback_file: str = None):
    """
    Apply user feedback to improve the matching model
    
    Args:
        matcher: FoodMatcher instance
        feedback_data: List of feedback entries with format:
                       [{'source_item': '...', 'target_item': '...', 'is_correct': True/False, 'features': {...}}]
        feedback_file: Optional path to save feedback
        
    Returns:
        Updated matcher and training results
    """
    if not feedback_data:
        logger.warning("No feedback data provided")
        return matcher, {}
    
    logger.info(f"Applying {len(feedback_data)} feedback entries to improve matcher")
    
    # If feedback_file is provided, load existing feedback and append
    all_feedback = []
    if feedback_file and os.path.exists(feedback_file):
        try:
            with open(feedback_file, 'r') as f:
                all_feedback = json.load(f)
            logger.info(f"Loaded {len(all_feedback)} existing feedback entries")
        except Exception as e:
            logger.warning(f"Error loading feedback file: {e}")
    
    # Add new feedback
    all_feedback.extend(feedback_data)
    
    # Save all feedback if file provided
    if feedback_file:
        try:
            os.makedirs(os.path.dirname(feedback_file), exist_ok=True)
            with open(feedback_file, 'w') as f:
                json.dump(all_feedback, f, indent=2)
            logger.info(f"Saved {len(all_feedback)} feedback entries to {feedback_file}")
        except Exception as e:
            logger.warning(f"Error saving feedback file: {e}")
    
    # Create training data from feedback
    X = []
    y = []
    for entry in all_feedback:
        if 'features' in entry and 'is_correct' in entry:
            X.append(entry['features'])
            y.append(1 if entry['is_correct'] else 0)
    
    if not X:
        logger.warning("No usable feedback data found")
        return matcher, {}
    
    # Train model directly with feedback
    logger.info(f"Training matcher with {len(X)} feedback entries")
    
    # Create a basic DataFrame for the matcher's train_from_matches method
    feedback_df = pd.DataFrame({
        'Source_Index': range(len(X)),
        'Target_Index': range(len(X)),
        'correct_match': y
    })
    
    # Add feature columns
    for i, features in enumerate(X):
        for feature_name, feature_value in features.items():
            feedback_df.loc[i, feature_name] = feature_value
    
    # Train model with feedback data
    training_results = matcher.train_from_matches(feedback_df, y)
    
    # Track performance metrics over time
    from shelfscale.visualization.performance import PerformanceTracker
    tracker = PerformanceTracker()
    
    # Add metrics and feature importance to history
    if isinstance(training_results, dict) and 'feature_importance' in training_results:
        tracker.add_metrics(training_results, source="feedback")
        tracker.add_feature_importance(training_results.get('feature_importance', {}))
    
    return matcher, training_results

def create_interactive_feedback_session(matcher, source_df: pd.DataFrame, target_df: pd.DataFrame, 
                                      source_col: str, target_col: str, num_samples: int = 10,
                                      output_file: str = "output/feedback_session.json"):
    """
                                      Generates and saves a set of sample matches for interactive user feedback to improve the matcher.
                                      
                                      Selects a balanced set of high and low confidence matches between the source and target datasets, formats them for user labeling, and saves the session to a JSON file for later review or retraining.
                                      
                                      Args:
                                          source_df: DataFrame containing source items to match.
                                          target_df: DataFrame containing target items to match against.
                                          source_col: Name of the column in the source DataFrame to use for matching.
                                          target_col: Name of the column in the target DataFrame to use for matching.
                                          num_samples: Total number of feedback samples to generate.
                                          output_file: Path to save the feedback session JSON file.
                                      
                                      Returns:
                                          A dictionary containing the feedback file path, the list of feedback samples, and the sample count.
                                      """
    if output_file is None:
        output_file = config.FEEDBACK_SESSION_PATH

    # 1. Create a matching to generate feedback samples
    matches = matcher.match_datasets(
        source_df,
        target_df,
        source_col,
        target_col
    )
    
    # 2. Select samples for feedback
    # Include a mix of high and low confidence matches
    high_conf = matches[matches['Similarity_Score'] > 0.8].sample(min(num_samples // 2, len(matches[matches['Similarity_Score'] > 0.8])))
    low_conf = matches[matches['Similarity_Score'] <= 0.8].sample(min(num_samples // 2, len(matches[matches['Similarity_Score'] <= 0.8])))
    sample_matches = pd.concat([high_conf, low_conf]).reset_index(drop=True)
    
    # 3. Convert to a feedback-friendly format
    feedback_samples = []
    for i, row in sample_matches.iterrows():
        source_item = source_df.iloc[row['Source_Index']]
        target_item = target_df.iloc[row['Target_Index']]
        
        # Create a feature dictionary
        feature_cols = [col for col in row.index if col not in ['Source_Index', 'Target_Index']]
        features = {col: row[col] for col in feature_cols}
        
        feedback_samples.append({
            'source_item': source_item[source_col],
            'target_item': target_item[target_col],
            'features': features,
            'similarity': row['Similarity_Score'],
            'source_index': int(row['Source_Index']),
            'target_index': int(row['Target_Index']),
            'is_correct': None  # To be filled by user
        })
    
    # 4. Save the feedback session
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(feedback_samples, f, indent=2)
        logger.info(f"Feedback session with {len(feedback_samples)} samples saved to {output_file}")
    except Exception as e:
        logger.warning(f"Error saving feedback session: {e}")
    
    return {
        'feedback_file': output_file,
        'samples': feedback_samples,
        'sample_count': len(feedback_samples)
    }

def process_feedback_file(feedback_file: str, matcher, apply_immediately: bool = True):
    """
    Process a feedback file to improve the matcher
    
    Args:
        feedback_file: Path to feedback file
        matcher: FoodMatcher instance
        apply_immediately: Whether to apply feedback immediately
        
    Returns:
        Updated matcher and training results
    """
    if not os.path.exists(feedback_file):
        logger.warning(f"Feedback file {feedback_file} not found")
        return matcher, {}
    
    # Load feedback
    try:
        with open(feedback_file, 'r') as f:
            feedback_data = json.load(f)
        logger.info(f"Loaded {len(feedback_data)} feedback entries from {feedback_file}")
    except Exception as e:
        logger.warning(f"Error loading feedback file: {e}")
        return matcher, {}
    
    # Filter to only entries with 'is_correct' filled out
    valid_feedback = [entry for entry in feedback_data if entry.get('is_correct') is not None]
    logger.info(f"Found {len(valid_feedback)} valid feedback entries")
    
    if not valid_feedback:
        logger.warning("No valid feedback entries found")
        return matcher, {}
    
    if apply_immediately:
        return apply_feedback_to_matches(matcher, valid_feedback, feedback_file)
    
    return matcher, {'feedback_count': len(valid_feedback)}

def visualize_feedback_impact(feedback_file: str, output_file: str = "output/feedback_impact.html"):
    """
    Generates an interactive dashboard visualizing the impact of user feedback on model performance.
    
    Creates and saves a performance history dashboard to the specified output file, and logs a summary of recent improvements based on feedback data.
    """
    if output_file is None:
        output_file = config.FEEDBACK_IMPACT_VISUALIZATION_PATH

    from shelfscale.visualization.performance import PerformanceTracker # Keep local import for now
    tracker = PerformanceTracker(history_file=config.MODEL_PERFORMANCE_HISTORY_PATH) # Pass history file
    
    # Generate interactive dashboard with performance history
    # Ensure output directory exists (though config.py should handle OUTPUT_DIR creation)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    tracker.create_interactive_dashboard(output_file)
    
    # Print summary of improvements
    summary = tracker.get_performance_summary()
    
    if 'error' in summary:
        logger.warning(f"Could not generate summary: {summary['error']}")
        return
    
    logger.info(f"Performance summary as of {summary['date']}:")
    for metric, value in summary['latest_metrics'].items():
        if metric in summary['improvements'] and isinstance(value, (int, float)):
            improvement = summary['improvements'][metric]
            sign = '+' if improvement > 0 else ''
            logger.info(f"  {metric}: {value:.4f} ({sign}{improvement:.4f})")
        elif isinstance(value, (int, float)):
            logger.info(f"  {metric}: {value:.4f}")
    
    logger.info(f"Performance dashboard saved to {output_file}")

if __name__ == "__main__":
    # If run directly, train a matcher and generate training data
    from fuzzywuzzy import fuzz
    matcher = train_matcher_from_existing_data()
    generate_training_data_from_matches(matcher)
    metrics = evaluate_matcher_performance()
    print("Matcher performance metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}") 