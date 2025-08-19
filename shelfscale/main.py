"""
ShelfScale Main Application - LLM-Enhanced Food Product Analysis

This script demonstrates the full capabilities of ShelfScale's LLM-Enhanced
matching system for food product analysis, weight prediction, and nutritional scoring.

Features:
- LLM-Enhanced product matching with semantic understanding
- Advanced weight extraction and prediction
- Multi-source data integration
- Nutritional scoring (Traffic Light, Nutri-Score)
- Interactive analysis with detailed reasoning
"""

import os
import pandas as pd
import numpy as np
import re
import argparse
import random
import logging
from typing import List, Dict, Tuple, Optional, Any, Union

import shelfscale.config as config
from shelfscale.data_sourcing.open_food_facts import OpenFoodFactsClient
from shelfscale.data_processing.weight_extraction import clean_weights
from shelfscale.data_processing.categorization import FoodCategorizer, clean_food_categories
from shelfscale.data_processing.cleaner import DataCleaner
from shelfscale.data_processing.transformation import normalize_weights, create_food_group_summary
from shelfscale.matching.algorithm import FoodMatcher
from shelfscale.utils.helpers import (
    load_data, 
    save_data, 
    split_data_by_group,
    extract_numeric_value
)
from shelfscale.utils.learning import (
    train_matcher_from_existing_data, 
    evaluate_matcher_performance,
    generate_training_data_from_matches,
    create_weight_predictions,
    apply_feedback_to_matches,
    load_existing_matches
)
from shelfscale.data_sourcing.excel_loader import ExcelLoader
from shelfscale.data_sourcing.csv_loader import CsvLoader
from shelfscale.scoring import score_traffic_lights, score_nutri

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Local data loading functions (extract_food_portion_data, extract_fruit_veg_survey_data, load_mccance_widdowson_data)
# are now removed. Their responsibilities are moved to PDFExtractor and ExcelLoader respectively.


def match_datasets(main_df, secondary_df, main_col, secondary_col, threshold=70, additional_cols=None):
    """Match items between datasets using fuzzy matching with machine learning"""
    print(f"Matching datasets based on {main_col} and {secondary_col}...")
    
    # Try to load a trained matcher first for better results
    # Initialize LLM-Enhanced Food Matcher (v3.0)
    logger.info("🤖 Initializing LLM-Enhanced Food Matching System")
    try:
        matcher = train_matcher_from_existing_data(model_dir=config.MODEL_DIR)
        logger.info("✅ Using trained LLM-enhanced matcher for dataset matching")
    except Exception as e:
        logger.warning(f"Could not load trained matcher: {e}. Using default LLM-enhanced matcher.")
        matcher = FoodMatcher(similarity_threshold=threshold/100, use_llm=True)
    
    # Log system status
    if hasattr(matcher, 'get_system_status'):
        status = matcher.get_system_status()
        logger.info(f"🚀 Primary matching system: {status.get('primary_system', 'unknown')}")
        if 'llm_enhanced' in status.get('primary_system', ''):
            logger.info("🧠 Using LLM for semantic food understanding")
    
    # Set up additional matching columns
    additional_match_cols = None
    if additional_cols:
        additional_match_cols = [(col, col) for col in additional_cols if col in main_df.columns and col in secondary_df.columns]
    
    # Perform matching
    matches = matcher.match_datasets(
        main_df, 
        secondary_df, 
        main_col, 
        secondary_col,
        additional_match_cols=additional_match_cols
    )
    
    # Merge matched datasets
    merged_df = matcher.merge_matched_datasets(
        main_df,
        secondary_df,
        matches
    )
    
    print(f"  Found {len(merged_df)} matches with similarity > {threshold}%")
    return merged_df


def process_weight_info(df, weight_cols):
    """
    Process weight information from one or more columns using enhanced weight extraction
    
    Args:
        df: Input DataFrame
        weight_cols: Column(s) containing weight information
            Can be a string or list of strings
            
    Returns:
        DataFrame with processed weight information
    """
    # Make a copy
    weight_processed_df = df.copy()
    
    # Handle single column as string
    if isinstance(weight_cols, str):
        weight_cols = [weight_cols]
    
    # Try to find weight columns if none provided
    if not weight_cols or not any(col in df.columns for col in weight_cols):
        # Look for columns that might contain weight information
        potential_weight_cols = []
        for col in df.columns:
            col_lower = col.lower()
            if any(term in col_lower for term in ['weight', 'size', 'portion', 'pack', 'g)', 'kg)', 'g', 'kg']):
                potential_weight_cols.append(col)
        
        if potential_weight_cols:
            print(f"  No valid weight columns provided, using detected columns: {', '.join(potential_weight_cols)}")
            weight_cols = potential_weight_cols
        else:
            print("  No weight columns found in the DataFrame")
            # Create empty weight column to ensure consistent output structure
            weight_processed_df['Normalized_Weight'] = np.nan
            return weight_processed_df
    
    # Use the enhanced weight extraction
    from shelfscale.data_processing.weight_extraction import WeightExtractor
    
    # Create weight extractor with default target unit (grams)
    extractor = WeightExtractor()
    
    # Process all weight columns at once using the enhanced extractor
    result_df = extractor.process_dataframe(
        weight_processed_df,
        weight_cols,
        new_weight_col='Normalized_Weight',
        new_unit_col='Weight_Unit',
        source_col='Weight_Source_Col'
    )
    
    # Get extraction statistics
    success_count = result_df['Normalized_Weight'].notna().sum()
    total_count = len(result_df)
    success_rate = (success_count / total_count) * 100 if total_count > 0 else 0
    
    print(f"  Successfully extracted {success_count} weights out of {total_count} items ({success_rate:.1f}%)")
    
    # Try to predict missing weights using group averages and similar items
    if success_count > 0 and success_count < total_count:
        from shelfscale.data_processing.weight_extraction import predict_missing_weights
        
        # Check if we have food group or name columns that could help with prediction
        group_col = None
        for col_name in ['Food_Group', 'Food Group', 'Category', 'Super_Group', 'Super Group']:
            if col_name in result_df.columns:
                group_col = col_name
                break
                
        name_col = None
        for col_name in ['Food_Name', 'Food Name', 'Product_Name', 'Product Name', 'Name', 'Description']:
            if col_name in result_df.columns:
                name_col = col_name
                break
                
        if group_col or name_col:
            print(f"  Attempting to predict missing weights using {'food groups and ' if group_col else ''}{'similar item names' if name_col else ''}")
            result_df = predict_missing_weights(
                result_df,
                weight_col='Normalized_Weight',
                group_col=group_col,
                name_col=name_col
            )
            
            # Report prediction results
            predicted_count = result_df['Weight_Prediction_Source'].notna().sum()
            if predicted_count > 0:
                print(f"  Successfully predicted {predicted_count} additional weights")
    
    return result_df


def score_input_file(input_file: str, score_type: str, output_file: str) -> pd.DataFrame:
    """
    Score an input CSV file directly
    
    Args:
        input_file: Path to input CSV file
        score_type: Type of scoring ('nutri', 'traffic', 'all')
        output_file: Path to output scores file
        
    Returns:
        Scored DataFrame
    """
    print(f"Loading input file: {input_file}")
    
    # Load the input data
    try:
        df = pd.read_csv(input_file)
        print(f"Loaded {len(df)} rows from {input_file}")
    except Exception as e:
        logger.error(f"Error loading input file: {e}")
        return pd.DataFrame()
    
    # Apply scoring
    scored_df = apply_nutrition_scoring(df, score_type)
    
    # Save results
    scored_df.to_csv(output_file, index=False)
    print(f"Saved scored results to {output_file}")
    
    return scored_df


def build_nutrient_view(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a nutrition view with standardized nutrient columns
    
    Args:
        df: Input DataFrame with product data
        
    Returns:
        DataFrame with nutrition columns ready for scoring
    """
    nutrient_df = df.copy()
    
    # Standardize nutrient column names - look for common variations
    nutrient_mappings = {
        # Energy
        'Energy_kJ': ['Energy (kJ)', 'Energy_kJ', 'Energy kJ', 'Energy_kilojoules'],
        'Energy_kcal': ['Energy (kcal)', 'Energy_kcal', 'Energy kcal', 'Energy_calories', 'Calories'],
        
        # Macronutrients
        'Fat_g': ['Fat (g)', 'Fat_g', 'Total Fat', 'Fat g', 'Fat'],
        'SatFat_g': ['SatFat (g)', 'SatFat_g', 'Saturated Fat', 'Saturated fat', 'SaturatedFat_g'],
        'Sugars_g': ['Sugars (g)', 'Sugars_g', 'Total Sugars', 'Sugars g', 'Sugars'],
        'Salt_g': ['Salt (g)', 'Salt_g', 'Salt g', 'Salt'],
        'Sodium_mg': ['Sodium (mg)', 'Sodium_mg', 'Sodium mg', 'Sodium'],
        'Fiber_g': ['Fiber (g)', 'Fiber_g', 'Fibre (g)', 'Fibre_g', 'Dietary Fiber', 'Fiber'],
        'Protein_g': ['Protein (g)', 'Protein_g', 'Protein g', 'Protein'],
        
        # Other nutrients
        'Carbs_g': ['Carbohydrate (g)', 'Carbs_g', 'Carbohydrates', 'Carbs'],
    }
    
    # Map columns to standardized names
    for standard_name, variations in nutrient_mappings.items():
        if standard_name not in nutrient_df.columns:
            for variation in variations:
                if variation in nutrient_df.columns:
                    nutrient_df[standard_name] = nutrient_df[variation]
                    logger.info(f"Mapped {variation} to {standard_name}")
                    break
    
    # Add nutrient source tracking
    nutrient_df['Nutrient_Source'] = 'MW'  # Default to McCance & Widdowson
    
    # For items without nutrition data, estimate from category medians if available
    missing_nutrition = nutrient_df[['Energy_kcal', 'Fat_g', 'SatFat_g', 'Sugars_g', 'Salt_g']].isna().all(axis=1)
    
    if missing_nutrition.any() and 'Food_Category' in nutrient_df.columns:
        logger.info(f"Estimating nutrition for {missing_nutrition.sum()} items using category medians")
        
        # Calculate category medians for available data
        category_nutrition = nutrient_df[~missing_nutrition].groupby('Food_Category')[
            ['Energy_kcal', 'Fat_g', 'SatFat_g', 'Sugars_g', 'Salt_g', 'Fiber_g', 'Protein_g']
        ].median()
        
        # Fill missing values with category medians
        for idx in nutrient_df[missing_nutrition].index:
            category = nutrient_df.loc[idx, 'Food_Category']
            if category in category_nutrition.index:
                for col in category_nutrition.columns:
                    if pd.isna(nutrient_df.loc[idx, col]):
                        nutrient_df.loc[idx, col] = category_nutrition.loc[category, col]
                        nutrient_df.loc[idx, 'Nutrient_Source'] = 'Category_Median'
    
    return nutrient_df


def apply_nutrition_scoring(df: pd.DataFrame, score_type: str) -> pd.DataFrame:
    """
    Apply nutrition scoring to a DataFrame
    
    Args:
        df: DataFrame with nutrition data
        score_type: Type of scoring ('nutri', 'traffic', 'all')
        
    Returns:
        DataFrame with scoring results
    """
    print(f"Applying {score_type} scoring to {len(df)} products...")
    
    # Build nutrition view
    nutrition_df = build_nutrient_view(df)
    
    # Add weight and provenance columns if not present
    weight_cols = ['Normalized_Weight', 'Weight_Unit', 'Weight_Source', 
                   'Weight_Prediction_Source', 'Weight_Prediction_Confidence']
    
    for col in weight_cols:
        if col not in nutrition_df.columns:
            if 'Confidence' in col:
                nutrition_df[col] = np.nan
            else:
                nutrition_df[col] = None
    
    # Apply scoring based on type
    if score_type in ['traffic', 'all']:
        print("Calculating Traffic Light scores...")
        nutrition_df = score_traffic_lights(nutrition_df)
    
    if score_type in ['nutri', 'all']:
        print("Calculating Nutri-Scores...")
        nutrition_df = score_nutri(nutrition_df)
    
    # Calculate overall score confidence
    if 'Score_Confidence' not in nutrition_df.columns:
        nutrition_df['Score_Confidence'] = 0.8  # Default confidence
    
    # Report coverage
    total_products = len(nutrition_df)
    
    if score_type in ['traffic', 'all']:
        traffic_coverage = (nutrition_df['Traffic_Lights_Summary'] != 'unknown').sum()
        print(f"Traffic Light coverage: {traffic_coverage}/{total_products} ({traffic_coverage/total_products*100:.1f}%)")
    
    if score_type in ['nutri', 'all']:
        nutri_coverage = nutrition_df['Nutri_Grade'].notna().sum()
        print(f"Nutri-Score coverage: {nutri_coverage}/{total_products} ({nutri_coverage/total_products*100:.1f}%)")
    
    return nutrition_df


def main():
    """Main function demonstrating ShelfScale workflow with multiple data sources"""
    print("ShelfScale - Comprehensive Food Product Weight Analysis")
    print("=" * 60)
    
    # Set up more detailed logging for debugging
    logging.basicConfig(level=logging.INFO, 
                      format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="ShelfScale - Food Product Weight Analysis")
    parser.add_argument("--output", default=os.path.join(config.OUTPUT_DIR, "weight_dataset.csv"), help="Output file path")
    parser.add_argument("--food-portion-pdf", default=config.FOOD_PORTION_PDF_PATH, help="Food Portion Sizes PDF path")
    parser.add_argument("--fruit-veg-pdf", default=config.FRUIT_VEG_SURVEY_PDF_PATH, help="Fruit and Veg Survey PDF path") # Corrected to use full path
    parser.add_argument("--mccance-widdowson", default=config.MCCANCE_WIDDOWSON_PATH, help="McCance and Widdowson data path")
    parser.add_argument("--matching-threshold", type=int, default=config.DEFAULT_MATCHING_THRESHOLD, help="Matching threshold percentage")
    parser.add_argument("--run-dashboard", action="store_true", help="Run interactive dashboard")
    parser.add_argument("--load-cache", action="store_true", help="Load cached data (faster)")
    # --skip-pdfs argument is removed
    parser.add_argument("--use-super-group", action="store_false", default=True, help="Use Super Group reduced data")
    parser.add_argument("--process-raw", action="store_true", help="Process all raw data and save to Processed directory")
    
    # Scoring arguments
    parser.add_argument("--score", choices=['nutri', 'traffic', 'all'], help="Calculate nutrition scores")
    parser.add_argument("--input-file", help="Input CSV file for scoring (optional)")
    parser.add_argument("--output-scores", default=config.NUTRITION_SCORES_PATH, help="Output file for nutrition scores")
    
    args = parser.parse_args()
    
    # If processing raw data was requested, do it and exit
    if args.process_raw:
        print("Processing all raw data files...")
        processed_data = process_raw_data()
        print("Raw data processing complete. Results saved to Processed directory.")
        return processed_data
    
    # If scoring was requested and input file provided, score that file directly
    if args.score and args.input_file:
        return score_input_file(args.input_file, args.score, args.output_scores)
    
    # Output directory is created by config.py
    
    # Instantiate loaders
    excel_loader = ExcelLoader()
    
    # Try to import and instantiate PDFExtractor conditionally
    try:
        from shelfscale.data_sourcing.pdf_extraction import PDFExtractor
        pdf_extractor = PDFExtractor(cache_dir=config.CACHE_DIR)
        pdf_available = True
    except ImportError as e:
        logger.warning(f"PDFExtractor not available: {e}. PDF processing will be skipped.")
        pdf_extractor = None
        pdf_available = False
    # csv_loader = CsvLoader() # Not directly used for initial data load in main, but available

    # Setup data sources
    data_sources = {}
    
    # 1. Load McCance and Widdowson's dataset (highest quality)
    # TODO: Consider moving caching logic into ExcelLoader for consistency.
    if args.load_cache and os.path.exists(config.MW_DATA_CACHED_PATH):
        logger.info(f"Loading cached McCance and Widdowson data from {config.MW_DATA_CACHED_PATH}...")
        data_sources["mw_data"] = pd.read_csv(config.MW_DATA_CACHED_PATH)
    else:
        logger.info("Loading McCance and Widdowson data using ExcelLoader...")
        # ExcelLoader uses config.MCCANCE_WIDDOWSON_PATH and config.MW_SHEET_NAME_FOR_MAIN_PY by default
        # if we pass sheet_name_primary=config.MW_SHEET_NAME_FOR_MAIN_PY explicitly.
        # Default in ExcelLoader is MW_FACTORS_SHEET_NAME, main.py used MW_SHEET_NAME_FOR_MAIN_PY (formerly config.MW_SHEET_NAME)
        data_sources["mw_data"] = excel_loader.load_mccance_widdowson(
            file_path=args.mccance_widdowson, # Use path from args, which defaults to config
            sheet_name_primary=config.MW_SHEET_NAME_FOR_MAIN_PY # Specify sheet for main.py context
        )
        if data_sources["mw_data"] is not None and not data_sources["mw_data"].empty:
            logger.info(f"Saving McCance and Widdowson data to cache: {config.MW_DATA_CACHED_PATH}")
            data_sources["mw_data"].to_csv(config.MW_DATA_CACHED_PATH, index=False)
        elif data_sources["mw_data"] is None: # Ensure it's an empty DataFrame if loading failed
             data_sources["mw_data"] = pd.DataFrame()

    # 2. Load McCance and Widdowson's Super Group reduced dataset
    # This logic remains largely the same as it loads processed CSVs.
    super_group_data = {}
    if args.use_super_group:
        print("Loading McCance and Widdowson Super Group reduced data...")
        # Using config.MW_PROCESSED_SUPER_GROUP_PATH
        super_group_dir_path = config.MW_PROCESSED_SUPER_GROUP_PATH 
        # get_path might be redundant if config paths are absolute or correctly relative
        # super_group_dir_path = get_path(super_group_dir_path) 
        if not os.path.isdir(super_group_dir_path):
            logger.warning(f"Super-group directory '{super_group_dir_path}' not found – skipping.")
            super_group_files = []
        else:
            super_group_files = [
                f for f in os.listdir(super_group_dir_path)
                if f.endswith(".csv") and not f.startswith(".")
            ]
        
        # Output directory created by config.py
        # os.makedirs(config.OUTPUT_DIR, exist_ok=True) 
        
        for file in super_group_files:
            file_path = os.path.join(super_group_dir_path, file)
            try:
                df = pd.read_csv(file_path)
                group_name = os.path.splitext(file)[0]
                super_group_data[group_name] = df
                print(f"  Loaded {len(df)} items from {file}")
            except Exception as e:
                logger.error(f"Error loading {file}: {e}")
        
        # Combine super groups
        if super_group_data:
            data_sources["super_group"] = pd.concat(super_group_data.values())
            print(f"  Total super group items: {len(data_sources['super_group'])}")
    
        # 3. Load Food Portion Sizes data (portion-specific)
    # 3. Load Food Portion Sizes data (portion-specific)
    # PDFExtractor handles its own caching. The --load-cache flag is implicitly handled by PDFExtractor.
    if pdf_available:
        logger.info("Loading Food Portion Sizes data using PDFExtractor...")
        try:
            data_sources["portion_data"] = pdf_extractor.extract_food_portion_sizes(
                pdf_path=args.food_portion_pdf
                # pages argument is handled by PDFExtractor using config.PDF_FOOD_PORTION_PAGES by default
            )
            if data_sources["portion_data"] is None: # Ensure it's an empty DF if loading failed
                data_sources["portion_data"] = pd.DataFrame()
        except FileNotFoundError as e: # Should be handled by PDFExtractor, but as a fallback
            logger.error(f"Could not find food portion PDF (main.py fallback): {e}")
            data_sources["portion_data"] = pd.DataFrame()
    else:
        logger.info("PDFExtractor not available - skipping Food Portion Sizes data")
        data_sources["portion_data"] = pd.DataFrame()

    # 4. Load Fruit and Vegetable Survey data
    # PDFExtractor handles its own caching.
    if pdf_available:
        logger.info("Loading Fruit and Vegetable Survey data using PDFExtractor...")
        try:
            data_sources["fruit_veg_data"] = pdf_extractor.extract_fruit_veg_survey(
                pdf_path=args.fruit_veg_pdf
                # pages argument is handled by PDFExtractor using config.FRUIT_VEG_SURVEY_PAGES by default
            )
            if data_sources["fruit_veg_data"] is None: # Ensure it's an empty DF if loading failed
                data_sources["fruit_veg_data"] = pd.DataFrame()
        except FileNotFoundError as e: # Should be handled by PDFExtractor, but as a fallback
            logger.error(f"Could not find fruit and veg PDF (main.py fallback): {e}")
            data_sources["fruit_veg_data"] = pd.DataFrame()
    else:
        logger.info("PDFExtractor not available - skipping Fruit and Vegetable Survey data")
        data_sources["fruit_veg_data"] = pd.DataFrame()
    
    # Create an integrated dataset by matching items across sources
    print("\nCreating integrated dataset from all sources...")
    
    # Use MW data as the base dataset
    base_dataset = data_sources.get("mw_data", pd.DataFrame()).copy()
    print(f"Base dataset from McCance and Widdowson: {len(base_dataset)} items")
    
    # Column name standardization will now be primarily handled by DataCleaner.
    # The local standardize_columns function is removed.
    # Loaders (ExcelLoader, PDFExtractor) are expected to provide data conforming to their schemas
    # which use consistent names like 'Food_Name', 'Weight_g'.
    # DataCleaner will then apply its own standardization (e.g., to snake_case or other rules).
    # Subsequent code in main.py needs to be aware of the column naming convention output by DataCleaner.

    # Example: If DataCleaner standardizes to snake_case, then 'Food_Name' becomes 'food_name'.
    # This will be handled when DataCleaner.clean() is called on base_dataset later.
    
    # Make sure Food Name is standardized in base dataset (initial check before DataCleaner)
    # This specific check might be redundant if ExcelLoader is correctly using MW_EXPECTED_COLUMNS
    # which should already specify 'Food_Name'.
    if "Food Name" in base_dataset.columns and "Food_Name" not in base_dataset.columns: # From M&W legacy loading
        base_dataset = base_dataset.rename(columns={"Food Name": "Food_Name"})
    
    # Check required columns
    if "Food_Name" not in base_dataset.columns and "Food Name" in base_dataset.columns:
        base_dataset["Food_Name"] = base_dataset["Food Name"]
    
    # Ensure we have a food name column for matching
    food_name_col = "Food_Name"
    if food_name_col not in base_dataset.columns:
        for col in base_dataset.columns:
            if 'food' in col.lower() and 'name' in col.lower():
                food_name_col = col
                break
            elif 'name' in col.lower():
                food_name_col = col
                break
            elif 'description' in col.lower():
                food_name_col = col
                break
    
    # Initialize matcher
    matcher = FoodMatcher(similarity_threshold=args.matching_threshold / 100)
    
    # Match with Food Portion Sizes data
    # portion_data = standardize_columns(data_sources.get("portion_data", pd.DataFrame()).copy()) # standardize_columns removed
    portion_data = data_sources.get("portion_data", pd.DataFrame()).copy()
    # PDFExtractor for portion_data should provide columns as per FPS_EXPECTED_SCHEMA (e.g., 'Food_Name', 'Weight_g')
    # If DataCleaner later standardizes portion_data (if it's part of base_dataset or cleaned separately),
    # then matching should occur on those DataCleaner-standardized names.
    # For now, assuming portion_data columns are as per its schema directly from PDFExtractor.
    
    print(f"Matching with Food Portion data: {len(portion_data)} items")
    
    if len(portion_data) > 0 and not portion_data.empty and food_name_col in base_dataset.columns:
        # Determine target column for matching in portion_data
        # PDFExtractor should ensure 'Food_Name' from FPS_EXPECTED_SCHEMA
        target_portion_food_col = 'Food_Name' 
        if target_portion_food_col not in portion_data.columns:
            # Fallback if schema-defined name isn't present (should not happen if PDFExtractor is correct)
            logger.warning(f"'{target_portion_food_col}' not in portion_data. Attempting to find alternative name for matching.")
            alt_cols = [col for col in portion_data.columns if 'food' in col.lower() and 'name' in col.lower()]
            if alt_cols: target_portion_food_col = alt_cols[0]
            else: target_portion_food_col = None # No suitable column
        
        if target_portion_food_col:
            portion_matches = matcher.match_datasets(
                base_dataset,
                portion_data,
                food_name_col,  # Source column from base_dataset
                target_portion_food_col,  # Target column from portion_data
                additional_match_cols=None
            )
        else:
            logger.warning("Could not find a suitable food name column in portion_data for matching. Skipping portion data matching.")
            portion_matches = pd.DataFrame() # Empty DataFrame if no column
            
        # Merge matches into base dataset
        merged_portion = matcher.merge_matched_datasets(
            base_dataset,
            portion_data,
            portion_matches,
            None,  # All source columns
            None,  # All target columns
            {"Weight_g": "Portion_Weight_g"}  # Rename target weights
        )
        
        if len(merged_portion) > 0:
            base_dataset = merged_portion
            print(f"Merged with portion data: {len(merged_portion)} items")
        else:
            print("No portion data matches found.")
    
    # Match with Fruit and Vegetable Survey data
    # fruit_veg_data = standardize_columns(data_sources.get("fruit_veg_data", pd.DataFrame()).copy()) # standardize_columns removed
    fruit_veg_data = data_sources.get("fruit_veg_data", pd.DataFrame()).copy()
    # PDFExtractor for fruit_veg_data should provide columns as per FVS_EXPECTED_SCHEMA (e.g., 'Sample_Name')

    print(f"Matching with Fruit and Veg data: {len(fruit_veg_data)} items")
    
    if len(fruit_veg_data) > 0 and not fruit_veg_data.empty and food_name_col in base_dataset.columns:
        # Determine target column for matching in fruit_veg_data
        # PDFExtractor should ensure 'Sample_Name' from FVS_EXPECTED_SCHEMA
        target_fvs_food_col = 'Sample_Name' 
        if target_fvs_food_col not in fruit_veg_data.columns:
            logger.warning(f"'{target_fvs_food_col}' not in fruit_veg_data. Attempting to find alternative name for matching.")
            alt_cols = [col for col in fruit_veg_data.columns if ('sample' in col.lower() or 'food' in col.lower()) and 'name' in col.lower()]
            if alt_cols: target_fvs_food_col = alt_cols[0]
            else: target_fvs_food_col = None # No suitable column

        if target_fvs_food_col:
            fruit_veg_matches = matcher.match_datasets(
                base_dataset,
                fruit_veg_data,
                food_name_col,  # Source column from base_dataset
                target_fvs_food_col,  # Target column from fruit_veg_data
                additional_match_cols=None
            )
        else:
            logger.warning("Could not find a suitable food name column in fruit_veg_data for matching. Skipping FVS data matching.")
            fruit_veg_matches = pd.DataFrame() # Empty DataFrame if no column

        # Merge matches into base dataset
        merged_fruit_veg = matcher.merge_matched_datasets(
            base_dataset,
            fruit_veg_data,
            fruit_veg_matches,
            None,  # All source columns
            None,  # All target columns
            {"Pack_Size": "Fruit_Veg_Size"}  # Rename target weights
        )
        
        if len(merged_fruit_veg) > 0:
            base_dataset = merged_fruit_veg
            print(f"Merged with fruit and veg data: {len(merged_fruit_veg)} items")
        else:
            print("No fruit and veg data matches found.")
    
    # Clean the integrated dataset
    print("\nCleaning and formatting the integrated dataset...")
    cleaner = DataCleaner()
    
    try:
        cleaned_dataset = cleaner.clean(base_dataset)
    except Exception as e:
        logger.error(f"Error cleaning dataset: {e}")
        logger.warning("Using original dataset without cleaning.")
        cleaned_dataset = base_dataset
    
    if len(cleaned_dataset) == 0:
        logger.warning("No data available after cleaning. Using original dataset.")
        cleaned_dataset = base_dataset
    
    # Categorize foods by type
    categorizer = FoodCategorizer()
    
    # Find food name column for categorization
    if food_name_col not in cleaned_dataset.columns:
        food_name_col = None
        # Try to find a suitable column for categorization
    for col_candidate in ['Food_Name', 'Food Name', 'name', 'description', 'Name', 'Description']: # Prioritize common names
        if col_candidate in cleaned_dataset.columns:
            food_name_col = col_candidate
            break
    else: # If no exact match, then search more broadly
        for col in cleaned_dataset.columns:
            if 'food' in col.lower() and 'name' in col.lower():
                food_name_col = col
                break
            elif 'name' in col.lower(): # Broader 'name' check
                food_name_col = col
                break
            elif 'description' in col.lower(): # Broader 'description' check
                food_name_col = col
                break
        else: # If still not found
            logger.warning("Could not find a suitable food name column for categorization.")
            logger.info(f"Available columns:\n{cleaned_dataset.columns.tolist()}")
            if len(cleaned_dataset) > 0: # Create a dummy column only if df is not empty
                cleaned_dataset['Food_Name_Fallback_For_Categorization'] = "Unknown Food"
                food_name_col = 'Food_Name_Fallback_For_Categorization'
            else:
                food_name_col = None # No column to use if df is empty

    if food_name_col:
        print(f"Categorizing foods using column: {food_name_col}")
        if food_name_col in cleaned_dataset.columns:
            categorized_dataset = categorizer.clean_food_categories(
                cleaned_dataset, 
                food_name_col, 
                'Food_Category', 
                'Super_Category'
            )
        else: # Should not happen if logic above is correct
            logger.warning(f"Food name column '{food_name_col}' selected but not found. Skipping categorization.")
            categorized_dataset = cleaned_dataset
            categorized_dataset['Food_Category'] = "Unknown"
            categorized_dataset['Super_Category'] = "Unknown"
    else:
        logger.warning("No food name column available for categorization. Skipping categorization.")
        categorized_dataset = cleaned_dataset
        if 'Food_Category' not in categorized_dataset.columns:
             categorized_dataset['Food_Category'] = "Unknown"
        if 'Super_Category' not in categorized_dataset.columns:
             categorized_dataset['Super_Category'] = "Unknown"

    # Process and normalize weight information
    weight_cols = []
    for col in categorized_dataset.columns:
        if 'weight' in col.lower() or 'size' in col.lower():
            weight_cols.append(col)
    
    if weight_cols:
        print(f"Processing weight information from columns: {', '.join(weight_cols)}")
        normalized_dataset = process_weight_info(categorized_dataset, weight_cols)
        print(f"Final dataset has {len(normalized_dataset)} items with weight information")
    else:
        logger.warning("No weight columns found in the dataset")
        normalized_dataset = categorized_dataset
        normalized_dataset['Normalized_Weight'] = np.nan
    
    # Apply nutrition scoring if requested
    if args.score:
        print(f"\nApplying {args.score} nutrition scoring...")
        scored_dataset = apply_nutrition_scoring(normalized_dataset, args.score)
        
        # Save scored dataset
        save_data(scored_dataset, args.output_scores)
        print(f"Saved nutrition scores to {args.output_scores}")
        
        # Update normalized_dataset for dashboard
        normalized_dataset = scored_dataset
    
    # Save the comprehensive dataset
    save_data(normalized_dataset, args.output)
    print(f"Saved integrated dataset to {args.output}")
    
    # Launch dashboard if requested
    if args.run_dashboard:
        print("\nLaunching interactive dashboard...")
        try:
            # Use the enhanced dashboard with pre-built data
            from shelfscale.visualization.enhanced_dashboard import create_enhanced_dashboard
            dashboard = create_enhanced_dashboard()
            dashboard.run_server(debug=False, port=8050)
        except ImportError as e:
            logger.error(f"Could not load dashboard. Dashboard dependencies may not be installed: {e}")
            print("Error: Dashboard dependencies not installed. Install with 'pip install dash plotly'.")
    
    print("\nProcess completed successfully!")
    return normalized_dataset


if __name__ == "__main__":
    main() 