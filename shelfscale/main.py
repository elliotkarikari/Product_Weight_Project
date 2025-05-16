"""
Main script demonstrating ShelfScale package usage with comprehensive data sources
with machine learning capabilities for continuous improvement
"""

import os
import pandas as pd
import numpy as np
import tabula
import PyPDF2
import re
import argparse
from fuzzywuzzy import process, fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
import logging

from shelfscale.data_sourcing.open_food_facts import OpenFoodFactsClient
from shelfscale.data_processing.weight_extraction import clean_weights
from shelfscale.data_processing.categorization import clean_food_categories, FoodCategorizer
from shelfscale.data_processing.cleaner import DataCleaner
from shelfscale.data_processing.transformation import normalize_weights, create_food_group_summary
from shelfscale.matching.algorithm import FoodMatcher
from shelfscale.utils.helpers import (
    load_data, 
    save_data, 
    split_data_by_group,
    get_path,
)
from shelfscale.utils.learning import (
    train_matcher_from_existing_data, 
    evaluate_matcher_performance,
    generate_training_data_from_matches,
    create_weight_predictions,
    apply_feedback_to_matches,
    load_existing_matches
)
from shelfscale.data_sourcing.pdf_extraction import PDFExtractor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_food_portion_data(pdf_path):
    """Extract data from Food Portion Sizes PDF"""
    print("Extracting data from Food Portion Sizes PDF...")
    
    try:
        # Extract tables from PDF
        tables = tabula.read_pdf(pdf_path, 
                                pages="12-114", 
                                stream=True, 
                                multiple_tables=True, 
                                guess=False, 
                                encoding="latin1",
                                pandas_options={"header": [0, 1, 2, 3]}
                            )
        
        # Combine tables into single DataFrame
        df = pd.DataFrame()
        for table in tables:
            df = pd.concat([df, table], ignore_index=True)
        
        # Clean up the DataFrame
        df.columns = ['Food_Name', 'Portion_Size', 'Weight_g', 'Notes']
        df = df.dropna(thresh=2)  # Drop rows with less than 2 non-NaN values
        
        print(f"  Extracted {len(df)} food portion records")
        return df
    except Exception as e:
        print(f"Error extracting data from PDF: {str(e)}")
        # Return empty DataFrame as fallback
        return pd.DataFrame(columns=['Food_Name', 'Portion_Size', 'Weight_g', 'Notes'])


def extract_fruit_veg_survey_data(pdf_path):
    """Extract data from Fruit and Vegetable survey PDF"""
    print("Extracting data from Fruit and Vegetable Survey PDF...")
    
    try:
        # Extract text from PDF
        extracted_data = {}
        with open(pdf_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            num_pages = len(pdf_reader.pages)
            
            # Extract data from pages with sample information
            start_page = 11
            end_page = min(1291, num_pages)
            
            for page_num in range(start_page - 1, end_page):
                try:
                    # Get the page
                    page = pdf_reader.pages[page_num]
                    # Extract text
                    page_text = page.extract_text()
                    # Store in dict
                    extracted_data[page_num + 1] = page_text
                except Exception as e:
                    print(f"  Warning: Error extracting text from page {page_num + 1}: {str(e)}")
                    continue
        
        # Parse extracted text into structured data
        samples = []
        
        # Regular expressions for key information
        sample_number_pattern = r"Composite Sample Number:\s*(\d+)"
        sample_name_pattern = r"Composite Sample Name:\s*(.*?)\n"
        weight_pattern = r"Pack size:\s*(.*?)\n"
        
        # Process each page
        for page_num, text in extracted_data.items():
            # Extract information using regex
            sample_number_match = re.search(sample_number_pattern, text)
            sample_name_match = re.search(sample_name_pattern, text)
            weight_match = re.search(weight_pattern, text)
            
            if sample_name_match:
                sample = {
                    'Page': page_num,
                    'Sample_Number': sample_number_match.group(1) if sample_number_match else None,
                    'Sample_Name': sample_name_match.group(1).strip() if sample_name_match else None,
                    'Pack_Size': weight_match.group(1).strip() if weight_match else None
                }
                samples.append(sample)
        
        # Create DataFrame
        df = pd.DataFrame(samples)
        print(f"  Extracted {len(df)} fruit and vegetable samples")
        return df
    except Exception as e:
        print(f"Error extracting data from fruit and vegetable survey PDF: {str(e)}")
        # Return empty DataFrame as fallback
        return pd.DataFrame(columns=['Page', 'Sample_Number', 'Sample_Name', 'Pack_Size'])


def load_mccance_widdowson_data(file_path):
    """Load McCance and Widdowson's food composition data"""
    print("Loading McCance and Widdowson's data...")
    try:
        if not os.path.exists(file_path):
            print(f"  Error: McCance and Widdowson data file not found: {file_path}")
            return pd.DataFrame()
            
        # Try different sheet names since the data format might vary
        sheet_names = ["1.2 Factors", "1.2_Factors", "Factors", "Food Composition"]
        df = None
        
        # First try to get all sheet names from the file
        try:
            available_sheets = pd.ExcelFile(file_path).sheet_names
            print(f"  Available sheets: {', '.join(available_sheets)}")
            
            # Add available sheets to our list to try
            for sheet in available_sheets:
                if sheet not in sheet_names:
                    sheet_names.append(sheet)
        except Exception as e:
            print(f"  Warning: Could not read sheet names: {str(e)}")
        
        # Try each sheet name until we find one that works
        for sheet in sheet_names:
            try:
                df = pd.read_excel(file_path, sheet_name=sheet)
                if len(df) > 0:
                    print(f"  Successfully loaded data from sheet: {sheet}")
                    break
            except Exception as e:
                print(f"  Could not load sheet '{sheet}': {str(e)}")
        
        if df is None or len(df) == 0:
            print("  Error: Could not load any data from the Excel file.")
            return pd.DataFrame()
        
        print(f"  Loaded {len(df)} food items")
        print(f"  Available columns: {', '.join(df.columns.tolist())}")
        
        # Check if data is valid
        non_null_counts = df.count()
        if non_null_counts.sum() == 0:
            print("  Warning: Data appears to be empty or corrupted (all values are NaN)")
            return pd.DataFrame()
            
        # Identify key columns that should have data
        key_cols = ['Food Name', 'Food Code', 'FoodName', 'Food_Name', 'Description']
        found_key_col = False
        
        for col in key_cols:
            if col in df.columns and df[col].notna().sum() > 0:
                found_key_col = True
                # Standardize the column name to 'Food Name'
                if col != 'Food Name':
                    df.rename(columns={col: 'Food Name'}, inplace=True)
                break
        
        if not found_key_col:
            print(f"  Warning: No key column with food names found. Available columns: {', '.join(df.columns)}")
        
        return df
    except Exception as e:
        print(f"Error loading McCance and Widdowson data: {str(e)}")
        return pd.DataFrame()


def match_datasets(main_df, secondary_df, main_col, secondary_col, threshold=70, additional_cols=None):
    """Match items between datasets using fuzzy matching with machine learning"""
    print(f"Matching datasets based on {main_col} and {secondary_col}...")
    
    # Try to load a trained matcher first for better results
    try:
        matcher = train_matcher_from_existing_data()
        logger.info("Using trained matcher for dataset matching")
    except Exception as e:
        logger.warning(f"Could not load trained matcher: {e}. Using default matcher.")
        matcher = FoodMatcher(similarity_threshold=threshold/100)
    
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


def process_weight_info(df, weight_col):
    """Extract and normalize weight information"""
    print("Processing weight information...")
    
    # Use the weight extraction module
    weight_processed_df = clean_weights(df, weight_col=weight_col)
    
    # Count successful extractions
    success_count = weight_processed_df['Weight_Value'].notna().sum()
    total_count = len(weight_processed_df)
    success_rate = (success_count / total_count) * 100 if total_count > 0 else 0
    
    print(f"  Successfully extracted {success_count} weights out of {total_count} entries ({success_rate:.2f}%)")
    return weight_processed_df


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
    parser.add_argument("--output", default="output/weight_dataset.csv", help="Output file path")
    parser.add_argument("--food-portion-pdf", default="D:/LIDA/Product_Weight_Project/Product_Weight_Project_Build/Data/Raw Data/Food_Portion_Sizes.pdf", help="Food Portion Sizes PDF path")
    parser.add_argument("--fruit-veg-pdf", default="D:/LIDA/Product_Weight_Project/Product_Weight_Project_Build/Data/Raw Data/fruit_and_vegetable_survey_2015_sampling_report.pdf", help="Fruit and Veg Survey PDF path")
    parser.add_argument("--mccance-widdowson", default="D:/LIDA/Product_Weight_Project/Product_Weight_Project_Build/Data/Raw Data/McCance_Widdowsons_2021.xlsx", help="McCance and Widdowson data path")
    parser.add_argument("--matching-threshold", type=int, default=70, help="Matching threshold percentage")
    parser.add_argument("--run-dashboard", action="store_true", help="Run interactive dashboard")
    parser.add_argument("--load-cache", action="store_true", help="Load cached data (faster)")
    parser.add_argument("--skip-pdfs", action="store_true", help="Skip PDF extraction (faster)")
    parser.add_argument("--use-super-group", action="store_true", default=False, help="Use Super Group reduced data")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Setup data sources
    data_sources = {}
    
    # 1. Load McCance and Widdowson's dataset (highest quality)
    if args.load_cache and os.path.exists("output/mw_data_cached.csv"):
        print("Loading cached McCance and Widdowson data...")
        data_sources["mw_data"] = pd.read_csv("output/mw_data_cached.csv")
    else:
        data_sources["mw_data"] = load_mccance_widdowson_data(args.mccance_widdowson)
        # Cache the data for future runs
        data_sources["mw_data"].to_csv("output/mw_data_cached.csv", index=False)
    
    # Print dataset info for debugging
    print(f"\nMcCance and Widdowson dataset shape: {data_sources['mw_data'].shape}")
    print(f"Columns: {', '.join(data_sources['mw_data'].columns.tolist())}")
    print(f"Sample row: \n{data_sources['mw_data'].iloc[0].to_dict()}\n")
    
    # 2. Load McCance and Widdowson's Super Group reduced dataset
    # This contains better structured and cleaned food group data
    super_group_data = {}
    if args.use_super_group:
        print("Loading McCance and Widdowson Super Group reduced data...")
        super_group_dir = "Data/Processed/MW_DataReduction/Reduced Super Group"
        super_group_dir = get_path(super_group_dir)
        if not os.path.isdir(super_group_dir):
            logger.warning(f"Super-group directory '{super_group_dir}' not found – skipping.")
            super_group_files = []
        else:
            super_group_files = [
                f for f in os.listdir(super_group_dir)
                if f.endswith(".csv") and not f.startswith(".")
            ]
        
        # Create output directory if it doesn't exist
        os.makedirs("output", exist_ok=True)
        
        for file in super_group_files:
            try:
                # Extract food group from filename
                food_group = os.path.splitext(file)[0]
                file_path = os.path.join(super_group_dir, file)
                
                if not os.path.exists(file_path):
                    logger.warning(f"Super-group file '{file_path}' not found – skipping.")
                    continue
                
                df = pd.read_csv(file_path)
                print(f"  Loaded {len(df)} items from {food_group}")
                
                # Store in our dict with food group as key
                super_group_data[food_group] = df
                
            except Exception as e:
                print(f"  Error loading {file}: {str(e)}")
        
        # Combine all super group data
        if super_group_data:
            all_super_group = pd.concat(super_group_data.values(), ignore_index=True)
            print(f"  Combined super group data: {len(all_super_group)} items")
            
            # Add this as a data source
            data_sources["super_group"] = all_super_group
        
        # Also load the cleaned version if available
        cleaned_dir = os.path.join(super_group_dir, "Cleaned")
        if os.path.exists(cleaned_dir):
            cleaned_files = [f for f in os.listdir(cleaned_dir) if f.endswith('.csv') and not f.startswith('.')]
            cleaned_dfs = []
            
            for file in cleaned_files:
                try:
                    file_path = os.path.join(cleaned_dir, file)
                    if not os.path.exists(file_path):
                        logger.warning(f"Cleaned super-group file '{file_path}' not found – skipping.")
                        continue
                    
                    df = pd.read_csv(file_path)
                    cleaned_dfs.append(df)
                except Exception as e:
                    print(f"  Error loading cleaned file {file}: {str(e)}")
            
            if cleaned_dfs:
                all_cleaned = pd.concat(cleaned_dfs, ignore_index=True)
                print(f"  Loaded cleaned super group data: {len(all_cleaned)} items")
                data_sources["super_group_cleaned"] = all_cleaned
    
    # 3. Load Food Portion Sizes data (portion-specific)
    if not args.skip_pdfs:
        try:
            food_portion_path = get_path(args.food_portion_pdf)
            pdf_extractor = PDFExtractor(cache_dir="output")
            data_sources["portion_data"] = pdf_extractor.extract_food_portion_sizes(food_portion_path)
        except FileNotFoundError as e:
            logger.error(f"Could not find food portion PDF: {e}")
            data_sources["portion_data"] = pd.DataFrame()
    elif args.load_cache and os.path.exists("output/food_portion_sizes.csv"):
        print("Loading cached Food Portion data...")
        data_sources["portion_data"] = pd.read_csv("output/food_portion_sizes.csv")
    
    # 4. Load Fruit and Veg survey data (retail-specific)
    if not args.skip_pdfs:
        try:
            fruit_veg_path = get_path(args.fruit_veg_pdf)
            pdf_extractor = PDFExtractor(cache_dir="output") if 'pdf_extractor' not in locals() else pdf_extractor
            data_sources["fruit_veg_data"] = pdf_extractor.extract_fruit_veg_survey(fruit_veg_path)
        except FileNotFoundError as e:
            logger.error(f"Could not find fruit and veg PDF: {e}")
            data_sources["fruit_veg_data"] = pd.DataFrame()
    elif args.load_cache and os.path.exists("output/fruit_veg_survey.csv"):
        print("Loading cached Fruit and Veg data...")
        data_sources["fruit_veg_data"] = pd.read_csv("output/fruit_veg_survey.csv")
    
    # Cache PDF-extracted data for future runs if not using cache
    if not args.skip_pdfs and not args.load_cache:
        if "portion_data" in data_sources:
            data_sources["portion_data"].to_csv("output/food_portion_sizes.csv", index=False)
        if "fruit_veg_data" in data_sources:
            data_sources["fruit_veg_data"].to_csv("output/fruit_veg_survey.csv", index=False)
    
    # Create a main dataset from the best available data
    print("\nCreating integrated dataset from all sources...")
    
    # Start with McCance and Widdowson as the base (highest quality)
    main_dataset = data_sources["mw_data"].copy()
    print(f"Base dataset from McCance and Widdowson: {len(main_dataset)} items")
    
    # Merge with super group data if available
    if "super_group_cleaned" in data_sources:
        super_group_data = data_sources["super_group_cleaned"]
        print(f"Adding Super Group cleaned data: {len(super_group_data)} items")
        
        # Set up columns for matching
        mw_col = 'Food Name'
        sg_col = 'Food Name'
        
        # Match datasets
        super_group_matcher = FoodMatcher(similarity_threshold=0.7)  # Higher threshold for high-quality data
        super_group_matches = super_group_matcher.match_datasets(
            main_dataset,
            super_group_data,
            mw_col,
            sg_col,
            additional_match_cols=[('Food Code', 'Food Code')] if 'Food Code' in main_dataset.columns and 'Food Code' in super_group_data.columns else None
        )
        
        # Merge the datasets
        integrated_dataset = super_group_matcher.merge_matched_datasets(
            main_dataset,
            super_group_data,
            super_group_matches,
            merged_cols={'Super_Group': 'Super Group', 'Sale_Format': 'Sale format(s)'}
        )
        
        print(f"Integrated dataset with Super Group data: {len(integrated_dataset)} items")
        main_dataset = integrated_dataset
    
    # Match with portion data if available
    if "portion_data" in data_sources and len(data_sources["portion_data"]) > 0:
        portion_data = data_sources["portion_data"]
        print(f"Matching with Food Portion data: {len(portion_data)} items")
        
        # Match food names
        mw_col = 'Food Name'
        portion_col = 'Food_Name'
        
        # Match datasets using the food matcher
        portion_matcher = FoodMatcher(similarity_threshold=args.matching_threshold/100)
        portion_matches = portion_matcher.match_datasets(
            main_dataset,
            portion_data,
            mw_col,
            portion_col
        )
        
        # Merge the datasets
        portion_merged = portion_matcher.merge_matched_datasets(
            main_dataset,
            portion_data,
            portion_matches
        )
        
        # Process weight information
        weight_col = next((c for c in portion_merged.columns if c.endswith('Weight_g')), None)
        if weight_col:
            portion_merged = process_weight_info(portion_merged, weight_col)
        
        print(f"Merged with portion data: {len(portion_merged)} items")
        main_dataset = portion_merged
    
    # Match with fruit and veg data if available
    if "fruit_veg_data" in data_sources and len(data_sources["fruit_veg_data"]) > 0:
        fruit_veg_data = data_sources["fruit_veg_data"]
        print(f"Matching with Fruit and Veg data: {len(fruit_veg_data)} items")
        
        # Match food names
        mw_col = 'Food Name'
        fv_col = 'Sample_Name'
        
        # Match datasets using the food matcher
        fv_matcher = FoodMatcher(similarity_threshold=args.matching_threshold/100)
        fv_matches = fv_matcher.match_datasets(
            main_dataset,
            fruit_veg_data,
            mw_col,
            fv_col
        )
        
        # Merge the datasets
        fv_merged = fv_matcher.merge_matched_datasets(
            main_dataset,
            fruit_veg_data,
            fv_matches
        )
        
        # Process weight information
        pack_size_col = next((c for c in fv_merged.columns if c.endswith('Pack_Size')), None)
        if pack_size_col:
            fv_merged = process_weight_info(fv_merged, pack_size_col)
        
        print(f"Merged with fruit and veg data: {len(fv_merged)} items")
        main_dataset = fv_merged
    
    # Create a data cleaner for the final cleanup
    print("\nCleaning and formatting the integrated dataset...")
    cleaner = DataCleaner()
    
    # Apply consistent naming and cleaning
    cleaned_dataset = cleaner.clean(main_dataset)
    
    # Check if we have any data to process
    if len(cleaned_dataset) == 0:
        print("Warning: No data available after cleaning. Using original dataset.")
        cleaned_dataset = main_dataset
    
    # Identify appropriate column for categorization
    food_name_col = None
    possible_cols = ['Food Name', 'Food_Name', 'Food Name_source', 'Food_Name_target']
    for col in possible_cols:
        if col in cleaned_dataset.columns:
            food_name_col = col
            break
    
    if food_name_col is None:
        print("Warning: Could not find a food name column for categorization.")
        print(f"Available columns: {', '.join(cleaned_dataset.columns.tolist())}")
        food_name_col = cleaned_dataset.columns[0] if len(cleaned_dataset.columns) > 0 else 'Food Name'
    
    # Categorize foods more precisely
    categorizer = FoodCategorizer()
    # Use clean_food_categories function with the identified column
    print(f"Categorizing foods using column: {food_name_col}")
    categorized_dataset = clean_food_categories(cleaned_dataset, food_name_col, 'Food_Category', 'Super_Category', categorizer)
    
    # Normalize weights across different units
    print("Normalizing weights...")
    if len(categorized_dataset) == 0:
        print("Warning: No data available for weight normalization.")
        normalized_dataset = categorized_dataset
    else:
        normalized_dataset = normalize_weights(categorized_dataset)
        
    # Generate summary stats
    print("\nGenerating food group summaries...")
    if len(normalized_dataset) == 0:
        print("Warning: No data available for summary generation.")
        summary = pd.DataFrame()
    else:
        summary = create_food_group_summary(normalized_dataset)
    
    # Save the final dataset
    print(f"\nSaving integrated dataset to {args.output}...")
    normalized_dataset.to_csv(args.output, index=False)
    
    # Also save the summary
    summary_path = args.output.replace('.csv', '_summary.csv')
    summary.to_csv(summary_path, index=False)
    print(f"Food group summary saved to {summary_path}")
    
    # Launch dashboard if requested
    if args.run_dashboard:
        print("\nLaunching interactive dashboard...")
        try:
            # Only import the dashboard when needed
            from shelfscale.visualization.dashboard import ShelfScaleDashboard
            dashboard = ShelfScaleDashboard(normalized_dataset)
            dashboard.run_server(debug=False, port=8050)
        except ImportError as e:
            logger.error(f"Could not load dashboard. Dashboard dependencies may not be installed: {e}")
            print("Error: Dashboard dependencies not installed. Install with 'pip install dash plotly'.")
    
    print("\nProcess completed successfully!")
    return normalized_dataset


if __name__ == "__main__":
    main() 