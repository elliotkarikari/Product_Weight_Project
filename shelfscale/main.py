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
from shelfscale.visualization.dashboard import ShelfScaleDashboard
from shelfscale.utils.helpers import load_data, save_data, split_data_by_group
from shelfscale.utils.learning import (
    train_matcher_from_existing_data, 
    evaluate_matcher_performance,
    generate_training_data_from_matches,
    create_weight_predictions
)

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
    df = pd.read_csv(file_path)
    print(f"  Loaded {len(df)} food items")
    return df


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
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='ShelfScale - Food Product Weight Analysis')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate matcher performance')
    parser.add_argument('--train', action='store_true', help='Train matcher from existing data')
    parser.add_argument('--generate-training', action='store_true', help='Generate training data')
    parser.add_argument('--predict', action='store_true', help='Run weight predictions on input file')
    parser.add_argument('--input-file', type=str, help='Input file for predictions')
    args = parser.parse_args()
    
    # Handle evaluation mode
    if args.evaluate:
        print("\nEvaluating matcher performance...")
        metrics = evaluate_matcher_performance()
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        return
    
    # Handle training mode
    if args.train:
        print("\nTraining matcher from existing data...")
        matcher = train_matcher_from_existing_data()
        print("Matcher training complete.")
        
        if args.generate_training:
            print("Generating training data...")
            generate_training_data_from_matches(matcher)
        return
    
    # Handle prediction mode
    if args.predict and args.input_file:
        if not os.path.exists(args.input_file):
            print(f"Error: Input file {args.input_file} not found.")
            return
        
        print(f"\nGenerating weight predictions for {args.input_file}...")
        try:
            # Determine file type and load
            if args.input_file.endswith('.csv'):
                df = pd.read_csv(args.input_file)
            elif args.input_file.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(args.input_file)
            else:
                print("Error: Input file must be CSV or Excel.")
                return
            
            # Get product name column
            name_cols = [col for col in df.columns if any(term in col.lower() for term in ['name', 'product', 'description', 'item'])]
            if not name_cols:
                print("Error: Could not find a product name column.")
                return
            
            product_name_col = name_cols[0]
            print(f"Using '{product_name_col}' as product name column")
            
            # Find potential weight columns
            weight_cols = [col for col in df.columns if any(term in col.lower() for term in ['weight', 'size', 'volume', 'amount'])]
            existing_weight_col = weight_cols[0] if weight_cols else None
            
            # Generate predictions
            result_df = create_weight_predictions(df, product_name_col, existing_weight_col)
            
            # Save results
            output_file = os.path.splitext(args.input_file)[0] + '_predicted.csv'
            result_df.to_csv(output_file, index=False)
            print(f"Predictions saved to {output_file}")
            return
        except Exception as e:
            print(f"Error generating predictions: {str(e)}")
            return
    
    # Create output directory
    os.makedirs("output", exist_ok=True)
    
    # Step 1: Data Loading from Multiple Sources
    print("\n1. Loading data from multiple sources...")
    
    # Set paths to data files (with both absolute and relative path options)
    base_path = os.environ.get('PRODUCT_WEIGHT_PROJECT_PATH', 'D:/LIDA/Product_Weight_Project')
    data_path = os.path.join(base_path, 'Product_Weight_Project_Build/Data')
    
    # Define paths with fallbacks between absolute and relative paths
    def get_path(rel_path):
        abs_path = os.path.join(data_path, rel_path)
        return abs_path if os.path.exists(abs_path) else rel_path
    
    mw_path = get_path('Raw Data/Labelling2021_watercress.csv')
    portion_sizes_path = get_path('Food_Portion_Sizes.pdf')
    fruit_veg_survey_path = get_path('fruit_and_vegetable_survey_2015_sampling_report.pdf')
    
    # Set up error recovery for data loading
    mw_df = None
    fps_df = None
    fvs_df = None
    api_df = None
    
    # 1.1. Load McCance and Widdowson data
    try:
        mw_df = load_mccance_widdowson_data(mw_path)
    except Exception as e:
        print(f"ERROR: Failed to load McCance and Widdowson data: {str(e)}")
        print("Creating minimal sample data...")
        # Create minimal sample data
        mw_df = pd.DataFrame({
            'Food Name': [f'Sample Food {i}' for i in range(1, 101)],
            'Food Code': [f'F{i:03d}' for i in range(1, 101)],
            'Food Group': ['Vegetables', 'Fruits', 'Cereals', 'Dairy'] * 25,
        })
    
    # 1.2. Extract data from Food Portion Sizes PDF
    try:
        fps_df = extract_food_portion_data(portion_sizes_path)
        # Check if we got empty results
        if fps_df.empty:
            raise ValueError("No data extracted from Food Portion Sizes PDF")
    except Exception as e:
        print(f"ERROR: Failed to extract Food Portion Sizes data: {str(e)}")
        print("Creating minimal sample data...")
        # Create minimal sample data
        fps_df = pd.DataFrame({
            'Food_Name': mw_df['Food Name'].sample(n=min(50, len(mw_df))).tolist(),
            'Portion_Size': ['Medium', 'Large', 'Small'] * 20,
            'Weight_g': [str(random.randint(50, 500)) for _ in range(50)],
            'Notes': [''] * 50
        })
    
    # 1.3. Extract data from Fruit and Vegetable Survey PDF
    try:
        fvs_df = extract_fruit_veg_survey_data(fruit_veg_survey_path)
        # Check if we got empty results
        if fvs_df.empty:
            raise ValueError("No data extracted from Fruit and Vegetable Survey PDF")
    except Exception as e:
        print(f"ERROR: Failed to extract Fruit and Vegetable Survey data: {str(e)}")
        print("Creating minimal sample data...")
        # Create minimal sample data
        fvs_df = pd.DataFrame({
            'Page': list(range(1, 31)),
            'Sample_Number': [f'S{i:03d}' for i in range(1, 31)],
            'Sample_Name': [f'Sample Fruit {i}' for i in range(1, 31)],
            'Pack_Size': [f'{random.randint(100, 1000)}g' for _ in range(30)]
        })
    
    # 1.4. Get data from Open Food Facts API for additional items
    print("\nFetching data from Open Food Facts API for supplementary information...")
    client = OpenFoodFactsClient(retry_delay=2.0, max_retries=3)
    
    # Get data for key food groups
    food_groups = ['vegetables', 'fruits', 'cereals', 'dairy']
    api_dfs = []
    
    for group in food_groups:
        try:
            print(f"  Fetching data for {group}...")
            df = client.search_by_food_group(group)
            df['Food Group'] = group.title()
            api_dfs.append(df)
        except Exception as e:
            print(f"  ERROR: Failed to fetch {group} data: {str(e)}")
            # Create minimal fallback data
            fallback_df = pd.DataFrame({
                'Product Name': [f'{group.title()} Sample {i}' for i in range(1, 11)],
                'Weight': [f'{random.randint(100, 1000)}g' for _ in range(10)],
                'Packaging Details': ['Plastic', 'Box', 'Jar', 'Bag', 'Container'] * 2,
                'Country': ['Sample Country'] * 10,
                'Food Group': [group.title()] * 10
            })
            api_dfs.append(fallback_df)
    
    # Combine API results
    api_df = pd.concat(api_dfs, ignore_index=True)
    print(f"  Total products fetched from API: {len(api_df)}")
    
    # Step 2: Data Processing and Cleaning
    print("\n2. Processing and cleaning data...")
    
    # 2.1 Clean McCance and Widdowson data
    try:
        print("  Processing McCance and Widdowson data...")
        mw_df = clean_food_categories(mw_df, text_col='Food Name', new_col='Food_Category')
    except Exception as e:
        print(f"  ERROR: Error cleaning McCance and Widdowson data: {str(e)}")
        # Add a basic Food_Category column if needed
        if 'Food_Category' not in mw_df.columns:
            mw_df['Food_Category'] = mw_df.get('Food Group', ['Uncategorized'] * len(mw_df))
    
    # 2.2 Clean Food Portion Sizes data
    try:
        print("  Processing Food Portion Sizes data...")
        fps_df = process_weight_info(fps_df, weight_col='Weight_g')
    except Exception as e:
        print(f"  ERROR: Error cleaning Food Portion Sizes data: {str(e)}")
        # Add basic Weight_Value if needed
        if 'Weight_Value' not in fps_df.columns:
            fps_df['Weight_Value'] = fps_df['Weight_g'].apply(
                lambda x: float(x.replace('g', '')) if isinstance(x, str) and 'g' in x else None
            )
    
    # 2.3 Clean Fruit and Vegetable Survey data
    try:
        print("  Processing Fruit and Vegetable Survey data...")
        fvs_df = process_weight_info(fvs_df, weight_col='Pack_Size')
    except Exception as e:
        print(f"  ERROR: Error cleaning Fruit and Vegetable Survey data: {str(e)}")
        # Add basic Weight_Value if needed
        if 'Weight_Value' not in fvs_df.columns:
            fvs_df['Weight_Value'] = fvs_df['Pack_Size'].apply(
                lambda x: float(x.replace('g', '')) if isinstance(x, str) and 'g' in x else None
            )
    
    # 2.4 Clean API data
    try:
        print("  Processing API data...")
        api_df = clean_food_categories(api_df, text_col='Product Name', new_col='Food_Category')
        api_df = clean_weights(api_df)
    except Exception as e:
        print(f"  ERROR: Error cleaning API data: {str(e)}")
        # Add basic Weight_Value if needed
        if 'Weight_Value' not in api_df.columns:
            api_df['Weight_Value'] = api_df['Weight'].apply(
                lambda x: float(x.replace('g', '')) if isinstance(x, str) and 'g' in x else None
            )
        # Add basic Food_Category if needed
        if 'Food_Category' not in api_df.columns:
            api_df['Food_Category'] = api_df['Food Group']
    
    # Step 3: Matching and Merging Datasets
    print("\n3. Matching and merging datasets...")
    
    # 3.1 Match McCance and Widdowson data with Food Portion Sizes
    try:
        mw_fps_df = match_datasets(
            mw_df, 
            fps_df, 
            'Food Name', 
            'Food_Name', 
            threshold=70
        )
    except Exception as e:
        print(f"  ERROR: Failed to match MW with FPS data: {str(e)}")
        # Create empty DataFrame with correct structure
        mw_fps_df = pd.DataFrame(columns=[
            'Source_Index', 'Target_Index', 'Source_Item', 'Target_Item', 'Similarity_Score'
        ])
    
    # 3.2 Match McCance and Widdowson data with Fruit and Vegetable Survey
    try:
        mw_fvs_df = match_datasets(
            mw_df, 
            fvs_df, 
            'Food Name', 
            'Sample_Name', 
            threshold=70
        )
    except Exception as e:
        print(f"  ERROR: Failed to match MW with FVS data: {str(e)}")
        # Create empty DataFrame with correct structure
        mw_fvs_df = pd.DataFrame(columns=[
            'Source_Index', 'Target_Index', 'Source_Item', 'Target_Item', 'Similarity_Score'
        ])
    
    # 3.3 Create consolidated weights dataset
    print("  Creating consolidated weights dataset...")
    
    # Start with McCance and Widdowson data
    consolidated_df = mw_df.copy()
    
    # Add weight information from Food Portion Sizes matches
    if 'Weight_Value_target' in mw_fps_df.columns:
        # Map weights back to main dataset
        weight_map = dict(zip(
            mw_fps_df['Food Name_source'], 
            mw_fps_df['Weight_Value_target']
        ))
        
        # Create new column for FPS weights
        consolidated_df['FPS_Weight'] = consolidated_df['Food Name'].map(weight_map)
    else:
        consolidated_df['FPS_Weight'] = None
    
    # Add weight information from Fruit and Vegetable Survey matches
    if 'Weight_Value_target' in mw_fvs_df.columns:
        # Map weights back to main dataset
        weight_map = dict(zip(
            mw_fvs_df['Food Name_source'], 
            mw_fvs_df['Weight_Value_target']
        ))
        
        # Create new column for FVS weights
        consolidated_df['FVS_Weight'] = consolidated_df['Food Name'].map(weight_map)
    else:
        consolidated_df['FVS_Weight'] = None
    
    # Use cleaner to handle missing values
    try:
        cleaner = DataCleaner(config={
            'categories': {'enabled': False},
            'weight': {'enabled': False},
            'text': {'enabled': False},
            'duplicates': {'enabled': True},
            'missing_values': {'enabled': False}  # We want to keep NaN values for analysis
        })
        consolidated_df = cleaner.clean(consolidated_df)
    except Exception as e:
        print(f"  WARNING: Error while cleaning consolidated data: {str(e)}")
        # Remove duplicate rows as a basic cleanup step
        consolidated_df = consolidated_df.drop_duplicates()
    
    # Step 4: Create final weight dataset with priority logic
    print("\n4. Creating final weight dataset with priority logic...")
    
    # Calculate standardized weight based on priority:
    # 1. FPS weight (most reliable)
    # 2. FVS weight (survey data)
    # 3. API weight (least reliable)
    consolidated_df['Standardized_Weight'] = consolidated_df['FPS_Weight']
    
    # Where FPS weight is missing, use FVS weight
    mask = consolidated_df['Standardized_Weight'].isna()
    consolidated_df.loc[mask, 'Standardized_Weight'] = consolidated_df.loc[mask, 'FVS_Weight']
    
    # Calculate weight coverage
    total_items = len(consolidated_df)
    items_with_weight = consolidated_df['Standardized_Weight'].notna().sum()
    coverage_pct = (items_with_weight / total_items) * 100 if total_items > 0 else 0
    
    print(f"  Total food items: {total_items}")
    print(f"  Items with standardized weight: {items_with_weight} ({coverage_pct:.1f}%)")
    
    # Step 5: Create summary statistics by food group
    print("\n5. Creating food group summary statistics...")
    try:
        summary_df = consolidated_df.groupby('Food_Category').agg({
            'Standardized_Weight': ['count', 'mean', 'median', 'min', 'max', 'std'],
            'Food Name': 'count'
        }).reset_index()
        
        # Flatten the MultiIndex columns
        summary_df.columns = [
            'Food_Category' if col[0] == 'Food_Category' else 
            f"{col[0]}_{col[1]}" for col in summary_df.columns
        ]
        
        # Calculate coverage by food group
        summary_df['Coverage_Pct'] = (summary_df['Standardized_Weight_count'] / summary_df['Food Name_count']) * 100
        
        # Display summary
        print("\nWeight coverage by food category:")
        sorted_summary = summary_df.sort_values('Coverage_Pct', ascending=False)
        for _, row in sorted_summary.iterrows():
            if row['Food Name_count'] > 0:
                print(f"  {row['Food_Category']}: {row['Coverage_Pct']:.1f}% coverage ({row['Standardized_Weight_count']}/{row['Food Name_count']} items)")
    except Exception as e:
        print(f"  ERROR: Failed to create summary statistics: {str(e)}")
        summary_df = pd.DataFrame()
    
    # Step 6: Save processed data
    print("\n6. Saving processed data...")
    try:
        save_data(consolidated_df, "output/consolidated_weights.csv")
        if not summary_df.empty:
            save_data(summary_df, "output/food_group_summary.csv")
        
        # Also save individual datasets
        save_data(mw_fps_df, "output/mw_fps_matches.csv")
        save_data(mw_fvs_df, "output/mw_fvs_matches.csv")
        print("  All data successfully saved to output directory")
    except Exception as e:
        print(f"  ERROR: Failed to save data: {str(e)}")
    
    # Step 7: Optional - Launch dashboard
    print("\n7. Would you like to launch the interactive dashboard? (y/n)")
    response = input().strip().lower()
    
    if response == 'y':
        try:
            print("  Launching dashboard...")
            dashboard = ShelfScaleDashboard(consolidated_df)
            dashboard.run_server(debug=True)
        except Exception as e:
            print(f"  ERROR: Failed to launch dashboard: {str(e)}")
    
    # Train the matcher with the latest matches for future use
    try:
        print("\n9. Training matcher from the new matches data...")
        matcher = train_matcher_from_existing_data()
        print("  Matcher trained successfully")
    except Exception as e:
        print(f"  Warning: Could not train matcher - {str(e)}")
    
    print("\nShelfScale workflow completed!")


if __name__ == "__main__":
    main() 