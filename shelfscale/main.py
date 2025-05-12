"""
Main script demonstrating ShelfScale package usage with comprehensive data sources
"""

import os
import pandas as pd
import numpy as np
import tabula
import PyPDF2
import re
from fuzzywuzzy import process, fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from shelfscale.data_sourcing.open_food_facts import OpenFoodFactsClient
from shelfscale.data_processing.weight_extraction import clean_weights
from shelfscale.data_processing.categorization import clean_food_categories, FoodCategorizer
from shelfscale.data_processing.cleaner import DataCleaner
from shelfscale.data_processing.transformation import normalize_weights, create_food_group_summary
from shelfscale.matching.algorithm import FoodMatcher
from shelfscale.visualization.dashboard import ShelfScaleDashboard
from shelfscale.utils.helpers import load_data, save_data, split_data_by_group


def extract_food_portion_data(pdf_path):
    """Extract data from Food Portion Sizes PDF"""
    print("Extracting data from Food Portion Sizes PDF...")
    
    # Extract tables from PDF
    tables = tabula.read_pdf(pdf_path, 
                            pages="12-114", 
                            stream=True, 
                            multiple_tables=True, 
                            guess=False, 
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


def extract_fruit_veg_survey_data(pdf_path):
    """Extract data from Fruit and Vegetable survey PDF"""
    print("Extracting data from Fruit and Vegetable Survey PDF...")
    
    # Extract text from PDF
    extracted_data = {}
    with open(pdf_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        num_pages = len(pdf_reader.pages)
        
        # Extract data from pages with sample information
        start_page = 11
        end_page = min(1291, num_pages)
        
        for page_num in range(start_page - 1, end_page):
            # Get the page
            page = pdf_reader.pages[page_num]
            # Extract text
            page_text = page.extract_text()
            # Store in dict
            extracted_data[page_num + 1] = page_text
    
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


def load_mccance_widdowson_data(file_path):
    """Load McCance and Widdowson's food composition data"""
    print("Loading McCance and Widdowson's data...")
    df = pd.read_csv(file_path)
    print(f"  Loaded {len(df)} food items")
    return df


def match_datasets(main_df, secondary_df, main_col, secondary_col, threshold=70):
    """Match items between datasets using fuzzy matching"""
    print(f"Matching datasets based on {main_col} and {secondary_col}...")
    
    matcher = FoodMatcher(similarity_threshold=threshold/100)
    
    # Perform matching
    matches = matcher.match_datasets(
        main_df, 
        secondary_df, 
        main_col, 
        secondary_col
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
    
    # 1.1. Load McCance and Widdowson data
    mw_df = load_mccance_widdowson_data(mw_path)
    
    # 1.2. Extract data from Food Portion Sizes PDF
    fps_df = extract_food_portion_data(portion_sizes_path)
    
    # 1.3. Extract data from Fruit and Vegetable Survey PDF
    fvs_df = extract_fruit_veg_survey_data(fruit_veg_survey_path)
    
    # 1.4. Get data from Open Food Facts API for additional items
    print("\nFetching data from Open Food Facts API for supplementary information...")
    client = OpenFoodFactsClient()
    
    # Get data for key food groups
    food_groups = ['vegetables', 'fruits', 'cereals', 'dairy']
    api_dfs = []
    
    for group in food_groups:
        print(f"  Fetching data for {group}...")
        df = client.search_by_food_group(group)
        df['Food Group'] = group.title()
        api_dfs.append(df)
    
    # Combine API results
    api_df = pd.concat(api_dfs, ignore_index=True)
    print(f"  Total products fetched from API: {len(api_df)}")
    
    # Step 2: Data Processing and Cleaning
    print("\n2. Processing and cleaning data...")
    
    # 2.1 Clean McCance and Widdowson data
    print("  Processing McCance and Widdowson data...")
    mw_df = clean_food_categories(mw_df, text_col='Food Name', new_col='Food_Category')
    
    # 2.2 Clean Food Portion Sizes data
    print("  Processing Food Portion Sizes data...")
    fps_df = process_weight_info(fps_df, weight_col='Weight_g')
    
    # 2.3 Clean Fruit and Vegetable Survey data
    print("  Processing Fruit and Vegetable Survey data...")
    fvs_df = process_weight_info(fvs_df, weight_col='Pack_Size')
    
    # 2.4 Clean API data
    print("  Processing API data...")
    api_df = clean_food_categories(api_df, text_col='Product Name', new_col='Food_Category')
    api_df = clean_weights(api_df)
    
    # Step 3: Matching and Merging Datasets
    print("\n3. Matching and merging datasets...")
    
    # 3.1 Match McCance and Widdowson data with Food Portion Sizes
    mw_fps_df = match_datasets(
        mw_df, 
        fps_df, 
        'Food Name', 
        'Food_Name', 
        threshold=70
    )
    
    # 3.2 Match McCance and Widdowson data with Fruit and Vegetable Survey
    mw_fvs_df = match_datasets(
        mw_df, 
        fvs_df, 
        'Food Name', 
        'Sample_Name', 
        threshold=70
    )
    
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
    
    # Add weight information from Fruit and Vegetable Survey matches
    if 'Weight_Value_target' in mw_fvs_df.columns:
        # Map weights back to main dataset
        weight_map = dict(zip(
            mw_fvs_df['Food Name_source'], 
            mw_fvs_df['Weight_Value_target']
        ))
        
        # Create new column for FVS weights
        consolidated_df['FVS_Weight'] = consolidated_df['Food Name'].map(weight_map)
    
    # Use cleaner to handle missing values
    cleaner = DataCleaner(config={
        'categories': {'enabled': False},
        'weight': {'enabled': False},
        'text': {'enabled': False},
        'duplicates': {'enabled': True},
        'missing_values': {'enabled': False}  # We want to keep NaN values for analysis
    })
    consolidated_df = cleaner.clean(consolidated_df)
    
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
    coverage_pct = (items_with_weight / total_items) * 100
    
    print(f"  Total food items: {total_items}")
    print(f"  Items with standardized weight: {items_with_weight} ({coverage_pct:.1f}%)")
    
    # Step 5: Create summary statistics by food group
    print("\n5. Creating food group summary statistics...")
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
    
    # Step 6: Save processed data
    print("\n6. Saving processed data...")
    save_data(consolidated_df, "output/consolidated_weights.csv")
    save_data(summary_df, "output/food_group_summary.csv")
    
    # Also save individual datasets
    save_data(mw_fps_df, "output/mw_fps_matches.csv")
    save_data(mw_fvs_df, "output/mw_fvs_matches.csv")
    
    # Step 7: Optional - Launch dashboard
    print("\n7. Would you like to launch the interactive dashboard? (y/n)")
    response = input().strip().lower()
    
    if response == 'y':
        print("  Launching dashboard...")
        dashboard = ShelfScaleDashboard(consolidated_df)
        dashboard.run_server(debug=True)
    
    print("\nShelfScale workflow completed!")


if __name__ == "__main__":
    main() 