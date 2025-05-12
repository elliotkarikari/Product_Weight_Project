"""
Main script demonstrating ShelfScale package usage
"""

import os
import pandas as pd
from shelfscale.data_sourcing.open_food_facts import OpenFoodFactsClient
from shelfscale.data_processing.cleaning import clean_weight_column, clean_food_groups, handle_missing_values
from shelfscale.data_processing.transformation import normalize_weights, create_food_group_summary
from shelfscale.matching.algorithm import FoodMatcher
from shelfscale.visualization.dashboard import ShelfScaleDashboard
from shelfscale.utils.helpers import load_data, save_data, split_data_by_group


def main():
    """Main function demonstrating ShelfScale workflow"""
    print("ShelfScale - Food Product Weight Analysis")
    print("=" * 50)
    
    # Step 1: Data Sourcing
    print("\n1. Sourcing data from Open Food Facts...")
    client = OpenFoodFactsClient()
    
    # Get data for multiple food groups
    food_groups = ['vegetables', 'fruits', 'cereals']
    dfs = []
    
    for group in food_groups:
        print(f"  Fetching data for {group}...")
        df = client.search_by_food_group(group)
        df['Food Group'] = group.title()
        dfs.append(df)
    
    # Combine into a single DataFrame
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"  Total products fetched: {len(combined_df)}")
    
    # Step 2: Data Processing - Cleaning
    print("\n2. Cleaning and processing data...")
    
    # Clean data
    print("  Cleaning food group names...")
    cleaned_df = clean_food_groups(combined_df)
    
    print("  Cleaning weight column...")
    cleaned_df = clean_weight_column(cleaned_df)
    
    print("  Handling missing values...")
    cleaned_df = handle_missing_values(cleaned_df, strategy='drop')
    
    # Step 3: Data Processing - Transformation
    print("\n3. Transforming data...")
    
    print("  Normalizing weights...")
    normalized_df = normalize_weights(cleaned_df)
    
    print("  Creating food group summary...")
    summary_df = create_food_group_summary(normalized_df)
    print(summary_df)
    
    # Step 4: Save processed data
    print("\n4. Saving processed data...")
    os.makedirs("output", exist_ok=True)
    save_data(normalized_df, "output/processed_data.csv")
    save_data(summary_df, "output/food_group_summary.csv")
    
    # Step 5: Optional - Launch dashboard
    print("\n5. Would you like to launch the interactive dashboard? (y/n)")
    response = input().strip().lower()
    
    if response == 'y':
        print("  Launching dashboard...")
        dashboard = ShelfScaleDashboard(normalized_df)
        dashboard.run_server(debug=True)
    
    print("\nShelfScale workflow completed!")


if __name__ == "__main__":
    main() 