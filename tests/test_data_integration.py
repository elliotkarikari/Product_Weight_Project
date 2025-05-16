#!/usr/bin/env python
"""
Test data integration between different sources.
Verifies that data from various sources can be loaded, processed and matched correctly.
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np
from pathlib import Path

# Add the project root to the path so we can import modules
sys.path.append(str(Path(__file__).parent.parent))

from shelfscale.data_sourcing.pdf_extraction import PDFExtractor
from shelfscale.data_processing.raw_processor import RawDataProcessor
from shelfscale.matching.algorithm import FoodMatcher
from shelfscale.utils.helpers import get_path

class TestDataIntegration(unittest.TestCase):
    """Test suite for data integration"""
    
    def setUp(self):
        """Set up test case"""
        # Set up paths
        self.raw_data_dir = get_path("Data/Raw Data")
        self.processed_data_dir = get_path("Data/Processed")
        self.output_dir = get_path("output")
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize helper classes
        self.pdf_extractor = PDFExtractor(cache_dir=self.output_dir)
        self.matcher = FoodMatcher(similarity_threshold=0.7)
    
    def test_pdf_extraction(self):
        """Test PDF extraction"""
        # Check if PDF files exist
        food_portion_pdf = os.path.join(self.raw_data_dir, "Food_Portion_Sizes.pdf")
        fruit_veg_pdf = None
        
        # Look for fruit and veg PDF
        for file in os.listdir(self.raw_data_dir):
            if "fruit" in file.lower() and "veg" in file.lower() and file.endswith(".pdf"):
                fruit_veg_pdf = os.path.join(self.raw_data_dir, file)
                break
        
        # Test food portion sizes extraction if file exists
        if os.path.exists(food_portion_pdf):
            df = self.pdf_extractor.extract_food_portion_sizes(food_portion_pdf)
            self.assertIsInstance(df, pd.DataFrame)
            self.assertGreater(len(df), 0)
            
            # Check required columns
            self.assertIn("Food_Name", df.columns)
            self.assertIn("Weight_g", df.columns)
            
            # Check that there's some actual data
            self.assertGreater(df["Food_Name"].notna().sum(), 0)
        
        # Test fruit and veg survey extraction if file exists
        if fruit_veg_pdf and os.path.exists(fruit_veg_pdf):
            df = self.pdf_extractor.extract_fruit_veg_survey(fruit_veg_pdf)
            self.assertIsInstance(df, pd.DataFrame)
            self.assertGreater(len(df), 0)
            
            # Check required columns
            self.assertIn("Sample_Name", df.columns)
            self.assertIn("Pack_Size", df.columns)
            
            # Check that there's some actual data
            self.assertGreater(df["Sample_Name"].notna().sum(), 0)
    
    def test_food_matching(self):
        """Test food matching between datasets"""
        # Create sample datasets
        foods1 = pd.DataFrame({
            "Food_Name": ["Apple", "Banana", "Orange", "Pear", "Grapes"],
            "Weight_g": [100, 120, 180, 150, 200]
        })
        
        foods2 = pd.DataFrame({
            "Sample_Name": ["Red apple", "Banana (fresh)", "Orange juice", "Conference pear", "White grapes"],
            "Pack_Size": ["95g", "125g", "200ml", "155g", "250g"]
        })
        
        # Test matching
        matches = self.matcher.match_datasets(
            foods1, 
            foods2,
            "Food_Name",
            "Sample_Name"
        )
        
        # Check that we got matches
        self.assertGreater(len(matches), 0)
        
        # Check specific matches
        self.assertIn(("Apple", "Red apple"), 
                     [(m["source_item"], m["target_item"]) for m in matches])
        self.assertIn(("Banana", "Banana (fresh)"), 
                     [(m["source_item"], m["target_item"]) for m in matches])
        
        # Test merging
        merged_df = self.matcher.merge_matched_datasets(
            foods1,
            foods2,
            matches
        )
        
        # Check that merged dataset has correct shape
        self.assertEqual(len(merged_df), len(foods1))
        
        # Check that columns were merged
        self.assertIn("Sample_Name", merged_df.columns)
        self.assertIn("Pack_Size", merged_df.columns)
    
    def test_column_standardization(self):
        """Test column name standardization and matching with variant names"""
        # Create datasets with variant column names
        df1 = pd.DataFrame({
            "Food Name": ["Apple", "Banana", "Orange"],
            "Weight": [100, 120, 180]
        })
        
        df2 = pd.DataFrame({
            "food_name": ["Red apple", "Banana (fresh)", "Orange juice"],
            "weight_g": [95, 125, 200]
        })
        
        # Create a processor to standardize
        processor = RawDataProcessor()
        
        # Standardize column names through a helper function
        def standardize_columns(df):
            rename_map = {}
            for col in df.columns:
                col_lower = col.lower()
                if 'food' in col_lower and 'name' in col_lower:
                    rename_map[col] = 'Food_Name'
                elif col_lower in ['name', 'foodname']:
                    rename_map[col] = 'Food_Name'
                elif 'weight' in col_lower:
                    rename_map[col] = 'Weight_g'
            
            if rename_map:
                return df.rename(columns=rename_map)
            return df
        
        # Apply standardization
        df1_std = standardize_columns(df1)
        df2_std = standardize_columns(df2)
        
        # Check that column names were standardized
        self.assertIn("Food_Name", df1_std.columns)
        self.assertIn("Weight_g", df1_std.columns)
        self.assertIn("Food_Name", df2_std.columns)
        self.assertIn("Weight_g", df2_std.columns)
        
        # Test matching with standardized column names
        matches = self.matcher.match_datasets(
            df1_std, 
            df2_std,
            "Food_Name",
            "Food_Name"
        )
        
        # Check that matches were found
        self.assertGreater(len(matches), 0)
    
    def test_raw_processor(self):
        """Test raw data processor"""
        # Initialize processor
        processor = RawDataProcessor(
            self.raw_data_dir,
            self.processed_data_dir
        )
        
        # Check directory creation
        super_group_dir = os.path.join(self.processed_data_dir, "MW_DataReduction/Reduced Super Group")
        self.assertTrue(os.path.exists(super_group_dir))
        
        # Try to process McCance Widdowson data if file exists
        mw_file = os.path.join(self.raw_data_dir, "McCance_Widdowsons_2021.xlsx")
        if os.path.exists(mw_file):
            # Process MW data
            df = processor.process_mccance_widdowson()
            
            # Check that processing worked
            self.assertIsInstance(df, pd.DataFrame)
            self.assertGreater(len(df), 0)
            
            # Check output files
            output_path = os.path.join(self.processed_data_dir, "MW_DataReduction/Reduced Total/McCance_Widdowson_Full.csv")
            self.assertTrue(os.path.exists(output_path))

if __name__ == "__main__":
    unittest.main() 