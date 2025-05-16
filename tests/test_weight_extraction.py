#!/usr/bin/env python
"""
Test script for the enhanced weight extraction functionality.
Tests various weight formats and extraction accuracy.
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np
from pathlib import Path

# Add the project root to the path so we can import modules
sys.path.append(str(Path(__file__).parent.parent))

from shelfscale.data_processing.weight_extraction import (
    WeightExtractor, 
    clean_weights,
    predict_missing_weights
)

class TestWeightExtraction(unittest.TestCase):
    """Test cases for weight extraction functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create weight extractor
        self.weight_extractor = WeightExtractor(target_unit='g')
        
        # Sample data with various weight formats
        self.test_cases = [
            # Simple cases
            ("100g", 100.0, "g"),
            ("250 g", 250.0, "g"),
            ("1kg", 1000.0, "g"),
            ("1.5 kg", 1500.0, "g"),
            ("500mg", 0.5, "g"),
            
            # Range formats
            ("100-150g", 150.0, "g"),
            ("1.2–1.5kg", 1500.0, "g"),
            
            # Multipack formats
            ("3 x 100g", 100.0, "g"),
            ("6x50g", 50.0, "g"),
            
            # Pack formats
            ("6 pack x 30g", 30.0, "g"),
            ("4pk x 125g", 125.0, "g"),
            
            # Fraction formats
            ("1/2 kg", 2000.0, "g"),
            
            # Mixed fractions
            ("1 1/2 kg", 1500.0, "g"),
            
            # Common approximations
            ("approx 100g", 100.0, "g"),
            
            # Various food product formats
            ("Cheese, cheddar (200g block)", 200.0, "g"),
            ("Bread, whole wheat, 400g loaf", 400.0, "g"),
            ("Yogurt pot (125g)", 125.0, "g"),
            ("Rice 1kg bag", 1000.0, "g"),
            ("Mini cookies 8 x 25g", 200.0, "g"),
            ("Chocolate bar, 3.5oz", 99.23, "g"),
            ("Cereal 750g box", 750.0, "g"),
            ("Coffee beans 1lb bag", 453.59, "g")
        ]
        
        # Create a test DataFrame
        self.test_df = pd.DataFrame({
            'Food_Name': [
                'Apples, red, raw',
                'Bananas, raw',
                'Carrot, raw',
                'Chicken breast, raw',
                'Rice, white, cooked',
                'Milk, whole',
                'Yogurt, plain',
                'Bread, whole wheat',
                'Chocolate bar',
                'Potato chips, regular'
            ],
            'Food_Group': [
                'Fruits',
                'Fruits',
                'Vegetables',
                'Meat',
                'Grains',
                'Dairy',
                'Dairy',
                'Grains',
                'Sweets',
                'Snacks'
            ],
            'Weight_Text': [
                '150g',
                '120g',
                '80g',
                '250g',
                '180g',
                '250g',
                '125g',
                '400g loaf',
                '50g',
                '30g'
            ],
            'Package_Size': [
                '1kg bag',
                '5 x 120g',
                '1kg bag',
                '500g package',
                '5kg bag',
                '1kg carton',
                '4 x 125g',
                '400g',
                'box of 24',
                '150g bag'
            ]
        })
    
    def test_individual_extraction(self):
        """Test extraction of individual weight strings"""
        for text, expected_value, expected_unit in self.test_cases:
            value, unit = self.weight_extractor.extract(text)
            
            # Allow small differences due to rounding
            self.assertIsNotNone(value, f"Failed to extract weight from '{text}'")
            self.assertAlmostEqual(value, expected_value, delta=1.0, 
                                  msg=f"Extracted {value} from '{text}', expected {expected_value}")
            self.assertEqual(unit, expected_unit, f"Extracted unit '{unit}' from '{text}', expected '{expected_unit}'")
    
    def test_dataframe_processing(self):
        """Test processing a DataFrame with weight information"""
        # Process the test DataFrame
        result = self.weight_extractor.process_dataframe(
            self.test_df,
            text_cols=['Weight_Text', 'Package_Size'],
            new_weight_col='Normalized_Weight',
            new_unit_col='Weight_Unit',
            source_col='Weight_Source'
        )
        
        # Check results
        self.assertEqual(len(result), len(self.test_df), "Output DataFrame should have the same number of rows")
        
        # Check that weight extraction worked on most rows
        successful_extractions = result['Normalized_Weight'].notna().sum()
        self.assertGreaterEqual(successful_extractions, 8, 
                              f"Expected at least 8 successful extractions, got {successful_extractions}")
        
        # Check a few specific values
        self.assertAlmostEqual(result.iloc[0]['Normalized_Weight'], 150.0, delta=1.0)
        self.assertEqual(result.iloc[0]['Weight_Unit'], 'g')
    
    def test_clean_weights(self):
        """Test the clean_weights function"""
        # Create a simple DataFrame with only weight values
        df = pd.DataFrame({
            'Weight': ['100g', '200 g', '1kg', '45oz', 'non-weight']
        })
        
        # Clean weights
        result = clean_weights(df, weight_col='Weight')
        
        # Check results
        self.assertEqual(len(result), len(df))
        self.assertEqual(result['Normalized_Weight'].notna().sum(), 4)
        
        # Check specific values
        self.assertAlmostEqual(result.iloc[0]['Normalized_Weight'], 100.0)
        self.assertAlmostEqual(result.iloc[2]['Normalized_Weight'], 1000.0)
        self.assertTrue(pd.isna(result.iloc[4]['Normalized_Weight']))
    
    def test_predict_missing_weights_simple(self):
        """Test the predict_missing_weights function with a simple prepared dataset"""
        # Create a simple DataFrame with foods in the same group
        # Some have weights and some don't
        df = pd.DataFrame({
            'Food_Name': [
                'Apple', 'Banana', 'Orange',  # Have weights
                'Pear', 'Grape', 'Kiwi'       # Missing weights
            ],
            'Food_Group': ['Fruit'] * 6,
            'Normalized_Weight': [
                150.0, 120.0, 180.0,  # Known weights
                np.nan, np.nan, np.nan  # Missing weights
            ]
        })
        
        # Predict missing weights
        result = predict_missing_weights(
            df,
            weight_col='Normalized_Weight',
            group_col='Food_Group',
            name_col='Food_Name'
        )
        
        # All rows should now have weights
        self.assertTrue(result['Normalized_Weight'].notna().all(),
                       "All rows should have weights after prediction")
        
        # Check that the prediction outputs are present
        self.assertTrue('Weight_Prediction_Source' in result.columns)
        self.assertTrue('Weight_Prediction_Confidence' in result.columns)
        
        # Verify the predicted values are reasonable (should be close to the group median/mean)
        for i in range(3, 6):
            self.assertGreater(result.iloc[i]['Normalized_Weight'], 0)
            self.assertIsNotNone(result.iloc[i]['Weight_Prediction_Source'])
            self.assertGreater(result.iloc[i]['Weight_Prediction_Confidence'], 0)


if __name__ == '__main__':
    unittest.main() 