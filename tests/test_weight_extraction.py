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
    predict_missing_weights,
    load_density_map,
    get_density_for_row
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


class TestVolumeConversion(unittest.TestCase):
    """Test cases for volume to weight conversion using density maps"""
    
    def setUp(self):
        """Set up test fixtures for volume conversion"""
        self.weight_extractor = WeightExtractor(target_unit='g')
        
        # Load density map
        self.density_df = load_density_map()
        
        # Volume conversion test cases
        self.volume_test_cases = [
            # Format: (text, product_row, expected_weight_approx, expected_unit)
            ("500ml", {"Super_Category": "Milk and milk products", "Food_Category": "Milk"}, 515.0, "g"),  # 500ml * 1.03 density
            ("1L", {"Super_Category": "Beverages", "Food_Category": "Water"}, 1000.0, "g"),  # 1000ml * 1.0 density
            ("250ml", {"Super_Category": "Fats and oils", "Food_Category": "Olive oil"}, 227.5, "g"),  # 250ml * 0.91 density
            ("100ml", {"Super_Category": "Sugars preserves and snacks", "Food_Category": "Honey"}, 140.0, "g"),  # 100ml * 1.4 density
            ("2L", {"Super_Category": "Beverages", "Food_Category": "Soft drinks"}, 2100.0, "g"),  # 2000ml * 1.05 density
        ]
        
        # Test DataFrame with volume units
        self.volume_test_df = pd.DataFrame({
            'Food_Name': ['Milk', 'Water', 'Olive Oil', 'Honey', 'Cola'],
            'Super_Category': ['Milk and milk products', 'Beverages', 'Fats and oils', 
                              'Sugars preserves and snacks', 'Beverages'],
            'Food_Category': ['Milk', 'Water', 'Olive oil', 'Honey', 'Soft drinks'],
            'Volume_Text': ['500ml', '1L', '250ml', '100ml', '2L']
        })
    
    def test_density_loading(self):
        """Test loading of density map"""
        self.assertIsInstance(self.density_df, pd.DataFrame)
        self.assertIn('Super_Category', self.density_df.columns)
        self.assertIn('Food_Category', self.density_df.columns)
        self.assertIn('density_g_per_ml', self.density_df.columns)
        self.assertGreater(len(self.density_df), 0)
    
    def test_get_density_for_row(self):
        """Test density lookup for product rows"""
        # Test specific match
        milk_row = pd.Series({'Super_Category': 'Milk and milk products', 'Food_Category': 'Milk'})
        milk_density = get_density_for_row(milk_row, self.density_df)
        self.assertAlmostEqual(milk_density, 1.03, places=2)
        
        # Test super category match
        beverage_row = pd.Series({'Super_Category': 'Beverages', 'Food_Category': 'Unknown'})
        beverage_density = get_density_for_row(beverage_row, self.density_df)
        self.assertGreater(beverage_density, 0.9)
        self.assertLess(beverage_density, 1.1)
        
        # Test fallback to default
        unknown_row = pd.Series({'Super_Category': 'Unknown', 'Food_Category': 'Unknown'})
        default_density = get_density_for_row(unknown_row, self.density_df)
        self.assertEqual(default_density, 1.0)
    
    def test_volume_to_weight_conversion(self):
        """Test volume to weight conversion for individual cases"""
        for text, product_data, expected_weight, expected_unit in self.volume_test_cases:
            product_row = pd.Series(product_data)
            
            # Extract with density conversion
            weight, unit = self.weight_extractor.extract_from_text(text, product_row)
            
            self.assertIsNotNone(weight, f"Failed to extract weight from volume '{text}'")
            self.assertAlmostEqual(weight, expected_weight, delta=5.0,
                                  msg=f"Volume conversion: {text} → {weight}g, expected ~{expected_weight}g")
            self.assertEqual(unit, expected_unit)
    
    def test_volume_dataframe_processing(self):
        """Test processing DataFrame with volume units"""
        # Process DataFrame with volume→weight conversion
        result = self.weight_extractor.process_dataframe(
            self.volume_test_df,
            text_cols=['Volume_Text'],
            new_weight_col='Normalized_Weight',
            new_unit_col='Weight_Unit',
            source_col='Weight_Source'
        )
        
        # Check that all volumes were converted to weights
        self.assertTrue(result['Normalized_Weight'].notna().all(),
                       "All volume entries should be converted to weights")
        
        # Check specific conversions
        milk_weight = result[result['Food_Name'] == 'Milk']['Normalized_Weight'].iloc[0]
        self.assertAlmostEqual(milk_weight, 515.0, delta=10.0)  # 500ml milk ≈ 515g
        
        water_weight = result[result['Food_Name'] == 'Water']['Normalized_Weight'].iloc[0]
        self.assertAlmostEqual(water_weight, 1000.0, delta=5.0)  # 1L water = 1000g
        
        oil_weight = result[result['Food_Name'] == 'Olive Oil']['Normalized_Weight'].iloc[0]
        self.assertAlmostEqual(oil_weight, 227.5, delta=5.0)  # 250ml oil ≈ 227.5g
    
    def test_mixed_units_dataframe(self):
        """Test DataFrame with mixed weight and volume units"""
        mixed_df = pd.DataFrame({
            'Food_Name': ['Bread', 'Milk', 'Cheese', 'Water', 'Honey'],
            'Super_Category': ['Cereals and cereal products', 'Milk and milk products', 
                              'Milk and milk products', 'Beverages', 'Sugars preserves and snacks'],
            'Food_Category': ['Bread', 'Milk', 'Cheese', 'Water', 'Honey'],
            'Weight_Text': ['400g', '500ml', '200g', '1.5L', '250ml']
        })
        
        result = self.weight_extractor.process_dataframe(
            mixed_df,
            text_cols=['Weight_Text'],
            new_weight_col='Normalized_Weight',
            new_unit_col='Weight_Unit',
            source_col='Weight_Source'
        )
        
        # All should have weights
        self.assertTrue(result['Normalized_Weight'].notna().all())
        
        # Weight units should remain as weights
        bread_weight = result[result['Food_Name'] == 'Bread']['Normalized_Weight'].iloc[0]
        self.assertAlmostEqual(bread_weight, 400.0, delta=1.0)
        
        cheese_weight = result[result['Food_Name'] == 'Cheese']['Normalized_Weight'].iloc[0]
        self.assertAlmostEqual(cheese_weight, 200.0, delta=1.0)
        
        # Volume units should be converted to weights
        milk_weight = result[result['Food_Name'] == 'Milk']['Normalized_Weight'].iloc[0]
        self.assertAlmostEqual(milk_weight, 515.0, delta=10.0)  # 500ml * 1.03
        
        water_weight = result[result['Food_Name'] == 'Water']['Normalized_Weight'].iloc[0]
        self.assertAlmostEqual(water_weight, 1500.0, delta=10.0)  # 1.5L * 1.0
        
        honey_weight = result[result['Food_Name'] == 'Honey']['Normalized_Weight'].iloc[0]
        self.assertAlmostEqual(honey_weight, 350.0, delta=10.0)  # 250ml * 1.4
    
    def test_unit_aliases(self):
        """Test new unit aliases are recognized"""
        unit_alias_cases = [
            ("100cl", 1000.0, "g"),  # centiliters
            ("5dl", 500.0, "g"),     # deciliters  
            ("1 litre", 1000.0, "g"), # liter variants
            ("2 litres", 2000.0, "g"),
        ]
        
        water_row = pd.Series({'Super_Category': 'Beverages', 'Food_Category': 'Water'})
        
        for text, expected_weight, expected_unit in unit_alias_cases:
            weight, unit = self.weight_extractor.extract_from_text(text, water_row)
            
            self.assertIsNotNone(weight, f"Failed to extract from '{text}'")
            self.assertAlmostEqual(weight, expected_weight, delta=5.0,
                                  msg=f"Unit alias test: {text} → {weight}g, expected {expected_weight}g")
            self.assertEqual(unit, expected_unit)


if __name__ == '__main__':
    unittest.main() 