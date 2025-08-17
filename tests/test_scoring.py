"""
Tests for nutrition scoring functionality
"""

import unittest
import pandas as pd
import numpy as np
from shelfscale.scoring.traffic_lights import TrafficLightsScorer, score_traffic_lights
from shelfscale.scoring.nutri_score import NutriScorer, score_nutri


class TestTrafficLights(unittest.TestCase):
    """Test cases for Traffic Light scoring"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.scorer = TrafficLightsScorer()
        
        # Test data representing different nutrition scenarios
        self.test_data = pd.DataFrame({
            'Food_Name': ['Apple', 'Chocolate Bar', 'Cola', 'Bread', 'Water'],
            'Super_Category': ['Fruit', 'Sugars preserves and snacks', 'Beverages', 
                              'Cereals and cereal products', 'Beverages'],
            'Fat_g': [0.2, 31.0, 0.0, 3.2, 0.0],           # per 100g
            'SatFat_g': [0.1, 18.5, 0.0, 0.7, 0.0],        # per 100g
            'Sugars_g': [10.4, 43.0, 10.6, 4.5, 0.0],      # per 100g
            'Salt_g': [0.0, 0.02, 0.01, 1.1, 0.0],         # per 100g
            'Serving_Weight_g': [150, 50, 330, 30, 250]     # serving size
        })
    
    def test_classify_nutrient_foods(self):
        """Test nutrient classification for foods"""
        # Test fat classification
        self.assertEqual(self.scorer.classify_nutrient(2.0, 'fat', is_drink=False), 'green')
        self.assertEqual(self.scorer.classify_nutrient(10.0, 'fat', is_drink=False), 'amber')
        self.assertEqual(self.scorer.classify_nutrient(20.0, 'fat', is_drink=False), 'red')
        
        # Test saturated fat classification
        self.assertEqual(self.scorer.classify_nutrient(1.0, 'saturates', is_drink=False), 'green')
        self.assertEqual(self.scorer.classify_nutrient(3.0, 'saturates', is_drink=False), 'amber')
        self.assertEqual(self.scorer.classify_nutrient(6.0, 'saturates', is_drink=False), 'red')
        
        # Test sugars classification
        self.assertEqual(self.scorer.classify_nutrient(4.0, 'sugars', is_drink=False), 'green')
        self.assertEqual(self.scorer.classify_nutrient(15.0, 'sugars', is_drink=False), 'amber')
        self.assertEqual(self.scorer.classify_nutrient(25.0, 'sugars', is_drink=False), 'red')
        
        # Test salt classification
        self.assertEqual(self.scorer.classify_nutrient(0.2, 'salt', is_drink=False), 'green')
        self.assertEqual(self.scorer.classify_nutrient(1.0, 'salt', is_drink=False), 'amber')
        self.assertEqual(self.scorer.classify_nutrient(2.0, 'salt', is_drink=False), 'red')
    
    def test_classify_nutrient_drinks(self):
        """Test nutrient classification for drinks"""
        # Test sugars for drinks (different thresholds)
        self.assertEqual(self.scorer.classify_nutrient(2.0, 'sugars', is_drink=True), 'green')
        self.assertEqual(self.scorer.classify_nutrient(8.0, 'sugars', is_drink=True), 'amber')
        self.assertEqual(self.scorer.classify_nutrient(15.0, 'sugars', is_drink=True), 'red')
    
    def test_score_product_basic(self):
        """Test basic product scoring"""
        # Test apple (should be mostly green)
        apple_score = self.scorer.score_product(
            fat_g=0.2, saturates_g=0.1, sugars_g=10.4, salt_g=0.0
        )
        
        self.assertEqual(apple_score['fat'], 'green')
        self.assertEqual(apple_score['saturates'], 'green')
        self.assertEqual(apple_score['sugars'], 'amber')  # Natural sugars
        self.assertEqual(apple_score['salt'], 'green')
        self.assertEqual(apple_score['summary'], 'amber')  # Worst score
        
        # Test chocolate bar (should be mostly red)
        chocolate_score = self.scorer.score_product(
            fat_g=31.0, saturates_g=18.5, sugars_g=43.0, salt_g=0.02
        )
        
        self.assertEqual(chocolate_score['fat'], 'red')
        self.assertEqual(chocolate_score['saturates'], 'red')
        self.assertEqual(chocolate_score['sugars'], 'red')
        self.assertEqual(chocolate_score['salt'], 'green')
        self.assertEqual(chocolate_score['summary'], 'red')
    
    def test_score_product_beverage(self):
        """Test beverage scoring with different thresholds"""
        cola_score = self.scorer.score_product(
            fat_g=0.0, saturates_g=0.0, sugars_g=10.6, salt_g=0.01,
            is_drink=True
        )
        
        self.assertEqual(cola_score['fat'], 'green')
        self.assertEqual(cola_score['saturates'], 'green')
        self.assertEqual(cola_score['sugars'], 'amber')  # Drink threshold
        self.assertEqual(cola_score['salt'], 'green')
    
    def test_sodium_to_salt_conversion(self):
        """Test sodium to salt conversion"""
        # Test with sodium instead of salt (sodium_mg = salt_g * 400)
        score_with_sodium = self.scorer.score_product(
            fat_g=1.0, saturates_g=0.5, sugars_g=5.0, sodium_mg=600  # 600mg sodium = 1.5g salt
        )
        
        score_with_salt = self.scorer.score_product(
            fat_g=1.0, saturates_g=0.5, sugars_g=5.0, salt_g=1.5
        )
        
        # Should give same results
        self.assertEqual(score_with_sodium['salt'], score_with_salt['salt'])
        self.assertEqual(score_with_sodium['salt'], 'red')  # 1.5g salt is red
    
    def test_score_dataframe(self):
        """Test scoring a full DataFrame"""
        scored_df = score_traffic_lights(self.test_data)
        
        # Check that all traffic light columns were added
        expected_cols = ['Traffic_Lights_Fat', 'Traffic_Lights_SatFat', 
                        'Traffic_Lights_Sugars', 'Traffic_Lights_Salt', 'Traffic_Lights_Summary']
        
        for col in expected_cols:
            self.assertIn(col, scored_df.columns)
        
        # Check specific results
        # Apple should be mostly green
        apple_row = scored_df[scored_df['Food_Name'] == 'Apple'].iloc[0]
        self.assertEqual(apple_row['Traffic_Lights_Fat'], 'green')
        self.assertEqual(apple_row['Traffic_Lights_Salt'], 'green')
        
        # Chocolate should be mostly red
        chocolate_row = scored_df[scored_df['Food_Name'] == 'Chocolate Bar'].iloc[0]
        self.assertEqual(chocolate_row['Traffic_Lights_Fat'], 'red')
        self.assertEqual(chocolate_row['Traffic_Lights_SatFat'], 'red')
        
        # Cola should be identified as beverage and scored accordingly
        cola_row = scored_df[scored_df['Food_Name'] == 'Cola'].iloc[0]
        self.assertEqual(cola_row['Traffic_Lights_Fat'], 'green')


class TestNutriScore(unittest.TestCase):
    """Test cases for Nutri-Score calculation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.scorer = NutriScorer()
        
        # Test data with known Nutri-Score examples
        self.test_data = pd.DataFrame({
            'Food_Name': ['Apple', 'Chocolate Bar', 'Water', 'Whole Wheat Bread', 'Yogurt'],
            'Super_Category': ['Fruit', 'Sugars preserves and snacks', 'Beverages', 
                              'Cereals and cereal products', 'Milk and milk products'],
            'Energy_kcal': [52, 534, 0, 247, 59],            # per 100g/ml
            'Sugars_g': [10.4, 43.0, 0, 4.5, 4.7],          # per 100g/ml
            'SatFat_g': [0.1, 18.5, 0, 0.7, 1.0],           # per 100g/ml
            'Salt_g': [0.0, 0.02, 0, 1.1, 0.1],             # per 100g/ml
            'Fiber_g': [2.4, 7.0, 0, 8.5, 0.0],             # per 100g/ml
            'Protein_g': [0.3, 4.9, 0, 13.0, 3.5],          # per 100g/ml
            'FVN_percent': [100, 0, 0, 0, 0]                 # % fruits/veg/nuts
        })
    
    def test_calculate_threshold_points(self):
        """Test threshold point calculation"""
        # Test energy thresholds for foods
        self.assertEqual(self.scorer.calculate_threshold_points(300, self.scorer.energy_thresholds_food), 0)
        self.assertEqual(self.scorer.calculate_threshold_points(500, self.scorer.energy_thresholds_food), 1)
        self.assertEqual(self.scorer.calculate_threshold_points(1000, self.scorer.energy_thresholds_food), 2)
        
        # Test sugar thresholds
        self.assertEqual(self.scorer.calculate_threshold_points(4.0, self.scorer.sugars_thresholds_food), 0)
        self.assertEqual(self.scorer.calculate_threshold_points(8.0, self.scorer.sugars_thresholds_food), 1)
        self.assertEqual(self.scorer.calculate_threshold_points(50.0, self.scorer.sugars_thresholds_food), 10)
    
    def test_negative_points_calculation(self):
        """Test negative points (A) calculation"""
        # Test chocolate bar (high energy, sugar, sat fat)
        neg_points = self.scorer.calculate_negative_points(
            energy_kj=534 * 4.184,  # Convert kcal to kJ
            sugars_g=43.0,
            saturated_fat_g=18.5,
            salt_g=0.02
        )
        
        # Should have high negative points
        self.assertGreater(neg_points['total'], 15)
        self.assertGreater(neg_points['energy'], 5)
        self.assertGreater(neg_points['sugars'], 8)
        self.assertGreater(neg_points['saturated_fat'], 8)
        
        # Test apple (low negative points)
        neg_points_apple = self.scorer.calculate_negative_points(
            energy_kj=52 * 4.184,
            sugars_g=10.4,
            saturated_fat_g=0.1,
            salt_g=0.0
        )
        
        self.assertLess(neg_points_apple['total'], 5)
    
    def test_positive_points_calculation(self):
        """Test positive points (C) calculation"""
        # Test apple (high FVN, some fiber)
        pos_points = self.scorer.calculate_positive_points(
            fruits_veg_nuts_percent=100,
            fiber_g=2.4,
            protein_g=0.3
        )
        
        self.assertEqual(pos_points['fvn'], 5)  # Maximum FVN points
        self.assertGreater(pos_points['fiber'], 0)
        self.assertEqual(pos_points['total'], pos_points['fvn'] + pos_points['fiber'] + pos_points['protein'])
        
        # Test bread (high fiber and protein)
        pos_points_bread = self.scorer.calculate_positive_points(
            fruits_veg_nuts_percent=0,
            fiber_g=8.5,
            protein_g=13.0
        )
        
        self.assertEqual(pos_points_bread['fvn'], 0)
        self.assertGreater(pos_points_bread['fiber'], 3)
        self.assertEqual(pos_points_bread['protein'], 5)  # Maximum protein points
    
    def test_nutri_score_calculation(self):
        """Test complete Nutri-Score calculation"""
        # Test water (should be automatic A)
        water_score = self.scorer.calculate_nutri_score(is_water=True)
        self.assertEqual(water_score['grade'], 'A')
        self.assertTrue(water_score['is_water'])
        
        # Test apple (should be A or B)
        apple_score = self.scorer.calculate_nutri_score(
            energy_kcal=52,
            sugars_g=10.4,
            saturated_fat_g=0.1,
            salt_g=0.0,
            fiber_g=2.4,
            protein_g=0.3,
            fruits_veg_nuts_percent=100
        )
        
        self.assertIn(apple_score['grade'], ['A', 'B'])
        self.assertLess(apple_score['score'], 5)  # Should have low score
        
        # Test chocolate bar (should be D or E)
        chocolate_score = self.scorer.calculate_nutri_score(
            energy_kcal=534,
            sugars_g=43.0,
            saturated_fat_g=18.5,
            salt_g=0.02,
            fiber_g=7.0,
            protein_g=4.9,
            fruits_veg_nuts_percent=0
        )
        
        self.assertIn(chocolate_score['grade'], ['D', 'E'])
        self.assertGreater(chocolate_score['score'], 10)  # Should have high score
    
    def test_protein_exception_rule(self):
        """Test protein exception rule (A >= 11 and FVN < 5)"""
        # Create a scenario where A >= 11 and FVN < 5
        score_result = self.scorer.calculate_nutri_score(
            energy_kcal=600,  # High energy
            sugars_g=30.0,    # High sugars  
            saturated_fat_g=10.0,  # High sat fat
            salt_g=2.0,       # High salt
            fiber_g=1.0,
            protein_g=15.0,   # High protein but should be ignored
            fruits_veg_nuts_percent=0  # Low FVN
        )
        
        # Protein should be excluded due to exception rule
        self.assertTrue(score_result['protein_exception_applied'])
        self.assertEqual(score_result['positive_points']['protein'], 0)
        self.assertGreater(score_result['positive_points']['protein_raw'], 0)
    
    def test_sodium_to_salt_conversion(self):
        """Test sodium to salt conversion in Nutri-Score"""
        # Test with salt
        score_with_salt = self.scorer.calculate_nutri_score(
            energy_kcal=100, sugars_g=5.0, saturated_fat_g=2.0, salt_g=1.0
        )
        
        # Test with equivalent sodium (salt_g * 400 = sodium_mg)
        score_with_sodium = self.scorer.calculate_nutri_score(
            energy_kcal=100, sugars_g=5.0, saturated_fat_g=2.0, sodium_mg=400
        )
        
        # Should give same sodium points
        self.assertEqual(
            score_with_salt['negative_points']['sodium'],
            score_with_sodium['negative_points']['sodium']
        )
    
    def test_score_dataframe(self):
        """Test scoring a full DataFrame"""
        scored_df = score_nutri(self.test_data)
        
        # Check that Nutri-Score columns were added
        self.assertIn('Nutri_Score', scored_df.columns)
        self.assertIn('Nutri_Grade', scored_df.columns)
        self.assertIn('Score_Confidence', scored_df.columns)
        
        # Check specific results
        # Water should be A
        water_row = scored_df[scored_df['Food_Name'] == 'Water'].iloc[0]
        self.assertEqual(water_row['Nutri_Grade'], 'A')
        
        # Apple should be A or B (healthy fruit)
        apple_row = scored_df[scored_df['Food_Name'] == 'Apple'].iloc[0]
        self.assertIn(apple_row['Nutri_Grade'], ['A', 'B'])
        
        # Chocolate should be D or E (unhealthy)
        chocolate_row = scored_df[scored_df['Food_Name'] == 'Chocolate Bar'].iloc[0]
        self.assertIn(chocolate_row['Nutri_Grade'], ['D', 'E'])


if __name__ == '__main__':
    unittest.main()