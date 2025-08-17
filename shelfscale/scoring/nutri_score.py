"""
Nutri-Score nutrition labelling implementation
Based on the official Nutri-Score algorithm from Santé Publique France
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Union, Tuple
import logging

logger = logging.getLogger(__name__)


class NutriScorer:
    """
    Nutri-Score nutrition scoring system
    
    Implements the official Nutri-Score algorithm as defined by 
    Santé Publique France and the European Commission.
    """
    
    def __init__(self):
        """Initialize the Nutri-Score calculator with official thresholds"""
        
        # Negative points (A) - energy and nutrients to limit
        # Based on official Nutri-Score calculation tables
        
        # Energy thresholds (kJ per 100g) for foods
        self.energy_thresholds_food = [
            (335, 0), (670, 1), (1005, 2), (1340, 3), (1675, 4),
            (2010, 5), (2345, 6), (2680, 7), (3015, 8), (3350, 9), (float('inf'), 10)
        ]
        
        # Energy thresholds (kJ per 100ml) for beverages
        self.energy_thresholds_beverage = [
            (0, 0), (30, 1), (60, 2), (90, 3), (120, 4), 
            (150, 5), (180, 6), (210, 7), (240, 8), (270, 9), (float('inf'), 10)
        ]
        
        # Sugars thresholds (g per 100g/ml)
        self.sugars_thresholds_food = [
            (4.5, 0), (9, 1), (13.5, 2), (18, 3), (22.5, 4),
            (27, 5), (31, 6), (36, 7), (40, 8), (45, 9), (float('inf'), 10)
        ]
        
        self.sugars_thresholds_beverage = [
            (0, 0), (1.5, 1), (3, 2), (4.5, 3), (6, 4),
            (7.5, 5), (9, 6), (10.5, 7), (12, 8), (13.5, 9), (float('inf'), 10)
        ]
        
        # Saturated fat thresholds (g per 100g/ml)
        self.satfat_thresholds = [
            (1, 0), (2, 1), (3, 2), (4, 3), (5, 4),
            (6, 5), (7, 6), (8, 7), (9, 8), (10, 9), (float('inf'), 10)
        ]
        
        # Sodium thresholds (mg per 100g/ml)
        self.sodium_thresholds = [
            (90, 0), (180, 1), (270, 2), (360, 3), (450, 4),
            (540, 5), (630, 6), (720, 7), (810, 8), (900, 9), (float('inf'), 10)
        ]
        
        # Positive points (C) - beneficial nutrients
        
        # Fruits, vegetables, nuts percentage thresholds
        self.fvn_thresholds = [
            (40, 0), (60, 1), (80, 2), (float('inf'), 5)
        ]
        
        # Fiber thresholds (g per 100g/ml)
        self.fiber_thresholds = [
            (0.9, 0), (1.9, 1), (2.8, 2), (3.7, 3), (4.7, 4), (float('inf'), 5)
        ]
        
        # Protein thresholds (g per 100g/ml)
        self.protein_thresholds = [
            (1.6, 0), (3.2, 1), (4.8, 2), (6.4, 3), (8.0, 4), (float('inf'), 5)
        ]
        
        # Final score to grade mapping
        self.food_grade_thresholds = [
            (-1, 'A'), (2, 'B'), (10, 'C'), (18, 'D'), (float('inf'), 'E')
        ]
        
        self.beverage_grade_thresholds = [
            (1, 'B'), (5, 'C'), (9, 'D'), (float('inf'), 'E')
        ]
    
    def calculate_threshold_points(self, value: float, thresholds: list) -> int:
        """
        Calculate points based on threshold table
        
        Args:
            value: Value to score
            thresholds: List of (threshold, points) tuples
            
        Returns:
            Points for the given value
        """
        if pd.isna(value) or value < 0:
            return 0
            
        for threshold, points in thresholds:
            if value <= threshold:
                return points
        
        return thresholds[-1][1]  # Return max points if over all thresholds
    
    def calculate_negative_points(self, 
                                 energy_kj: Optional[float] = None,
                                 sugars_g: Optional[float] = None,
                                 saturated_fat_g: Optional[float] = None,
                                 sodium_mg: Optional[float] = None,
                                 salt_g: Optional[float] = None,
                                 is_beverage: bool = False) -> Dict[str, int]:
        """
        Calculate negative points (A) for nutrients to limit
        
        Args:
            energy_kj: Energy in kJ per 100g/ml
            sugars_g: Total sugars in g per 100g/ml
            saturated_fat_g: Saturated fat in g per 100g/ml
            sodium_mg: Sodium in mg per 100g/ml
            salt_g: Salt in g per 100g/ml (alternative to sodium)
            is_beverage: Whether this is a beverage
            
        Returns:
            Dictionary with negative points breakdown
        """
        negative_points = {}
        
        # Energy points
        if energy_kj is not None:
            thresholds = self.energy_thresholds_beverage if is_beverage else self.energy_thresholds_food
            negative_points['energy'] = self.calculate_threshold_points(energy_kj, thresholds)
        else:
            negative_points['energy'] = 0
        
        # Sugar points
        if sugars_g is not None:
            thresholds = self.sugars_thresholds_beverage if is_beverage else self.sugars_thresholds_food
            negative_points['sugars'] = self.calculate_threshold_points(sugars_g, thresholds)
        else:
            negative_points['sugars'] = 0
        
        # Saturated fat points
        if saturated_fat_g is not None:
            negative_points['saturated_fat'] = self.calculate_threshold_points(saturated_fat_g, self.satfat_thresholds)
        else:
            negative_points['saturated_fat'] = 0
        
        # Sodium points (convert salt to sodium if needed)
        if sodium_mg is None and salt_g is not None:
            sodium_mg = salt_g * 400  # Convert salt to sodium: sodium_mg = salt_g * 400
        
        if sodium_mg is not None:
            negative_points['sodium'] = self.calculate_threshold_points(sodium_mg, self.sodium_thresholds)
        else:
            negative_points['sodium'] = 0
        
        # Total negative points
        negative_points['total'] = sum(negative_points[k] for k in negative_points if k != 'total')
        
        return negative_points
    
    def calculate_positive_points(self,
                                 fruits_veg_nuts_percent: Optional[float] = None,
                                 fiber_g: Optional[float] = None,
                                 protein_g: Optional[float] = None) -> Dict[str, int]:
        """
        Calculate positive points (C) for beneficial nutrients
        
        Args:
            fruits_veg_nuts_percent: Percentage of fruits, vegetables, nuts (0-100)
            fiber_g: Fiber in g per 100g/ml
            protein_g: Protein in g per 100g/ml
            
        Returns:
            Dictionary with positive points breakdown
        """
        positive_points = {}
        
        # Fruits, vegetables, nuts points
        if fruits_veg_nuts_percent is not None:
            positive_points['fvn'] = self.calculate_threshold_points(fruits_veg_nuts_percent, self.fvn_thresholds)
        else:
            positive_points['fvn'] = 0
        
        # Fiber points
        if fiber_g is not None:
            positive_points['fiber'] = self.calculate_threshold_points(fiber_g, self.fiber_thresholds)
        else:
            positive_points['fiber'] = 0
        
        # Protein points
        if protein_g is not None:
            positive_points['protein'] = self.calculate_threshold_points(protein_g, self.protein_thresholds)
        else:
            positive_points['protein'] = 0
        
        # Total positive points
        positive_points['total'] = sum(positive_points[k] for k in positive_points if k != 'total')
        
        return positive_points
    
    def calculate_nutri_score(self,
                             energy_kj: Optional[float] = None,
                             energy_kcal: Optional[float] = None,
                             sugars_g: Optional[float] = None,
                             saturated_fat_g: Optional[float] = None,
                             sodium_mg: Optional[float] = None,
                             salt_g: Optional[float] = None,
                             fruits_veg_nuts_percent: Optional[float] = None,
                             fiber_g: Optional[float] = None,
                             protein_g: Optional[float] = None,
                             is_beverage: bool = False,
                             is_water: bool = False) -> Dict[str, Union[int, str, Dict]]:
        """
        Calculate complete Nutri-Score for a product
        
        Args:
            energy_kj: Energy in kJ per 100g/ml
            energy_kcal: Energy in kcal per 100g/ml (alternative to kJ)
            sugars_g: Total sugars in g per 100g/ml
            saturated_fat_g: Saturated fat in g per 100g/ml
            sodium_mg: Sodium in mg per 100g/ml
            salt_g: Salt in g per 100g/ml (alternative to sodium)
            fruits_veg_nuts_percent: Percentage of fruits, vegetables, nuts (0-100)
            fiber_g: Fiber in g per 100g/ml
            protein_g: Protein in g per 100g/ml
            is_beverage: Whether this is a beverage
            is_water: Whether this is water (automatic A grade)
            
        Returns:
            Dictionary with Nutri-Score calculation details
        """
        # Water gets automatic A grade
        if is_water:
            return {
                'score': -15,  # Very low score for water
                'grade': 'A',
                'negative_points': {'energy': 0, 'sugars': 0, 'saturated_fat': 0, 'sodium': 0, 'total': 0},
                'positive_points': {'fvn': 0, 'fiber': 0, 'protein': 0, 'total': 0},
                'is_beverage': True,
                'is_water': True
            }
        
        # Convert kcal to kJ if needed (1 kcal = 4.184 kJ)
        if energy_kj is None and energy_kcal is not None:
            energy_kj = energy_kcal * 4.184
        
        # Calculate negative points (A)
        negative_points = self.calculate_negative_points(
            energy_kj=energy_kj,
            sugars_g=sugars_g,
            saturated_fat_g=saturated_fat_g,
            sodium_mg=sodium_mg,
            salt_g=salt_g,
            is_beverage=is_beverage
        )
        
        # Calculate positive points (C)
        positive_points = self.calculate_positive_points(
            fruits_veg_nuts_percent=fruits_veg_nuts_percent,
            fiber_g=fiber_g,
            protein_g=protein_g
        )
        
        # Apply protein exception rule:
        # If A >= 11 and FVN < 5, then protein points are not counted
        total_negative = negative_points['total']
        fvn_points = positive_points['fvn']
        protein_points = positive_points['protein']
        
        if total_negative >= 11 and fvn_points < 5:
            protein_points_applied = 0
            logger.debug(f"Protein exception applied: A={total_negative}, FVN={fvn_points}")
        else:
            protein_points_applied = protein_points
        
        # Calculate final score: A - C
        total_positive = positive_points['fvn'] + positive_points['fiber'] + protein_points_applied
        final_score = total_negative - total_positive
        
        # Determine grade
        if is_beverage:
            grade_thresholds = self.beverage_grade_thresholds
        else:
            grade_thresholds = self.food_grade_thresholds
        
        grade = 'E'  # Default
        for threshold, grade_letter in grade_thresholds:
            if final_score <= threshold:
                grade = grade_letter
                break
        
        return {
            'score': final_score,
            'grade': grade,
            'negative_points': negative_points,
            'positive_points': {
                'fvn': positive_points['fvn'],
                'fiber': positive_points['fiber'],
                'protein': protein_points_applied,
                'protein_raw': protein_points,
                'total': total_positive
            },
            'is_beverage': is_beverage,
            'is_water': is_water,
            'protein_exception_applied': protein_points_applied != protein_points
        }


def score_nutri(df: pd.DataFrame,
               energy_kj_col: Optional[str] = 'Energy_kJ',
               energy_kcal_col: Optional[str] = 'Energy_kcal',
               sugars_col: str = 'Sugars_g',
               saturated_fat_col: str = 'SatFat_g',
               sodium_col: Optional[str] = 'Sodium_mg',
               salt_col: Optional[str] = 'Salt_g',
               fiber_col: Optional[str] = 'Fiber_g',
               protein_col: Optional[str] = 'Protein_g',
               fvn_col: Optional[str] = 'FVN_percent',
               is_beverage_col: Optional[str] = None,
               is_water_col: Optional[str] = None,
               beverage_categories: Optional[list] = None,
               water_categories: Optional[list] = None) -> pd.DataFrame:
    """
    Apply Nutri-Score calculation to a DataFrame of products
    
    Args:
        df: DataFrame with nutrition data (values should be per 100g/ml)
        energy_kj_col: Column name for energy in kJ per 100g/ml
        energy_kcal_col: Column name for energy in kcal per 100g/ml
        sugars_col: Column name for total sugars (g per 100g/ml)
        saturated_fat_col: Column name for saturated fat (g per 100g/ml)
        sodium_col: Column name for sodium (mg per 100g/ml)
        salt_col: Column name for salt (g per 100g/ml)
        fiber_col: Column name for fiber (g per 100g/ml)
        protein_col: Column name for protein (g per 100g/ml)
        fvn_col: Column name for fruits/vegetables/nuts percentage
        is_beverage_col: Column indicating if product is a beverage
        is_water_col: Column indicating if product is water
        beverage_categories: List of categories to treat as beverages
        water_categories: List of categories to treat as water
        
    Returns:
        DataFrame with added Nutri-Score columns
    """
    result_df = df.copy()
    scorer = NutriScorer()
    
    # Default beverage and water categories
    if beverage_categories is None:
        beverage_categories = [
            'Beverages', 'Alcoholic beverages', 'Soft drinks',
            'Fruit juice', 'Vegetable juice', 'Energy drinks'
        ]
    
    if water_categories is None:
        water_categories = ['Water', 'Mineral water', 'Spring water', 'Tap water']
    
    # Initialize output columns
    nutri_cols = ['Nutri_Score', 'Nutri_Grade', 'Score_Confidence']
    for col in nutri_cols:
        if col not in result_df.columns:
            if col == 'Score_Confidence':
                result_df[col] = np.nan
            else:
                result_df[col] = None
    
    # Process each row
    for idx, row in result_df.iterrows():
        # Determine if this is a beverage or water
        is_beverage = False
        is_water = False
        
        if is_beverage_col and is_beverage_col in result_df.columns:
            is_beverage = bool(row.get(is_beverage_col, False))
        elif 'Super_Category' in result_df.columns:
            is_beverage = row.get('Super_Category', '') in beverage_categories
        elif 'Food_Category' in result_df.columns:
            is_beverage = row.get('Food_Category', '') in beverage_categories
        
        if is_water_col and is_water_col in result_df.columns:
            is_water = bool(row.get(is_water_col, False))
        elif 'Food_Category' in result_df.columns:
            is_water = row.get('Food_Category', '') in water_categories
        elif 'Food_Name' in result_df.columns:
            food_name = str(row.get('Food_Name', '')).lower()
            is_water = any(water_term in food_name for water_term in ['water', 'mineral water'])
        
        # Get nutrient values
        energy_kj = row.get(energy_kj_col) if energy_kj_col and energy_kj_col in result_df.columns else None
        energy_kcal = row.get(energy_kcal_col) if energy_kcal_col and energy_kcal_col in result_df.columns else None
        sugars_g = row.get(sugars_col) if sugars_col in result_df.columns else None
        saturated_fat_g = row.get(saturated_fat_col) if saturated_fat_col in result_df.columns else None
        sodium_mg = row.get(sodium_col) if sodium_col and sodium_col in result_df.columns else None
        salt_g = row.get(salt_col) if salt_col and salt_col in result_df.columns else None
        fiber_g = row.get(fiber_col) if fiber_col and fiber_col in result_df.columns else None
        protein_g = row.get(protein_col) if protein_col and protein_col in result_df.columns else None
        fvn_percent = row.get(fvn_col) if fvn_col and fvn_col in result_df.columns else None
        
        # Calculate Nutri-Score
        nutri_result = scorer.calculate_nutri_score(
            energy_kj=energy_kj,
            energy_kcal=energy_kcal,
            sugars_g=sugars_g,
            saturated_fat_g=saturated_fat_g,
            sodium_mg=sodium_mg,
            salt_g=salt_g,
            fiber_g=fiber_g,
            protein_g=protein_g,
            fruits_veg_nuts_percent=fvn_percent,
            is_beverage=is_beverage,
            is_water=is_water
        )
        
        # Calculate confidence based on available data
        required_nutrients = [energy_kj or energy_kcal, sugars_g, saturated_fat_g, 
                             sodium_mg or salt_g, fiber_g, protein_g]
        available_count = sum(1 for val in required_nutrients if val is not None)
        confidence = available_count / len(required_nutrients)
        
        # Update DataFrame
        result_df.loc[idx, 'Nutri_Score'] = nutri_result['score']
        result_df.loc[idx, 'Nutri_Grade'] = nutri_result['grade']
        result_df.loc[idx, 'Score_Confidence'] = confidence
    
    return result_df


if __name__ == "__main__":
    # Example usage and testing
    
    # Create sample data with various food types
    sample_data = pd.DataFrame({
        'Food_Name': ['Apple', 'Chocolate Bar', 'Cola', 'Whole Wheat Bread', 'Cheese', 'Water'],
        'Super_Category': ['Fruit', 'Sugars preserves and snacks', 'Beverages', 
                          'Cereals and cereal products', 'Milk and milk products', 'Beverages'],
        'Food_Category': ['Fresh fruit', 'Chocolate', 'Soft drinks', 'Bread', 'Hard cheese', 'Water'],
        'Energy_kcal': [52, 534, 42, 247, 393, 0],         # per 100g/ml
        'Sugars_g': [10.4, 43.0, 10.6, 4.5, 0.1, 0],      # per 100g/ml
        'SatFat_g': [0.1, 18.5, 0.0, 0.7, 16.0, 0],       # per 100g/ml
        'Salt_g': [0.0, 0.02, 0.01, 1.1, 1.8, 0],         # per 100g/ml
        'Fiber_g': [2.4, 7.0, 0.0, 8.5, 0.0, 0],          # per 100g/ml
        'Protein_g': [0.3, 4.9, 0.0, 13.0, 24.9, 0],      # per 100g/ml
        'FVN_percent': [100, 0, 0, 0, 0, 0]                # % fruits/veg/nuts
    })
    
    # Apply Nutri-Score calculation
    scored_data = score_nutri(sample_data)
    
    # Display results
    print("Nutri-Score Calculation Results:")
    print("=" * 50)
    
    display_cols = ['Food_Name', 'Nutri_Score', 'Nutri_Grade', 'Score_Confidence']
    
    for idx, row in scored_data.iterrows():
        print(f"\n{row['Food_Name']}:")
        print(f"  Score: {row['Nutri_Score']}")
        print(f"  Grade: {row['Nutri_Grade']}")
        print(f"  Confidence: {row['Score_Confidence']:.2f}")