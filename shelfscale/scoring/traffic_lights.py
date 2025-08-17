"""
UK Front-of-Pack Traffic Light nutrition labelling implementation
Based on UK Food Standards Agency guidelines
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Union, Tuple
import logging

logger = logging.getLogger(__name__)


class TrafficLightsScorer:
    """
    UK Traffic Light nutrition scoring system
    
    Implements the UK Front-of-Pack Traffic Light labelling system
    based on guidelines from the UK Food Standards Agency.
    """
    
    def __init__(self):
        """Initialize the Traffic Lights scorer with UK thresholds"""
        
        # UK Traffic Light thresholds per 100g for foods
        # Source: UK Food Standards Agency Front-of-pack nutrition labelling guidance
        self.food_thresholds = {
            'fat': {'low': 3.0, 'high': 17.5},           # g per 100g
            'saturates': {'low': 1.5, 'high': 5.0},     # g per 100g  
            'sugars': {'low': 5.0, 'high': 22.5},       # g per 100g
            'salt': {'low': 0.3, 'high': 1.5}           # g per 100g
        }
        
        # UK Traffic Light thresholds per 100ml for drinks
        self.drink_thresholds = {
            'fat': {'low': 1.5, 'high': 8.75},          # g per 100ml
            'saturates': {'low': 0.75, 'high': 2.5},    # g per 100ml
            'sugars': {'low': 2.5, 'high': 11.25},      # g per 100ml
            'salt': {'low': 0.3, 'high': 0.75}          # g per 100ml
        }
        
        # Alternative serving size thresholds per serving
        self.serving_thresholds = {
            'fat': {'low': 1.0, 'high': 5.0},           # g per serving
            'saturates': {'low': 0.5, 'high': 1.5},     # g per serving
            'sugars': {'low': 2.5, 'high': 6.75},       # g per serving
            'salt': {'low': 0.3, 'high': 1.5}           # g per serving
        }
    
    def classify_nutrient(self, value: float, nutrient: str, 
                         is_drink: bool = False, 
                         per_serving: bool = False) -> str:
        """
        Classify a single nutrient value into traffic light category
        
        Args:
            value: Nutrient value to classify
            nutrient: Nutrient name ('fat', 'saturates', 'sugars', 'salt')
            is_drink: Whether this is a drink/beverage
            per_serving: Whether value is per serving (vs per 100g/ml)
            
        Returns:
            Traffic light classification: 'green', 'amber', or 'red'
        """
        if pd.isna(value) or value < 0:
            return 'unknown'
            
        # Select appropriate thresholds
        if per_serving:
            thresholds = self.serving_thresholds
        elif is_drink:
            thresholds = self.drink_thresholds
        else:
            thresholds = self.food_thresholds
            
        if nutrient not in thresholds:
            logger.warning(f"Unknown nutrient '{nutrient}' for traffic light classification")
            return 'unknown'
            
        low_threshold = thresholds[nutrient]['low']
        high_threshold = thresholds[nutrient]['high']
        
        if value <= low_threshold:
            return 'green'
        elif value <= high_threshold:
            return 'amber' 
        else:
            return 'red'
    
    def score_product(self, 
                     fat_g: Optional[float] = None,
                     saturates_g: Optional[float] = None, 
                     sugars_g: Optional[float] = None,
                     salt_g: Optional[float] = None,
                     sodium_mg: Optional[float] = None,
                     serving_weight_g: Optional[float] = None,
                     is_drink: bool = False) -> Dict[str, Union[str, Dict]]:
        """
        Score a single product using UK Traffic Lights system
        
        Args:
            fat_g: Total fat in grams per 100g/ml
            saturates_g: Saturated fat in grams per 100g/ml
            sugars_g: Total sugars in grams per 100g/ml  
            salt_g: Salt in grams per 100g/ml
            sodium_mg: Sodium in mg per 100g/ml (alternative to salt)
            serving_weight_g: Serving size in grams (for per-serving calculation)
            is_drink: Whether this is a beverage
            
        Returns:
            Dictionary with traffic light scores and summary
        """
        # Convert sodium to salt if provided (salt_g = sodium_mg / 400)
        if salt_g is None and sodium_mg is not None:
            salt_g = sodium_mg / 400.0
        
        # Initialize results
        results = {
            'fat': 'unknown',
            'saturates': 'unknown', 
            'sugars': 'unknown',
            'salt': 'unknown',
            'per_100g': {},
            'per_serving': {},
            'summary': 'unknown'
        }
        
        # Score per 100g/ml (standard)
        nutrients = {
            'fat': fat_g,
            'saturates': saturates_g,
            'sugars': sugars_g, 
            'salt': salt_g
        }
        
        for nutrient, value in nutrients.items():
            if value is not None:
                score = self.classify_nutrient(value, nutrient, is_drink=is_drink)
                results[nutrient] = score
                results['per_100g'][nutrient] = {
                    'value': value,
                    'score': score,
                    'unit': 'g per 100g' if not is_drink else 'g per 100ml'
                }
        
        # Score per serving if serving weight provided
        if serving_weight_g is not None and serving_weight_g > 0:
            serving_factor = serving_weight_g / 100.0
            
            for nutrient, value in nutrients.items():
                if value is not None:
                    serving_value = value * serving_factor
                    serving_score = self.classify_nutrient(
                        serving_value, nutrient, 
                        is_drink=is_drink, per_serving=True
                    )
                    results['per_serving'][nutrient] = {
                        'value': serving_value,
                        'score': serving_score,
                        'unit': 'g per serving'
                    }
        
        # Calculate summary (worst/most restrictive score)
        scores = [results[n] for n in ['fat', 'saturates', 'sugars', 'salt'] 
                 if results[n] != 'unknown']
        
        if scores:
            # Priority: red > amber > green 
            if 'red' in scores:
                results['summary'] = 'red'
            elif 'amber' in scores:
                results['summary'] = 'amber'
            else:
                results['summary'] = 'green'
        
        return results
    
    def convert_per_serving_to_per_100g(self, 
                                       value: float, 
                                       serving_weight_g: float) -> float:
        """
        Convert per-serving value to per-100g value
        
        Args:
            value: Nutrient value per serving
            serving_weight_g: Serving weight in grams
            
        Returns:
            Value per 100g
        """
        if serving_weight_g <= 0:
            raise ValueError("Serving weight must be positive")
        return (value * 100.0) / serving_weight_g


def score_traffic_lights(df: pd.DataFrame,
                        fat_col: str = 'Fat_g',
                        saturates_col: str = 'SatFat_g', 
                        sugars_col: str = 'Sugars_g',
                        salt_col: str = 'Salt_g',
                        sodium_col: Optional[str] = 'Sodium_mg',
                        serving_weight_col: Optional[str] = 'Serving_Weight_g',
                        is_drink_col: Optional[str] = None,
                        beverage_categories: Optional[list] = None) -> pd.DataFrame:
    """
    Apply UK Traffic Light scoring to a DataFrame of products
    
    Args:
        df: DataFrame with nutrition data (values should be per 100g/ml)
        fat_col: Column name for total fat (g per 100g/ml)
        saturates_col: Column name for saturated fat (g per 100g/ml)
        sugars_col: Column name for total sugars (g per 100g/ml)
        salt_col: Column name for salt (g per 100g/ml)
        sodium_col: Column name for sodium (mg per 100g/ml), alternative to salt
        serving_weight_col: Column name for serving weight (g)
        is_drink_col: Column name indicating if product is a drink
        beverage_categories: List of categories to treat as beverages
        
    Returns:
        DataFrame with added Traffic Light score columns
    """
    result_df = df.copy()
    scorer = TrafficLightsScorer()
    
    # Default beverage categories if not provided
    if beverage_categories is None:
        beverage_categories = [
            'Beverages', 'Alcoholic beverages', 'Soft drinks', 
            'Fruit juice', 'Vegetable juice', 'Energy drinks'
        ]
    
    # Initialize output columns
    traffic_light_cols = [
        'Traffic_Lights_Fat', 'Traffic_Lights_SatFat', 
        'Traffic_Lights_Sugars', 'Traffic_Lights_Salt',
        'Traffic_Lights_Summary'
    ]
    
    for col in traffic_light_cols:
        if col not in result_df.columns:
            result_df[col] = 'unknown'
    
    # Process each row
    for idx, row in result_df.iterrows():
        # Determine if this is a drink
        is_drink = False
        if is_drink_col and is_drink_col in result_df.columns:
            is_drink = bool(row.get(is_drink_col, False))
        elif 'Super_Category' in result_df.columns:
            is_drink = row.get('Super_Category', '') in beverage_categories
        elif 'Food_Category' in result_df.columns:
            is_drink = row.get('Food_Category', '') in beverage_categories
        
        # Get nutrient values
        fat_g = row.get(fat_col) if fat_col in result_df.columns else None
        saturates_g = row.get(saturates_col) if saturates_col in result_df.columns else None
        sugars_g = row.get(sugars_col) if sugars_col in result_df.columns else None
        salt_g = row.get(salt_col) if salt_col in result_df.columns else None
        sodium_mg = row.get(sodium_col) if sodium_col and sodium_col in result_df.columns else None
        serving_weight_g = row.get(serving_weight_col) if serving_weight_col and serving_weight_col in result_df.columns else None
        
        # Score the product
        scores = scorer.score_product(
            fat_g=fat_g,
            saturates_g=saturates_g,
            sugars_g=sugars_g,
            salt_g=salt_g,
            sodium_mg=sodium_mg,
            serving_weight_g=serving_weight_g,
            is_drink=is_drink
        )
        
        # Update DataFrame with scores
        result_df.loc[idx, 'Traffic_Lights_Fat'] = scores['fat']
        result_df.loc[idx, 'Traffic_Lights_SatFat'] = scores['saturates'] 
        result_df.loc[idx, 'Traffic_Lights_Sugars'] = scores['sugars']
        result_df.loc[idx, 'Traffic_Lights_Salt'] = scores['salt']
        result_df.loc[idx, 'Traffic_Lights_Summary'] = scores['summary']
    
    return result_df


if __name__ == "__main__":
    # Example usage and testing
    
    # Create sample data
    sample_data = pd.DataFrame({
        'Food_Name': ['Apple', 'Chocolate Bar', 'Cola', 'Bread', 'Cheese'],
        'Super_Category': ['Fruit', 'Sugars preserves and snacks', 'Beverages', 
                          'Cereals and cereal products', 'Milk and milk products'],
        'Fat_g': [0.2, 31.0, 0.0, 3.2, 25.0],           # per 100g
        'SatFat_g': [0.1, 18.5, 0.0, 0.7, 16.0],        # per 100g
        'Sugars_g': [10.4, 43.0, 10.6, 4.5, 0.1],       # per 100g
        'Salt_g': [0.0, 0.02, 0.01, 1.1, 1.8],          # per 100g
        'Serving_Weight_g': [150, 50, 330, 30, 30]       # serving size
    })
    
    # Apply Traffic Light scoring
    scored_data = score_traffic_lights(sample_data)
    
    # Display results
    print("Traffic Light Scoring Results:")
    print("=" * 50)
    
    display_cols = ['Food_Name', 'Traffic_Lights_Fat', 'Traffic_Lights_SatFat', 
                   'Traffic_Lights_Sugars', 'Traffic_Lights_Salt', 'Traffic_Lights_Summary']
    
    for col in display_cols:
        if col in scored_data.columns:
            print(f"\n{col}:")
            for idx, val in scored_data[col].items():
                print(f"  {scored_data.loc[idx, 'Food_Name']}: {val}")