"""
Food-specific preprocessing utilities for ShelfScale datasets
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Optional, Union, Tuple, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FoodTextNormalizer:
    """Class for normalizing food text descriptions"""
    
    def __init__(self):
        """Initialize the food text normalizer with domain-specific rules"""
        # Common food preparation terms and their normalized forms
        self.prep_terms = {
            'raw': 'raw',
            'fresh': 'raw',
            'uncooked': 'raw',
            'cooked': 'cooked',
            'boiled': 'cooked',
            'fried': 'cooked',
            'roasted': 'cooked',
            'baked': 'cooked',
            'grilled': 'cooked',
            'steamed': 'cooked',
            'sauteed': 'cooked',
            'sautéed': 'cooked',
            'frozen': 'frozen',
            'freeze': 'frozen',
            'freezing': 'frozen',
            'dried': 'dried',
            'dry': 'dried',
            'dehydrated': 'dried',
            'canned': 'canned',
            'tinned': 'canned',
            'in water': 'canned',
            'in brine': 'canned',
            'in oil': 'canned',
            'in syrup': 'canned',
            'processed': 'processed',
            'preserved': 'processed',
            'whole': 'whole',
            'intact': 'whole',
            'unpeeled': 'whole',
            'peeled': 'peeled',
            'skinless': 'peeled',
            'skin removed': 'peeled',
            'sliced': 'sliced',
            'diced': 'sliced',
            'chopped': 'sliced',
            'cubed': 'sliced',
            'pieces': 'sliced'
        }
        
        # Common unit terms and their normalized forms
        self.unit_terms = {
            'g': 'g',
            'gram': 'g',
            'grams': 'g',
            'kg': 'kg',
            'kilogram': 'kg',
            'kilograms': 'kg',
            'mg': 'mg',
            'milligram': 'mg',
            'milligrams': 'mg',
            'ml': 'ml',
            'milliliter': 'ml',
            'millilitre': 'ml',
            'milliliters': 'ml',
            'millilitres': 'ml',
            'l': 'l',
            'liter': 'l',
            'litre': 'l',
            'liters': 'l',
            'litres': 'l',
            'oz': 'oz',
            'ounce': 'oz',
            'ounces': 'oz',
            'lb': 'lb',
            'pound': 'lb',
            'pounds': 'lb',
            'cup': 'cup',
            'cups': 'cup',
            'tbsp': 'tbsp',
            'tablespoon': 'tbsp',
            'tablespoons': 'tbsp',
            'tsp': 'tsp',
            'teaspoon': 'tsp',
            'teaspoons': 'tsp',
            'piece': 'piece',
            'pieces': 'piece',
            'pack': 'pack',
            'packet': 'pack',
            'packets': 'pack',
            'packs': 'pack'
        }
        
        # Common food group terms and their normalized forms
        self.food_group_terms = {
            'fruit': 'fruits',
            'fruits': 'fruits',
            'vegetable': 'vegetables',
            'vegetables': 'vegetables',
            'veg': 'vegetables',
            'meat': 'meat',
            'meats': 'meat',
            'poultry': 'poultry',
            'fish': 'fish',
            'seafood': 'fish',
            'dairy': 'dairy',
            'milk': 'dairy',
            'cheese': 'dairy',
            'grain': 'grains',
            'grains': 'grains',
            'cereal': 'grains',
            'cereals': 'grains',
            'bread': 'grains',
            'pasta': 'grains',
            'rice': 'grains',
            'nut': 'nuts',
            'nuts': 'nuts',
            'seed': 'nuts',
            'seeds': 'nuts',
            'legume': 'legumes',
            'legumes': 'legumes',
            'bean': 'legumes',
            'beans': 'legumes',
            'sweet': 'sweets',
            'sweets': 'sweets',
            'dessert': 'sweets',
            'desserts': 'sweets',
            'confectionery': 'sweets',
            'snack': 'snacks',
            'snacks': 'snacks',
            'beverage': 'beverages',
            'beverages': 'beverages',
            'drink': 'beverages',
            'drinks': 'beverages'
        }
        
        # Common ingredient terms for matching
        self.ingredient_terms = set([
            'apple', 'banana', 'beef', 'bread', 'broccoli', 'butter', 'carrot', 'cheese', 'chicken', 
            'chocolate', 'cinnamon', 'cod', 'corn', 'cucumber', 'egg', 'fish', 'flour', 'garlic', 
            'ginger', 'grapes', 'honey', 'lamb', 'lemon', 'lettuce', 'milk', 'mushroom', 'mustard', 
            'oats', 'oil', 'olive', 'onion', 'orange', 'pasta', 'peanut', 'pepper', 'pork', 
            'potato', 'rice', 'salmon', 'salt', 'spinach', 'strawberry', 'sugar', 'tomato', 
            'tuna', 'vanilla', 'vinegar', 'wheat', 'yogurt'
        ])
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize food text description
        
        Args:
            text: Food item text description
            
        Returns:
            Normalized text
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        normalized = text.lower()
        
        # Replace special characters with spaces
        normalized = re.sub(r'[^\w\s]', ' ', normalized)
        
        # Replace multiple spaces with a single space
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Remove leading/trailing spaces
        normalized = normalized.strip()
        
        return normalized
    
    def extract_preparation_state(self, text: str) -> List[str]:
        """
        Extract food preparation state from text
        
        Args:
            text: Food item text description
            
        Returns:
            List of preparation states
        """
        if not isinstance(text, str):
            return []
        
        text = text.lower()
        states = []
        
        # Check for each preparation term
        for term, normalized in self.prep_terms.items():
            if term in text:
                states.append(normalized)
        
        # Remove duplicates while preserving order
        unique_states = []
        for state in states:
            if state not in unique_states:
                unique_states.append(state)
        
        return unique_states
    
    def extract_units(self, text: str) -> Tuple[str, float]:
        """
        Extract unit and amount from text
        
        Args:
            text: Food item text description
            
        Returns:
            Tuple of (unit, amount)
        """
        if not isinstance(text, str):
            return ("", 0.0)
        
        text = text.lower()
        
        # Define regex patterns for different formats
        patterns = [
            # Simple pattern: "100g" or "100 g"
            r'(\d+(?:\.\d+)?)\s*(g|kg|ml|l|mg|oz|lb)\b',
            
            # Multipack pattern: "3 x 100g"
            r'(\d+)\s*x\s*(\d+(?:\.\d+)?)\s*(g|kg|ml|l|mg|oz|lb)\b',
            
            # Pack pattern: "6pk x 25g"
            r'(\d+)\s*(?:pk|pack)s?\s*(?:x\s*)?(\d+(?:\.\d+)?)\s*(g|kg|ml|l|mg|oz|lb)\b',
            
            # Range pattern: "100-150g" (take average)
            r'(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)\s*(g|kg|ml|l|mg|oz|lb)\b'
        ]
        
        # Try each pattern
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                if len(match.groups()) == 2:
                    # Simple pattern
                    amount = float(match.group(1))
                    unit = match.group(2)
                    return (self.normalize_unit(unit), amount)
                elif len(match.groups()) == 3:
                    if 'x' in match.group(0) or 'pk' in match.group(0) or 'pack' in match.group(0):
                        # Multipack or pack pattern
                        multiplier = float(match.group(1))
                        amount = float(match.group(2))
                        unit = match.group(3)
                        return (self.normalize_unit(unit), multiplier * amount)
                    else:
                        # Range pattern
                        min_val = float(match.group(1))
                        max_val = float(match.group(2))
                        unit = match.group(3)
                        return (self.normalize_unit(unit), (min_val + max_val) / 2)
        
        # Check for unit terms
        for term, normalized in self.unit_terms.items():
            if term in text:
                # Try to find a number before the term
                pattern = r'(\d+(?:\.\d+)?)\s*' + re.escape(term) + r'\b'
                match = re.search(pattern, text)
                if match:
                    return (normalized, float(match.group(1)))
        
        return ("", 0.0)
    
    def normalize_unit(self, unit: str) -> str:
        """
        Normalize unit to standard form
        
        Args:
            unit: Unit string
            
        Returns:
            Normalized unit
        """
        if not unit:
            return ""
        
        unit = unit.lower()
        
        # Check if unit is already in normalized form
        if unit in self.unit_terms.values():
            return unit
        
        # Check if unit is in our mapping
        if unit in self.unit_terms:
            return self.unit_terms[unit]
        
        # Default to original unit
        return unit
    
    def extract_food_group(self, text: str) -> str:
        """
        Extract food group from text
        
        Args:
            text: Food item text description
            
        Returns:
            Food group
        """
        if not isinstance(text, str):
            return ""
        
        text = text.lower()
        
        # Check for each food group term
        for term, normalized in self.food_group_terms.items():
            if term in text.split():  # Match whole words only
                return normalized
        
        return ""
    
    def extract_ingredients(self, text: str) -> List[str]:
        """
        Extract ingredients from text
        
        Args:
            text: Food item text description
            
        Returns:
            List of ingredients
        """
        if not isinstance(text, str):
            return []
        
        text = text.lower()
        words = set(text.split())
        
        # Find ingredients
        ingredients = []
        for word in words:
            if word in self.ingredient_terms:
                ingredients.append(word)
        
        return ingredients


class FoodPortionSizeProcessor:
    """Class for processing Food Portion Sizes data"""
    
    def __init__(self):
        """Initialize the Food Portion Size processor"""
        self.normalizer = FoodTextNormalizer()
    
    def process_portion_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process Food Portion Sizes data
        
        Args:
            df: Food Portion Sizes DataFrame
            
        Returns:
            Processed DataFrame
        """
        # Create a copy to avoid modifying the original
        processed = df.copy()
        
        # Normalize food names
        if 'Food_Name' in processed.columns:
            processed['Food_Name_Normalized'] = processed['Food_Name'].apply(self.normalizer.normalize_text)
        
        # Extract preparation states
        if 'Food_Name' in processed.columns:
            processed['Preparation_State'] = processed['Food_Name'].apply(self.normalizer.extract_preparation_state)
            # Convert list to string for easier handling
            processed['Preparation_State'] = processed['Preparation_State'].apply(lambda x: ', '.join(x) if x else '')
        
        # Extract food groups
        if 'Food_Name' in processed.columns:
            processed['Food_Group'] = processed['Food_Name'].apply(self.normalizer.extract_food_group)
        
        # Process portion descriptions
        if 'Portion_Description' in processed.columns:
            # Normalize portion descriptions
            processed['Portion_Description_Normalized'] = processed['Portion_Description'].apply(self.normalizer.normalize_text)
            
            # Extract units and amounts
            unit_amount = processed['Portion_Description'].apply(self.normalizer.extract_units)
            processed['Extracted_Unit'] = [ua[0] for ua in unit_amount]
            processed['Extracted_Amount'] = [ua[1] for ua in unit_amount]
        
        # Process weights
        if 'Weight_g' in processed.columns:
            # Convert to numeric
            processed['Weight_g'] = pd.to_numeric(processed['Weight_g'], errors='coerce')
            
            # Fill missing weights with extracted amounts if units match
            mask = (processed['Weight_g'].isna() & 
                   (processed['Extracted_Unit'] == 'g') & 
                   (processed['Extracted_Amount'] > 0))
            
            processed.loc[mask, 'Weight_g'] = processed.loc[mask, 'Extracted_Amount']
            processed.loc[mask, 'Weight_Source'] = 'Extracted'
        
        return processed


class FruitVegSurveyProcessor:
    """Class for processing Fruit & Vegetable Survey data"""
    
    def __init__(self):
        """Initialize the Fruit & Vegetable Survey processor"""
        self.normalizer = FoodTextNormalizer()
    
    def process_survey_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process Fruit & Vegetable Survey data
        
        Args:
            df: Fruit & Vegetable Survey DataFrame
            
        Returns:
            Processed DataFrame
        """
        # Create a copy to avoid modifying the original
        processed = df.copy()
        
        # Normalize sample names
        if 'Sample_Name' in processed.columns:
            processed['Sample_Name_Normalized'] = processed['Sample_Name'].apply(self.normalizer.normalize_text)
        
        # Extract preparation states
        if 'Sample_Name' in processed.columns:
            processed['Preparation_State'] = processed['Sample_Name'].apply(self.normalizer.extract_preparation_state)
            # Convert list to string for easier handling
            processed['Preparation_State'] = processed['Preparation_State'].apply(lambda x: ', '.join(x) if x else '')
        
        # Extract food groups
        if 'Sample_Name' in processed.columns:
            processed['Food_Group'] = processed['Sample_Name'].apply(self.normalizer.extract_food_group)
            
            # If Sample_Type column exists, use it to refine the food group
            if 'Sample_Type' in processed.columns:
                # Where Food_Group is empty, try to extract from Sample_Type
                mask = processed['Food_Group'] == ''
                processed.loc[mask, 'Food_Group'] = processed.loc[mask, 'Sample_Type'].apply(self.normalizer.extract_food_group)
        
        # Process pack sizes
        if 'Pack_Size' in processed.columns:
            # Normalize pack size descriptions
            processed['Pack_Size_Normalized'] = processed['Pack_Size'].apply(self.normalizer.normalize_text)
            
            # Extract units and amounts
            unit_amount = processed['Pack_Size'].apply(self.normalizer.extract_units)
            processed['Extracted_Unit'] = [ua[0] for ua in unit_amount]
            processed['Extracted_Amount'] = [ua[1] for ua in unit_amount]
            
            # Create Weight_g column from extracted amounts
            processed['Weight_g'] = np.nan
            
            # Fill weights based on unit
            for unit, multiplier in [('g', 1), ('kg', 1000), ('mg', 0.001)]:
                mask = processed['Extracted_Unit'] == unit
                processed.loc[mask, 'Weight_g'] = processed.loc[mask, 'Extracted_Amount'] * multiplier
            
            # Mark source
            processed.loc[processed['Weight_g'].notna(), 'Weight_Source'] = 'Extracted'
        
        return processed


class McCanceWiddowsonProcessor:
    """Class for processing McCance & Widdowson data"""
    
    def __init__(self):
        """Initialize the McCance & Widdowson processor"""
        self.normalizer = FoodTextNormalizer()
    
    def process_mw_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process McCance & Widdowson data
        
        Args:
            df: McCance & Widdowson DataFrame
            
        Returns:
            Processed DataFrame
        """
        # Create a copy to avoid modifying the original
        processed = df.copy()
        
        # Normalize food names
        food_name_col = next((col for col in processed.columns if 'food name' in col.lower()), None)
        if food_name_col:
            processed['Food_Name_Normalized'] = processed[food_name_col].apply(self.normalizer.normalize_text)
        
        # Extract preparation states
        if food_name_col:
            processed['Preparation_State'] = processed[food_name_col].apply(self.normalizer.extract_preparation_state)
            # Convert list to string for easier handling
            processed['Preparation_State'] = processed['Preparation_State'].apply(lambda x: ', '.join(x) if x else '')
        
        # Process food groups
        food_group_col = next((col for col in processed.columns if 'food group' in col.lower() or 'super group' in col.lower()), None)
        if not food_group_col:
            # If no food group column, try to extract from food name
            if food_name_col:
                processed['Food_Group'] = processed[food_name_col].apply(self.normalizer.extract_food_group)
        else:
            # Normalize existing food group values
            processed['Food_Group'] = processed[food_group_col].apply(
                lambda x: self.normalizer.food_group_terms.get(str(x).lower(), x) if pd.notna(x) else ""
            )
        
        # Process food codes
        food_code_col = next((col for col in processed.columns if 'food code' in col.lower()), None)
        if food_code_col:
            # Ensure food codes are strings
            processed['Food_Code'] = processed[food_code_col].astype(str)
        
        # Extract ingredients
        if food_name_col:
            processed['Ingredients'] = processed[food_name_col].apply(self.normalizer.extract_ingredients)
            # Convert list to string for easier handling
            processed['Ingredients'] = processed['Ingredients'].apply(lambda x: ', '.join(x) if x else '')
        
        return processed


def create_integrated_dataset(data_sources: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Create an integrated dataset from multiple sources with domain-specific preprocessing
    
    Args:
        data_sources: Dictionary of DataFrames from different sources
        
    Returns:
        Integrated DataFrame
    """
    # Check if we have any data sources
    if not data_sources:
        logger.warning("No data sources provided")
        return pd.DataFrame()
    
    # Initialize processors
    mw_processor = McCanceWiddowsonProcessor()
    fps_processor = FoodPortionSizeProcessor()
    fvs_processor = FruitVegSurveyProcessor()
    
    processed_sources = {}
    
    # Process each data source
    for source_name, df in data_sources.items():
        if df.empty:
            continue
            
        if 'mw' in source_name.lower():
            # McCance & Widdowson data
            processed_sources[source_name] = mw_processor.process_mw_data(df)
        elif 'portion' in source_name.lower():
            # Food Portion Sizes data
            processed_sources[source_name] = fps_processor.process_portion_data(df)
        elif 'fruit' in source_name.lower() or 'veg' in source_name.lower():
            # Fruit & Vegetable Survey data
            processed_sources[source_name] = fvs_processor.process_survey_data(df)
        else:
            # Unknown source, just add as is
            processed_sources[source_name] = df
    
    # Start with McCance & Widdowson as the base
    base_source = next((s for s in processed_sources.keys() if 'mw' in s.lower()), None)
    
    if not base_source:
        # If no MW data, use the first available source
        if processed_sources:
            base_source = list(processed_sources.keys())[0]
        else:
            logger.warning("No processed data sources available")
            return pd.DataFrame()
    
    # Use the base source as our starting point
    integrated = processed_sources[base_source].copy()
    logger.info(f"Using {base_source} as base dataset with {len(integrated)} items")
    
    # Add a source column
    integrated['Data_Source'] = base_source
    
    # Return the integrated dataset
    return integrated


if __name__ == "__main__":
    # Test the module
    normalizer = FoodTextNormalizer()
    
    test_text = "Fresh apple, 100g pack"
    print(f"Original: {test_text}")
    print(f"Normalized: {normalizer.normalize_text(test_text)}")
    print(f"Preparation: {normalizer.extract_preparation_state(test_text)}")
    print(f"Units: {normalizer.extract_units(test_text)}")
    print(f"Food Group: {normalizer.extract_food_group(test_text)}")
    print(f"Ingredients: {normalizer.extract_ingredients(test_text)}") 