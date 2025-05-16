"""
Food categorization utilities for ShelfScale datasets
"""

import pandas as pd
import os
import logging
from typing import Dict, Tuple
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FoodCategorizer:
    """Categorizes food products based on their descriptions"""
    
    def __init__(self, food_categories_path: str = None):
        """
        Initialize the food categorizer
        
        Args:
            food_categories_path: Path to food categories CSV file
        """
        self.food_categories = self._load_default_categories()
        
        # Load custom categories if provided
        if food_categories_path and os.path.exists(food_categories_path):
            try:
                custom_categories = pd.read_csv(food_categories_path)
                self.food_categories.update(
                    {row['term']: (row['category'], row['super_category']) 
                     for _, row in custom_categories.iterrows()}
                )
            except Exception as e:
                logger.error(f"Failed to load custom food categories: {e}")
    
    def clean_food_categories(self, df: pd.DataFrame, text_col: str, 
                           category_col: str = 'Food_Category', 
                           super_category_col: str = 'Super_Category') -> pd.DataFrame:
        """
        Categorize foods in a DataFrame based on text descriptions
        
        Args:
            df: Input DataFrame
            text_col: Column containing food descriptions
            category_col: Output column for food categories
            super_category_col: Output column for super categories
            
        Returns:
            DataFrame with added category columns
        """
        if text_col not in df.columns:
            raise ValueError(f"Column '{text_col}' not found in DataFrame")
            
        # Make a copy to avoid modifying the original
        categorized_df = df.copy()
        
        # Initialize category columns
        categorized_df[category_col] = 'Unknown'
        categorized_df[super_category_col] = 'Unknown'
        
        # Apply categorization to each food item
        for idx, row in categorized_df.iterrows():
            food_name = str(row[text_col]) if not pd.isna(row[text_col]) else ""
            category, super_category = self._categorize_food_item(food_name)
            categorized_df.loc[idx, category_col] = category
            categorized_df.loc[idx, super_category_col] = super_category
            
        return categorized_df
    
    def _categorize_food_item(self, food_name: str) -> Tuple[str, str]:
        """
        Categorize a single food item based on its name
        
        Args:
            food_name: Food item name or description
            
        Returns:
            Tuple of (category, super_category)
        """
        # Default if no match is found
        default_result = ("Other", "Miscellaneous")
        
        if not food_name or pd.isna(food_name):
            return default_result
            
        # Convert to lowercase for matching
        food_name_lower = food_name.lower()
        
        # Try to match with category terms
        for term, (category, super_category) in self.food_categories.items():
            if term.lower() in food_name_lower:
                return category, super_category
        
        # Fall back to broader categories using word-based matching
        food_words = set(re.findall(r'\b\w+\b', food_name_lower))
        
        if any(word in food_words for word in ['fruit', 'apple', 'banana', 'orange', 'berry', 'grape']):
            return "Fruits", "Produce"
        
        if any(word in food_words for word in ['vegetable', 'veg', 'carrot', 'broccoli', 'lettuce']):
            return "Vegetables", "Produce"
        
        if any(word in food_words for word in ['meat', 'beef', 'pork', 'lamb', 'chicken', 'turkey']):
            return "Meat", "Animal Products"
        
        if any(word in food_words for word in ['fish', 'salmon', 'tuna', 'cod', 'seafood', 'shellfish']):
            return "Seafood", "Animal Products"
        
        if any(word in food_words for word in ['dairy', 'milk', 'cheese', 'yogurt', 'butter', 'cream']):
            return "Dairy", "Animal Products"
        
        if any(word in food_words for word in ['bread', 'loaf', 'roll', 'bun', 'bagel', 'pastry']):
            return "Breads & Bakery", "Grains & Starches"
        
        if any(word in food_words for word in ['pasta', 'noodle', 'spaghetti', 'rice', 'grain']):
            return "Pasta & Rice", "Grains & Starches"
        
        if any(word in food_words for word in ['snack', 'crisp', 'chip', 'cracker', 'cookie']):
            return "Snacks", "Processed Foods"
        
        if any(word in food_words for word in ['sweet', 'candy', 'chocolate', 'dessert']):
            return "Sweets & Desserts", "Processed Foods"
        
        if any(word in food_words for word in ['beverage', 'drink', 'juice', 'water', 'soda', 'tea', 'coffee']):
            return "Beverages", "Drinks"
        
        if any(word in food_words for word in ['alcohol', 'wine', 'beer', 'spirit', 'liquor']):
            return "Alcoholic Beverages", "Drinks"
        
        # No specific match found
        return default_result
    
    def _load_default_categories(self) -> Dict[str, Tuple[str, str]]:
        """
        Load default food categories
            
        Returns:
            Dictionary mapping food terms to category and super_category
        """
        return {
            # Fruits
            'apple': ('Fruits', 'Produce'),
            'banana': ('Fruits', 'Produce'),
            'orange': ('Fruits', 'Produce'),
            'grape': ('Fruits', 'Produce'),
            'berry': ('Fruits', 'Produce'),
            'strawberry': ('Fruits', 'Produce'),
            'blueberry': ('Fruits', 'Produce'),
            'raspberry': ('Fruits', 'Produce'),
            'melon': ('Fruits', 'Produce'),
            'pineapple': ('Fruits', 'Produce'),
            'citrus': ('Fruits', 'Produce'),
            'fruit': ('Fruits', 'Produce'),
            
            # Vegetables
            'vegetable': ('Vegetables', 'Produce'),
            'carrot': ('Vegetables', 'Produce'),
            'broccoli': ('Vegetables', 'Produce'),
            'lettuce': ('Vegetables', 'Produce'),
            'spinach': ('Vegetables', 'Produce'),
            'potato': ('Vegetables', 'Produce'),
            'tomato': ('Vegetables', 'Produce'),
            'onion': ('Vegetables', 'Produce'),
            'pepper': ('Vegetables', 'Produce'),
            'garlic': ('Vegetables', 'Produce'),
            'cucumber': ('Vegetables', 'Produce'),
            'veg': ('Vegetables', 'Produce'),
            
            # Meat
            'beef': ('Meat', 'Animal Products'),
            'pork': ('Meat', 'Animal Products'),
            'lamb': ('Meat', 'Animal Products'),
            'chicken': ('Meat', 'Animal Products'),
            'turkey': ('Meat', 'Animal Products'),
            'meat': ('Meat', 'Animal Products'),
            'sausage': ('Meat', 'Animal Products'),
            'bacon': ('Meat', 'Animal Products'),
            'ham': ('Meat', 'Animal Products'),
            'steak': ('Meat', 'Animal Products'),
            'burger': ('Meat', 'Animal Products'),
            
            # Seafood
            'fish': ('Seafood', 'Animal Products'),
            'salmon': ('Seafood', 'Animal Products'),
            'tuna': ('Seafood', 'Animal Products'),
            'cod': ('Seafood', 'Animal Products'),
            'seafood': ('Seafood', 'Animal Products'),
            'shellfish': ('Seafood', 'Animal Products'),
            'prawn': ('Seafood', 'Animal Products'),
            'shrimp': ('Seafood', 'Animal Products'),
            'crab': ('Seafood', 'Animal Products'),
            'lobster': ('Seafood', 'Animal Products'),
            
            # Dairy
            'milk': ('Dairy', 'Animal Products'),
            'cheese': ('Dairy', 'Animal Products'),
            'yogurt': ('Dairy', 'Animal Products'),
            'butter': ('Dairy', 'Animal Products'),
            'cream': ('Dairy', 'Animal Products'),
            'dairy': ('Dairy', 'Animal Products'),
            'yoghurt': ('Dairy', 'Animal Products'),
            'custard': ('Dairy', 'Animal Products'),
            
            # Breads & Bakery
            'bread': ('Breads & Bakery', 'Grains & Starches'),
            'loaf': ('Breads & Bakery', 'Grains & Starches'),
            'roll': ('Breads & Bakery', 'Grains & Starches'),
            'bun': ('Breads & Bakery', 'Grains & Starches'),
            'bagel': ('Breads & Bakery', 'Grains & Starches'),
            'pastry': ('Breads & Bakery', 'Grains & Starches'),
            'croissant': ('Breads & Bakery', 'Grains & Starches'),
            'cake': ('Breads & Bakery', 'Grains & Starches'),
            'biscuit': ('Breads & Bakery', 'Grains & Starches'),
            'cookie': ('Breads & Bakery', 'Grains & Starches'),
            
            # Pasta & Rice
            'pasta': ('Pasta & Rice', 'Grains & Starches'),
            'noodle': ('Pasta & Rice', 'Grains & Starches'),
            'spaghetti': ('Pasta & Rice', 'Grains & Starches'),
            'rice': ('Pasta & Rice', 'Grains & Starches'),
            'grain': ('Pasta & Rice', 'Grains & Starches'),
            'cereal': ('Pasta & Rice', 'Grains & Starches'),
            'oat': ('Pasta & Rice', 'Grains & Starches'),
            'wheat': ('Pasta & Rice', 'Grains & Starches'),
            
            # Snacks
            'snack': ('Snacks', 'Processed Foods'),
            'crisp': ('Snacks', 'Processed Foods'),
            'chip': ('Snacks', 'Processed Foods'),
            'cracker': ('Snacks', 'Processed Foods'),
            'popcorn': ('Snacks', 'Processed Foods'),
            'pretzel': ('Snacks', 'Processed Foods'),
            'nut': ('Snacks', 'Processed Foods'),
            
            # Sweets & Desserts
            'sweet': ('Sweets & Desserts', 'Processed Foods'),
            'candy': ('Sweets & Desserts', 'Processed Foods'),
            'chocolate': ('Sweets & Desserts', 'Processed Foods'),
            'dessert': ('Sweets & Desserts', 'Processed Foods'),
            'ice cream': ('Sweets & Desserts', 'Processed Foods'),
            'pudding': ('Sweets & Desserts', 'Processed Foods'),
            'pie': ('Sweets & Desserts', 'Processed Foods'),
            'tart': ('Sweets & Desserts', 'Processed Foods'),
            
            # Beverages
            'beverage': ('Beverages', 'Drinks'),
            'drink': ('Beverages', 'Drinks'),
            'juice': ('Beverages', 'Drinks'),
            'water': ('Beverages', 'Drinks'),
            'soda': ('Beverages', 'Drinks'),
            'tea': ('Beverages', 'Drinks'),
            'coffee': ('Beverages', 'Drinks'),
            'squash': ('Beverages', 'Drinks'),
            'cordial': ('Beverages', 'Drinks'),
            'lemonade': ('Beverages', 'Drinks'),
            
            # Alcoholic Beverages
            'alcohol': ('Alcoholic Beverages', 'Drinks'),
            'wine': ('Alcoholic Beverages', 'Drinks'),
            'beer': ('Alcoholic Beverages', 'Drinks'),
            'spirit': ('Alcoholic Beverages', 'Drinks'),
            'liquor': ('Alcoholic Beverages', 'Drinks'),
            'whisky': ('Alcoholic Beverages', 'Drinks'),
            'vodka': ('Alcoholic Beverages', 'Drinks'),
            'rum': ('Alcoholic Beverages', 'Drinks'),
            'gin': ('Alcoholic Beverages', 'Drinks'),
            
            # Condiments & Sauces
            'sauce': ('Condiments & Sauces', 'Processed Foods'),
            'condiment': ('Condiments & Sauces', 'Processed Foods'),
            'ketchup': ('Condiments & Sauces', 'Processed Foods'),
            'mustard': ('Condiments & Sauces', 'Processed Foods'),
            'mayonnaise': ('Condiments & Sauces', 'Processed Foods'),
            'dressing': ('Condiments & Sauces', 'Processed Foods'),
            'gravy': ('Condiments & Sauces', 'Processed Foods'),
            'spice': ('Condiments & Sauces', 'Processed Foods'),
            'herb': ('Condiments & Sauces', 'Processed Foods'),
            
            # Prepared Foods
            'soup': ('Prepared Foods', 'Processed Foods'),
            'stew': ('Prepared Foods', 'Processed Foods'),
            'ready meal': ('Prepared Foods', 'Processed Foods'),
            'frozen meal': ('Prepared Foods', 'Processed Foods'),
            'pizza': ('Prepared Foods', 'Processed Foods'),
            'sandwich': ('Prepared Foods', 'Processed Foods'),
            'wrap': ('Prepared Foods', 'Processed Foods'),
            'salad': ('Prepared Foods', 'Processed Foods'),
        }

def clean_food_categories(
    df: pd.DataFrame, 
    text_col: str, 
    new_col: str = 'Food_Category',
    super_category_col: str = 'Super_Category',
    categorizer: FoodCategorizer = None
) -> pd.DataFrame:
    """
    Clean and standardize food categories with hierarchical categorization
    
    Args:
        df: Input DataFrame
        text_col: Column with text to categorize
        new_col: Column name for standardized categories
        super_category_col: Column name for super categories
        categorizer: Custom FoodCategorizer instance
        
    Returns:
        DataFrame with standardized food categories
    """
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found in DataFrame")
        
    # Create a copy
    cleaned_df = df.copy()
    
    # Create categorizer if not provided
    if categorizer is None:
        categorizer = FoodCategorizer()
    
    # Apply categorization
    cleaned_df = categorizer.clean_food_categories(cleaned_df, text_col, new_col, super_category_col)
    
    # Count categories
    category_counts = cleaned_df[new_col].value_counts()
    logger.info(f"Category distribution: {category_counts.to_dict()}")
    
    # Check uncategorized
    uncategorized_count = category_counts.get("Unknown", 0)
    if uncategorized_count > 0:
        uncategorized_pct = (uncategorized_count / len(cleaned_df)) * 100
        logger.warning(f"{uncategorized_count} items ({uncategorized_pct:.2f}%) could not be categorized")
    
    return cleaned_df