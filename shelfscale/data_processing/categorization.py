"""
Food categorization utilities for ShelfScale datasets
"""

import pandas as pd
import os
import logging
from typing import Dict

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FoodCategorizer:
    """Hierarchical food categorization system"""
    
    def __init__(self, mapping_file=None, mapping_dict=None):
        """
        Initialize with category mappings
        
        Args:
            mapping_file: Path to JSON/CSV file with mappings
            mapping_dict: Dictionary of mappings
        """
        self.mapping = {
            # Core food groups (standardized names)
            'vegetables': 'Vegetables',
            'vegetable': 'Vegetables',
            'fruit': 'Fruit',
            'fruits': 'Fruit',
            'meat': 'Meat and meat products',
            'meat product': 'Meat and meat products',
            'meat products': 'Meat and meat products',
            'cereal': 'Cereals',
            'cereals': 'Cereals',
            'dairy': 'Milk and milk products',
            'milk': 'Milk and milk products',
            'milk product': 'Milk and milk products',
            'milk products': 'Milk and milk products',
            'fish': 'Fish and fish products',
            'fish product': 'Fish and fish products',
            'fish products': 'Fish and fish products',
            'alcoholic': 'Alcoholic beverages',
            'alcoholic beverage': 'Alcoholic beverages',
            'alcohol': 'Alcoholic beverages',
            'nuts': 'Nuts and seeds',
            'seeds': 'Nuts and seeds',
            'nut': 'Nuts and seeds',
            'seed': 'Nuts and seeds',
            'legume': 'Legumes',
            'legumes': 'Legumes',
            'bean': 'Legumes',
            'beans': 'Legumes',
            'oil': 'Oils and fats',
            'fat': 'Oils and fats',
            'oils': 'Oils and fats',
            'fats': 'Oils and fats',
            'sugar': 'Sugars and confectionery',
            'sugars': 'Sugars and confectionery',
            'confectionery': 'Sugars and confectionery',
            'sweet': 'Sugars and confectionery',
            'sweets': 'Sugars and confectionery',
            'candy': 'Sugars and confectionery',
            'bakery': 'Bakery products',
            'baked': 'Bakery products',
            'bread': 'Bakery products',
            'pastry': 'Bakery products',
            'spice': 'Herbs and spices',
            'spices': 'Herbs and spices',
            'herb': 'Herbs and spices',
            'herbs': 'Herbs and spices',
            'beverage': 'Non-alcoholic beverages',
            'beverages': 'Non-alcoholic beverages',
            'drink': 'Non-alcoholic beverages',
            'drinks': 'Non-alcoholic beverages',
            'water': 'Non-alcoholic beverages',
            'juice': 'Non-alcoholic beverages',
            'tea': 'Non-alcoholic beverages',
            'coffee': 'Non-alcoholic beverages',
            'prepared': 'Prepared meals',
            'meal': 'Prepared meals',
            'dish': 'Prepared meals',
            'ready': 'Prepared meals',
            'snack': 'Snack foods',
            'snacks': 'Snack foods',
            'chip': 'Snack foods',
            'chips': 'Snack foods',
            'crisp': 'Snack foods',
            'crisps': 'Snack foods'
        }
        
        # Load additional mappings from file if provided
        if mapping_file:
            self._load_mappings(mapping_file)
            
        # Update with provided dictionary if available
        if mapping_dict:
            self.mapping.update(mapping_dict)
            
        # Create hierarchical mappings (super categories)
        self.hierarchy = {
            'Plant-based foods': [
                'Vegetables', 'Fruit', 'Nuts and seeds', 'Legumes', 'Cereals', 'Herbs and spices'
            ],
            'Animal-based foods': [
                'Meat and meat products', 'Fish and fish products', 'Milk and milk products', 'Eggs'
            ],
            'Processed foods': [
                'Bakery products', 'Prepared meals', 'Snack foods', 'Sugars and confectionery'
            ],
            'Beverages': [
                'Alcoholic beverages', 'Non-alcoholic beverages'
            ],
            'Other': [
                'Oils and fats', 'Supplements'
            ]
        }
        
        # Build reverse lookup for super categories
        self.super_category_lookup = {}
        for super_cat, categories in self.hierarchy.items():
            for category in categories:
                self.super_category_lookup[category] = super_cat
    
    def _load_mappings(self, file_path):
        """Load category mappings from file"""
        _, ext = os.path.splitext(file_path)
        
        if ext.lower() == '.json':
            with open(file_path, 'r') as f:
                import json
                self.mapping.update(json.load(f))
                
        elif ext.lower() == '.csv':
            mapping_df = pd.read_csv(file_path)
            # Assume CSV has 'term' and 'category' columns
            for _, row in mapping_df.iterrows():
                self.mapping[row['term'].lower()] = row['category']
    
    def categorize(self, text: str) -> str:
        """
        Categorize food item based on text description
        
        Args:
            text: Food description text
            
        Returns:
            Standardized food category
        """
        if pd.isna(text) or not isinstance(text, str):
            return "Uncategorized"
            
        text = text.lower()
        
        # Try direct matches first
        if text in self.mapping:
            return self.mapping[text]
            
        # Try substring matches
        for term, category in self.mapping.items():
            if term in text:
                return category
                
        return "Uncategorized"
    
    def get_super_category(self, category: str) -> str:
        """
        Get the super category for a given category
        
        Args:
            category: Standard food category
            
        Returns:
            Super category
        """
        return self.super_category_lookup.get(category, "Other")
    

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
    cleaned_df[new_col] = cleaned_df[text_col].apply(categorizer.categorize)
    
    # Add super categories
    cleaned_df[super_category_col] = cleaned_df[new_col].apply(categorizer.get_super_category)
    
    # Count categories
    category_counts = cleaned_df[new_col].value_counts()
    logger.info(f"Category distribution: {category_counts.to_dict()}")
    
    # Check uncategorized
    uncategorized_count = category_counts.get("Uncategorized", 0)
    if uncategorized_count > 0:
        uncategorized_pct = (uncategorized_count / len(cleaned_df)) * 100
        logger.warning(f"{uncategorized_count} items ({uncategorized_pct:.2f}%) could not be categorized")
    
    return cleaned_df