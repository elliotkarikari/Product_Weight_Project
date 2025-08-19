"""
Enhanced text preprocessing for AI-powered food product matching

This module provides advanced text cleaning, normalization, and feature extraction
capabilities that go far beyond the original clean_composite_name() function.
"""

import re
import logging
import unicodedata
from typing import Dict, List, Optional, Tuple, Set, Union
import pandas as pd

logger = logging.getLogger(__name__)


class EnhancedTextPreprocessor:
    """
    Advanced text preprocessor for food product names with AI-enhanced features
    
    Features:
    - Comprehensive cleaning and normalization
    - Brand extraction and standardization
    - Unit and weight standardization
    - Category-specific preprocessing rules
    - Multi-language support
    - Deterministic results for reproducibility
    """
    
    def __init__(self):
        """Initialize the preprocessor with comprehensive rule sets"""
        self.setup_patterns()
        self.setup_brand_mappings()
        self.setup_unit_mappings()
        self.setup_category_rules()
        
    def setup_patterns(self):
        """Set up regex patterns for various cleaning operations"""
        
        # Common food processing terms to remove/standardize
        self.processing_terms = {
            r'\b(ready to eat|rte)\b': '',
            r'\b(semi-?dried|semi dried)\b': 'dried',
            r'\b(fresh steamed|steamed fresh)\b': 'steamed',
            r'\b(boiled flesh and skin|flesh and skin boiled)\b': 'boiled',
            r'\b(boiled flesh an d skin)\b': 'boiled',  # typo fix
            r'\b(raw flesh only|flesh only raw)\b': 'raw',
            r'\b(fresh boiled|boiled fresh)\b': 'boiled',
            r'\b(stewed without sugar|without sugar stewed)\b': 'stewed',
            r'\b(flesh flesh and skin)\b': 'flesh and skin',  # duplicate fix
            r'\b(homemade|home made|home-made)\b': '',
            r'\b(bfresh)\b': 'fresh',  # typo fix
            r'\b(retial|retail)\b': '',  # typo fix
            r'\b(canned in brine|in brine)\b': 'canned',
            r'\b(canned in oil|in oil)\b': 'canned',
            r'\b(canned in water|in water)\b': 'canned',
        }
        
        # Cooking method standardization
        self.cooking_methods = {
            r'\b(boiled|steamed|poached|simmered)\b': 'cooked',
            r'\b(grilled|roasted|baked|fried|pan-fried)\b': 'cooked',
            r'\b(raw|fresh|uncooked)\b': 'raw',
            r'\b(dried|dehydrated|freeze-dried)\b': 'dried',
            r'\b(canned|tinned|jarred|bottled)\b': 'preserved',
            r'\b(frozen|chilled|refrigerated)\b': 'frozen'
        }
        
        # Weight and size patterns
        self.weight_patterns = [
            r'\b(\d+(?:\.\d+)?)\s*(g|grams?|grammes?)\b',
            r'\b(\d+(?:\.\d+)?)\s*(kg|kilograms?|kilogrammes?)\b', 
            r'\b(\d+(?:\.\d+)?)\s*(ml|millilitres?|milliliters?)\b',
            r'\b(\d+(?:\.\d+)?)\s*(l|litres?|liters?)\b',
            r'\b(\d+(?:\.\d+)?)\s*(oz|ounces?)\b',
            r'\b(\d+(?:\.\d+)?)\s*(lb|lbs|pounds?)\b'
        ]
        
        # Brand detection patterns
        self.brand_indicators = [
            r'\b(brand|trademark|tm|®|©)\b',
            r'\b(ltd|limited|inc|incorporated|corp|corporation)\b',
            r'\b(co\.?|company)\b'
        ]
        
        # Packaging/format indicators
        self.packaging_terms = {
            r'\b(canned?|tinned?|jarred?|bottled?)\b': 'canned',
            r'\b(bagged?|packet|sachet)\b': 'packaged',
            r'\b(fresh|raw|unprocessed)\b': 'fresh',
            r'\b(frozen|chilled)\b': 'frozen',
            r'\b(dried|dehydrated)\b': 'dried'
        }
        
    def setup_brand_mappings(self):
        """Set up common brand name mappings and corrections"""
        self.brand_mappings = {
            # Common brand variations
            'sainsburys': "sainsbury's",
            'sainsbury': "sainsbury's", 
            'tescos': 'tesco',
            'asdas': 'asda',
            'morrisons': 'morrison',
            # Add more as needed
        }
        
        # Known food brands (for extraction)
        self.known_brands = {
            'tesco', "sainsbury's", 'asda', 'morrisons', 'waitrose',
            'marks & spencer', 'm&s', 'iceland', 'aldi', 'lidl',
            'heinz', 'nestle', 'unilever', 'kellogg', 'coca-cola',
            'pepsi', 'cadbury', 'mars', 'ferrero'
        }
        
    def setup_unit_mappings(self):
        """Set up unit standardization mappings"""
        self.unit_mappings = {
            # Weight units
            'grams': 'g', 'gram': 'g', 'grammes': 'g', 'gramme': 'g',
            'kilograms': 'kg', 'kilogram': 'kg', 'kilogrammes': 'kg', 'kilogramme': 'kg',
            'ounces': 'oz', 'ounce': 'oz',
            'pounds': 'lb', 'pound': 'lb', 'lbs': 'lb',
            
            # Volume units  
            'millilitres': 'ml', 'millilitre': 'ml', 'milliliters': 'ml', 'milliliter': 'ml',
            'litres': 'l', 'litre': 'l', 'liters': 'l', 'liter': 'l',
            'pints': 'pint', 'fluid ounces': 'fl oz', 'fluid ounce': 'fl oz',
            
            # Count units
            'pieces': 'pcs', 'piece': 'pc',
            'items': 'item', 'units': 'unit'
        }
        
    def setup_category_rules(self):
        """Set up category-specific preprocessing rules"""
        self.category_rules = {
            'fruit': {
                'standardize': {
                    r'\b(apple|apples)\b': 'apple',
                    r'\b(banana|bananas)\b': 'banana',
                    r'\b(orange|oranges)\b': 'orange',
                },
                'remove': [r'\b(class\s+[1-9]|grade\s+[a-z])\b']
            },
            'vegetables': {
                'standardize': {
                    r'\b(potato|potatoes|spud|spuds)\b': 'potato',
                    r'\b(carrot|carrots)\b': 'carrot',
                    r'\b(onion|onions)\b': 'onion',
                },
                'remove': [r'\b(organic|bio|eco)\b']
            },
            'meat': {
                'standardize': {
                    r'\b(chicken|poultry)\b': 'chicken',
                    r'\b(beef|cow)\b': 'beef',
                    r'\b(pork|pig)\b': 'pork',
                },
                'remove': [r'\b(free range|free-range|outdoor bred)\b']
            },
            'dairy': {
                'standardize': {
                    r'\b(milk|dairy)\b': 'milk',
                    r'\b(cheese|cheddar|brie)\b': 'cheese',
                    r'\b(yogurt|yoghurt)\b': 'yogurt',
                },
                'remove': [r'\b(full fat|low fat|skimmed|semi-skimmed)\b']
            }
        }
        
    def normalize_unicode(self, text: str) -> str:
        """Normalize unicode characters and remove accents"""
        if not isinstance(text, str):
            return ""
            
        # Normalize unicode
        text = unicodedata.normalize('NFKD', text)
        
        # Remove accents
        text = ''.join(c for c in text if not unicodedata.combining(c))
        
        return text
        
    def extract_weights_and_sizes(self, text: str) -> Tuple[str, List[Dict[str, Union[float, str]]]]:
        """
        Extract weight/size information from text
        
        Returns:
            Tuple of (cleaned_text, list_of_extracted_weights)
        """
        weights = []
        cleaned_text = text
        
        for pattern in self.weight_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                value = float(match.group(1))
                unit = match.group(2).lower()
                
                # Standardize unit
                if unit in self.unit_mappings:
                    unit = self.unit_mappings[unit]
                    
                weights.append({
                    'value': value,
                    'unit': unit,
                    'original': match.group(0)
                })
                
                # Remove from text
                cleaned_text = cleaned_text.replace(match.group(0), ' ')
                
        return cleaned_text, weights
        
    def extract_brands(self, text: str) -> Tuple[str, List[str]]:
        """
        Extract brand names from text
        
        Returns:
            Tuple of (cleaned_text, list_of_brands)
        """
        brands = []
        cleaned_text = text.lower()
        
        # Check for known brands
        for brand in self.known_brands:
            pattern = r'\b' + re.escape(brand) + r'\b'
            if re.search(pattern, cleaned_text, re.IGNORECASE):
                brands.append(brand)
                cleaned_text = re.sub(pattern, ' ', cleaned_text, flags=re.IGNORECASE)
                
        # Apply brand mappings
        for old, new in self.brand_mappings.items():
            if old in cleaned_text:
                cleaned_text = cleaned_text.replace(old, new)
                if new not in brands:
                    brands.append(new)
                    
        return cleaned_text, brands
        
    def apply_category_rules(self, text: str, category: str) -> str:
        """Apply category-specific preprocessing rules"""
        if not category or category.lower() not in self.category_rules:
            return text
            
        rules = self.category_rules[category.lower()]
        
        # Apply standardizations
        if 'standardize' in rules:
            for pattern, replacement in rules['standardize'].items():
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
                
        # Remove category-specific terms
        if 'remove' in rules:
            for pattern in rules['remove']:
                text = re.sub(pattern, ' ', text, flags=re.IGNORECASE)
                
        return text
        
    def clean_comprehensive(self, text: str, category: str = None) -> str:
        """
        Comprehensive text cleaning with all preprocessing steps
        
        Args:
            text: Input text to clean
            category: Optional food category for category-specific rules
            
        Returns:
            Cleaned and normalized text
        """
        if pd.isna(text) or not isinstance(text, str):
            return ""
            
        logger.debug(f"Cleaning text: '{text}' (category: {category})")
        
        # 1. Unicode normalization
        text = self.normalize_unicode(text)
        
        # 2. Basic cleaning
        text = text.lower().strip()
        
        # 3. Remove weights/sizes (we extract them separately)
        text, _ = self.extract_weights_and_sizes(text)
        
        # 4. Remove brands (we extract them separately) 
        text, _ = self.extract_brands(text)
        
        # 5. Apply processing term standardization
        for pattern, replacement in self.processing_terms.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
            
        # 6. Apply cooking method standardization
        for pattern, replacement in self.cooking_methods.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
            
        # 7. Apply packaging term standardization
        for pattern, replacement in self.packaging_terms.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
            
        # 8. Apply category-specific rules
        if category:
            text = self.apply_category_rules(text, category)
            
        # 9. Remove punctuation and special characters
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # 10. Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 11. Remove common stop words specific to food
        stop_words = {
            'and', 'or', 'with', 'in', 'on', 'at', 'to', 'for', 'of', 'the', 'a', 'an',
            'food', 'product', 'item', 'pack', 'package', 'portion', 'serving'
        }
        
        words = text.split()
        words = [word for word in words if word not in stop_words and len(word) > 1]
        text = ' '.join(words)
        
        logger.debug(f"Cleaned result: '{text}'")
        return text
        
    def extract_features(self, text: str, category: str = None) -> Dict[str, Union[str, List, float]]:
        """
        Extract comprehensive features from text for ML matching
        
        Returns:
            Dictionary containing:
            - cleaned_text: Fully processed text
            - original_text: Original input
            - weights: List of extracted weights/sizes
            - brands: List of extracted brands
            - word_count: Number of words
            - char_count: Number of characters
            - has_numbers: Whether text contains numbers
            - processing_type: Detected processing method
            - category: Food category
        """
        if pd.isna(text) or not isinstance(text, str):
            return self._empty_features()
            
        original_text = text
        
        # Extract components
        temp_text, weights = self.extract_weights_and_sizes(text)
        temp_text, brands = self.extract_brands(temp_text)
        
        # Get cleaned text
        cleaned_text = self.clean_comprehensive(text, category)
        
        # Detect processing type
        processing_type = self._detect_processing_type(original_text)
        
        # Calculate features
        features = {
            'cleaned_text': cleaned_text,
            'original_text': original_text,
            'weights': weights,
            'brands': brands,
            'word_count': len(cleaned_text.split()) if cleaned_text else 0,
            'char_count': len(cleaned_text),
            'has_numbers': bool(re.search(r'\d', original_text)),
            'processing_type': processing_type,
            'category': category or 'unknown',
            'has_weights': len(weights) > 0,
            'has_brands': len(brands) > 0,
            'complexity_score': self._calculate_complexity_score(original_text)
        }
        
        return features
        
    def _detect_processing_type(self, text: str) -> str:
        """Detect the primary processing type from text"""
        text_lower = text.lower()
        
        if any(term in text_lower for term in ['raw', 'fresh', 'uncooked']):
            return 'raw'
        elif any(term in text_lower for term in ['canned', 'tinned', 'jarred', 'preserved']):
            return 'preserved'
        elif any(term in text_lower for term in ['frozen', 'chilled']):
            return 'frozen'
        elif any(term in text_lower for term in ['dried', 'dehydrated']):
            return 'dried'
        elif any(term in text_lower for term in ['cooked', 'boiled', 'steamed', 'fried', 'baked']):
            return 'cooked'
        else:
            return 'unknown'
            
    def _calculate_complexity_score(self, text: str) -> float:
        """Calculate a complexity score for the product name"""
        if not text:
            return 0.0
            
        score = 0.0
        
        # Length factor
        score += min(len(text) / 100, 1.0) * 0.3
        
        # Word count factor  
        word_count = len(text.split())
        score += min(word_count / 10, 1.0) * 0.3
        
        # Special character factor
        special_chars = len(re.findall(r'[^\w\s]', text))
        score += min(special_chars / 5, 1.0) * 0.2
        
        # Number presence
        if re.search(r'\d', text):
            score += 0.2
            
        return min(score, 1.0)
        
    def _empty_features(self) -> Dict[str, Union[str, List, float]]:
        """Return empty feature dictionary for invalid input"""
        return {
            'cleaned_text': '',
            'original_text': '',
            'weights': [],
            'brands': [],
            'word_count': 0,
            'char_count': 0,
            'has_numbers': False,
            'processing_type': 'unknown',
            'category': 'unknown',
            'has_weights': False,
            'has_brands': False,
            'complexity_score': 0.0
        }
        
    def preprocess_dataframe(self, df: pd.DataFrame, 
                           text_column: str, 
                           category_column: str = None) -> pd.DataFrame:
        """
        Preprocess an entire dataframe with enhanced features
        
        Args:
            df: Input dataframe
            text_column: Name of column containing text to preprocess
            category_column: Optional category column for category-specific rules
            
        Returns:
            Dataframe with additional preprocessing columns
        """
        logger.info(f"Preprocessing dataframe with {len(df)} rows")
        
        result_df = df.copy()
        
        # Extract features for each row
        features_list = []
        for idx, row in df.iterrows():
            text = row[text_column] if text_column in row else ""
            category = row[category_column] if category_column and category_column in row else None
            
            features = self.extract_features(text, category)
            features_list.append(features)
            
        # Create feature columns
        features_df = pd.DataFrame(features_list)
        
        # Add feature columns to result
        for col in features_df.columns:
            if col not in ['original_text']:  # Don't duplicate original
                result_df[f'preprocessed_{col}'] = features_df[col]
                
        logger.info(f"Preprocessing complete. Added {len(features_df.columns)} feature columns")
        return result_df