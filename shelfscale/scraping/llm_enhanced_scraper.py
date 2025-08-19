"""
LLM-Enhanced Scraping System

This module provides intelligent scraping capabilities using LLMs for better
data extraction, parsing, and validation of product information.
"""

import logging
import json
import re
import asyncio
import aiohttp
from typing import Dict, List, Optional, Union, Any
import pandas as pd
from datetime import datetime
from dataclasses import dataclass, asdict

# Import with graceful fallback
try:
    from ..utils.logging_config import get_logger, monitor_performance
except ImportError:
    import logging
    def get_logger(name):
        return logging.getLogger(name)
    def monitor_performance(name):
        def decorator(func):
            return func
        return decorator

try:
    from .retail_scraper import RetailScraper
except ImportError:
    # Create a minimal RetailScraper fallback
    class RetailScraper:
        def __init__(self, **kwargs):
            pass
        def extract_weight_from_text(self, text):
            return None
        def scrape_products(self, query, retailer, max_pages):
            return []

logger = get_logger(__name__)


@dataclass
class ProductInfo:
    """Structured product information extracted by LLM"""
    name: str
    brand: Optional[str] = None
    weight: Optional[str] = None
    weight_value: Optional[float] = None
    weight_unit: Optional[str] = None
    price: Optional[str] = None
    price_value: Optional[float] = None
    currency: Optional[str] = None
    category: Optional[str] = None
    description: Optional[str] = None
    ingredients: Optional[List[str]] = None
    nutritional_info: Optional[Dict[str, Any]] = None
    allergens: Optional[List[str]] = None
    country_of_origin: Optional[str] = None
    confidence_score: float = 0.0
    extraction_notes: Optional[str] = None


class LLMEnhancedScraper(RetailScraper):
    """
    Enhanced scraper that uses LLMs for intelligent data extraction and parsing
    """
    
    def __init__(self, **kwargs):
        """Initialize the LLM-enhanced scraper"""
        super().__init__(**kwargs)
        
        # LLM configuration
        self.llm_model = "gpt-4"
        self.max_tokens = 2000
        self.temperature = 0.1  # Low temperature for consistent extraction
        
        # System prompt for product extraction
        self.extraction_prompt = self._create_extraction_prompt()
        
        # Enhanced statistics
        self.llm_stats = {
            'total_llm_extractions': 0,
            'successful_extractions': 0,
            'failed_extractions': 0,
            'token_usage': 0,
            'extraction_time': 0.0
        }
        
    def _create_extraction_prompt(self) -> str:
        """Create system prompt for product information extraction"""
        return """You are an expert product information extraction system. Extract structured product information from unstructured text data.

Your task is to identify and extract:

1. PRODUCT NAME: Clean, standardized product name
2. BRAND: Brand name if clearly identifiable
3. WEIGHT/SIZE: Package weight or size with value and unit
4. PRICE: Price information with value and currency
5. CATEGORY: Food/product category
6. NUTRITIONAL INFO: Any nutritional information (calories, fat, etc.)
7. INGREDIENTS: List of ingredients if available
8. ALLERGENS: Allergen information
9. COUNTRY OF ORIGIN: If mentioned

Key extraction rules:
- Be precise and conservative - only extract information you're confident about
- Standardize units (g, kg, ml, l, oz, lb)
- Clean and normalize text (remove extra spaces, fix capitalization)
- For weights: extract both numeric value and unit separately
- For prices: extract numeric value and currency separately
- For categories: use standard food categories (dairy, meat, vegetables, etc.)

Output format (JSON):
{
    "name": "Clean product name",
    "brand": "Brand name or null",
    "weight": "Full weight string",
    "weight_value": 500,
    "weight_unit": "g",
    "price": "Full price string", 
    "price_value": 2.99,
    "currency": "GBP",
    "category": "Standard category",
    "description": "Product description",
    "ingredients": ["ingredient1", "ingredient2"],
    "nutritional_info": {"calories_per_100g": 250, "fat_per_100g": 15},
    "allergens": ["milk", "eggs"],
    "country_of_origin": "UK",
    "confidence_score": 0.85,
    "extraction_notes": "Any relevant notes about extraction quality"
}

If information is not clearly available, use null for that field."""

    @monitor_performance("llm_product_extraction")
    async def extract_product_info_llm(self, 
                                      raw_text: str,
                                      context: Dict = None) -> ProductInfo:
        """
        Extract structured product information using LLM
        
        Args:
            raw_text: Raw scraped product text
            context: Additional context (source, category hints, etc.)
            
        Returns:
            Structured ProductInfo object
        """
        try:
            # Prepare extraction prompt
            prompt = self._create_product_extraction_prompt(raw_text, context)
            
            # Make LLM call
            self.llm_stats['total_llm_extractions'] += 1
            result = await self._call_llm_for_extraction(prompt)
            
            # Parse and validate result
            product_info = self._parse_extraction_result(result, raw_text)
            
            self.llm_stats['successful_extractions'] += 1
            logger.info(f"Successfully extracted product info: {product_info.name}")
            
            return product_info
            
        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")
            self.llm_stats['failed_extractions'] += 1
            return self._fallback_extraction(raw_text)
            
    def _create_product_extraction_prompt(self, 
                                        raw_text: str, 
                                        context: Dict = None) -> str:
        """Create specific extraction prompt for product text"""
        
        context_info = ""
        if context:
            context_info = "\nAdditional context:\n"
            for key, value in context.items():
                context_info += f"- {key}: {value}\n"
                
        prompt = f"""Extract product information from this text:

Raw product text:
"{raw_text}"
{context_info}

Please extract all available product information following the specified JSON format.
Focus on accuracy and only include information you're confident about."""

        return prompt
        
    async def _call_llm_for_extraction(self, prompt: str) -> str:
        """Make LLM API call for extraction"""
        
        # Simulate LLM call with mock response for demonstration
        await asyncio.sleep(0.2)  # Simulate API delay
        
        # Generate mock response based on text analysis
        return self._generate_mock_extraction_response(prompt)
        
    def _generate_mock_extraction_response(self, prompt: str) -> str:
        """Generate mock LLM response for demonstration"""
        
        # Extract the raw text from prompt
        text_match = re.search(r'Raw product text:\s*"(.*?)"', prompt, re.DOTALL)
        if not text_match:
            return '{"name": "Unknown Product", "confidence_score": 0.1}'
            
        raw_text = text_match.group(1).lower()
        
        # Basic extraction logic for demonstration
        result = {
            "name": None,
            "brand": None,
            "weight": None,
            "weight_value": None,
            "weight_unit": None,
            "price": None,
            "price_value": None,
            "currency": None,
            "category": None,
            "description": raw_text[:100] + "..." if len(raw_text) > 100 else raw_text,
            "ingredients": None,
            "nutritional_info": None,
            "allergens": None,
            "country_of_origin": None,
            "confidence_score": 0.7,
            "extraction_notes": "Mock extraction for demonstration"
        }
        
        # Extract name (first meaningful part)
        name_patterns = [
            r'([a-zA-Z\s]+(?:chicken|beef|bread|milk|cheese|apple|banana|pasta|rice)[a-zA-Z\s]*)',
            r'^([a-zA-Z\s]{10,50})',  # First 10-50 characters if no food words
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, raw_text, re.IGNORECASE)
            if match:
                result["name"] = match.group(1).strip().title()
                break
                
        if not result["name"]:
            result["name"] = raw_text[:50].strip().title()
            
        # Extract weight
        weight_patterns = [
            r'(\d+(?:\.\d+)?)\s*(g|kg|ml|l|oz|lb)',
            r'(\d+)\s*x\s*(\d+(?:\.\d+)?)\s*(g|kg|ml|l)',
        ]
        
        for pattern in weight_patterns:
            match = re.search(pattern, raw_text)
            if match:
                if 'x' in pattern:
                    # Multi-pack: multiply quantity × unit weight
                    quantity = float(match.group(1))
                    unit_weight = float(match.group(2))
                    unit = match.group(3)
                    total_weight = quantity * unit_weight
                    result["weight"] = f"{total_weight}{unit}"
                    result["weight_value"] = total_weight
                    result["weight_unit"] = unit
                else:
                    result["weight"] = match.group(0)
                    result["weight_value"] = float(match.group(1))
                    result["weight_unit"] = match.group(2)
                break
                
        # Extract price
        price_patterns = [
            r'£(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)\s*p(?:ence)?',
            r'\$(\d+(?:\.\d+)?)',
        ]
        
        for pattern in price_patterns:
            match = re.search(pattern, raw_text)
            if match:
                price_value = float(match.group(1))
                if 'p' in pattern:
                    price_value /= 100  # Convert pence to pounds
                    currency = "GBP"
                elif '£' in pattern:
                    currency = "GBP"
                elif '$' in pattern:
                    currency = "USD"
                else:
                    currency = "GBP"  # Default
                    
                result["price"] = match.group(0)
                result["price_value"] = price_value
                result["currency"] = currency
                break
                
        # Extract brand (common UK brands)
        brands = ['tesco', 'sainsbury', 'asda', 'morrisons', 'organic', 'premium', 'value']
        for brand in brands:
            if brand in raw_text:
                result["brand"] = brand.title()
                break
                
        # Categorize based on keywords
        categories = {
            'dairy': ['milk', 'cheese', 'yogurt', 'butter', 'cream'],
            'meat': ['chicken', 'beef', 'pork', 'lamb', 'turkey'],
            'vegetables': ['carrot', 'potato', 'onion', 'tomato', 'lettuce'],
            'fruit': ['apple', 'banana', 'orange', 'grape', 'strawberry'],
            'bakery': ['bread', 'cake', 'biscuit', 'pastry'],
            'beverages': ['juice', 'water', 'tea', 'coffee', 'soda']
        }
        
        for category, keywords in categories.items():
            if any(keyword in raw_text for keyword in keywords):
                result["category"] = category
                break
                
        # Extract allergens
        allergen_keywords = ['milk', 'eggs', 'nuts', 'gluten', 'soy', 'fish']
        found_allergens = [allergen for allergen in allergen_keywords if allergen in raw_text]
        if found_allergens:
            result["allergens"] = found_allergens
            
        return json.dumps(result)
        
    def _parse_extraction_result(self, 
                               llm_response: str, 
                               original_text: str) -> ProductInfo:
        """Parse and validate LLM extraction result"""
        try:
            data = json.loads(llm_response)
            
            # Create ProductInfo with validation
            product_info = ProductInfo(
                name=data.get('name', 'Unknown Product'),
                brand=data.get('brand'),
                weight=data.get('weight'),
                weight_value=data.get('weight_value'),
                weight_unit=data.get('weight_unit'),
                price=data.get('price'),
                price_value=data.get('price_value'),
                currency=data.get('currency'),
                category=data.get('category'),
                description=data.get('description'),
                ingredients=data.get('ingredients'),
                nutritional_info=data.get('nutritional_info'),
                allergens=data.get('allergens'),
                country_of_origin=data.get('country_of_origin'),
                confidence_score=data.get('confidence_score', 0.5),
                extraction_notes=data.get('extraction_notes')
            )
            
            # Validate extracted data
            product_info = self._validate_extracted_data(product_info, original_text)
            
            return product_info
            
        except (json.JSONDecodeError, TypeError) as e:
            logger.error(f"Failed to parse LLM response: {e}")
            return self._fallback_extraction(original_text)
            
    def _validate_extracted_data(self, 
                               product_info: ProductInfo, 
                               original_text: str) -> ProductInfo:
        """Validate and clean extracted product information"""
        
        # Validate weight
        if product_info.weight_value is not None:
            if product_info.weight_value <= 0 or product_info.weight_value > 50000:  # Unrealistic weights
                product_info.weight_value = None
                product_info.weight_unit = None
                product_info.confidence_score *= 0.8
                
        # Validate price
        if product_info.price_value is not None:
            if product_info.price_value <= 0 or product_info.price_value > 1000:  # Unrealistic prices
                product_info.price_value = None
                product_info.currency = None
                product_info.confidence_score *= 0.8
                
        # Validate name
        if not product_info.name or len(product_info.name.strip()) < 3:
            product_info.name = original_text[:50].strip() or "Unknown Product"
            product_info.confidence_score *= 0.6
            
        # Clean text fields
        if product_info.name:
            product_info.name = re.sub(r'\s+', ' ', product_info.name.strip())
            
        if product_info.brand:
            product_info.brand = product_info.brand.strip().title()
            
        return product_info
        
    def _fallback_extraction(self, raw_text: str) -> ProductInfo:
        """Fallback extraction when LLM fails"""
        
        # Use parent class extraction methods
        weight_info = self.extract_weight_from_text(raw_text)
        
        return ProductInfo(
            name=raw_text[:50].strip() or "Unknown Product",
            weight=weight_info.get('original') if weight_info else None,
            weight_value=weight_info.get('value') if weight_info else None,
            weight_unit=weight_info.get('unit') if weight_info else None,
            description=raw_text,
            confidence_score=0.3,
            extraction_notes="Fallback extraction used due to LLM failure"
        )
        
    async def scrape_and_extract_products(self, 
                                        query: str,
                                        retailer: str = 'tesco',
                                        max_pages: int = 2) -> List[ProductInfo]:
        """
        Scrape products and enhance with LLM extraction
        
        Args:
            query: Search query
            retailer: Retailer to scrape
            max_pages: Maximum pages
            
        Returns:
            List of enhanced ProductInfo objects
        """
        logger.info(f"Enhanced scraping for '{query}' from {retailer}")
        
        # First, scrape using parent class method
        raw_products = self.scrape_products(query, retailer, max_pages)
        
        if not raw_products:
            logger.warning("No products scraped")
            return []
            
        # Enhance each product with LLM extraction
        enhanced_products = []
        
        for raw_product in raw_products:
            try:
                # Combine all product text for extraction
                product_text = f"""
                Product: {raw_product.get('name', '')}
                Weight: {raw_product.get('weight', '')}
                Price: {raw_product.get('price', '')}
                Retailer: {raw_product.get('retailer', '')}
                """.strip()
                
                # Add context
                context = {
                    'retailer': raw_product.get('retailer', ''),
                    'source_url': raw_product.get('url', ''),
                    'query': query
                }
                
                # Extract with LLM
                product_info = await self.extract_product_info_llm(product_text, context)
                
                # Add original data
                if not product_info.name or product_info.name == "Unknown Product":
                    product_info.name = raw_product.get('name', 'Unknown Product')
                    
                enhanced_products.append(product_info)
                
            except Exception as e:
                logger.error(f"Error enhancing product: {e}")
                # Add fallback product info
                fallback_info = self._fallback_extraction(
                    raw_product.get('name', 'Unknown Product')
                )
                enhanced_products.append(fallback_info)
                
        logger.info(f"Enhanced {len(enhanced_products)} products with LLM extraction")
        return enhanced_products
        
    def products_to_dataframe(self, products: List[ProductInfo]) -> pd.DataFrame:
        """Convert ProductInfo objects to DataFrame"""
        
        data = []
        for product in products:
            product_dict = asdict(product)
            
            # Flatten nested dictionaries
            if product_dict['nutritional_info']:
                for key, value in product_dict['nutritional_info'].items():
                    product_dict[f'nutrition_{key}'] = value
                    
            # Convert lists to strings
            if product_dict['ingredients']:
                product_dict['ingredients_str'] = ', '.join(product_dict['ingredients'])
            if product_dict['allergens']:
                product_dict['allergens_str'] = ', '.join(product_dict['allergens'])
                
            data.append(product_dict)
            
        df = pd.DataFrame(data)
        df['extraction_timestamp'] = datetime.now()
        
        return df
        
    def validate_extraction_quality(self, 
                                   products: List[ProductInfo],
                                   min_confidence: float = 0.6) -> Dict[str, Any]:
        """
        Validate the quality of LLM extractions
        
        Args:
            products: List of extracted products
            min_confidence: Minimum confidence threshold
            
        Returns:
            Quality metrics
        """
        if not products:
            return {'total_products': 0, 'quality_score': 0.0}
            
        # Calculate metrics
        total_products = len(products)
        high_confidence_products = sum(1 for p in products if p.confidence_score >= min_confidence)
        
        # Field completeness
        fields_to_check = ['name', 'weight_value', 'price_value', 'category', 'brand']
        field_completeness = {}
        
        for field in fields_to_check:
            complete_count = sum(1 for p in products if getattr(p, field) is not None)
            field_completeness[field] = complete_count / total_products
            
        # Overall quality score
        avg_confidence = sum(p.confidence_score for p in products) / total_products
        avg_completeness = sum(field_completeness.values()) / len(field_completeness)
        quality_score = (avg_confidence + avg_completeness) / 2
        
        return {
            'total_products': total_products,
            'high_confidence_products': high_confidence_products,
            'high_confidence_rate': high_confidence_products / total_products,
            'average_confidence': avg_confidence,
            'field_completeness': field_completeness,
            'average_completeness': avg_completeness,
            'quality_score': quality_score,
            'llm_stats': self.llm_stats.copy()
        }
        
    def export_enhanced_data(self, 
                           products: List[ProductInfo], 
                           filename: str = None,
                           format: str = 'csv') -> str:
        """
        Export enhanced product data
        
        Args:
            products: List of ProductInfo objects
            filename: Output filename
            format: Export format ('csv', 'json', 'excel')
            
        Returns:
            Path to exported file
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"enhanced_products_{timestamp}"
            
        if format == 'csv':
            df = self.products_to_dataframe(products)
            filepath = f"{filename}.csv"
            df.to_csv(filepath, index=False)
            
        elif format == 'json':
            data = [asdict(product) for product in products]
            filepath = f"{filename}.json"
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
        elif format == 'excel':
            df = self.products_to_dataframe(products)
            filepath = f"{filename}.xlsx"
            df.to_excel(filepath, index=False)
            
        else:
            raise ValueError(f"Unsupported format: {format}")
            
        logger.info(f"Enhanced data exported to {filepath}")
        return filepath
        
    def get_llm_stats(self) -> Dict[str, Any]:
        """Get LLM usage statistics"""
        stats = self.llm_stats.copy()
        
        if stats['total_llm_extractions'] > 0:
            stats['success_rate'] = stats['successful_extractions'] / stats['total_llm_extractions']
        else:
            stats['success_rate'] = 0.0
            
        return stats