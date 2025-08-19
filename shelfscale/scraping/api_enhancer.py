"""
API enhancement tools for ShelfScale

This module provides enhanced API integration capabilities with intelligent
data processing and validation.
"""

import logging
import requests
import time
import json
from typing import Dict, List, Optional, Union, Any
import pandas as pd
from datetime import datetime

from ..utils.logging_config import get_logger, monitor_performance

logger = get_logger(__name__)


class APIEnhancer:
    """
    Enhanced API integration with intelligent data processing
    """
    
    def __init__(self, rate_limit_delay: float = 1.0, max_retries: int = 3):
        """
        Initialize API enhancer
        
        Args:
            rate_limit_delay: Delay between API calls (seconds)
            max_retries: Maximum retry attempts
        """
        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries
        self.session = requests.Session()
        
        # API configurations
        self.apis = {
            'open_food_facts': {
                'base_url': 'https://world.openfoodfacts.org/api/v0',
                'rate_limit': 1.0,  # seconds
                'headers': {
                    'User-Agent': 'ShelfScale/1.0 (https://github.com/yourusername/shelfscale)'
                }
            },
            'spoonacular': {
                'base_url': 'https://api.spoonacular.com',
                'rate_limit': 1.0,
                'requires_key': True
            },
            'usda_food_data': {
                'base_url': 'https://api.nal.usda.gov/fdc/v1',
                'rate_limit': 0.5,
                'requires_key': True
            }
        }
        
        # Statistics tracking
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'cached_responses': 0
        }
        
    @monitor_performance("api_data_fetching")
    def enhanced_open_food_facts_search(self, query: str, 
                                       country: str = 'en:united-kingdom',
                                       page_size: int = 100,
                                       max_pages: int = 10) -> List[Dict]:
        """
        Enhanced Open Food Facts search with pagination and filtering
        
        Args:
            query: Search query
            country: Country filter
            page_size: Results per page
            max_pages: Maximum pages to fetch
            
        Returns:
            List of product data
        """
        logger.info(f"Enhanced OFF search for: '{query}'")
        
        all_products = []
        api_config = self.apis['open_food_facts']
        
        for page in range(1, max_pages + 1):
            try:
                url = f"{api_config['base_url']}/search"
                params = {
                    'search_terms': query,
                    'search_simple': 1,
                    'action': 'process',
                    'countries_tags_en': country,
                    'page_size': page_size,
                    'page': page,
                    'json': 1
                }
                
                response = self._make_request(url, params, api_config)
                
                if response and 'products' in response:
                    products = response['products']
                    
                    if not products:  # No more results
                        break
                        
                    # Enhanced data processing
                    enhanced_products = self._process_off_products(products)
                    all_products.extend(enhanced_products)
                    
                    logger.info(f"Fetched {len(products)} products from page {page}")
                    
                    # Rate limiting
                    time.sleep(api_config['rate_limit'])
                else:
                    break
                    
            except Exception as e:
                logger.error(f"Error fetching OFF page {page}: {e}")
                continue
                
        logger.info(f"Total enhanced OFF products: {len(all_products)}")
        return all_products
        
    def _process_off_products(self, products: List[Dict]) -> List[Dict]:
        """
        Process and enhance Open Food Facts products
        
        Args:
            products: Raw product data
            
        Returns:
            Enhanced product data
        """
        enhanced_products = []
        
        for product in products:
            try:
                enhanced = {
                    'id': product.get('id', ''),
                    'product_name': product.get('product_name', ''),
                    'product_name_en': product.get('product_name_en', ''),
                    'brands': product.get('brands', ''),
                    'categories': product.get('categories', ''),
                    'quantity': product.get('quantity', ''),
                    'packaging': product.get('packaging', ''),
                    'countries': product.get('countries', ''),
                    'stores': product.get('stores', ''),
                    'url': product.get('url', ''),
                    
                    # Nutritional data
                    'energy_kcal': product.get('nutriments', {}).get('energy-kcal_100g'),
                    'fat': product.get('nutriments', {}).get('fat_100g'),
                    'carbohydrates': product.get('nutriments', {}).get('carbohydrates_100g'),
                    'proteins': product.get('nutriments', {}).get('proteins_100g'),
                    'salt': product.get('nutriments', {}).get('salt_100g'),
                    
                    # Enhanced fields
                    'weight_extracted': self._extract_weight_from_off(product),
                    'brand_extracted': self._extract_brand_from_off(product),
                    'category_mapped': self._map_off_category(product.get('categories', '')),
                    'data_quality_score': self._calculate_off_quality_score(product),
                    'fetched_at': datetime.now().isoformat()
                }
                
                enhanced_products.append(enhanced)
                
            except Exception as e:
                logger.warning(f"Error processing OFF product: {e}")
                continue
                
        return enhanced_products
        
    def _extract_weight_from_off(self, product: Dict) -> Optional[Dict]:
        """Extract and parse weight information from OFF product"""
        quantity = product.get('quantity', '')
        
        if not quantity:
            return None
            
        # Use the weight extraction from retail scraper
        from .retail_scraper import RetailScraper
        scraper = RetailScraper()
        return scraper.extract_weight_from_text(quantity)
        
    def _extract_brand_from_off(self, product: Dict) -> str:
        """Extract primary brand from OFF product"""
        brands = product.get('brands', '')
        if brands:
            # Take first brand if multiple
            return brands.split(',')[0].strip()
        return ''
        
    def _map_off_category(self, categories: str) -> str:
        """Map OFF categories to our standard categories"""
        if not categories:
            return 'unknown'
            
        categories_lower = categories.lower()
        
        # Category mapping
        category_mapping = {
            'fruit': ['fruits', 'fruit'],
            'vegetables': ['vegetables', 'legumes'],
            'meat': ['meat', 'poultry', 'beef', 'chicken', 'pork'],
            'fish': ['fish', 'seafood', 'salmon', 'tuna'],
            'dairy': ['dairy', 'milk', 'cheese', 'yogurt'],
            'cereals': ['cereals', 'bread', 'pasta', 'rice'],
            'beverages': ['beverages', 'drinks', 'juice'],
            'snacks': ['snacks', 'biscuits', 'chocolate']
        }
        
        for category, keywords in category_mapping.items():
            if any(keyword in categories_lower for keyword in keywords):
                return category
                
        return 'other'
        
    def _calculate_off_quality_score(self, product: Dict) -> float:
        """Calculate data quality score for OFF product"""
        score = 0.0
        max_score = 10.0
        
        # Check presence of key fields
        if product.get('product_name'): score += 2
        if product.get('brands'): score += 1
        if product.get('quantity'): score += 2
        if product.get('categories'): score += 1
        if product.get('nutriments'): score += 2
        if product.get('packaging'): score += 1
        if product.get('stores'): score += 1
        
        return score / max_score
        
    def _make_request(self, url: str, params: Dict, 
                     api_config: Dict) -> Optional[Dict]:
        """
        Make API request with retries and error handling
        
        Args:
            url: API endpoint URL
            params: Request parameters
            api_config: API configuration
            
        Returns:
            API response data
        """
        headers = api_config.get('headers', {})
        
        for attempt in range(self.max_retries):
            try:
                self.stats['total_requests'] += 1
                
                response = self.session.get(
                    url, params=params, headers=headers, timeout=30
                )
                response.raise_for_status()
                
                self.stats['successful_requests'] += 1
                return response.json()
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"API request attempt {attempt + 1} failed: {e}")
                self.stats['failed_requests'] += 1
                
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    
        return None
        
    def bulk_enhance_data(self, df: pd.DataFrame, 
                         text_column: str,
                         api_name: str = 'open_food_facts') -> pd.DataFrame:
        """
        Bulk enhance data using API lookups
        
        Args:
            df: Input dataframe
            text_column: Column containing product names
            api_name: API to use for enhancement
            
        Returns:
            Enhanced dataframe
        """
        logger.info(f"Bulk enhancing {len(df)} items using {api_name}")
        
        enhanced_data = []
        
        for idx, row in df.iterrows():
            query = row[text_column]
            
            if pd.isna(query) or not query.strip():
                continue
                
            try:
                if api_name == 'open_food_facts':
                    results = self.enhanced_open_food_facts_search(
                        query, max_pages=1, page_size=10
                    )
                    
                    if results:
                        # Take best match based on name similarity
                        best_match = self._find_best_api_match(query, results)
                        if best_match:
                            enhanced_data.append({
                                'original_index': idx,
                                'original_name': query,
                                **best_match
                            })
                            
                # Rate limiting
                time.sleep(self.rate_limit_delay)
                
            except Exception as e:
                logger.warning(f"Error enhancing '{query}': {e}")
                continue
                
        # Convert to DataFrame and merge back
        if enhanced_data:
            enhanced_df = pd.DataFrame(enhanced_data)
            enhanced_df = enhanced_df.set_index('original_index')
            
            # Merge with original data
            result_df = df.join(enhanced_df, rsuffix='_enhanced')
            logger.info(f"Enhanced {len(enhanced_data)} items")
            
        else:
            result_df = df.copy()
            logger.warning("No data enhanced")
            
        return result_df
        
    def _find_best_api_match(self, query: str, results: List[Dict]) -> Optional[Dict]:
        """Find best matching result from API response"""
        if not results:
            return None
            
        # Simple name-based matching (could be enhanced with ML)
        query_lower = query.lower()
        
        best_match = None
        best_score = 0
        
        for result in results:
            name = result.get('product_name', '').lower()
            name_en = result.get('product_name_en', '').lower()
            
            # Simple word overlap scoring
            query_words = set(query_lower.split())
            name_words = set(name.split()) | set(name_en.split())
            
            overlap = len(query_words & name_words)
            score = overlap / max(len(query_words), 1)
            
            if score > best_score:
                best_score = score
                best_match = result
                
        return best_match if best_score > 0.3 else None
        
    def get_api_stats(self) -> Dict[str, Any]:
        """Get API usage statistics"""
        stats = self.stats.copy()
        
        if stats['total_requests'] > 0:
            stats['success_rate'] = stats['successful_requests'] / stats['total_requests']
        else:
            stats['success_rate'] = 0.0
            
        return stats