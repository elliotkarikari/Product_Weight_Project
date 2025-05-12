"""
Client for the Open Food Facts API
Handles fetching and processing data from the Open Food Facts database
"""

import requests
import pandas as pd
from typing import Dict, List, Optional, Union, Any


class OpenFoodFactsClient:
    """Client for interacting with the Open Food Facts API"""
    
    BASE_URL = "https://world.openfoodfacts.org/cgi/search.pl"
    
    def __init__(self):
        """Initialize the Open Food Facts client"""
        self.session = requests.Session()
    
    def search_products(self, 
                       query: str, 
                       category: Optional[str] = None,
                       page_size: int = 100, 
                       max_pages: int = 10) -> pd.DataFrame:
        """
        Search for products in the Open Food Facts database
        
        Args:
            query: Search term
            category: Optional category filter
            page_size: Number of results per page
            max_pages: Maximum number of pages to fetch
            
        Returns:
            DataFrame containing product information
        """
        product_names = []
        weights = []
        packaging_details = []
        countries = []
        
        for page_number in range(1, max_pages + 1):
            params = {
                "search_terms": query,
                "json": 1,
                "page_size": page_size,
                "page": page_number
            }
            
            # Add category filter if provided
            if category:
                params.update({
                    "tagtype_0": "categories",
                    "tag_contains_0": "contains",
                    "tag_0": category
                })
            
            try:
                response = self.session.get(self.BASE_URL, params=params)
                response.raise_for_status()
                data = response.json()
                products = data.get("products", [])
                
                if not products:
                    # No more results
                    break
                    
                for product in products:
                    # Check if the product has weight information and its language is English
                    if "quantity" in product and product.get("lang", "unknown") == "en":
                        name = product.get("product_name", f"Unknown {query.title()}").strip()
                        
                        # Skip if the product name is unknown
                        if name == f"Unknown {query.title()}":
                            continue
                            
                        weight = product.get("quantity", "Unknown Weight").strip()
                        packaging = product.get("packaging", "Unknown Packaging").strip()
                        country = product.get("countries", "Unknown Country").strip()
                        
                        # Append data to lists
                        product_names.append(name)
                        weights.append(weight)
                        packaging_details.append(packaging)
                        countries.append(country)
                        
            except requests.exceptions.RequestException as e:
                print(f"Error occurred while processing page {page_number}: {e}")
                break
                
            except Exception as e:
                print(f"Unexpected error occurred while processing page {page_number}: {e}")
                break
                
        # Create a DataFrame
        data_dict = {
            'Product Name': product_names,
            'Weight': weights,
            'Packaging Details': packaging_details,
            'Country': countries
        }
        
        return pd.DataFrame(data_dict)
    
    def search_by_food_group(self, food_group: str) -> pd.DataFrame:
        """
        Search for products based on a specific food group
        
        Args:
            food_group: The food group to search for (e.g., 'vegetables', 'fruits')
            
        Returns:
            DataFrame containing product information
        """
        return self.search_products(food_group, category=food_group) 