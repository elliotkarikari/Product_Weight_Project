"""PDF extraction utilities for ShelfScale food data"""

import os
import re
import pandas as pd
import PyPDF2
import tabula
import logging
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)

class PDFExtractor:
    """Extract food and weight data from PDF documents"""
    
    def __init__(self, cache_dir: str = 'output'):
        """Initialize the extractor
        
        Args:
            cache_dir: Directory to cache extracted data
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
    def extract_food_portion_sizes(self, pdf_path: str, 
                                  pages: str = "12-114",
                                  force_extract: bool = False,
                                  cache_path: Optional[str] = None) -> pd.DataFrame:
        """Extract food portion size data from PDF
        
        Args:
            pdf_path: Path to the PDF file
            pages: Page range to extract (e.g., "12-114")
            force_extract: Force extraction even if cache exists
            cache_path: Optional path to save cached data
            
        Returns:
            DataFrame with food portion size data
        """
        if cache_path is None:
            cache_path = os.path.join(self.cache_dir, "food_portion_sizes.csv")
        
        # Check if cached data exists
        if os.path.exists(cache_path) and not force_extract:
            logger.info(f"Loading food portion sizes from cache: {cache_path}")
            return pd.read_csv(cache_path)
            
        logger.info(f"Extracting food portion sizes from PDF: {pdf_path}")
        
        try:
            # Extract tables from PDF
            tables = tabula.read_pdf(pdf_path, 
                                    pages=pages, 
                                    stream=True, 
                                    multiple_tables=True, 
                                    guess=False, 
                                    encoding="latin1",
                                    pandas_options={"header": [0, 1, 2, 3]}
                                    )
            
            # Combine tables into single DataFrame
            df = pd.DataFrame()
            for table in tables:
                df = pd.concat([df, table], axis=0, ignore_index=True)
            
            # Clean up the DataFrame
            df.columns = ['Food_Name', 'Portion_Size', 'Weight_g', 'Notes']
            df = df.dropna(thresh=2)  # Drop rows with less than 2 non-NaN values
            
            # Clean and normalize text columns
            for col in ['Food_Name', 'Portion_Size', 'Notes']:
                if col in df.columns:
                    df[col] = df[col].astype(str).str.strip()
            
            # Convert weight to numeric
            if 'Weight_g' in df.columns:
                df['Weight_g'] = pd.to_numeric(df['Weight_g'], errors='coerce')
            
            # Save to cache
            df.to_csv(cache_path, index=False)
            logger.info(f"Extracted {len(df)} food portion items and cached to {cache_path}")
            
            return df
        except Exception as e:
            logger.error(f"Error extracting food portion sizes: {str(e)}")
            # Return empty DataFrame as fallback
            return pd.DataFrame(columns=['Food_Name', 'Portion_Size', 'Weight_g', 'Notes'])
    
    def extract_fruit_veg_survey(self, pdf_path: str,
                               start_page: int = 11,
                               end_page: int = 1291,
                               force_extract: bool = False,
                               cache_path: Optional[str] = None) -> pd.DataFrame:
        """Extract fruit and vegetable survey data from PDF
        
        Args:
            pdf_path: Path to the PDF file
            start_page: First page to extract
            end_page: Last page to extract
            force_extract: Force extraction even if cache exists
            cache_path: Optional path to save cached data
            
        Returns:
            DataFrame with fruit and vegetable data
        """
        if cache_path is None:
            cache_path = os.path.join(self.cache_dir, "fruit_veg_survey.csv")
        
        # Check if cached data exists
        if os.path.exists(cache_path) and not force_extract:
            logger.info(f"Loading fruit and veg survey from cache: {cache_path}")
            return pd.read_csv(cache_path)
            
        logger.info(f"Extracting fruit and veg survey from PDF: {pdf_path}")
        
        try:
            # Extract text from PDF
            extracted_data = {}
            with open(pdf_path, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)
                
                # Adjust end page if needed
                end_page = min(end_page, num_pages)
                
                # Extract data from pages with sample information
                for page_num in range(start_page - 1, end_page):
                    try:
                        # Get the page
                        page = pdf_reader.pages[page_num]
                        # Extract text
                        page_text = page.extract_text()
                        # Store in dict
                        extracted_data[page_num + 1] = page_text
                    except Exception as e:
                        logger.warning(f"Error extracting text from page {page_num + 1}: {str(e)}")
                        continue
            
            # Parse extracted text into structured data
            samples = []
            
            # Regular expressions for key information
            sample_number_pattern = r"Composite Sample Number:\s*(\d+)"
            sample_name_pattern = r"Composite Sample Name:\s*(.*?)\n"
            weight_pattern = r"Pack size:\s*(.*?)\n"
            
            # Process each page
            for page_num, text in extracted_data.items():
                # Extract information using regex
                sample_number_match = re.search(sample_number_pattern, text)
                sample_name_match = re.search(sample_name_pattern, text)
                weight_match = re.search(weight_pattern, text)
                
                if sample_name_match:
                    sample = {
                        'Page': page_num,
                        'Sample_Number': sample_number_match.group(1) if sample_number_match else None,
                        'Sample_Name': sample_name_match.group(1).strip() if sample_name_match else None,
                        'Pack_Size': weight_match.group(1).strip() if weight_match else None
                    }
                    samples.append(sample)
            
            # Create DataFrame
            df = pd.DataFrame(samples)
            
            # Save to cache
            df.to_csv(cache_path, index=False)
            logger.info(f"Extracted {len(df)} fruit and veg samples and cached to {cache_path}")
            
            return df
        except Exception as e:
            logger.error(f"Error extracting fruit and veg survey: {str(e)}")
            # Return empty DataFrame as fallback
            return pd.DataFrame(columns=['Page', 'Sample_Number', 'Sample_Name', 'Pack_Size'])
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text fields
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        if pd.isna(text) or not isinstance(text, str):
            return ""
            
        # List of patterns to clean
        patterns = [
            # Remove common qualifiers
            (r'\b(raw|cooked|boiled|fresh|dried|canned|frozen)\b', ''),
            # Remove preparation details
            (r'\b(ready to eat|semi-dried|flesh only|flesh and skin)\b', ''),
            # Clean up punctuation
            (r'\s*,\s*', ' '), 
            # Normalize whitespace
            (r'\s+', ' ')
        ]
        
        # Apply patterns
        cleaned = text.lower()
        for pattern, replacement in patterns:
            cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
            
        return cleaned.strip()
    
    def extract_weight_from_text(self, text: str) -> Optional[float]:
        """Extract weight value from text description
        
        Args:
            text: Text containing weight information
            
        Returns:
            Extracted weight or None if not found
        """
        if pd.isna(text) or not isinstance(text, str):
            return None
            
        # Common patterns for weight
        patterns = [
            # Match "X g" or "X grams"
            r'(\d+\.?\d*)\s*(?:g|grams|gram)\b',
            # Match "X kg" or "X kilograms"
            r'(\d+\.?\d*)\s*(?:kg|kilograms|kilogram)\b',
            # Match just number followed by unit
            r'(\d+\.?\d*)\s*(?:[a-zA-Z]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    value = float(match.group(1))
                    # Convert kg to g if needed
                    if 'kg' in text.lower():
                        value *= 1000
                    return value
                except ValueError:
                    continue
                    
        return None 