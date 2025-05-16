"""Module for extracting structured data from food-related PDFs"""

import os
import re
import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any, Union, Tuple
import tabula
import PyPDF2
import traceback

# Set up logging
logger = logging.getLogger(__name__)

class PDFExtractor:
    """Extracts structured data from food-related PDFs"""
    
    def __init__(self, cache_dir: str = "output"):
        """
        Initialize the PDF extractor
        
        Args:
            cache_dir: Directory to store cached data
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Extended list of encodings to try
        self.encodings = [
            'utf-8', 
            'latin1', 
            'cp1252', 
            'ISO-8859-1', 
            'ISO-8859-15',
            'Windows-1252',
            'utf-16',
            'ascii'
        ]
    
    def extract_food_portion_sizes(self, pdf_path: str) -> pd.DataFrame:
        """
        Extract food portion sizes from PDF
        
        Args:
            pdf_path: Path to Food Portion Sizes PDF
            
        Returns:
            DataFrame with food portion data
        """
        cache_path = os.path.join(self.cache_dir, "food_portion_sizes.csv")
        
        # Use cache if available
        if os.path.exists(cache_path):
            logger.info(f"Loading cached food portion sizes from {cache_path}")
            return pd.read_csv(cache_path)
        
        logger.info(f"Extracting food portion sizes from PDF: {pdf_path}")
        
        # Check if file exists
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file not found: {pdf_path}")
            return pd.DataFrame(columns=["Food_Name", "Portion_Size", "Weight_g", "Notes"])
        
        extraction_methods = [
            self._extract_food_portion_lattice,
            self._extract_food_portion_stream,
            self._extract_food_portion_simple
        ]
        
        # Try extraction methods in sequence until one succeeds
        for i, method in enumerate(extraction_methods):
            try:
                logger.info(f"Trying extraction method {i+1}/{len(extraction_methods)}")
                df = method(pdf_path)
                
                if isinstance(df, pd.DataFrame) and len(df) > 0:
                    logger.info(f"Method {i+1} succeeded, extracted {len(df)} rows")
                    
                    # Clean data
                    df = self._clean_food_portion_data(df)
                    
                    # Cache for future use
                    df.to_csv(cache_path, index=False)
                    
                    logger.info(f"Extracted {len(df)} food portion items and cached to {cache_path}")
                    return df
                else:
                    logger.warning(f"Method {i+1} produced no results, trying next method")
            except Exception as e:
                logger.error(f"Method {i+1} failed: {e}")
                logger.debug(traceback.format_exc())
        
        logger.error("All extraction methods failed")
        return pd.DataFrame(columns=["Food_Name", "Portion_Size", "Weight_g", "Notes"])
    
    def _extract_food_portion_lattice(self, pdf_path: str) -> pd.DataFrame:
        """Extract food portion sizes using lattice mode"""
        for encoding in self.encodings:
            try:
                logger.info(f"Trying lattice mode with encoding: {encoding}")
                tables = tabula.read_pdf(
                    pdf_path, 
                    pages="all", 
                    multiple_tables=True,
                    encoding=encoding,
                    lattice=True,
                    guess=True
                )
                
                if tables and len(tables) > 0 and sum(len(t) for t in tables) > 0:
                    logger.info(f"Lattice mode: found {len(tables)} tables with {sum(len(t) for t in tables)} total rows")
                    df = pd.concat(tables, ignore_index=True)
                    
                    # Standardize column names
                    return self._standardize_food_portion_columns(df)
            except Exception as e:
                logger.warning(f"Lattice extraction with {encoding} failed: {e}")
        
        raise ValueError("Lattice extraction failed with all encodings")
    
    def _extract_food_portion_stream(self, pdf_path: str) -> pd.DataFrame:
        """Extract food portion sizes using stream mode"""
        for encoding in self.encodings:
            try:
                logger.info(f"Trying stream mode with encoding: {encoding}")
                tables = tabula.read_pdf(
                    pdf_path, 
                    pages="all", 
                    multiple_tables=True,
                    encoding=encoding,
                    stream=True,
                    guess=False
                )
                
                if tables and len(tables) > 0 and sum(len(t) for t in tables) > 0:
                    logger.info(f"Stream mode: found {len(tables)} tables with {sum(len(t) for t in tables)} total rows")
                    df = pd.concat(tables, ignore_index=True)
                    
                    # Standardize column names
                    return self._standardize_food_portion_columns(df)
            except Exception as e:
                logger.warning(f"Stream extraction with {encoding} failed: {e}")
        
        raise ValueError("Stream extraction failed with all encodings")
    
    def _extract_food_portion_simple(self, pdf_path: str) -> pd.DataFrame:
        """Simple extraction for when other methods fail"""
        try:
            logger.info("Trying simple extraction with fixed settings")
            tables = tabula.read_pdf(
                pdf_path, 
                pages="12-114",  # Target most likely pages with portion data
                multiple_tables=False,  # Try treating it as one large table
                encoding="latin1",  # Common encoding for older PDFs
                pandas_options={"header": None}  # No header to avoid issues
            )
            
            if isinstance(tables, list) and len(tables) > 0:
                df = pd.concat(tables, ignore_index=True)
            else:
                df = tables
                
            # For simple extraction, assign standard columns based on position
            if len(df.columns) >= 3:
                column_names = ["Food_Name", "Portion_Size", "Weight_g"]
                if len(df.columns) >= 4:
                    column_names.append("Notes")
                    
                # Use only the number of columns that exist
                df.columns = column_names[:len(df.columns)]
                
                # Add missing expected columns
                for col in ["Food_Name", "Portion_Size", "Weight_g", "Notes"]:
                    if col not in df.columns:
                        df[col] = None
                
                return df
        except Exception as e:
            logger.warning(f"Simple extraction failed: {e}")
        
        raise ValueError("Simple extraction failed")
    
    def _standardize_food_portion_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names for food portion data"""
        # Standardize column names
        renamed_cols = {}
        for col in df.columns:
            col_lower = str(col).lower()
            if "food" in col_lower or "name" in col_lower or "item" in col_lower:
                renamed_cols[col] = "Food_Name"
            elif "portion" in col_lower or "serving" in col_lower:
                renamed_cols[col] = "Portion_Size"
            elif "weight" in col_lower or "g)" in col_lower or "gram" in col_lower:
                renamed_cols[col] = "Weight_g"
            elif "note" in col_lower or "comment" in col_lower:
                renamed_cols[col] = "Notes"
        
        # Rename columns
        if renamed_cols:
            df = df.rename(columns=renamed_cols)
        
        # Add missing columns
        for col in ["Food_Name", "Portion_Size", "Weight_g", "Notes"]:
            if col not in df.columns:
                df[col] = None
        
        return df
    
    def extract_fruit_veg_survey(self, pdf_path: str) -> pd.DataFrame:
        """
        Extract fruit and vegetable survey data from PDF
        
        Args:
            pdf_path: Path to Fruit & Vegetable Survey PDF
            
        Returns:
            DataFrame with fruit and vegetable data
        """
        cache_path = os.path.join(self.cache_dir, "fruit_veg_survey.csv")
        
        # Use cache if available
        if os.path.exists(cache_path):
            logger.info(f"Loading cached fruit and veg survey from {cache_path}")
            return pd.read_csv(cache_path)
        
        logger.info(f"Extracting fruit and veg survey from PDF: {pdf_path}")
        
        # Check if file exists
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file not found: {pdf_path}")
            return pd.DataFrame(columns=["Page", "Sample_Number", "Sample_Name", "Pack_Size"])
        
        extraction_methods = [
            self._extract_fruit_veg_tables,
            self._extract_fruit_veg_text_based,
            self._extract_fruit_veg_simple
        ]
        
        # Try extraction methods in sequence until one succeeds
        for i, method in enumerate(extraction_methods):
            try:
                logger.info(f"Trying fruit/veg extraction method {i+1}/{len(extraction_methods)}")
                df = method(pdf_path)
                
                if isinstance(df, pd.DataFrame) and len(df) > 0:
                    logger.info(f"Method {i+1} succeeded, extracted {len(df)} rows")
                    
                    # Clean and standardize data
                    df = self._clean_fruit_veg_data(df)
                    
                    # Cache for future use
                    df.to_csv(cache_path, index=False)
                    
                    logger.info(f"Extracted {len(df)} fruit and veg samples and cached to {cache_path}")
                    return df
                else:
                    logger.warning(f"Method {i+1} produced no results, trying next method")
            except Exception as e:
                logger.error(f"Method {i+1} failed: {e}")
                logger.debug(traceback.format_exc())
        
        logger.error("All fruit/veg extraction methods failed")
        return pd.DataFrame(columns=["Page", "Sample_Number", "Sample_Name", "Pack_Size"])
    
    def _extract_fruit_veg_tables(self, pdf_path: str) -> pd.DataFrame:
        """Extract fruit and vegetable data using table extraction"""
        # Try table-based approach first
        sample_tables = []
        
        # Determine page range
        try:
            with open(pdf_path, 'rb') as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                total_pages = len(pdf_reader.pages)
                
                # Look for appendix sections
                appendix_pages = []
                for i in range(min(50, total_pages)):  # Check first 50 pages
                    text = pdf_reader.pages[i].extract_text()
                    if "appendix" in text.lower() and "sample" in text.lower():
                        # Extract page numbers mentioned
                        page_matches = re.findall(r'appendix.*?(\d+)', text.lower())
                        if page_matches:
                            for match in page_matches:
                                try:
                                    page_num = int(match)
                                    if page_num <= total_pages:
                                        appendix_pages.append(page_num)
                                except ValueError:
                                    pass
                            
                if appendix_pages:
                    logger.info(f"Found potential appendix pages: {appendix_pages}")
                    # Include a range around the appendix start
                    start_page = max(1, min(appendix_pages) - 2)
                    end_page = min(total_pages, start_page + 50)  # Look at ~50 pages
                    page_range = f"{start_page}-{end_page}"
                else:
                    # Default to checking the latter part of the document
                    start_page = max(1, total_pages - 100)
                    page_range = f"{start_page}-{total_pages}"
        except Exception as e:
            logger.warning(f"Error determining page range: {e}")
            page_range = "all"
        
        # Try different encodings for table extraction
        for encoding in self.encodings:
            try:
                logger.info(f"Trying table extraction with encoding: {encoding} on pages {page_range}")
                tables = tabula.read_pdf(
                    pdf_path, 
                    pages=page_range, 
                    multiple_tables=True,
                    encoding=encoding,
                    guess=True
                )
                
                # Filter for sample tables
                for table in tables:
                    if len(table) == 0:
                        continue
                        
                    # Check if this looks like a sample table
                    has_sample_col = any(col.lower() in ["sample", "product", "item"] 
                                      for col in table.columns if isinstance(col, str))
                    
                    has_detail_col = any("sample" in str(col).lower() 
                                       for col in table.columns if isinstance(col, str))
                    
                    # Check first row for sample indicators
                    has_sample_row = False
                    if len(table) > 0:
                        first_row = table.iloc[0]
                        has_sample_row = any("sample" in str(val).lower() 
                                           for val in first_row if isinstance(val, str))
                    
                    if has_sample_col or has_detail_col or has_sample_row:
                        sample_tables.append(table)
                
                if sample_tables:
                    logger.info(f"Found {len(sample_tables)} sample tables using encoding: {encoding}")
                    break
            
            except Exception as e:
                logger.warning(f"Failed to extract tables with encoding {encoding}: {e}")
        
        # If we found tables, process them
        if sample_tables:
            # Combine tables and standardize columns
            df = pd.concat(sample_tables, ignore_index=True)
            
            # Standardize columns for fruit/veg data
            return self._standardize_fruit_veg_columns(df)
        
        raise ValueError("Table extraction failed for fruit and veg survey")
    
    def _extract_fruit_veg_simple(self, pdf_path: str) -> pd.DataFrame:
        """Simple extraction method for fruit and vegetable data"""
        try:
            logger.info("Trying simple fruit/veg extraction")
            # Try to extract tables from all pages
            tables = tabula.read_pdf(
                pdf_path, 
                pages="all", 
                multiple_tables=True,
                encoding="latin1",
                stream=True,  # Stream mode often works better for loosely structured data
                guess=False
            )
            
            # Keep only tables with sufficient columns (at least 2)
            valid_tables = [table for table in tables if len(table.columns) >= 2]
            
            if valid_tables:
                df = pd.concat(valid_tables, ignore_index=True)
                
                # For simple extraction, assign standard columns based on position
                if len(df.columns) >= 2:
                    # Keep only the first few columns that are most likely to contain relevant info
                    df = df.iloc[:, :min(4, len(df.columns))]
                    
                    # Assign standard names based on position
                    column_names = ["Sample_Number", "Sample_Name", "Pack_Size", "Notes"]
                    df.columns = column_names[:len(df.columns)]
                    
                    # Add missing expected columns
                    for col in ["Page", "Sample_Number", "Sample_Name", "Pack_Size"]:
                        if col not in df.columns:
                            df[col] = None
                    
                    # Add page numbers if missing
                    if "Page" in df.columns and df["Page"].isna().all():
                        df["Page"] = range(1, len(df) + 1)
                    
                    return df
        except Exception as e:
            logger.warning(f"Simple fruit/veg extraction failed: {e}")
        
        raise ValueError("Simple extraction failed for fruit and veg survey")
    
    def _standardize_fruit_veg_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names for fruit and vegetable data"""
        # Standardize column names
        renamed_cols = {}
        for col in df.columns:
            col_lower = str(col).lower()
            if "sample" in col_lower and "name" in col_lower:
                renamed_cols[col] = "Sample_Name"
            elif "product" in col_lower or "item" in col_lower:
                renamed_cols[col] = "Sample_Name"
            elif "sample" in col_lower and "number" in col_lower:
                renamed_cols[col] = "Sample_Number"
            elif "size" in col_lower or "weight" in col_lower or "pack" in col_lower:
                renamed_cols[col] = "Pack_Size"
            elif "page" in col_lower:
                renamed_cols[col] = "Page"
        
        # Rename columns
        if renamed_cols:
            df = df.rename(columns=renamed_cols)
        
        # Add missing columns
        for col in ["Page", "Sample_Number", "Sample_Name", "Pack_Size"]:
            if col not in df.columns:
                df[col] = None
        
        return df
    
    def _extract_fruit_veg_text_based(self, pdf_path: str) -> pd.DataFrame:
        """
        Extract fruit and vegetable data using text-based approach
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            DataFrame with extracted data
        """
        logger.info("Using text-based extraction for fruit and veg survey")
        
        try:
            # Extract text from PDF
            extracted_data = {}
            with open(pdf_path, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)
                
                # Focus on pages likely to contain sample information
                # Usually in appendices or later in the document
                start_page = max(1, int(num_pages * 0.5))  # Start from halfway through
                end_page = num_pages
                
                for page_num in range(start_page - 1, end_page):
                    try:
                        # Get the page
                        page = pdf_reader.pages[page_num]
                        # Extract text
                        page_text = page.extract_text()
                        # Only store pages that might contain sample information
                        if any(term in page_text.lower() for term in ["sample", "product", "weight", "pack size", "g)", "kg)"]):
                            extracted_data[page_num + 1] = page_text
                    except Exception as e:
                        logger.warning(f"Error extracting text from page {page_num + 1}: {str(e)}")
                        continue
            
            # Parse extracted text into structured data
            samples = []
            
            # More comprehensive patterns
            sample_patterns = [
                r"(?:Sample|Number|ID)[\s#:]*(\d+)",
                r"Sample[\s#:]*(\w+)",
                r"Number[\s#:]*(\w+)",
                r"ID[\s#:]*(\w+)"
            ]
            
            name_patterns = [
                r"(?:Sample Name|Product|Item)[:\s]+(.*?)(?:\n|$)",
                r"Name[:\s]+(.*?)(?:\n|$)",
                r"Description[:\s]+(.*?)(?:\n|$)"
            ]
            
            weight_patterns = [
                r"(?:Pack size|Weight|Size)[:\s]+(.*?)(?:\n|$)",
                r"(?:Pack|Weight|Size)[:\s]+(.*?)(?:\n|$)",
                r"(\d+\s*g)",
                r"(\d+\s*kg)"
            ]
            
            # Process each page
            for page_num, text in extracted_data.items():
                # Try each pattern
                sample_number = None
                for pattern in sample_patterns:
                    match = re.search(pattern, text, re.IGNORECASE)
                    if match:
                        sample_number = match.group(1).strip()
                        break
                
                sample_name = None
                for pattern in name_patterns:
                    match = re.search(pattern, text, re.IGNORECASE)
                    if match:
                        sample_name = match.group(1).strip()
                        break
                
                pack_size = None
                for pattern in weight_patterns:
                    match = re.search(pattern, text, re.IGNORECASE)
                    if match:
                        pack_size = match.group(1).strip()
                        break
                
                # If we found any information, consider it a valid sample
                if sample_name or pack_size:
                    sample = {
                        'Page': page_num,
                        'Sample_Number': sample_number,
                        'Sample_Name': sample_name,
                        'Pack_Size': pack_size
                    }
                    samples.append(sample)
            
            # Create DataFrame
            df = pd.DataFrame(samples)
            
            # Remove duplicates
            if 'Sample_Name' in df.columns and not df['Sample_Name'].isna().all():
                df = df.drop_duplicates(subset=['Sample_Name'], keep='first')
            
            if len(df) > 0:
                return df
            
        except Exception as e:
            logger.error(f"Text-based extraction failed: {e}")
        
        raise ValueError("Text-based extraction failed for fruit and veg survey")
    
    def _clean_food_portion_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize food portion data
        
        Args:
            df: Raw food portion DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        # Make a copy
        cleaned_df = df.copy()
        
        # Fill NA values in Food_Name with empty string
        cleaned_df["Food_Name"] = cleaned_df["Food_Name"].fillna("").astype(str)
        
        # Handle Weight_g column to ensure it's numeric
        if "Weight_g" in cleaned_df.columns:
            # Convert to string first to handle mixed types
            cleaned_df["Weight_g"] = cleaned_df["Weight_g"].astype(str)
            
            # Replace common non-numeric characters
            cleaned_df["Weight_g"] = cleaned_df["Weight_g"].str.replace('g', '')
            cleaned_df["Weight_g"] = cleaned_df["Weight_g"].str.replace('kg', '000')
            cleaned_df["Weight_g"] = cleaned_df["Weight_g"].str.replace(',', '.')
            
            # Extract numeric values using regex
            cleaned_df["Weight_g"] = cleaned_df["Weight_g"].apply(self._extract_numeric)
        
        return cleaned_df
    
    def _clean_fruit_veg_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize fruit and vegetable survey data
        
        Args:
            df: Raw fruit and veg DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        # Make a copy
        cleaned_df = df.copy()
        
        # Fill NA values in Sample_Name with empty string
        if "Sample_Name" in cleaned_df.columns:
            cleaned_df["Sample_Name"] = cleaned_df["Sample_Name"].fillna("").astype(str)
        
        # Handle Pack_Size column to ensure it contains extractable weight info
        if "Pack_Size" in cleaned_df.columns:
            # Convert to string first to handle mixed types
            cleaned_df["Pack_Size"] = cleaned_df["Pack_Size"].fillna("").astype(str)
            
            # Standardize weight formats
            cleaned_df["Pack_Size"] = cleaned_df["Pack_Size"].str.replace('g', 'g ')
            cleaned_df["Pack_Size"] = cleaned_df["Pack_Size"].str.replace('kg', 'kg ')
            cleaned_df["Pack_Size"] = cleaned_df["Pack_Size"].str.replace('Kg', 'kg ')
        
        return cleaned_df
    
    def _extract_numeric(self, text: str) -> Optional[float]:
        """
        Extract numeric value from text
        
        Args:
            text: Text containing a number
            
        Returns:
            Extracted number or None
        """
        if pd.isna(text) or not text:
            return None
        
        try:
            # First try direct conversion (handles simple cases)
            return float(text)
        except (ValueError, TypeError):
            # Then try more complex pattern matching
            try:
                # Handle common unit markers
                text = str(text).lower()
                
                # Convert kg to g
                if 'kg' in text:
                    text = text.replace('kg', '')
                    multiplier = 1000
                else:
                    multiplier = 1
                    
                # Remove g unit
                if 'g' in text:
                    text = text.replace('g', '')
                    
                # Clean the text
                text = text.replace(',', '.').strip()
                
                # Extract decimal number pattern
                match = re.search(r'(\d+\.?\d*)', text)
                if match:
                    value = float(match.group(1)) * multiplier
                    return value
                    
                # If there's no match and text contains numbers, try to extract them
                numbers = re.findall(r'\d+', text)
                if numbers:
                    # Return the first number found
                    return float(numbers[0]) * multiplier
            except Exception:
                pass
            
            # Still no match, try last resort pattern
            try:
                # Look for number-range format (e.g., "50-100g") and take average
                range_match = re.search(r'(\d+\.?\d*)\s*[-–—]\s*(\d+\.?\d*)', text)
                if range_match:
                    start, end = float(range_match.group(1)), float(range_match.group(2))
                    return (start + end) / 2
            except Exception:
                pass
            
        return None 