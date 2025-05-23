"""
Raw data processing module for ShelfScale datasets
Processes data from Raw Data folder to the Processed folder structure
"""

import os
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
import re
import glob

import shelfscale.config as config
from shelfscale.data_sourcing.pdf_extraction import PDFExtractor
from shelfscale.data_sourcing.excel_loader import ExcelLoader # Import ExcelLoader
from shelfscale.data_sourcing.csv_loader import CsvLoader # Import CsvLoader
from shelfscale.data_processing.cleaner import DataCleaner
from shelfscale.data_processing.categorization import FoodCategorizer
from shelfscale.utils.helpers import extract_numeric_value


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RawDataProcessor:
    """
    Processes raw data from various sources into standardized formats
    for use in the ShelfScale system
    """
    
    def __init__(self, 
                 raw_data_dir: Optional[str] = None, 
                 processed_data_dir: Optional[str] = None,
                 pdf_cache_dir: Optional[str] = None):
        """
        Initialize the processor
        
        Args:
            raw_data_dir: Path to raw data directory. Defaults to config.RAW_DATA_DIR.
            processed_data_dir: Path to processed data directory. Defaults to config.PROCESSED_DATA_DIR.
            pdf_cache_dir: Path for PDFExtractor's cache. Defaults to config.RAW_PROCESSOR_TEMP_CACHE_DIR.
        """
        self.raw_data_dir = raw_data_dir if raw_data_dir is not None else config.RAW_DATA_DIR
        self.processed_data_dir = processed_data_dir if processed_data_dir is not None else config.PROCESSED_DATA_DIR
        pdf_temp_cache = pdf_cache_dir if pdf_cache_dir is not None else config.RAW_PROCESSOR_TEMP_CACHE_DIR
        
        # Directory creation is handled by config.py, so _create_directory_structure is removed.
        
        # Initialize helper classes
        self.pdf_extractor = PDFExtractor(cache_dir=pdf_temp_cache)
        self.excel_loader = ExcelLoader()
        self.csv_loader = CsvLoader()
        self.data_cleaner = DataCleaner(categorize_foods=True) 
        self.food_categorizer = FoodCategorizer()
    
    # _create_directory_structure method is removed as directories are created in config.py
    
    def process_all(self) -> Dict[str, pd.DataFrame]:
        """
        Process all available raw data
        
        Returns:
            Dictionary of processed datasets
        """
        processed_data = {}
        
        # Process McCance Widdowson data
        logger.info("Processing McCance Widdowson data...")
        mw_data = self.process_mccance_widdowson()
        if mw_data is not None:
            processed_data["mw_data"] = mw_data
        
        # Process Food Portion Sizes data
        logger.info("Processing Food Portion Sizes data...")
        fps_data = self.process_food_portion_sizes()
        if fps_data is not None:
            processed_data["food_portion_sizes"] = fps_data
        
        # Process Fruit and Vegetable Survey data
        logger.info("Processing Fruit and Vegetable Survey data...")
        fruit_veg_data = self.process_fruit_veg_survey()
        if fruit_veg_data is not None:
            processed_data["fruit_veg_survey"] = fruit_veg_data
        
        # Process Labelling dataset
        logger.info("Processing Labelling dataset...")
        labelling_data = self.process_labelling_data()
        if labelling_data is not None:
            processed_data["labelling_data"] = labelling_data
            
        return processed_data
    
    def process_mccance_widdowson(self) -> Optional[pd.DataFrame]:
        """
        Process McCance Widdowson Excel data
        
        Returns:
            Processed DataFrame or None if processing failed
        """
        try:
            # Use ExcelLoader to load and validate McCance & Widdowson data
            # It uses config.MCCANCE_WIDDOWSON_PATH, config.MW_FACTORS_SHEET_NAME (as primary), 
            # and config.MW_EXPECTED_COLUMNS by default.
            df = self.excel_loader.load_mccance_widdowson()

            if df is None:
                logger.error("Failed to load McCance & Widdowson data using ExcelLoader.")
                return None
            
            logger.info(f"Successfully loaded and validated McCance & Widdowson data using ExcelLoader. Shape: {df.shape}")

            # Standardize column names (if not already perfectly matching expected schema names)
            # Note: ExcelLoader validation checks for presence of expected columns, 
            # but names might have slight variations if not strictly enforced by a rename step in loader.
            # The current ExcelLoader does not standardize column names post-load, so we do it here if needed.
            # config.MW_EXPECTED_COLUMNS are ['Food_Name', 'Food_Code', 'Food_Group']
            col_renames = {}
            for col in df.columns:
                col_lower = col.lower()
                if "food" in col_lower and "name" in col_lower:
                    col_renames[col] = "Food_Name"
                elif col == "Food Name":
                    col_renames[col] = "Food_Name"
                elif "food" in col_lower and "code" in col_lower:
                    col_renames[col] = "Food_Code"
                elif "group" in col_lower:
                    col_renames[col] = "Food_Group"
                elif "description" in col_lower:
                    col_renames[col] = "Description"
            
            if col_renames:
                df = df.rename(columns=col_renames)
            
            # Ensure the main expected columns are present (as per config.MW_EXPECTED_COLUMNS)
            # This is already done by excel_loader.load_mccance_widdowson()
            # If column names need standardization beyond what ExcelLoader provides, it can be done here.
            # For example, if ExcelLoader loaded "Food name" but config expects "Food_Name"
            # This step is more about ensuring consistency for the next steps.
            # Let's assume the loader provides columns as per config.MW_EXPECTED_COLUMNS
            # or we add a renaming step here based on a mapping if needed.
            # For now, we trust the loader provides the expected names or compatible ones.

            # Process by food group (super group) if 'Food_Group' column exists
            if config.MW_EXPECTED_COLUMNS[2] in df.columns: # Assuming Food_Group is the third expected column
                food_group_col_name = config.MW_EXPECTED_COLUMNS[2] # typically 'Food_Group'
                food_groups = df[food_group_col_name].unique()
                super_group_output_dir = os.path.join(self.processed_data_dir, config.MW_REDUCED_SUPER_GROUP_SUBDIR)

                for group in food_groups:
                    if pd.isna(group) or not str(group).strip():
                        logger.warning("Skipping NA or empty food group name during M&W processing.")
                        continue
                        
                    group_df = df[df[food_group_col_name] == group].copy()
                    if not group_df.empty:
                        # Clean and save the group
                        clean_group_df = self.data_cleaner.clean(group_df) 
                        
                        group_filename = str(group).replace("/", " and ").replace("&", "and").replace(",", "")
                        group_filename = re.sub(r'[^\w\s-]', '', group_filename).strip().replace(' ', '_')
                        
                        output_path = os.path.join(super_group_output_dir, f"{group_filename}.csv")
                        clean_group_df.to_csv(output_path, index=False)
                        logger.info(f"Saved {len(clean_group_df)} items for M&W group '{group}' to {output_path}")
            else:
                logger.warning(f"'{config.MW_EXPECTED_COLUMNS[2]}' column not found in McCance & Widdowson data. Skipping group-specific processing.")

            # Save the full dataset using configured path
            full_dataset_output_path = config.MW_PROCESSED_FULL_FILE
            # Apply DataCleaner to the whole dataframe before saving the full version
            df_to_save = self.data_cleaner.clean(df.copy())
            df_to_save.to_csv(full_dataset_output_path, index=False)
            logger.info(f"Saved full M&W dataset ({len(df_to_save)} items) to {full_dataset_output_path}")
            
            return df_to_save # Return the cleaned full dataframe
        
        except Exception as e:
            logger.error(f"Error processing McCance Widdowson data: {e}", exc_info=True)
            return None
    
    def process_food_portion_sizes(self) -> Optional[pd.DataFrame]:
        """
        Process Food Portion Sizes PDF
        
        Returns:
            Processed DataFrame or None if processing failed
        """
        fps_file_path = config.FOOD_PORTION_PDF_PATH
        
        if not os.path.exists(fps_file_path):
            logger.error(f"Food Portion Sizes PDF not found: {fps_file_path}")
            return None
        
        try:
            # extract_food_portion_sizes now handles initial cleaning (incl. numeric Weight_g) and schema validation.
            df_extracted = self.pdf_extractor.extract_food_portion_sizes(pdf_path=fps_file_path)
            
            if df_extracted is None or df_extracted.empty: # Check if None or empty
                logger.error("No data extracted or data was empty from Food Portion Sizes PDF by PDFExtractor.")
                return None # Return None if extraction failed
            
            logger.info(f"Successfully extracted and schema-validated Food Portion Sizes data. Shape: {df_extracted.shape}")

            # Apply further general cleaning using DataCleaner
            df_cleaned_further = self.data_cleaner.clean(df_extracted.copy()) 
            
            # Ensure 'Normalized_Weight' column exists, using 'Weight_g' if available and numeric.
            # PDFExtractor's _clean_food_portion_data should have made 'Weight_g' numeric.
            if 'Weight_g' in df_cleaned_further.columns:
                if pd.api.types.is_numeric_dtype(df_cleaned_further['Weight_g']):
                    df_cleaned_further['Normalized_Weight'] = df_cleaned_further['Weight_g']
                    logger.info("Assigned 'Weight_g' to 'Normalized_Weight' for Food Portion Sizes data.")
                else:
                    # Attempt conversion again if it's not numeric for some reason (e.g. object type after cleaning)
                    df_cleaned_further['Normalized_Weight'] = pd.to_numeric(df_cleaned_further['Weight_g'], errors='coerce')
                    if df_cleaned_further['Normalized_Weight'].isnull().all() and not df_cleaned_further['Weight_g'].isnull().all(): # check if all became NaN vs some
                         logger.warning(f"'Weight_g' (dtype: {df_cleaned_further['Weight_g'].dtype}) could not be fully converted to numeric for 'Normalized_Weight'. Check data.")
                    elif not df_cleaned_further['Normalized_Weight'].isnull().all(): # At least some values converted
                         logger.info("Converted 'Weight_g' to numeric and assigned to 'Normalized_Weight'.")
                    # else all were null to begin with or became null, no specific message needed beyond earlier checks

            elif 'Normalized_Weight' not in df_cleaned_further.columns: # If Weight_g wasn't there, ensure col exists
                 df_cleaned_further['Normalized_Weight'] = np.nan # Use np.nan for consistency
                 logger.warning("'Weight_g' column not found. 'Normalized_Weight' column added with NaNs.")
            
            # Save "processed" data (after DataCleaner)
            processed_output_path = config.PROCESSED_FPS_PATH
            df_cleaned_further.to_csv(processed_output_path, index=False)
            logger.info(f"Saved processed (after DataCleaner) Food Portion Sizes data ({len(df_cleaned_further)} items) to {processed_output_path}")
            
            # Save the "raw extracted" data (output from PDFExtractor, before DataCleaner)
            raw_extracted_output_path = config.RAW_EXTRACTED_FPS_PATH
            df_extracted.to_csv(raw_extracted_output_path, index=False)
            logger.info(f"Saved raw extracted (from PDFExtractor) Food Portion Sizes data ({len(df_extracted)} items) to {raw_extracted_output_path}")
            
            return df_cleaned_further
        
        except Exception as e:
            logger.error(f"Error processing Food Portion Sizes data: {e}", exc_info=True)
            return None
    
    def process_fruit_veg_survey(self) -> Optional[pd.DataFrame]:
        """
        Process Fruit and Vegetable Survey PDF
        
        Returns:
            Processed DataFrame or None if processing failed
        """
        fvs_glob_pattern = os.path.join(self.raw_data_dir, config.FRUIT_VEG_PDF_GLOB_PATTERN)
        fvs_files = glob.glob(fvs_glob_pattern)
        
        if not fvs_files:
            logger.error(f"No Fruit and Vegetable Survey PDF found using pattern: {fvs_glob_pattern}")
            return None
        
        all_extracted_dfs = []
        for fvs_file_path in fvs_files:
            logger.info(f"Processing Fruit and Vegetable Survey PDF: {fvs_file_path}")
            try:
                # extract_fruit_veg_survey now handles initial cleaning and schema validation.
                df_single_pdf = self.pdf_extractor.extract_fruit_veg_survey(pdf_path=fvs_file_path)
                
                if df_single_pdf is not None and not df_single_pdf.empty:
                    logger.info(f"Extracted {len(df_single_pdf)} items from {os.path.basename(fvs_file_path)}")
                    df_single_pdf['Source_PDF'] = os.path.basename(fvs_file_path) # Corrected variable name
                    all_extracted_dfs.append(df_single_pdf)
                else:
                    logger.warning(f"No data extracted or empty DataFrame from {os.path.basename(fvs_file_path)}")
            except Exception as e:
                logger.error(f"Error processing FVS PDF {os.path.basename(fvs_file_path)}: {e}", exc_info=True)
        
        if not all_extracted_dfs:
            logger.error("No data extracted from any Fruit and Vegetable Survey PDF files.")
            return None # Return None if no data from any PDF
        
        # Combine all extracted data
        combined_raw_extracted_df = pd.concat(all_extracted_dfs, ignore_index=True)
        logger.info(f"Combined data from {len(all_extracted_dfs)} FVS PDF(s). Total rows: {len(combined_raw_extracted_df)}.")

        # Save the combined "raw extracted" data (after PDFExtractor's internal cleaning and validation)
        raw_extracted_output_path = config.RAW_EXTRACTED_FVS_TEXT_PATH # Path for combined raw FVS data
        combined_raw_extracted_df.to_csv(raw_extracted_output_path, index=False)
        logger.info(f"Saved combined raw extracted FVS data ({len(combined_raw_extracted_df)} items) to {raw_extracted_output_path}")

        # Apply further general cleaning using DataCleaner
        df_cleaned_further = self.data_cleaner.clean(combined_raw_extracted_df.copy())
            
        # Manual Normalized_Weight loop is removed. 
        # Numeric weight extraction from 'Pack_Size' (string) should be handled by WeightExtractor later.
        # If a temporary numeric weight column was needed here for some intermediate step:
        # if 'Pack_Size' in df_cleaned_further.columns:
        #    df_cleaned_further['Normalized_Weight_Temp'] = df_cleaned_further['Pack_Size'].apply(lambda x: extract_numeric_value(str(x)))
        #    logger.info("Created temporary numeric weight 'Normalized_Weight_Temp' for FVS data.")
        
        # Save "processed" data (after DataCleaner)
        processed_output_path = config.PROCESSED_FVS_TABLES_PATH # Path for combined processed FVS data
        df_cleaned_further.to_csv(processed_output_path, index=False)
        logger.info(f"Saved processed (after DataCleaner) FVS data ({len(df_cleaned_further)} items) to {processed_output_path}")
        
        return df_cleaned_further
    
    def process_labelling_data(self) -> Optional[pd.DataFrame]:
        """
        Process Labelling dataset CSV/Excel files
        
        Returns:
            Processed DataFrame or None if processing failed
        """
        try:
            # CsvLoader.load_labelling_data uses config.RAW_DATA_DIR, 
            # config.LABELLING_CSV_GLOB_PATTERN, and config.LABELLING_EXPECTED_COLUMNS by default.
            # It currently handles only CSV files.
            logger.info("Processing Labelling dataset (CSVs using CsvLoader)...")
            combined_df = self.csv_loader.load_labelling_data()

            # Note: The original code also searched for XLSX files. 
            # If XLSX files for labelling data are also required, this method needs extension:
            # 1. Add a similar glob pattern for XLSX to config (e.g., config.LABELLING_XLSX_GLOB_PATTERN).
            # 2. Use self.excel_loader (perhaps a new generic method in it) to load them.
            # 3. Concatenate DataFrames from CSV and XLSX sources.
            # For now, proceeding with CSV-only as per CsvLoader's current capability.

            if combined_df is None or combined_df.empty:
                logger.warning("No Labelling data loaded from CSV files or data was empty by CsvLoader.")
                return None 
            
            logger.info(f"Successfully loaded Labelling data using CsvLoader. Shape: {combined_df.shape}")
            # CsvLoader.load_labelling_data combines multiple CSVs. If 'Source_File' is needed,
            # CsvLoader would need to be enhanced to add it, or files loaded individually here.
            # Assuming for now that a combined DataFrame is the desired starting point.

            # Clean the data using DataCleaner
            clean_df = self.data_cleaner.clean(combined_df.copy()) # Use a copy
        
            # Categorize the data
            food_name_col = None
            # Prioritize specific column names, then try broader search
            for col_candidate in ['Food_Name', 'Food Name', 'name', 'description', 'Name', 'Description', 'Foodname']:
                if col_candidate in clean_df.columns:
                    food_name_col = col_candidate
                    break
            if not food_name_col: # If no exact match, then search more broadly
                for col in clean_df.columns:
                    col_lower = col.lower()
                    if 'food' in col_lower and 'name' in col_lower: food_name_col = col; break
                    elif 'name' in col_lower: food_name_col = col; break
                    elif 'description' in col_lower: food_name_col = col; break
            
            if food_name_col:
                logger.info(f"Using column '{food_name_col}' for food categorization in Labelling data.")
                categorized_df = self.food_categorizer.clean_food_categories(
                    clean_df, food_name_col, 'Food_Category', 'Super_Category'
                )
            else:
                logger.warning("Could not find a suitable food name column for categorization in Labelling data. Skipping.")
                categorized_df = clean_df # Assign clean_df to categorized_df before adding columns
                if 'Food_Category' not in categorized_df.columns: categorized_df['Food_Category'] = 'Unknown'
                if 'Super_Category' not in categorized_df.columns: categorized_df['Super_Category'] = 'Unknown'
            
            # Save processed data using configured path
            processed_output_path = config.PROCESSED_LABELLING_PATH
            categorized_df.to_csv(processed_output_path, index=False)
            logger.info(f"Saved processed Labelling data ({len(categorized_df)} items) to {processed_output_path}")
            
            return categorized_df
            
        except Exception as e:
            logger.error(f"Error processing Labelling data: {e}", exc_info=True)
            return None # Return None if error during processing


# Helper function to run the processor
def process_raw_data(
    raw_data_dir: Optional[str] = None, 
    processed_data_dir: Optional[str] = None
    # pdf_cache_dir is not exposed here, uses default from RawDataProcessor init
) -> Dict[str, pd.DataFrame]:
    """
    Process all raw data. Uses paths from config.py by default.
    
    Args:
        raw_data_dir: Optional. Path to raw data directory.
        processed_data_dir: Optional. Path to processed data directory.
        
    Returns:
        Dictionary of processed datasets
    """
    # RawDataProcessor will use config defaults if these are None
    processor = RawDataProcessor(raw_data_dir=raw_data_dir, processed_data_dir=processed_data_dir)
    return processor.process_all()


if __name__ == "__main__":
    # If run as a script, process all raw data using default (config-based) paths
    process_raw_data() 