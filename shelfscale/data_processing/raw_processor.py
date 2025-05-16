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

from shelfscale.data_sourcing.pdf_extraction import PDFExtractor
from shelfscale.data_processing.cleaner import DataCleaner
from shelfscale.data_processing.categorization import FoodCategorizer
from shelfscale.utils.helpers import get_path, extract_numeric_value

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RawDataProcessor:
    """
    Processes raw data from various sources into standardized formats
    for use in the ShelfScale system
    """
    
    def __init__(self, raw_data_dir: str = "Data/Raw Data", processed_data_dir: str = "Data/Processed"):
        """
        Initialize the processor
        
        Args:
            raw_data_dir: Path to raw data directory
            processed_data_dir: Path to processed data directory
        """
        self.raw_data_dir = get_path(raw_data_dir)
        self.processed_data_dir = get_path(processed_data_dir)
        
        # Create required directories if they don't exist
        self._create_directory_structure()
        
        # Initialize helper classes
        self.pdf_extractor = PDFExtractor(cache_dir=os.path.join(self.processed_data_dir, "temp"))
        self.data_cleaner = DataCleaner(categorize_foods=True)
        self.food_categorizer = FoodCategorizer()
    
    def _create_directory_structure(self) -> None:
        """Create the necessary directory structure for processed data"""
        # Main directories
        os.makedirs(self.processed_data_dir, exist_ok=True)
        
        # Subdirectories for each data type
        subdirs = [
            "FoodPortionSized",
            "Fruit and Veg Sample reports/tables",
            "Fruit and Veg Sample reports/text",
            "MW_DataReduction/Reduced Individual Tables",
            "MW_DataReduction/Reduced Super Group",
            "MW_DataReduction/Reduced Super Group/Cleaned",
            "MW_DataReduction/Reduced Total",
            "ReducedwithWeights",
            "Sample Reports"
        ]
        
        for subdir in subdirs:
            os.makedirs(os.path.join(self.processed_data_dir, subdir), exist_ok=True)
    
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
        # Look for the McCance Widdowson Excel file
        mw_file = os.path.join(self.raw_data_dir, "McCance_Widdowsons_2021.xlsx")
        
        if not os.path.exists(mw_file):
            logger.error(f"McCance Widdowson file not found: {mw_file}")
            return None
        
        try:
            # Load the Excel file
            logger.info(f"Loading McCance Widdowson data from {mw_file}")
            
            # Get all sheets
            excel_file = pd.ExcelFile(mw_file)
            available_sheets = excel_file.sheet_names
            logger.info(f"Available sheets: {', '.join(available_sheets)}")
            
            # Find the Factors sheet which contains food data
            factors_sheet = None
            for sheet in available_sheets:
                if "factor" in sheet.lower():
                    factors_sheet = sheet
                    break
            
            if factors_sheet is None:
                logger.warning("Could not find Factors sheet, using first sheet")
                factors_sheet = available_sheets[0]
            
            # Load the data
            df = pd.read_excel(mw_file, sheet_name=factors_sheet)
            logger.info(f"Loaded {len(df)} items from {factors_sheet} sheet")
            
            # Check if data is valid
            if len(df) == 0:
                logger.error("No data found in McCance Widdowson file")
                return None
            
            # Standardize column names
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
            
            # Process by food group (super group)
            food_groups = df["Food_Group"].unique() if "Food_Group" in df.columns else []
            
            # Create an individual table for each food group
            group_dfs = {}
            super_group_dir = os.path.join(self.processed_data_dir, "MW_DataReduction/Reduced Super Group")
            for group in food_groups:
                if pd.isna(group):
                    continue
                    
                group_df = df[df["Food_Group"] == group].copy()
                if len(group_df) > 0:
                    # Clean and save the group
                    clean_group_df = self.data_cleaner.clean(group_df)
                    
                    # Create a filename-friendly version of the group name
                    group_filename = group.replace("/", " and ").replace("&", "and").replace(",", "")
                    group_filename = re.sub(r'[^\w\s]', '', group_filename)
                    group_filename = group_filename.strip()
                    
                    # Save the group to CSV
                    output_path = os.path.join(super_group_dir, f"{group_filename}.csv")
                    clean_group_df.to_csv(output_path, index=False)
                    logger.info(f"Saved {len(clean_group_df)} items to {output_path}")
                    
                    group_dfs[group] = clean_group_df
            
            # Save the full dataset
            output_path = os.path.join(self.processed_data_dir, "MW_DataReduction/Reduced Total/McCance_Widdowson_Full.csv")
            df.to_csv(output_path, index=False)
            logger.info(f"Saved full dataset ({len(df)} items) to {output_path}")
            
            return df
        
        except Exception as e:
            logger.error(f"Error processing McCance Widdowson data: {e}")
            return None
    
    def process_food_portion_sizes(self) -> Optional[pd.DataFrame]:
        """
        Process Food Portion Sizes PDF
        
        Returns:
            Processed DataFrame or None if processing failed
        """
        # Look for the Food Portion Sizes PDF
        fps_file = os.path.join(self.raw_data_dir, "Food_Portion_Sizes.pdf")
        
        if not os.path.exists(fps_file):
            logger.error(f"Food Portion Sizes PDF not found: {fps_file}")
            return None
        
        try:
            # Extract data from PDF
            df = self.pdf_extractor.extract_food_portion_sizes(fps_file)
            logger.info(f"Extracted {len(df)} items from Food Portion Sizes PDF")
            
            # Check if data is valid
            if len(df) == 0:
                logger.error("No data extracted from Food Portion Sizes PDF")
                return None
            
            # Clean the data
            clean_df = self.data_cleaner.clean(df)
            
            # Process weight information
            weight_cols = [col for col in clean_df.columns if "weight" in col.lower() or "size" in col.lower()]
            
            if weight_cols:
                # Add a Normalized_Weight column
                clean_df['Normalized_Weight'] = np.nan
                
                # Process each row
                for idx, row in clean_df.iterrows():
                    for col in weight_cols:
                        if col in row and not pd.isna(row[col]):
                            try:
                                weight = extract_numeric_value(str(row[col]))
                                if weight is not None and weight > 0:
                                    clean_df.loc[idx, 'Normalized_Weight'] = weight
                                    break
                            except:
                                pass
            
            # Save processed data
            output_dir = os.path.join(self.processed_data_dir, "FoodPortionSized")
            output_path = os.path.join(output_dir, "Processed_Food_Portion_Sizes.csv")
            clean_df.to_csv(output_path, index=False)
            logger.info(f"Saved processed food portion sizes ({len(clean_df)} items) to {output_path}")
            
            # Save the raw extracted data
            raw_output_path = os.path.join(output_dir, "FoodPortionSizes_PDF.csv")
            df.to_csv(raw_output_path, index=False)
            logger.info(f"Saved raw food portion sizes ({len(df)} items) to {raw_output_path}")
            
            return clean_df
        
        except Exception as e:
            logger.error(f"Error processing Food Portion Sizes data: {e}")
            return None
    
    def process_fruit_veg_survey(self) -> Optional[pd.DataFrame]:
        """
        Process Fruit and Vegetable Survey PDF
        
        Returns:
            Processed DataFrame or None if processing failed
        """
        # Look for the Fruit and Vegetable Survey PDF files
        fvs_files = glob.glob(os.path.join(self.raw_data_dir, "*fruit*veg*.pdf"))
        
        if not fvs_files:
            logger.error("No Fruit and Vegetable Survey PDF found")
            return None
        
        # Process each found PDF
        all_dfs = []
        
        for fvs_file in fvs_files:
            logger.info(f"Processing Fruit and Vegetable Survey PDF: {fvs_file}")
            
            try:
                # Extract data from PDF
                df = self.pdf_extractor.extract_fruit_veg_survey(fvs_file)
                logger.info(f"Extracted {len(df)} items from {os.path.basename(fvs_file)}")
                
                # Add source column
                df['Source_PDF'] = os.path.basename(fvs_file)
                
                all_dfs.append(df)
            except Exception as e:
                logger.error(f"Error processing {os.path.basename(fvs_file)}: {e}")
        
        if not all_dfs:
            logger.error("No data extracted from any Fruit and Vegetable Survey PDF")
            return None
        
        # Combine all extracted data
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        # Clean the data
        clean_df = self.data_cleaner.clean(combined_df)
        
        # Process weight information from Pack_Size
        if 'Pack_Size' in clean_df.columns:
            # Add a Normalized_Weight column
            clean_df['Normalized_Weight'] = np.nan
            
            # Process each row
            for idx, row in clean_df.iterrows():
                if not pd.isna(row['Pack_Size']):
                    try:
                        weight = extract_numeric_value(str(row['Pack_Size']))
                        if weight is not None and weight > 0:
                            clean_df.loc[idx, 'Normalized_Weight'] = weight
                    except:
                        pass
        
        # Save data to table and text folders
        table_dir = os.path.join(self.processed_data_dir, "Fruit and Veg Sample reports/tables")
        text_dir = os.path.join(self.processed_data_dir, "Fruit and Veg Sample reports/text")
        
        # Save combined data
        output_path = os.path.join(table_dir, "Combined_Fruit_Veg_Samples.csv")
        clean_df.to_csv(output_path, index=False)
        logger.info(f"Saved combined fruit and veg samples ({len(clean_df)} items) to {output_path}")
        
        # Save raw data
        raw_output_path = os.path.join(text_dir, "Raw_Fruit_Veg_Samples.csv")
        combined_df.to_csv(raw_output_path, index=False)
        
        return clean_df
    
    def process_labelling_data(self) -> Optional[pd.DataFrame]:
        """
        Process Labelling dataset CSV/Excel files
        
        Returns:
            Processed DataFrame or None if processing failed
        """
        # Look for the Labelling dataset files
        labelling_files = []
        labelling_files.extend(glob.glob(os.path.join(self.raw_data_dir, "Labelling*.csv")))
        labelling_files.extend(glob.glob(os.path.join(self.raw_data_dir, "Labelling*.xlsx")))
        
        if not labelling_files:
            logger.error("No Labelling dataset files found")
            return None
        
        # Process each found file
        all_dfs = []
        
        for labelling_file in labelling_files:
            logger.info(f"Processing Labelling dataset: {labelling_file}")
            
            try:
                # Load the data
                if labelling_file.endswith('.csv'):
                    df = pd.read_csv(labelling_file)
                else:  # Excel file
                    df = pd.read_excel(labelling_file)
                
                logger.info(f"Loaded {len(df)} items from {os.path.basename(labelling_file)}")
                
                # Add source column
                df['Source_File'] = os.path.basename(labelling_file)
                
                all_dfs.append(df)
            except Exception as e:
                logger.error(f"Error processing {os.path.basename(labelling_file)}: {e}")
        
        if not all_dfs:
            logger.error("No data loaded from any Labelling dataset")
            return None
        
        # Combine all loaded data
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        # Clean the data
        clean_df = self.data_cleaner.clean(combined_df)
        
        # Categorize the data
        food_name_col = None
        for col in clean_df.columns:
            if 'food' in col.lower() and 'name' in col.lower():
                food_name_col = col
                break
            elif 'name' in col.lower():
                food_name_col = col
                break
            elif 'description' in col.lower():
                food_name_col = col
                break
        
        if food_name_col:
            categorized_df = self.food_categorizer.clean_food_categories(
                clean_df,
                food_name_col,
                'Food_Category',
                'Super_Category'
            )
        else:
            categorized_df = clean_df
            categorized_df['Food_Category'] = 'Unknown'
            categorized_df['Super_Category'] = 'Unknown'
        
        # Save processed data
        output_path = os.path.join(self.processed_data_dir, "ReducedwithWeights/Processed_Labelling_Data.csv")
        categorized_df.to_csv(output_path, index=False)
        logger.info(f"Saved processed labelling data ({len(categorized_df)} items) to {output_path}")
        
        return categorized_df


# Helper function to run the processor
def process_raw_data(
    raw_data_dir: str = "Data/Raw Data", 
    processed_data_dir: str = "Data/Processed"
) -> Dict[str, pd.DataFrame]:
    """
    Process all raw data
    
    Args:
        raw_data_dir: Path to raw data directory
        processed_data_dir: Path to processed data directory
        
    Returns:
        Dictionary of processed datasets
    """
    processor = RawDataProcessor(raw_data_dir, processed_data_dir)
    return processor.process_all()


if __name__ == "__main__":
    # If run as a script, process all raw data
    process_raw_data() 