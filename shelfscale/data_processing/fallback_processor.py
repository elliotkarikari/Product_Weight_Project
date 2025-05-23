"""
Fallback data processor for ShelfScale
Handles cases where processed data files are empty or missing
"""

import os
import pandas as pd
import logging
from typing import Dict, List, Optional
from shelfscale.config_manager import get_config

logger = logging.getLogger(__name__)


class FallbackDataProcessor:
    """
    Processes raw data when processed files are empty or missing
    """
    
    def __init__(self):
        self.mw_data = None
        self.labelling_data = None
    
    def load_mw_raw_data(self) -> Optional[pd.DataFrame]:
        """Load McCance & Widdowson data from raw Excel file"""
        try:
            logger.info(f"Loading raw McCance & Widdowson data from {config.MCCANCE_WIDDOWSON_PATH}")
            
            # Try multiple sheets to find the best data source
            sheet_candidates = [
                config.MW_SHEET_NAME_FOR_MAIN_PY,  # "1.3 Proximates"
                "1.2 Factors",
                "1.4 Inorganics"
            ]
            
            for sheet_name in sheet_candidates:
                try:
                    df = pd.read_excel(config.MCCANCE_WIDDOWSON_PATH, sheet_name=sheet_name)
                    logger.info(f"Successfully loaded {len(df)} rows from sheet '{sheet_name}'")
                    
                    # Clean column names
                    df.columns = df.columns.str.strip()
                    
                    # Standardize key column names
                    column_mapping = {
                        'Food Code': 'Food_Code',
                        'Food Name': 'Food_Name', 
                        'Group': 'Food_Group',
                        'Description': 'Description'
                    }
                    
                    df = df.rename(columns=column_mapping)
                    
                    # Filter out empty rows
                    df = df.dropna(subset=['Food_Code'], how='all')
                    df = df[df['Food_Code'].notna()]
                    
                    logger.info(f"After cleaning: {len(df)} rows with valid food codes")
                    self.mw_data = df
                    return df
                    
                except Exception as e:
                    logger.warning(f"Could not load from sheet '{sheet_name}': {e}")
                    continue
                    
            logger.error("Could not load data from any McCance & Widdowson sheet")
            return None
            
        except Exception as e:
            logger.error(f"Error loading McCance & Widdowson raw data: {e}")
            return None
    
    def create_super_groups_from_raw(self) -> Dict[str, pd.DataFrame]:
        """Create super group data from raw McCance & Widdowson data"""
        if self.mw_data is None:
            self.load_mw_raw_data()
            
        if self.mw_data is None:
            logger.error("No raw McCance & Widdowson data available for super group creation")
            return {}
        
        super_groups = {}
        
        try:
            # Group by Food_Group if available
            if 'Food_Group' in self.mw_data.columns:
                logger.info("Creating super groups based on Food_Group column")
                
                for group_code, group_data in self.mw_data.groupby('Food_Group'):
                    if pd.notna(group_code) and len(group_data) > 0:
                        # Clean group code for filename
                        clean_code = str(group_code).strip()
                        super_groups[clean_code] = group_data.copy()
                        logger.debug(f"Created super group '{clean_code}' with {len(group_data)} items")
                        
            else:
                # Fallback: create groups based on Food_Code patterns
                logger.info("Creating super groups based on Food_Code patterns")
                
                for _, row in self.mw_data.iterrows():
                    food_code = str(row.get('Food_Code', '')).strip()
                    if food_code:
                        # Extract group from food code (first 1-3 characters)
                        group_code = food_code[:2] if len(food_code) >= 2 else food_code
                        
                        if group_code not in super_groups:
                            super_groups[group_code] = []
                        super_groups[group_code].append(row)
                
                # Convert lists to DataFrames
                for code in list(super_groups.keys()):
                    super_groups[code] = pd.DataFrame(super_groups[code])
                    if len(super_groups[code]) == 0:
                        del super_groups[code]
            
            logger.info(f"Created {len(super_groups)} super groups from raw data")
            return super_groups
            
        except Exception as e:
            logger.error(f"Error creating super groups: {e}")
            return {}
    
    def save_super_groups(self, super_groups: Dict[str, pd.DataFrame]) -> int:
        """Save super group data to CSV files"""
        saved_count = 0
        
        try:
            super_group_dir = config.MW_PROCESSED_SUPER_GROUP_PATH
            os.makedirs(super_group_dir, exist_ok=True)
            
            for group_code, group_data in super_groups.items():
                if len(group_data) > 0:
                    filename = f"{group_code}.csv"
                    filepath = os.path.join(super_group_dir, filename)
                    
                    # Add metadata columns if missing
                    if 'Food_Category' not in group_data.columns:
                        group_data['Food_Category'] = 'Unknown'
                    if 'Super_Category' not in group_data.columns:
                        group_data['Super_Category'] = 'Unknown'
                    
                    group_data.to_csv(filepath, index=False)
                    logger.debug(f"Saved {len(group_data)} rows to {filename}")
                    saved_count += 1
                    
            logger.info(f"Successfully saved {saved_count} super group files")
            return saved_count
            
        except Exception as e:
            logger.error(f"Error saving super groups: {e}")
            return 0
    
    def load_labelling_data(self) -> Optional[pd.DataFrame]:
        """Load labelling dataset as alternative data source"""
        try:
            labelling_files = []
            
            # Check for CSV files
            for file in os.listdir(config.RAW_DATA_DIR):
                if file.startswith('Labelling') and file.endswith('.csv'):
                    labelling_files.append(os.path.join(config.RAW_DATA_DIR, file))
            
            if labelling_files:
                # Load the first labelling file found
                filepath = labelling_files[0]
                logger.info(f"Loading labelling data from {filepath}")
                
                df = pd.read_csv(filepath)
                logger.info(f"Loaded {len(df)} rows from labelling dataset")
                
                # Standardize column names
                column_mapping = {
                    'Food name': 'Food_Name',
                    'food_name': 'Food_Name',
                    'Product name': 'Food_Name',
                    'product_name': 'Food_Name'
                }
                
                df = df.rename(columns=column_mapping)
                self.labelling_data = df
                return df
                
            else:
                logger.info("No labelling data files found")
                return None
                
        except Exception as e:
            logger.error(f"Error loading labelling data: {e}")
            return None
    
    def process_all_fallback_data(self) -> Dict[str, pd.DataFrame]:
        """Process all available fallback data sources"""
        results = {}
        
        # 1. Process McCance & Widdowson data
        logger.info("Processing McCance & Widdowson fallback data...")
        mw_data = self.load_mw_raw_data()
        if mw_data is not None:
            results['mw_data'] = mw_data
            
            # Create and save super groups
            super_groups = self.create_super_groups_from_raw()
            if super_groups:
                saved_count = self.save_super_groups(super_groups)
                logger.info(f"Generated {saved_count} super group files from raw data")
                results['super_groups'] = super_groups
        
        # 2. Process labelling data
        logger.info("Processing labelling fallback data...")
        labelling_data = self.load_labelling_data()
        if labelling_data is not None:
            results['labelling_data'] = labelling_data
        
        logger.info(f"Fallback processing complete. Generated {len(results)} data sources.")
        return results


def check_and_repair_empty_super_groups() -> bool:
    """
    Check if super group files are empty and repair them using fallback processing
    
    Returns:
        True if repair was successful, False otherwise
    """
    try:
        super_group_dir = config.MW_PROCESSED_SUPER_GROUP_PATH
        
        if not os.path.exists(super_group_dir):
            logger.warning(f"Super group directory does not exist: {super_group_dir}")
            return False
        
        # Check if files are empty
        csv_files = [f for f in os.listdir(super_group_dir) if f.endswith('.csv')]
        total_rows = 0
        
        for csv_file in csv_files[:5]:  # Check first 5 files
            filepath = os.path.join(super_group_dir, csv_file)
            try:
                df = pd.read_csv(filepath)
                total_rows += len(df)
            except:
                continue
        
        if total_rows == 0:
            logger.warning("Super group files are empty. Initiating fallback processing...")
            
            processor = FallbackDataProcessor()
            results = processor.process_all_fallback_data()
            
            if 'super_groups' in results:
                logger.info("Successfully repaired empty super group files")
                return True
            else:
                logger.error("Failed to generate super group data from fallback processing")
                return False
        else:
            logger.info(f"Super group files contain {total_rows} rows - no repair needed")
            return True
            
    except Exception as e:
        logger.error(f"Error during super group repair: {e}")
        return False


if __name__ == "__main__":
    # Test the fallback processor
    logging.basicConfig(level=logging.INFO)
    
    processor = FallbackDataProcessor()
    results = processor.process_all_fallback_data()
    
    print("Fallback processing results:")
    for key, data in results.items():
        if isinstance(data, pd.DataFrame):
            print(f"  {key}: {len(data)} rows")
        else:
            print(f"  {key}: {type(data)}") 