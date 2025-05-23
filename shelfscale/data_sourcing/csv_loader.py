import pandas as pd
import os
import glob
import logging
from typing import Optional, List, Union

import shelfscale.config as config
# from shelfscale.utils import validation # Assuming a validation module might exist or be created later

logger = logging.getLogger(__name__)

class CsvLoader:
    """
    Loads data from CSV files, with specific methods for known datasets.
    """

    def __init__(self):
        """
        Initialize the CsvLoader.
        Configuration for paths and patterns are primarily sourced from shelfscale.config.
        """
        pass # Constructor might be expanded later if needed

    def load_csv(
        self, 
        file_path: str, 
        required_columns: Optional[List[str]] = None, 
        encoding: str = 'utf-8'
    ) -> Optional[pd.DataFrame]:
        """
        Load data from a single CSV file and optionally validate required columns.

        Args:
            file_path: Absolute path to the CSV file.
            required_columns: A list of column names that must be present in the CSV.
            encoding: The encoding to use when reading the CSV. Defaults to 'utf-8'.

        Returns:
            A pandas DataFrame containing the loaded data, or None if loading or validation fails.
        """
        if not os.path.exists(file_path):
            logger.error(f"CSV file not found at: {file_path}")
            return None

        logger.info(f"Loading CSV data from: {file_path} with encoding '{encoding}'")
        
        try:
            df = pd.read_csv(file_path, encoding=encoding)
        except Exception as e:
            logger.error(f"Error loading CSV file '{file_path}': {e}")
            # Try with a different common encoding as a fallback for simple cases
            if encoding == 'utf-8':
                logger.info(f"Attempting fallback encoding 'latin1' for '{file_path}'")
                try:
                    df = pd.read_csv(file_path, encoding='latin1')
                except Exception as e_fallback:
                    logger.error(f"Fallback encoding 'latin1' also failed for '{file_path}': {e_fallback}")
                    return None
            else: # If initial encoding wasn't utf-8, don't try another fallback here
                return None

        if df.empty:
            logger.warning(f"Loaded CSV file '{file_path}' is empty.")
            # Return empty DataFrame as it's valid but contains no data
            return df 

        # --- Validation ---
        if required_columns:
            logger.info(f"Performing column validation for: {file_path}")
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.error(f"Required columns missing from CSV '{file_path}': {missing_columns}. Available columns: {df.columns.tolist()}")
                return None # Or raise error, depending on desired strictness
            else:
                logger.info(f"All required columns found in '{file_path}'.")
        
        # Optional: Use validation.validate_data for a general report
        # if hasattr(validation, 'validate_data'):
        #     report = validation.validate_data(df, dataset_name=os.path.basename(file_path))
        #     logger.info(f"General data validation report for '{file_path}':\n{report}")
        # else:
        #     logger.info("General validation module/function not available. Skipping general validation report.")

        logger.info(f"CSV data loaded successfully from '{file_path}'. Shape: {df.shape}")
        return df

    def load_labelling_data(
        self, 
        file_pattern_csv: Optional[str] = None,
        base_dir: Optional[str] = None,
        required_columns: Optional[List[str]] = None
    ) -> Optional[pd.DataFrame]:
        """
        Load and combine Labelling data from multiple CSV files matching a pattern.

        Args:
            file_pattern_csv: Glob pattern for CSV files. Defaults to config.LABELLING_CSV_GLOB_PATTERN.
            base_dir: Base directory to search for files. Defaults to config.RAW_DATA_DIR.
            required_columns: A list of column names expected in the combined DataFrame.
                              Defaults to config.LABELLING_EXPECTED_COLUMNS.

        Returns:
            A pandas DataFrame containing the combined data from all found CSVs, 
            or None if no files are found or validation fails.
        """
        actual_base_dir = base_dir if base_dir is not None else config.RAW_DATA_DIR
        actual_pattern_csv = file_pattern_csv if file_pattern_csv is not None else config.LABELLING_CSV_GLOB_PATTERN
        actual_required_columns = required_columns if required_columns is not None else config.LABELLING_EXPECTED_COLUMNS

        full_csv_pattern = os.path.join(actual_base_dir, actual_pattern_csv)
        
        logger.info(f"Searching for Labelling CSV files matching pattern: {full_csv_pattern}")
        csv_files = glob.glob(full_csv_pattern)

        if not csv_files:
            logger.warning(f"No Labelling CSV files found matching pattern: {full_csv_pattern}")
            # As per instructions, this method focuses on CSVs first.
            # If XLSX handling were added, we'd check for XLSX files here if no CSVs found.
            return None 

        all_dfs = []
        for csv_file_path in csv_files:
            df = self.load_csv(file_path=csv_file_path, required_columns=None) # Validate columns on combined DF
            if df is not None:
                all_dfs.append(df)
        
        if not all_dfs:
            logger.error(f"Failed to load any data from the found Labelling CSV files: {csv_files}")
            return None

        combined_df = pd.concat(all_dfs, ignore_index=True)
        logger.info(f"Combined {len(all_dfs)} Labelling CSV files. Total rows: {combined_df.shape[0]}")

        # --- Validation for Combined DataFrame ---
        if actual_required_columns:
            logger.info("Performing column validation on combined Labelling data.")
            missing_columns = [col for col in actual_required_columns if col not in combined_df.columns]
            if missing_columns:
                logger.error(f"Required columns missing from combined Labelling data: {missing_columns}. Available columns: {combined_df.columns.tolist()}")
                return None # Or raise error
            else:
                logger.info("All required columns found in combined Labelling data.")

        # Optional: General validation on combined_df
        # if hasattr(validation, 'validate_data'):
        #     report = validation.validate_data(combined_df, dataset_name="Combined Labelling Data")
        #     logger.info(f"General data validation report for combined Labelling data:\n{report}")

        logger.info(f"Labelling data loaded and combined successfully. Shape: {combined_df.shape}")
        return combined_df

if __name__ == '__main__':
    # Example Usage (assuming config.py is set up and files exist)
    logging.basicConfig(level=logging.INFO)

    # This setup is for running this script directly.
    # Ensure PROJECT_ROOT and other necessary config paths are discoverable.
    # This might require setting PYTHONPATH or running from the project root.
    
    csv_loader = CsvLoader()

    # 1. Test generic load_csv (Create a dummy CSV for this to work)
    dummy_csv_path = os.path.join(config.RAW_DATA_DIR, "dummy_test.csv")
    # Create a dummy CSV for testing:
    # pd.DataFrame({'colA': [1, 2], 'colB': ['x', 'y']}).to_csv(dummy_csv_path, index=False)
    if os.path.exists(dummy_csv_path): # Check if dummy exists before trying to load
        print(f"\n--- Loading a generic CSV: {dummy_csv_path} ---")
        generic_df = csv_loader.load_csv(dummy_csv_path, required_columns=['colA'])
        if generic_df is not None:
            print(f"Loaded generic CSV. Shape: {generic_df.shape}")
            print(generic_df.head())
        else:
            print(f"Failed to load generic CSV from {dummy_csv_path}")
    else:
        print(f"Skipping generic CSV load test: dummy file '{dummy_csv_path}' not found.")


    # 2. Test load_labelling_data (This requires actual Labelling CSV files in RAW_DATA_DIR)
    print("\n--- Loading Labelling data (CSVs) ---")
    # Ensure config.LABELLING_CSV_GLOB_PATTERN and config.RAW_DATA_DIR are set.
    # And that config.LABELLING_EXPECTED_COLUMNS are appropriate for your test files.
    # Example: Create dummy Labelling_A.csv and Labelling_B.csv in RAW_DATA_DIR
    # pd.DataFrame({'Food_Name': ['apple', 'banana'], 'Weight_g': [100,120], 'Portion_Size': ['1 medium', '1 large']}).to_csv(os.path.join(config.RAW_DATA_DIR, "Labelling_TestA.csv"), index=False)
    # pd.DataFrame({'Food_Name': ['orange'], 'Weight_g': [150], 'Portion_Size': ['1 medium'], 'Extra_Col': [1]}).to_csv(os.path.join(config.RAW_DATA_DIR, "Labelling_TestB.csv"), index=False)
    
    # Modify config for test if needed, e.g., by setting expected columns to what dummy files have
    # original_labelling_cols = config.LABELLING_EXPECTED_COLUMNS
    # config.LABELLING_EXPECTED_COLUMNS = ['Food_Name', 'Weight_g'] # Example adjustment for dummy files
    
    labelling_df = csv_loader.load_labelling_data() # Uses defaults from config
    if labelling_df is not None:
        print(f"Loaded Labelling data. Shape: {labelling_df.shape}")
        print(labelling_df.head())
    else:
        print("Failed to load Labelling data (CSVs) or no files found matching pattern.")
    
    # config.LABELLING_EXPECTED_COLUMNS = original_labelling_cols # Restore if changed for test
