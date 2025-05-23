import pandas as pd
import os
import logging
from typing import Optional, List

import shelfscale.config as config
# from shelfscale.utils import validation # Assuming a validation module might exist or be created later

logger = logging.getLogger(__name__)

class ExcelLoader:
    """
    Loads data from Excel files, with specific methods for known datasets.
    """

    def __init__(self):
        """
        Initialize the ExcelLoader.
        Configuration for paths and sheet names are primarily sourced from shelfscale.config.
        """
        pass # Constructor might be expanded later if needed

    def load_mccance_widdowson(
        self, 
        file_path: Optional[str] = None, 
        sheet_name_primary: Optional[str] = None,
        fallback_sheet_names: Optional[List[str]] = None,
        expected_columns: Optional[List[str]] = None
    ) -> Optional[pd.DataFrame]:
        """
        Load and validate McCance and Widdowson's food composition data from an Excel file.

        Args:
            file_path: Path to the Excel file. Defaults to config.MCCANCE_WIDDOWSON_PATH.
            sheet_name_primary: Primary sheet name to try. Defaults to config.MW_FACTORS_SHEET_NAME.
                                (Note: main.py uses config.MW_SHEET_NAME_FOR_MAIN_PY, raw_processor uses config.MW_FACTORS_SHEET_NAME.
                                 This loader will default to MW_FACTORS_SHEET_NAME as it's more for detailed data loading)
            fallback_sheet_names: A list of alternative sheet names to try if the primary one fails.
                                  Defaults to a common list e.g. ["Factors", "Sheet1", etc.]
            expected_columns: A list of column names expected to be in the DataFrame. 
                              Defaults to config.MW_EXPECTED_COLUMNS.

        Returns:
            A pandas DataFrame containing the loaded data, or None if loading or validation fails.
        """
        actual_file_path = file_path if file_path is not None else config.MCCANCE_WIDDOWSON_PATH
        primary_sheet = sheet_name_primary if sheet_name_primary is not None else config.MW_FACTORS_SHEET_NAME
        
        # Define default fallback sheets if none provided. This can be expanded.
        default_fallbacks = ["Factors", "A1. Calculated values", "Sheet1", "Data"] 
        actual_fallback_sheets = fallback_sheet_names if fallback_sheet_names is not None else default_fallbacks
        
        actual_expected_columns = expected_columns if expected_columns is not None else config.MW_EXPECTED_COLUMNS

        if not os.path.exists(actual_file_path):
            logger.error(f"McCance & Widdowson file not found at: {actual_file_path}")
            return None

        logger.info(f"Loading McCance & Widdowson data from: {actual_file_path}")

        df = None
        sheets_to_try = [primary_sheet] + actual_fallback_sheets
        
        # Try to get all sheet names from the file to add them to the list of sheets to try
        try:
            excel_file_sheets = pd.ExcelFile(actual_file_path).sheet_names
            logger.info(f"Available sheets in the Excel file: {excel_file_sheets}")
            for s_name in excel_file_sheets:
                if s_name not in sheets_to_try:
                    sheets_to_try.append(s_name)
        except Exception as e:
            logger.warning(f"Could not read all sheet names from Excel file '{actual_file_path}': {e}")

        loaded_sheet_name = None
        for sheet in sheets_to_try:
            try:
                logger.info(f"Attempting to load sheet: '{sheet}'")
                temp_df = pd.read_excel(actual_file_path, sheet_name=sheet)
                if not temp_df.empty:
                    # Basic check: ensure it's not just headers with no data
                    if temp_df.shape[0] > 0: 
                        df = temp_df
                        loaded_sheet_name = sheet
                        logger.info(f"Successfully loaded data from sheet: '{sheet}'. Shape: {df.shape}")
                        break
                    else:
                        logger.info(f"Sheet '{sheet}' was loaded but found to be empty (only headers or no rows).")
                else:
                    logger.info(f"Sheet '{sheet}' is empty.")
            except Exception as e:
                logger.warning(f"Could not load sheet '{sheet}': {e}")
        
        if df is None:
            logger.error(f"Failed to load any valid data from the McCance & Widdowson Excel file using tried sheet names: {sheets_to_try}")
            return None

        # --- Validation ---
        logger.info("Performing validation on loaded McCance & Widdowson data...")

        # 1. Check for essential columns
        missing_columns = [col for col in actual_expected_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Essential columns missing from the loaded data: {missing_columns}. Available columns: {df.columns.tolist()}")
            # Depending on strictness, one might return None here or allow processing to continue with a warning.
            # For now, returning None as these are "expected".
            return None 
        else:
            logger.info(f"All expected columns found: {actual_expected_columns}")

        # 2. Check for empty DataFrame (already done implicitly, but good to be explicit)
        if df.empty:
            logger.error("Loaded DataFrame is empty after attempting all sheets.")
            return None
            
        # 3. Optional: Use validation.validate_data for a general report (if available)
        # if hasattr(validation, 'validate_data'):
        #     report = validation.validate_data(df, dataset_name="McCance & Widdowson")
        #     logger.info(f"General data validation report:\n{report}")
        # else:
        #     logger.info("General validation module/function not available. Skipping general validation report.")

        logger.info(f"McCance & Widdowson data loaded and validated successfully from sheet '{loaded_sheet_name}'.")
        return df

if __name__ == '__main__':
    # Example Usage (assuming config.py is set up and the file exists)
    logging.basicConfig(level=logging.INFO)
    
    # Ensure PROJECT_ROOT and necessary config paths are discoverable if running this file directly
    # This might require setting up PYTHONPATH or running from the project root for config to work as expected
    # For simplicity, this example assumes config is correctly loaded.
    
    loader = ExcelLoader()
    
    # Test with default paths from config
    print("\n--- Loading M&W with default config paths ---")
    mw_df_default = loader.load_mccance_widdowson()
    if mw_df_default is not None:
        print(f"Loaded default M&W data. Shape: {mw_df_default.shape}")
        print(mw_df_default.head())
    else:
        print("Failed to load default M&W data.")

    # Example: Test with a specific (potentially non-default) sheet name if you know one exists
    # print("\n--- Loading M&W with a specific sheet name ---")
    # mw_df_specific_sheet = loader.load_mccance_widdowson(sheet_name_primary="YourSpecificSheetNameHere")
    # if mw_df_specific_sheet is not None:
    #     print(f"Loaded M&W data from specific sheet. Shape: {mw_df_specific_sheet.shape}")
    # else:
    #     print("Failed to load M&W data from specific sheet.")

    # Example: Test with custom expected columns (if different from config)
    # print("\n--- Loading M&W with custom expected columns ---")
    # custom_cols = ['Some Other Column', 'Food_Name'] 
    # mw_df_custom_cols = loader.load_mccance_widdowson(expected_columns=custom_cols)
    # if mw_df_custom_cols is not None:
    #     print(f"Loaded M&W data with custom column check. Shape: {mw_df_custom_cols.shape}")
    # else:
    #     print("Failed to load M&W data with custom column check.")
