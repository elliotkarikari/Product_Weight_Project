import os

# Project Root
# Resolves to the 'shelfscale' directory.
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

# Data Directories
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "Data", "Raw Data")
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "Data", "Processed")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")

# Ensure directories exist
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# File Names
MCCANCE_WIDDOWSON_FILE = "McCance_Widdowsons_2021.xlsx"
FOOD_PORTION_PDF = "Food_Portion_Sizes.pdf"
FRUIT_VEG_SURVEY_PDF = "Fruit_and_Veg_Portion_Sizes.pdf" # Assuming a name for this file

# Full File Paths
MCCANCE_WIDDOWSON_PATH = os.path.join(RAW_DATA_DIR, MCCANCE_WIDDOWSON_FILE)
FOOD_PORTION_PDF_PATH = os.path.join(RAW_DATA_DIR, FOOD_PORTION_PDF)
FRUIT_VEG_SURVEY_PDF_PATH = os.path.join(RAW_DATA_DIR, FRUIT_VEG_SURVEY_PDF) # Assuming a path

# Cached File Names
MW_DATA_CACHED_FILE = "mw_data_cached.csv"
FOOD_PORTION_SIZES_CACHED_FILE = "food_portion_sizes.csv"
FRUIT_VEG_SURVEY_CACHED_FILE = "fruit_veg_survey.csv"

MW_DATA_CACHED_PATH = os.path.join(CACHE_DIR, MW_DATA_CACHED_FILE)
FOOD_PORTION_SIZES_CACHED_PATH = os.path.join(CACHE_DIR, FOOD_PORTION_SIZES_CACHED_FILE)
FRUIT_VEG_SURVEY_CACHED_PATH = os.path.join(CACHE_DIR, FRUIT_VEG_SURVEY_CACHED_FILE)

# Parameters
DEFAULT_MATCHING_THRESHOLD = 70
PDF_FOOD_PORTION_PAGES = "12-114" # As per current main.py
FRUIT_VEG_SURVEY_PAGES = "1-5" # Placeholder, adjust as needed - specific to main.py PDFExtractor usage

# --- McCance and Widdowson (MW) specific parameters ---
MW_SHEET_NAME_FOR_MAIN_PY = "A1. Calculated values" # Used in main.py's load_mccance_widdowson_data
MW_FACTORS_SHEET_NAME = "Factors" # Typically used in raw_processor for detailed processing
MW_REDUCED_SUPER_GROUP_DIR_NAME = "Reduced Super Group"
MW_DATA_REDUCTION_SUBDIR = "MW_DataReduction"
MW_REDUCED_INDIVIDUAL_TABLES_SUBDIR = os.path.join(MW_DATA_REDUCTION_SUBDIR, "Reduced Individual Tables")
MW_REDUCED_SUPER_GROUP_SUBDIR = os.path.join(MW_DATA_REDUCTION_SUBDIR, MW_REDUCED_SUPER_GROUP_DIR_NAME)
MW_REDUCED_SUPER_GROUP_CLEANED_SUBDIR = os.path.join(MW_REDUCED_SUPER_GROUP_SUBDIR, "Cleaned")
MW_REDUCED_TOTAL_SUBDIR = os.path.join(MW_DATA_REDUCTION_SUBDIR, "Reduced Total")

MW_PROCESSED_SUPER_GROUP_PATH = os.path.join(PROCESSED_DATA_DIR, MW_REDUCED_SUPER_GROUP_SUBDIR) # Path for main.py
MW_PROCESSED_FULL_FILE = os.path.join(PROCESSED_DATA_DIR, MW_REDUCED_TOTAL_SUBDIR, "McCance_Widdowson_Full.csv")

# --- Food Portion Sizes (FPS) specific parameters ---
FPS_SUBDIR = "FoodPortionSized"
PROCESSED_FPS_FILE_NAME = "Processed_Food_Portion_Sizes.csv"
RAW_EXTRACTED_FPS_FILE_NAME = "FoodPortionSizes_PDF.csv"
PROCESSED_FPS_PATH = os.path.join(PROCESSED_DATA_DIR, FPS_SUBDIR, PROCESSED_FPS_FILE_NAME)
RAW_EXTRACTED_FPS_PATH = os.path.join(PROCESSED_DATA_DIR, FPS_SUBDIR, RAW_EXTRACTED_FPS_FILE_NAME)

# --- Fruit and Vegetable Survey (FVS) specific parameters ---
FVS_REPORTS_SUBDIR = "Fruit and Veg Sample reports"
FVS_TABLES_SUBDIR = os.path.join(FVS_REPORTS_SUBDIR, "tables")
FVS_TEXT_SUBDIR = os.path.join(FVS_REPORTS_SUBDIR, "text")
FRUIT_VEG_PDF_GLOB_PATTERN = "*fruit*veg*.pdf" # Glob pattern for FVS PDFs in RAW_DATA_DIR
PROCESSED_FVS_COMBINED_FILE_NAME = "Combined_Fruit_Veg_Samples.csv"
RAW_EXTRACTED_FVS_COMBINED_FILE_NAME = "Raw_Fruit_Veg_Samples.csv"
PROCESSED_FVS_TABLES_PATH = os.path.join(PROCESSED_DATA_DIR, FVS_TABLES_SUBDIR, PROCESSED_FVS_COMBINED_FILE_NAME)
RAW_EXTRACTED_FVS_TEXT_PATH = os.path.join(PROCESSED_DATA_DIR, FVS_TEXT_SUBDIR, RAW_EXTRACTED_FVS_COMBINED_FILE_NAME)

# --- Labelling specific parameters ---
LABELLING_DATA_SUBDIR = "ReducedwithWeights" # As per raw_processor.py
LABELLING_CSV_GLOB_PATTERN = "Labelling*.csv"
LABELLING_XLSX_GLOB_PATTERN = "Labelling*.xlsx"
PROCESSED_LABELLING_FILE_NAME = "Processed_Labelling_Data.csv"
PROCESSED_LABELLING_PATH = os.path.join(PROCESSED_DATA_DIR, LABELLING_DATA_SUBDIR, PROCESSED_LABELLING_FILE_NAME)

# --- General Processed Subdirectories ---
# These are based on the structure in raw_processor.py's _create_directory_structure
# Some are already defined above.
PROCESSED_SUBDIRS = [
    FPS_SUBDIR,
    FVS_TABLES_SUBDIR,
    FVS_TEXT_SUBDIR,
    MW_REDUCED_INDIVIDUAL_TABLES_SUBDIR,
    MW_REDUCED_SUPER_GROUP_SUBDIR,
    MW_REDUCED_SUPER_GROUP_CLEANED_SUBDIR,
    MW_REDUCED_TOTAL_SUBDIR,
    LABELLING_DATA_SUBDIR,
    "Sample Reports" # This one seems generic, keeping as string for now
]

# Ensure all necessary processed directories exist
for subdir_name in PROCESSED_SUBDIRS:
    os.makedirs(os.path.join(PROCESSED_DATA_DIR, subdir_name), exist_ok=True)

# Ensure MW processed super group directory exists (used by main.py, might be redundant if PROCESSED_SUBDIRS covers it)
os.makedirs(MW_PROCESSED_SUPER_GROUP_PATH, exist_ok=True)

# Temporary cache for PDFExtractor within RawDataProcessor
RAW_PROCESSOR_TEMP_CACHE_DIR = os.path.join(PROCESSED_DATA_DIR, "temp_pdf_cache")
os.makedirs(RAW_PROCESSOR_TEMP_CACHE_DIR, exist_ok=True)

# --- Machine Learning Model files ---
FOOD_MATCHER_MODEL_FILE = "food_matcher_model.pkl"
FOOD_MATCHER_FEATURES_FILE = "food_matcher_features.pkl"
MODEL_PERFORMANCE_HISTORY_FILE = "model_performance_history.json"

FOOD_MATCHER_MODEL_PATH = os.path.join(MODEL_DIR, FOOD_MATCHER_MODEL_FILE)
FOOD_MATCHER_FEATURES_PATH = os.path.join(MODEL_DIR, FOOD_MATCHER_FEATURES_FILE)
MODEL_PERFORMANCE_HISTORY_PATH = os.path.join(MODEL_DIR, MODEL_PERFORMANCE_HISTORY_FILE)

# --- McCance and Widdowson specific parameters (continued) ---
MW_EXPECTED_COLUMNS = ['Food_Name', 'Food_Code', 'Food_Group'] # Expected columns for validation

# --- Labelling specific parameters (continued) ---
LABELLING_EXPECTED_COLUMNS = ['Food_Name', 'Weight_g', 'Portion_Size'] # Placeholder, adjust as needed

# --- Schema Definitions for Validation ---
FPS_EXPECTED_SCHEMA = {
    'Food_Name': {'type': 'str', 'required': True},
    'Portion_Size': {'type': 'str', 'required': False},
    'Weight_g': {'type': 'float', 'required': True, 'nullable': True},
    'Notes': {'type': 'str', 'required': False}
}

FVS_EXPECTED_SCHEMA = {
    'Page': {'type': 'int', 'required': False, 'nullable': True},
    'Sample_Number': {'type': 'str', 'required': False},
    'Sample_Name': {'type': 'str', 'required': True},
    'Pack_Size': {'type': 'str', 'required': False}
}

# --- Notebooks Directory ---
NOTEBOOKS_DIR = os.path.join(PROJECT_ROOT, "Jupyter_notebooks") # Corrected spelling
os.makedirs(NOTEBOOKS_DIR, exist_ok=True)
DATA_PRODUCT_NOTEBOOKS_SUBDIR = "Data_Product" # Subdirectory within NOTEBOOKS_DIR

# --- Output files for learning utilities ---
MW_FPS_MATCHES_FILE = "mw_fps_matches.csv"
MW_FVS_MATCHES_FILE = "mw_fvs_matches.csv"
SUPERGROUP_MATCHES_FILE = "supergroup_matches.csv"
API_MATCHES_FILE = "api_matches.csv"
TRAINING_DATA_FILE = "training_data.csv"
FEEDBACK_SESSION_FILE = "feedback_session.json" # For interactive feedback
FEEDBACK_IMPACT_VISUALIZATION_FILE = "feedback_impact.html"

MW_FPS_MATCHES_PATH = os.path.join(OUTPUT_DIR, MW_FPS_MATCHES_FILE)
MW_FVS_MATCHES_PATH = os.path.join(OUTPUT_DIR, MW_FVS_MATCHES_FILE)
SUPERGROUP_MATCHES_PATH = os.path.join(OUTPUT_DIR, SUPERGROUP_MATCHES_FILE)
API_MATCHES_PATH = os.path.join(OUTPUT_DIR, API_MATCHES_FILE)
TRAINING_DATA_PATH = os.path.join(OUTPUT_DIR, TRAINING_DATA_FILE)
FEEDBACK_SESSION_PATH = os.path.join(OUTPUT_DIR, FEEDBACK_SESSION_FILE)
FEEDBACK_IMPACT_VISUALIZATION_PATH = os.path.join(OUTPUT_DIR, FEEDBACK_IMPACT_VISUALIZATION_FILE)

# List of match file paths for load_existing_matches
EXISTING_MATCH_FILES_TO_LOAD = {
    "mw_fps_matches": MW_FPS_MATCHES_PATH,
    "mw_fvs_matches": MW_FVS_MATCHES_PATH,
    "supergroup_matches": SUPERGROUP_MATCHES_PATH,
    "api_matches": API_MATCHES_PATH
}
