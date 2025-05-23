"""
Configuration Manager for Product Weight Project
Handles environment variables, API keys, and application settings
"""

import os
import logging
from typing import Any, Optional, Union
from pathlib import Path

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

logger = logging.getLogger(__name__)


class ConfigManager:
    """
    Centralized configuration management with environment variable support
    """
    
    def __init__(self, env_file: Optional[str] = None):
        # Load environment variables from .env file if available
        if DOTENV_AVAILABLE:
            if env_file:
                load_dotenv(env_file)
            else:
                # Look for .env file in project root
                project_root = Path(__file__).parent.parent
                env_path = project_root / '.env'
                if env_path.exists():
                    load_dotenv(env_path)
                    logger.info(f"Loaded environment variables from {env_path}")
                else:
                    logger.info("No .env file found, using system environment variables")
        else:
            logger.warning("python-dotenv not installed, using system environment variables only")
        
        # Initialize configuration
        self._load_config()
    
    def _load_config(self):
        """Load all configuration settings"""
        
        # OpenAI Configuration
        self.OPENAI_API_KEY = self.get_env('OPENAI_API_KEY', '')
        self.OPENAI_MODEL = self.get_env('OPENAI_MODEL', 'gpt-4.1-mini')
        self.OPENAI_MAX_TOKENS = self.get_env_int('OPENAI_MAX_TOKENS', 300)
        self.OPENAI_TEMPERATURE = self.get_env_float('OPENAI_TEMPERATURE', 0.3)
        
        # LLM Curation Settings
        self.LLM_BATCH_SIZE = self.get_env_int('LLM_BATCH_SIZE', 10)
        self.LLM_DEMO_MODE = self.get_env_bool('LLM_DEMO_MODE', True)
        self.LLM_DEMO_SIZE = self.get_env_int('LLM_DEMO_SIZE', 50)
        self.LLM_CACHE_ENABLED = self.get_env_bool('LLM_CACHE_ENABLED', True)
        
        # Cost Management
        self.MAX_ESTIMATED_COST = self.get_env_float('MAX_ESTIMATED_COST', 5.0)
        self.COST_WARNING_THRESHOLD = self.get_env_float('COST_WARNING_THRESHOLD', 2.0)
        
        # Rate Limiting
        self.RATE_LIMIT_DELAY = self.get_env_float('RATE_LIMIT_DELAY', 3.0)
        self.MAX_RETRIES = self.get_env_int('MAX_RETRIES', 3)
        
        # File Paths - Legacy compatibility
        self.PROJECT_ROOT = Path(__file__).parent
        self.RAW_DATA_DIR = self.PROJECT_ROOT.parent / "Data" / "Raw Data"
        self.PROCESSED_DATA_DIR = self.PROJECT_ROOT.parent / "Data" / "Processed"
        self.MODEL_DIR = self.PROJECT_ROOT / "models"
        self.CACHE_DIR = self.get_env('CACHE_DIR', str(self.PROJECT_ROOT / "cache"))
        
        # Ensure directories exist
        for dir_path in [self.RAW_DATA_DIR, self.PROCESSED_DATA_DIR, self.MODEL_DIR, self.CACHE_DIR]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        # File Names
        self.MCCANCE_WIDDOWSON_FILE = "McCance_Widdowsons_2021.xlsx"
        self.FOOD_PORTION_PDF = "Food_Portion_Sizes.pdf"
        self.FRUIT_VEG_SURVEY_PDF = "fruit_and_vegetable_survey_2015_sampling_report.pdf"
        
        # Full File Paths
        self.MCCANCE_WIDDOWSON_PATH = str(self.RAW_DATA_DIR / self.MCCANCE_WIDDOWSON_FILE)
        self.FOOD_PORTION_PDF_PATH = str(self.RAW_DATA_DIR / self.FOOD_PORTION_PDF)
        self.FRUIT_VEG_SURVEY_PDF_PATH = str(self.RAW_DATA_DIR / self.FRUIT_VEG_SURVEY_PDF)
        
        # McCance and Widdowson specific parameters
        self.MW_SHEET_NAME_FOR_MAIN_PY = "1.3 Proximates"
        self.MW_FACTORS_SHEET_NAME = "1.2 Factors"
        self.DEFAULT_MATCHING_THRESHOLD = 70
        self.PDF_FOOD_PORTION_PAGES = "12-114"
        
        # Cached File Names
        self.MW_DATA_CACHED_FILE = "mw_data_cached.csv"
        self.FOOD_PORTION_SIZES_CACHED_FILE = "food_portion_sizes.csv"
        self.FRUIT_VEG_SURVEY_CACHED_FILE = "fruit_veg_survey.csv"
        
        # Cached File Paths
        cache_dir = Path(self.CACHE_DIR)
        self.MW_DATA_CACHED_PATH = str(cache_dir / self.MW_DATA_CACHED_FILE)
        self.FOOD_PORTION_SIZES_CACHED_PATH = str(cache_dir / self.FOOD_PORTION_SIZES_CACHED_FILE)
        self.FRUIT_VEG_SURVEY_CACHED_PATH = str(cache_dir / self.FRUIT_VEG_SURVEY_CACHED_FILE)
        
        # Processed data subdirectories
        self.FPS_SUBDIR = "FoodPortionSized"
        self.MW_DATA_REDUCTION_SUBDIR = "MW_DataReduction"
        self.MW_REDUCED_SUPER_GROUP_SUBDIR = os.path.join(self.MW_DATA_REDUCTION_SUBDIR, "Reduced Super Group")
        self.MW_REDUCED_TOTAL_SUBDIR = os.path.join(self.MW_DATA_REDUCTION_SUBDIR, "Reduced Total")
        
        # Processed file paths
        self.PROCESSED_FPS_FILE_NAME = "Processed_Food_Portion_Sizes.csv"
        self.RAW_EXTRACTED_FPS_FILE_NAME = "FoodPortionSizes_PDF.csv"
        self.PROCESSED_FPS_PATH = str(self.PROCESSED_DATA_DIR / self.FPS_SUBDIR / self.PROCESSED_FPS_FILE_NAME)
        self.RAW_EXTRACTED_FPS_PATH = str(self.PROCESSED_DATA_DIR / self.FPS_SUBDIR / self.RAW_EXTRACTED_FPS_FILE_NAME)
        self.MW_PROCESSED_SUPER_GROUP_PATH = str(self.PROCESSED_DATA_DIR / self.MW_REDUCED_SUPER_GROUP_SUBDIR)
        self.MW_PROCESSED_FULL_FILE = str(self.PROCESSED_DATA_DIR / self.MW_REDUCED_TOTAL_SUBDIR / "McCance_Widdowson_Full.csv")
        
        # Schema definitions
        self.FPS_EXPECTED_SCHEMA = {
            'Food_Name': {'type': 'str', 'required': True},
            'Portion_Size': {'type': 'str', 'required': False},
            'Weight_g': {'type': 'float', 'required': True, 'nullable': True},
            'Notes': {'type': 'str', 'required': False}
        }
        
        self.FVS_EXPECTED_SCHEMA = {
            'Page': {'type': 'int', 'required': False, 'nullable': True},
            'Sample_Number': {'type': 'str', 'required': False},
            'Sample_Name': {'type': 'str', 'required': True},
            'Pack_Size': {'type': 'str', 'required': False}
        }
        
        # Override OUTPUT_DIR if not set via environment
        if not self.get_env('OUTPUT_DIR'):
            self.OUTPUT_DIR = str(self.PROJECT_ROOT / "output")
        else:
            self.OUTPUT_DIR = self.get_env('OUTPUT_DIR', str(self.PROJECT_ROOT / "output"))
        
        # Database Configuration (for future use)
        self.DATABASE_URL = self.get_env('DATABASE_URL', 'sqlite:///product_weights.db')
        self.DATABASE_HOST = self.get_env('DATABASE_HOST', 'localhost')
        self.DATABASE_PORT = self.get_env_int('DATABASE_PORT', 5432)
        self.DATABASE_NAME = self.get_env('DATABASE_NAME', 'product_weights')
        self.DATABASE_USER = self.get_env('DATABASE_USER', '')
        self.DATABASE_PASSWORD = self.get_env('DATABASE_PASSWORD', '')
        
        # External APIs
        self.USDA_API_KEY = self.get_env('USDA_API_KEY', '')
        self.EDAMAM_APP_ID = self.get_env('EDAMAM_APP_ID', '')
        self.EDAMAM_APP_KEY = self.get_env('EDAMAM_APP_KEY', '')
        
        # Email Configuration
        self.SMTP_HOST = self.get_env('SMTP_HOST', 'smtp.gmail.com')
        self.SMTP_PORT = self.get_env_int('SMTP_PORT', 587)
        self.SMTP_USER = self.get_env('SMTP_USER', '')
        self.SMTP_PASSWORD = self.get_env('SMTP_PASSWORD', '')
        self.NOTIFICATION_EMAIL = self.get_env('NOTIFICATION_EMAIL', '')
        
        # Log Level
        self.LOG_LEVEL = self.get_env('LOG_LEVEL', 'INFO')
    
    def get_env(self, key: str, default: str = '') -> str:
        """Get string environment variable with default"""
        return os.getenv(key, default)
    
    def get_env_int(self, key: str, default: int = 0) -> int:
        """Get integer environment variable with default"""
        try:
            return int(os.getenv(key, str(default)))
        except (ValueError, TypeError):
            logger.warning(f"Invalid integer value for {key}, using default: {default}")
            return default
    
    def get_env_float(self, key: str, default: float = 0.0) -> float:
        """Get float environment variable with default"""
        try:
            return float(os.getenv(key, str(default)))
        except (ValueError, TypeError):
            logger.warning(f"Invalid float value for {key}, using default: {default}")
            return default
    
    def get_env_bool(self, key: str, default: bool = False) -> bool:
        """Get boolean environment variable with default"""
        value = os.getenv(key, str(default)).lower()
        return value in ('true', 'yes', '1', 'on', 'enabled')
    
    def validate_openai_config(self) -> bool:
        """Validate OpenAI configuration"""
        if not self.OPENAI_API_KEY or self.OPENAI_API_KEY == 'your-openai-api-key-here':
            return False
        return True
    
    def get_openai_config(self) -> dict:
        """Get OpenAI configuration as dict"""
        return {
            'api_key': self.OPENAI_API_KEY,
            'model': self.OPENAI_MODEL,
            'max_tokens': self.OPENAI_MAX_TOKENS,
            'temperature': self.OPENAI_TEMPERATURE
        }
    
    def get_llm_settings(self) -> dict:
        """Get LLM curation settings as dict"""
        return {
            'batch_size': self.LLM_BATCH_SIZE,
            'demo_mode': self.LLM_DEMO_MODE,
            'demo_size': self.LLM_DEMO_SIZE,
            'cache_enabled': self.LLM_CACHE_ENABLED,
            'rate_limit_delay': self.RATE_LIMIT_DELAY,
            'max_retries': self.MAX_RETRIES
        }
    
    def get_cost_settings(self) -> dict:
        """Get cost management settings as dict"""
        return {
            'max_estimated_cost': self.MAX_ESTIMATED_COST,
            'cost_warning_threshold': self.COST_WARNING_THRESHOLD
        }
    
    def get_file_paths(self) -> dict:
        """Get file path configurations as dict"""
        return {
            'output_dir': self.OUTPUT_DIR,
            'cache_dir': self.CACHE_DIR
        }
    
    def setup_logging(self):
        """Setup logging based on configuration"""
        level = getattr(logging, self.LOG_LEVEL.upper(), logging.INFO)
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    def print_config_summary(self):
        """Print configuration summary (without sensitive data)"""
        print("🔧 CONFIGURATION SUMMARY")
        print("=" * 40)
        print(f"OpenAI Model: {self.OPENAI_MODEL}")
        print(f"API Key Set: {'Yes' if self.validate_openai_config() else 'No'}")
        print(f"Demo Mode: {self.LLM_DEMO_MODE}")
        if self.LLM_DEMO_MODE:
            print(f"Demo Size: {self.LLM_DEMO_SIZE} items")
        print(f"Batch Size: {self.LLM_BATCH_SIZE}")
        print(f"Cache Enabled: {self.LLM_CACHE_ENABLED}")
        print(f"Max Cost: ${self.MAX_ESTIMATED_COST}")
        print(f"Output Dir: {self.OUTPUT_DIR}")
        print(f"Log Level: {self.LOG_LEVEL}")
        print("=" * 40)


# Global configuration instance
config = ConfigManager()


def get_config() -> ConfigManager:
    """Get the global configuration instance"""
    return config


def reload_config(env_file: Optional[str] = None) -> ConfigManager:
    """Reload configuration from environment"""
    global config
    config = ConfigManager(env_file)
    return config 