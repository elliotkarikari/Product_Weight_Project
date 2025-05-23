"""
Data validation utilities for ShelfScale
Provides comprehensive data integrity checks and helpful error reporting
"""

import os
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any
from shelfscale.config_manager import get_config

logger = logging.getLogger(__name__)


class DataValidator:
    """
    Comprehensive data validation for ShelfScale pipeline
    """
    
    def __init__(self):
        self.validation_results = {}
        self.warnings = []
        self.errors = []
    
    def validate_file_availability(self) -> Dict[str, Any]:
        """Validate that all expected files are available"""
        file_status = {}
        
        # Required files
        required_files = {
            'McCance & Widdowson Excel': get_config.MCCANCE_WIDDOWSON_PATH,
            'Food Portion PDF': get_config.FOOD_PORTION_PDF_PATH,
            'Fruit & Veg PDF': get_config.FRUIT_VEG_SURVEY_PDF_PATH
        }
        
        for name, path in required_files.items():
            exists = os.path.exists(path)
            file_status[name] = {
                'path': path,
                'exists': exists,
                'size_mb': round(os.path.getsize(path) / (1024*1024), 2) if exists else 0
            }
            
            if not exists:
                self.errors.append(f"Required file missing: {name} at {path}")
            else:
                logger.info(f"✓ Found {name}: {file_status[name]['size_mb']} MB")
        
        return file_status
    
    def validate_excel_data(self, file_path: str) -> Dict[str, Any]:
        """Validate Excel file structure and content"""
        validation_result = {
            'file_readable': False,
            'sheets_available': [],
            'target_sheet_exists': False,
            'data_rows': 0,
            'key_columns': []
        }
        
        try:
            # Check if file is readable
            xl = pd.ExcelFile(file_path)
            validation_result['file_readable'] = True
            validation_result['sheets_available'] = xl.sheet_names
            
            # Check if target sheet exists
            target_sheet = get_config.MW_SHEET_NAME_FOR_MAIN_PY
            if target_sheet in xl.sheet_names:
                validation_result['target_sheet_exists'] = True
                
                # Load and validate data
                df = pd.read_excel(file_path, sheet_name=target_sheet)
                validation_result['data_rows'] = len(df)
                validation_result['key_columns'] = list(df.columns)
                
                # Check for essential columns
                essential_columns = ['Food Code', 'Food Name', 'Group']
                missing_columns = [col for col in essential_columns if col not in df.columns]
                
                if missing_columns:
                    self.warnings.append(f"Missing essential columns in {target_sheet}: {missing_columns}")
                else:
                    logger.info(f"✓ Excel data validated: {len(df)} rows with all essential columns")
            else:
                self.errors.append(f"Target sheet '{target_sheet}' not found. Available: {xl.sheet_names}")
                
        except Exception as e:
            self.errors.append(f"Error reading Excel file: {e}")
            
        return validation_result
    
    def validate_super_group_data(self) -> Dict[str, Any]:
        """Validate super group CSV files"""
        validation_result = {
            'directory_exists': False,
            'file_count': 0,
            'total_data_rows': 0,
            'empty_files': [],
            'valid_files': []
        }
        
        super_group_dir = get_config.MW_PROCESSED_SUPER_GROUP_PATH
        
        if os.path.exists(super_group_dir):
            validation_result['directory_exists'] = True
            
            csv_files = [f for f in os.listdir(super_group_dir) if f.endswith('.csv')]
            validation_result['file_count'] = len(csv_files)
            
            for csv_file in csv_files:
                filepath = os.path.join(super_group_dir, csv_file)
                try:
                    df = pd.read_csv(filepath)
                    row_count = len(df)
                    
                    if row_count == 0:
                        validation_result['empty_files'].append(csv_file)
                    else:
                        validation_result['valid_files'].append({
                            'file': csv_file,
                            'rows': row_count
                        })
                        validation_result['total_data_rows'] += row_count
                        
                except Exception as e:
                    self.warnings.append(f"Error reading super group file {csv_file}: {e}")
            
            if validation_result['total_data_rows'] == 0:
                self.warnings.append("All super group files are empty - fallback processing will be triggered")
            else:
                logger.info(f"✓ Super group data: {validation_result['total_data_rows']} total rows across {len(validation_result['valid_files'])} files")
        else:
            self.errors.append(f"Super group directory does not exist: {super_group_dir}")
            
        return validation_result
    
    def validate_pdf_accessibility(self, pdf_path: str) -> Dict[str, Any]:
        """Validate PDF file accessibility and basic properties"""
        validation_result = {
            'exists': False,
            'readable': False,
            'size_mb': 0,
            'page_count': 0
        }
        
        if os.path.exists(pdf_path):
            validation_result['exists'] = True
            validation_result['size_mb'] = round(os.path.getsize(pdf_path) / (1024*1024), 2)
            
            try:
                # Try basic PDF reading
                import PyPDF2
                with open(pdf_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    validation_result['page_count'] = len(reader.pages)
                    validation_result['readable'] = True
                    logger.info(f"✓ PDF validated: {pdf_path} ({validation_result['page_count']} pages)")
                    
            except Exception as e:
                self.warnings.append(f"PDF may not be readable: {pdf_path} - {e}")
        else:
            self.errors.append(f"PDF file not found: {pdf_path}")
            
        return validation_result
    
    def validate_alternative_data_sources(self) -> Dict[str, Any]:
        """Check for alternative data sources like labelling datasets"""
        alternatives = {
            'labelling_csv_files': [],
            'labelling_xlsx_files': [],
            'other_data_files': []
        }
        
        try:
            raw_files = os.listdir(get_config.RAW_DATA_DIR)
            
            for file in raw_files:
                if file.startswith('Labelling') and file.endswith('.csv'):
                    filepath = os.path.join(get_config.RAW_DATA_DIR, file)
                    size_mb = round(os.path.getsize(filepath) / (1024*1024), 2)
                    alternatives['labelling_csv_files'].append({
                        'file': file,
                        'size_mb': size_mb
                    })
                elif file.startswith('Labelling') and file.endswith('.xlsx'):
                    filepath = os.path.join(get_config.RAW_DATA_DIR, file)
                    size_mb = round(os.path.getsize(filepath) / (1024*1024), 2)
                    alternatives['labelling_xlsx_files'].append({
                        'file': file,
                        'size_mb': size_mb
                    })
                elif file.endswith(('.csv', '.xlsx')) and 'McCance' not in file:
                    alternatives['other_data_files'].append(file)
            
            total_alternatives = len(alternatives['labelling_csv_files']) + len(alternatives['labelling_xlsx_files'])
            if total_alternatives > 0:
                logger.info(f"✓ Found {total_alternatives} alternative labelling data files")
                
        except Exception as e:
            self.warnings.append(f"Error checking alternative data sources: {e}")
            
        return alternatives
    
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run all validation checks and return comprehensive report"""
        logger.info("Running comprehensive data validation...")
        
        validation_report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'file_availability': self.validate_file_availability(),
            'excel_validation': {},
            'super_group_validation': self.validate_super_group_data(),
            'pdf_validations': {},
            'alternative_sources': self.validate_alternative_data_sources(),
            'summary': {
                'errors': [],
                'warnings': [],
                'data_source_count': 0,
                'total_data_rows': 0
            }
        }
        
        # Validate Excel files
        if os.path.exists(get_config.MCCANCE_WIDDOWSON_PATH):
            validation_report['excel_validation'] = self.validate_excel_data(get_config.MCCANCE_WIDDOWSON_PATH)
            
        # Validate PDFs
        pdf_files = {
            'food_portion_pdf': get_config.FOOD_PORTION_PDF_PATH,
            'fruit_veg_pdf': get_config.FRUIT_VEG_SURVEY_PDF_PATH
        }
        
        for name, path in pdf_files.items():
            validation_report['pdf_validations'][name] = self.validate_pdf_accessibility(path)
        
        # Compile summary
        validation_report['summary']['errors'] = self.errors
        validation_report['summary']['warnings'] = self.warnings
        
        # Count available data sources
        data_sources = 0
        total_rows = 0
        
        if validation_report['excel_validation'].get('data_rows', 0) > 0:
            data_sources += 1
            total_rows += validation_report['excel_validation']['data_rows']
            
        if validation_report['super_group_validation'].get('total_data_rows', 0) > 0:
            data_sources += 1
            total_rows += validation_report['super_group_validation']['total_data_rows']
            
        validation_report['summary']['data_source_count'] = data_sources
        validation_report['summary']['total_data_rows'] = total_rows
        
        # Log summary
        if self.errors:
            logger.error(f"Validation completed with {len(self.errors)} errors and {len(self.warnings)} warnings")
            for error in self.errors:
                logger.error(f"  ERROR: {error}")
        else:
            logger.info(f"✓ Validation successful: {data_sources} data sources with {total_rows} total rows")
            
        if self.warnings:
            for warning in self.warnings:
                logger.warning(f"  WARNING: {warning}")
        
        return validation_report
    
    def suggest_fixes(self, validation_report: Dict[str, Any]) -> List[str]:
        """Suggest fixes based on validation results"""
        suggestions = []
        
        # File missing suggestions
        for name, status in validation_report['file_availability'].items():
            if not status['exists']:
                if 'PDF' in name:
                    suggestions.append(f"Check if {name} is in the Raw Data directory with the correct filename")
                else:
                    suggestions.append(f"Ensure {name} file is available at: {status['path']}")
        
        # Excel data suggestions
        excel_val = validation_report.get('excel_validation', {})
        if not excel_val.get('target_sheet_exists', False):
            available_sheets = excel_val.get('sheets_available', [])
            if available_sheets:
                suggestions.append(f"Update config.MW_SHEET_NAME_FOR_MAIN_PY to one of: {available_sheets}")
        
        # Super group suggestions
        super_group_val = validation_report.get('super_group_validation', {})
        if super_group_val.get('total_data_rows', 0) == 0:
            suggestions.append("Run fallback processing to generate super group data from raw Excel file")
        
        # Alternative data suggestions
        alternatives = validation_report.get('alternative_sources', {})
        if alternatives.get('labelling_csv_files'):
            suggestions.append("Consider using labelling datasets as alternative data source")
        
        return suggestions


def validate_and_report() -> Tuple[bool, Dict[str, Any]]:
    """
    Run validation and return success status with detailed report
    
    Returns:
        Tuple of (validation_success, validation_report)
    """
    validator = DataValidator()
    report = validator.run_comprehensive_validation()
    
    success = len(validator.errors) == 0 and report['summary']['data_source_count'] > 0
    
    if not success:
        suggestions = validator.suggest_fixes(report)
        logger.info("Suggested fixes:")
        for suggestion in suggestions:
            logger.info(f"  💡 {suggestion}")
    
    return success, report


if __name__ == "__main__":
    # Test the validator
    logging.basicConfig(level=logging.INFO)
    
    success, report = validate_and_report()
    
    print(f"\nValidation Summary:")
    print(f"Success: {success}")
    print(f"Data sources: {report['summary']['data_source_count']}")
    print(f"Total rows: {report['summary']['total_data_rows']}")
    print(f"Errors: {len(report['summary']['errors'])}")
    print(f"Warnings: {len(report['summary']['warnings'])}") 