#!/usr/bin/env python
"""
Utility script to process all raw data sources and save to processed directory.
"""

import os
import sys
import argparse
import logging
import pandas as pd
from pathlib import Path

# Add the project root to the path so we can import modules
sys.path.append(str(Path(__file__).parent.parent.parent))

from shelfscale.data_processing.raw_processor import RawDataProcessor
from shelfscale.data_sourcing.pdf_extraction import PDFExtractor
from shelfscale.utils.helpers import get_path

def setup_logging(log_file=None):
    """Configure logging"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Create formatter
    formatter = logging.Formatter(log_format)
    
    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    
    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def process_data(raw_data_dir="Data/Raw Data", processed_data_dir="Data/Processed", log_file="raw_data_processing.log"):
    """Process all raw data"""
    # Setup logging
    logger = setup_logging(log_file)
    
    # Create processor
    logger.info("Initializing RawDataProcessor")
    processor = RawDataProcessor(raw_data_dir, processed_data_dir)
    
    # Process all data
    logger.info("Processing all raw data sources")
    processed_data = processor.process_all()
    
    # Report results
    total_items = sum(len(df) for df in processed_data.values())
    logger.info(f"Processing complete. Processed {len(processed_data)} datasets with {total_items} total items.")
    
    for source, df in processed_data.items():
        logger.info(f"  {source}: {len(df)} items")
    
    return processed_data

def verify_extractions(raw_data_dir="Data/Raw Data", cache_dir="output", log_file="pdf_verification.log"):
    """Verify PDF extractions using different methods"""
    # Setup logging
    logger = setup_logging(log_file)
    
    # Initialize extractor
    extractor = PDFExtractor(cache_dir=cache_dir)
    
    # Find PDF files
    pdf_files = []
    raw_dir = get_path(raw_data_dir)
    
    for file in os.listdir(raw_dir):
        if file.lower().endswith('.pdf'):
            pdf_files.append(os.path.join(raw_dir, file))
    
    # Process each PDF with different encodings
    for pdf_file in pdf_files:
        logger.info(f"Testing extraction methods on {os.path.basename(pdf_file)}")
        
        # Determine PDF type
        if "portion" in pdf_file.lower() or "size" in pdf_file.lower():
            logger.info("Detected as Food Portion Sizes PDF")
            
            # Extract using different methods
            try:
                # Extract using lattice mode
                df_lattice = extractor._extract_food_portion_lattice(pdf_file)
                logger.info(f"Lattice extraction: {len(df_lattice)} rows")
            except Exception as e:
                logger.error(f"Lattice extraction failed: {e}")
                df_lattice = None
            
            try:
                # Extract using stream mode
                df_stream = extractor._extract_food_portion_stream(pdf_file)
                logger.info(f"Stream extraction: {len(df_stream)} rows")
            except Exception as e:
                logger.error(f"Stream extraction failed: {e}")
                df_stream = None
            
            try:
                # Extract using simple mode
                df_simple = extractor._extract_food_portion_simple(pdf_file)
                logger.info(f"Simple extraction: {len(df_simple)} rows")
            except Exception as e:
                logger.error(f"Simple extraction failed: {e}")
                df_simple = None
                
        elif "fruit" in pdf_file.lower() or "veg" in pdf_file.lower():
            logger.info("Detected as Fruit and Vegetable Survey PDF")
            
            try:
                # Extract using table mode
                df_table = extractor._extract_fruit_veg_tables(pdf_file)
                logger.info(f"Table extraction: {len(df_table)} rows")
            except Exception as e:
                logger.error(f"Table extraction failed: {e}")
                df_table = None
            
            try:
                # Extract using text-based approach
                df_text = extractor._extract_fruit_veg_text_based(pdf_file)
                logger.info(f"Text-based extraction: {len(df_text)} rows")
            except Exception as e:
                logger.error(f"Text-based extraction failed: {e}")
                df_text = None
            
            try:
                # Extract using simple mode
                df_simple = extractor._extract_fruit_veg_simple(pdf_file)
                logger.info(f"Simple extraction: {len(df_simple)} rows")
            except Exception as e:
                logger.error(f"Simple extraction failed: {e}")
                df_simple = None
    
    logger.info("PDF extraction verification complete")

def main():
    """Main function to run from command line"""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Process raw data for ShelfScale")
    parser.add_argument("--raw-dir", default="Data/Raw Data", help="Raw data directory")
    parser.add_argument("--processed-dir", default="Data/Processed", help="Processed data directory")
    parser.add_argument("--log-file", default="raw_data_processing.log", help="Log file path")
    parser.add_argument("--verify", action="store_true", help="Verify PDF extraction methods")
    parser.add_argument("--cache-dir", default="output", help="Cache directory for PDF extraction")
    
    args = parser.parse_args()
    
    if args.verify:
        verify_extractions(args.raw_dir, args.cache_dir, args.log_file)
    else:
        process_data(args.raw_dir, args.processed_dir, args.log_file)

if __name__ == "__main__":
    main() 