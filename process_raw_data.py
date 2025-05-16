#!/usr/bin/env python
"""
Script to process all raw data from Raw Data folder to Processed folder
"""

import os
import sys
import logging
from shelfscale.data_processing.raw_processor import RawDataProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("raw_data_processing.log")
    ]
)
logger = logging.getLogger(__name__)


def main():
    """
    Main function to process all raw data
    """
    print("\n" + "=" * 80)
    print("ShelfScale Raw Data Processor")
    print("=" * 80)
    
    # Default paths relative to project root
    raw_data_dir = "Data/Raw Data"
    processed_data_dir = "Data/Processed"
    
    # Check if directories exist
    if not os.path.isdir(raw_data_dir):
        raw_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), raw_data_dir)
    
    if not os.path.isdir(processed_data_dir):
        processed_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), processed_data_dir)
    
    print(f"Raw data directory: {raw_data_dir}")
    print(f"Processed data directory: {processed_data_dir}")
    
    # Initialize processor
    processor = RawDataProcessor(raw_data_dir, processed_data_dir)
    
    # Process all available data sources
    print("\nProcessing all raw data sources...")
    processed_data = processor.process_all()
    
    # Print summary
    print("\n" + "=" * 80)
    print("Processing Summary")
    print("=" * 80)
    
    for source, data in processed_data.items():
        if data is not None and len(data) > 0:
            print(f"✓ {source}: {len(data)} items processed successfully")
        else:
            print(f"✗ {source}: Processing failed or no data available")
    
    print("\nRaw data processing complete. Results saved to Processed directory.")
    print("=" * 80)
    
    return processed_data


if __name__ == "__main__":
    main() 