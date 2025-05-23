#!/usr/bin/env python3
"""
Test script to check data loading capabilities
"""
import sys
import os
sys.path.append('shelfscale')

import pandas as pd
from shelfscale.config_manager import get_config

print("Testing data loading...")
print(f"Raw data dir: {get_config.RAW_DATA_DIR}")
print(f"Raw data dir exists: {os.path.exists(get_config.RAW_DATA_DIR)}")

# Test McCance & Widdowson loading
try:
    print(f"\nTesting McCance & Widdowson Excel file...")
    print(f"File path: {get_config.MCCANCE_WIDDOWSON_PATH}")
    print(f"File exists: {os.path.exists(get_config.MCCANCE_WIDDOWSON_PATH)}")
    
    xl = pd.ExcelFile(get_config.MCCANCE_WIDDOWSON_PATH)
    print(f"Available sheets: {xl.sheet_names}")
    
    print(f"Attempting to load sheet: {get_config.MW_SHEET_NAME_FOR_MAIN_PY}")
    df = pd.read_excel(get_config.MCCANCE_WIDDOWSON_PATH, sheet_name=get_config.MW_SHEET_NAME_FOR_MAIN_PY)
    print(f"Successfully loaded {len(df)} rows with columns: {list(df.columns)[:5]}...")
    
except Exception as e:
    print(f"Error loading McCance & Widdowson data: {e}")

# Test PDF files
pdf_files = [
    ("Food Portion PDF", get_config.FOOD_PORTION_PDF_PATH),
    ("Fruit & Veg PDF", get_config.FRUIT_VEG_SURVEY_PDF_PATH)
]

for name, path in pdf_files:
    print(f"\n{name}: {path}")
    print(f"Exists: {os.path.exists(path)}")

# Test super group files
print(f"\nTesting super group data...")
super_group_dir = get_config.MW_PROCESSED_SUPER_GROUP_PATH
print(f"Super group dir: {super_group_dir}")
print(f"Super group dir exists: {os.path.exists(super_group_dir)}")

if os.path.exists(super_group_dir):
    csv_files = [f for f in os.listdir(super_group_dir) if f.endswith('.csv')]
    print(f"Found {len(csv_files)} CSV files")
    
    total_rows = 0
    for csv_file in csv_files[:3]:  # Test first 3 files
        try:
            file_path = os.path.join(super_group_dir, csv_file)
            df = pd.read_csv(file_path)
            rows = len(df)
            print(f"  {csv_file}: {rows} rows")
            total_rows += rows
        except Exception as e:
            print(f"  {csv_file}: Error - {e}")
    
    print(f"Total data rows from sample files: {total_rows}")

print("\nData loading test complete!") 