# ShelfScale Data Processing

This document explains the data processing system for the ShelfScale product weight analysis project.

## Overview

The ShelfScale project integrates multiple data sources to create a comprehensive dataset of food products with accurate weight information. The data processing system handles:

- PDF extraction with robust error handling and multiple encoding options
- Excel data processing and standardization
- Data matching between different sources
- Weight extraction and normalization
- Food categorization

## Data Sources

The system processes four main data sources:

1. **McCance and Widdowson's Food Composition Data** (Excel)
   - Primary source of standardized food data
   - Contains detailed food composition information
   - Found in `Data/Raw Data/McCance_Widdowsons_2021.xlsx`

2. **Food Portion Sizes** (PDF)
   - Contains information about standard portion sizes for different foods
   - Includes weight in grams for each portion
   - Found in `Data/Raw Data/Food_Portion_Sizes.pdf`

3. **Fruit and Vegetable Survey** (PDF)
   - Contains survey data on fruit and vegetable products
   - Includes pack sizes and weights
   - Found in `Data/Raw Data/fruit_and_vegetable_survey_*.pdf`

4. **Labelling Dataset** (CSV/Excel)
   - Contains food labelling information
   - May include multiple files with variations in format
   - Found in `Data/Raw Data/Labelling*.csv` or `Data/Raw Data/Labelling*.xlsx`

## Directory Structure

The processed data is organized into the following structure:

'''
Data/
  Raw Data/
    McCance_Widdowsons_2021.xlsx
    Food_Portion_Sizes.pdf
    fruit_and_vegetable_survey_*.pdf
    Labelling*.csv
  Processed/
    FoodPortionSized/
      Processed_Food_Portion_Sizes.csv
    Fruit and Veg Sample reports/
      tables/
      text/
    MW_DataReduction/
      Reduced Individual Tables/
      Reduced Super Group/
        Cleaned/
      Reduced Total/
        McCance_Widdowson_Full.csv
    ReducedwithWeights/
      Processed_Labelling_Data.csv
```

## Key Components

### 1. PDF Extraction (pdf_extraction.py)

Handles extraction of structured data from PDFs using multiple methods:
- Lattice-based extraction for table-structured content
- Stream-based extraction for loosely structured content
- Text-based extraction as a fallback
- Supports multiple encodings to handle different PDF formats

### 2. Raw Data Processing (raw_processor.py)

The `RawDataProcessor` class manages the entire raw data processing workflow:
- Creates the necessary directory structure
- Processes each data source into a standardized format
- Cleans and categorizes the data
- Saves processed data to the appropriate location

### 3. Data Integration (main.py)

The main script integrates all data sources:
- Loads processed data from each source
- Matches items across datasets using fuzzy matching
- Normalizes weight information
- Creates a comprehensive integrated dataset

### 4. Matching Algorithm (matching/algorithm.py)

The `FoodMatcher` class provides sophisticated matching between datasets:
- Uses fuzzy string matching with TF-IDF vectorization
- Handles common variations in food names
- Supports additional columns for improved matching accuracy
- Configurable similarity threshold

## Using the Scripts

### Processing Raw Data

To process all raw data sources:

```bash
python shelfscale/data_processing/process_raw_data.py
```

Options:
- `--raw-dir`: Path to raw data directory (default: "Data/Raw Data")
- `--processed-dir`: Path to processed data directory (default: "Data/Processed")
- `--log-file`: Path to log file (default: "raw_data_processing.log")

### Verifying PDF Extraction

To test different PDF extraction methods:

```bash
python shelfscale/data_processing/process_raw_data.py --verify
```

This will try all extraction methods on each PDF and report success rates.

### Fixing Indentation Issues

For Python files with indentation problems:

```bash
python shelfscale/utils/fix_indentation.py path/to/file.py
```

### Running the Main Process

To run the complete data processing and integration:

```bash
python shelfscale/main.py --process-raw
```

This will process all raw data and integrate it into a comprehensive dataset.

## Advanced Usage

### Adding New Data Sources

To add a new data source:

1. Create a new method in `RawDataProcessor` to handle the specific format
2. Add the new source to the `process_all()` method
3. Update the main script to include the new data in the integration

### Customizing Matching

To customize the matching algorithm:

1. Create a custom instance of `FoodMatcher` with your desired parameters
2. Adjust the similarity threshold (default: 0.7 or 70%)
3. Add additional matching columns for improved accuracy

## Troubleshooting

### Empty Datasets

If processing results in empty datasets:
- Check that the raw data files exist in the expected location
- Try different PDF extraction methods with the `--verify` flag
- Check the log file for specific error messages

### PDF Extraction Failures

If PDF extraction fails:
- Try converting the PDF to a different format
- Use the text-based extraction method as a fallback
- Check for password protection or security settings in the PDF

### Column Matching Issues

If columns aren't matched correctly across datasets:
- Use the standardize_columns function to normalize column names
- Check for variations in column naming (e.g., "Food Name" vs "Food_Name")
- Add custom column mappings in the matcher.merge_matched_datasets call

## Testing

Run the test suite to verify data integration:

```bash
python -m unittest tests/test_data_integration.py
```

This will test:
- PDF extraction functionality
- Dataset matching and merging
- Column standardization
- Raw data processing 

# Enhanced Weight Extraction

One of the key challenges in processing food product data is extracting and standardizing weight information from varied text formats. The ShelfScale project includes an enhanced weight extraction system with the following capabilities:

## Key Features

### 1. Robust Pattern Recognition
The enhanced weight extractor can identify and parse a wide variety of weight formats:
- Simple formats: "100g", "250 g", "1kg", "500ml"
- Range formats: "100-150g" (takes average)
- Multipack formats: "3 x 100g", "6-pack x 30g"
- Fraction formats: "1/2 kg", "1/4 cup"
- Mixed fractions: "1 1/2 kg", "2 1/4 cups"
- Embedded in text: "Cheese, cheddar (200g block)", "Bread, whole wheat, 400g loaf"
- Common approximations: "approx 100g", "approximately 250ml"

### 2. Unit Standardization
The extractor automatically standardizes units, with comprehensive support for:
- Weight units: g, kg, mg, oz, lb, pounds, etc.
- Volume units: ml, l, cups, tbsp, tsp, fluid oz, etc.
- Consistent conversions between compatible units

### 3. Missing Value Prediction
The system includes predictive capabilities for estimating missing weight values:
- Group-based prediction: Uses food category averages
- Similarity-based prediction: Matches similar products by name
- Confidence scoring: Provides reliability metrics for predictions

## Usage Examples

### Basic Weight Extraction
```python
from shelfscale.data_processing.weight_extraction import WeightExtractor

# Create an extractor with default target unit (grams)
extractor = WeightExtractor()

# Extract from a single text string
weight, unit = extractor.extract("6pk x 25g")
print(f"Weight: {weight}{unit}")  # Output: Weight: 150g
```

### Processing a DataFrame
```python
from shelfscale.data_processing.weight_extraction import WeightExtractor

# Create the extractor
extractor = WeightExtractor(target_unit='g')

# Process weight information from multiple columns
result_df = extractor.process_dataframe(
    df,
    text_cols=['Product_Description', 'Package_Info'],
    new_weight_col='Normalized_Weight',
    new_unit_col='Weight_Unit',
    source_col='Weight_Source'
)
```

### Predicting Missing Weights
```python
from shelfscale.data_processing.weight_extraction import predict_missing_weights

# Fill missing weights using group averages and similar items
enriched_df = predict_missing_weights(
    df,
    weight_col='Normalized_Weight',
    group_col='Food_Group',
    name_col='Food_Name'
)
```

## Integration with Data Pipeline

The enhanced weight extraction is seamlessly integrated with the data processing pipeline and can be used directly from the main.py script by using the `--process-weights` option.

```
python main.py --input-data your_data.csv --process-weights
``` 