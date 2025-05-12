# ShelfScale

A standardized data product that enables understanding of nutrition and sustainability metrics at the basket level.

## Overview

ShelfScale is a Python package that provides tools for analyzing food product weights across different food groups. It helps in standardizing food product data, making it easier to compare and analyze nutritional and sustainability metrics.

## Background

Unpackaged product often contains missing weight information. Their nature (size and weight) doesn't allow for consitent weight information/data. ShelfScale addresses this by producing a generally accepted standard product weight database to use in the absence of retailer-specific data (Sold weights and Portion weights). This enables exploration of various components of human interaction with retail purchases, such as understanding the carbon footprint of a person's grocery shop or calculating the total number of calories from a basket shop.

## Project Aims

Goal 1: Produce a comprehensive product list with standardized weights and meta-data documentation of information sources and decisions made.

## Key Considerations

- Imputation Methods: Various statistical and machine learning methods are implemented to impute missing weight data.
- Matching Algorithms: Fuzzy matching algorithms are used to match retail data to our generic product list, with configurable similarity thresholds.
- Granularity: The database handles different levels of product specificity (e.g., 'crisps' - individual or share bag) and incorporates options for standard large/small portion sizes.
- Product Selection: Products are based primarily on McCance & Widdowson's composition of food integrated dataset.
- Quantity Measurement: Products are measured in consistent weight units (primarily grams) for easy comparison.

## Methodology & Decisions Made

*Workflow Approach: Initial work was done in alphabetical order to build a systematic workflow which was then automated.
*Data Grouping: Foods were grouped by McCance, Widdowson Group (2 or 3 letter codes assigned to every food) and counted.
*Data Reduction: Product list was reduced to its most basic form by research and verification against retail sources (e.g., Tesco website).
*Hierarchical Categorization: Added "Super Group" classification to create a larger category for easier searching, aligned with the Eatwell Guide groupings.
\*Data Integration: All tables were joined together to create a comprehensive database.

## Features

- **Data Sourcing**: Fetch food product data from Open Food Facts API
- **Data Processing**: Clean and transform food product data
- **Matching Algorithm**: Match food products across different datasets
- **Visualization**: Interactive dashboard for data exploration
- **Utilities**: Helper functions for working with food data

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/shelfscale.git
cd shelfscale

# Install the package
pip install -e .
```

## Usage

### Basic Usage

```python
from shelfscale.data_sourcing.open_food_facts import OpenFoodFactsClient
from shelfscale.data_processing.cleaning import clean_weight_column

# Get data from Open Food Facts
client = OpenFoodFactsClient()
vegetables = client.search_by_food_group('vegetables')

# Clean the weight column
cleaned_data = clean_weight_column(vegetables)
```

### Run the Demo

```bash
# Run the main demo script
python -m shelfscale.main
```

### Launch Dashboard

```python
from shelfscale.visualization.dashboard import ShelfScaleDashboard
import pandas as pd

# Load your data
df = pd.read_csv('your_data.csv')

# Create and launch dashboard
dashboard = ShelfScaleDashboard(df)
dashboard.run_server(debug=True)
```

## Project Structure

```
shelfscale/
├── data_sourcing/     # Data sourcing modules
├── data_processing/   # Data cleaning and transformation
├── matching/          # Food matching algorithms
├── visualization/     # Data visualization components
├── utils/             # Utility functions
├── main.py            # Example script
└── __init__.py        # Package initialization
```

## Dependencies

- pandas
- numpy
- requests
- plotly
- dash
- scikit-learn
- fuzzywuzzy
- python-Levenshtein
- openpyxl

## References

McCance & Widdowson's composition of food integrated dataset

## License

MIT
