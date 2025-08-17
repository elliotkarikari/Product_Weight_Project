# ShelfScale: Food Product Weight Analysis

ShelfScale is a comprehensive tool for analyzing and predicting food product weights by leveraging data from multiple sources and machine learning techniques.

## Features

- **Multi-source data integration**: Combines data from McCance & Widdowson's food composition tables, Food Portion Sizes PDFs, Fruit and Vegetable Survey data, and the Open Food Facts API.
- **Machine learning matching**: Uses advanced text matching algorithms with machine learning to match food products across different datasets.
- **Self-improving algorithm**: The matching algorithm learns from existing matches to improve accuracy over time.
- **Enhanced weight extraction**: Advanced pattern recognition for extracting weight information from text descriptions in various formats.
- **Volume→weight conversion**: Automatic conversion of volume measurements to weight using category-specific density mappings.
- **Nutrition scoring**: Implements UK Traffic Light and Nutri-Score nutrition labelling systems.
- **Interactive dashboard**: Web-based dashboard with upload functionality and nutrition scoring visualization.
- **REST API**: FastAPI-based API for weight extraction and nutrition scoring services.
- **CLI tools**: Command-line interface for batch processing and scoring.
- **Weight prediction**: Predicts product weights for new items based on similar products in the database.
- **Comprehensive data processing**: Cleans, transforms, and normalizes weight data from various formats.
- **Food categorization**: Categorizes products into food groups for better analysis.

## Enhanced Weight Extraction

The system includes a robust weight extraction module with:

1. **Advanced pattern recognition**: Intelligently extracts weight information from diverse text formats 
2. **Support for multiple formats**:
   - Simple weights: "100g", "1kg"
   - Ranges: "100-150g"
   - Multipacks: "3 x 100g"
   - Fractions: "1/2 kg", "1 1/2 kg"
   - Mixed units: "1kg 500g"
3. **Unit standardization**: Converts various units (g, kg, oz, lb, ml, l) to standard units
4. **Volume→weight conversion**: Automatically converts volume measurements using density data:
   - Category-specific densities (milk: 1.03 g/ml, oil: 0.92 g/ml, etc.)
   - Fallback to generic density for unknown categories
   - Supports cl, dl, litre/liter variants
5. **Weight prediction**: For products with missing weights using:
   - Group-based prediction using food category averages
   - Similarity-based matching based on product names
6. **Confidence scoring**: Indicates reliability of extracted and predicted weights

### Using Weight Extraction in Code

```python
from shelfscale import WeightExtractor, predict_missing_weights

# Extract weights from text
extractor = WeightExtractor(target_unit='g')
weight, unit = extractor.extract("Chocolate bar, 3.5oz")
print(f"Extracted: {weight} {unit}")  # Output: 99.23 g

# Process a DataFrame with multiple columns
result_df = extractor.process_dataframe(
    df, 
    text_cols=['Product_Name', 'Description', 'Package_Size']
)

# Predict missing weights based on groups and similar items
result_df = predict_missing_weights(
    result_df,
    weight_col='Normalized_Weight',
    group_col='Food_Group',
    name_col='Food_Name'
)
```

## Machine Learning Capabilities

ShelfScale's matching algorithm incorporates several advanced features:

1. **Self-learning from matches**: The system automatically learns from previous matches to improve future matching accuracy.
2. **Feature-based similarity**: Uses multiple text features beyond simple matching, including fuzzy ratios, token sorting, and partial matching.
3. **Confidence scoring**: Provides confidence scores for matches and weight predictions.
4. **Performance evaluation**: Includes tools to evaluate matching performance and track improvements.
5. **Feedback incorporation**: Can incorporate user feedback to improve matching for specific items.

## Installation

```bash
# Create conda environment
conda create -n product_weight python=3.8 -y
conda activate product_weight

# Install dependencies
pip install -e .
```

## Nutrition Scoring

ShelfScale implements two major nutrition labelling systems:

### UK Traffic Light System
- **Green/Amber/Red** classification for fat, saturated fat, sugars, and salt
- Different thresholds for foods vs. beverages
- Per 100g/ml and per serving calculations
- Summary score highlighting the worst nutrient

### Nutri-Score (A-E)
- **Official algorithm** from Santé Publique France
- Negative points for energy, sugars, saturated fat, sodium
- Positive points for fruits/vegetables/nuts, fiber, protein
- Automatic A grade for water
- Protein exception rule implementation

## Usage

### CLI Scoring

Score products using the command line:

```bash
# Score all products using both systems
python -m shelfscale.main --score all --output-scores nutrition_scores.csv

# Score specific file with Traffic Light only
python -m shelfscale.main --score traffic --input-file products.csv --output-scores results.csv

# Score with Nutri-Score only
python -m shelfscale.main --score nutri --output-scores nutri_results.csv
```

### Basic Processing

Run the main script to process data and generate weight information:

```bash
python -m shelfscale.main
```

### Machine Learning Features

#### Training the Model

Train the matching model using existing data:

```bash
python -m shelfscale.main --train
```

#### Generate Training Data

Create a training dataset from high-quality matches:

```bash
python -m shelfscale.main --train --generate-training
```

#### Evaluate Performance

Evaluate the current performance of the matching algorithm:

```bash
python -m shelfscale.main --evaluate
```

#### Predict Weights for New Products

Predict weights for a new list of products:

```bash
python -m shelfscale.main --predict --input-file your_products.csv
```

### Interactive Dashboard

Launch the web dashboard for upload and interactive analysis:

```bash
python -m shelfscale.main --run-dashboard
```

The dashboard provides:
- **File upload** for CSV/Excel product data
- **Automatic scoring** with Traffic Light and Nutri-Score
- **Interactive visualizations** of nutrition score distributions
- **Data table** with sortable/filterable results
- **Export functionality** for processed results

### REST API

Start the FastAPI server for programmatic access:

```bash
# Using uvicorn directly
uvicorn shelfscale.api:app --host 0.0.0.0 --port 8000

# Using the convenience command
shelfscale-api
```

#### API Endpoints

- `POST /weights/extract` - Extract weights from text descriptions
- `POST /products/score` - Calculate nutrition scores for products
- `POST /batch/score` - Batch process CSV file upload
- `GET /health` - Health check endpoint
- `GET /docs` - Interactive API documentation

#### Example API Usage

```python
import requests

# Extract weights
response = requests.post("http://localhost:8000/weights/extract", json={
    "items": [
        {"text": "500ml milk", "super_category": "Milk and milk products"},
        "250g bread"
    ]
})

# Score products
response = requests.post("http://localhost:8000/products/score", json={
    "items": [
        {
            "name": "Apple",
            "energy_kcal": 52,
            "sugars_g": 10.4,
            "saturated_fat_g": 0.1,
            "salt_g": 0.0
        }
    ]
})
```

### Outputs

The system generates several output files:

- `weight_dataset.csv`: Integrated dataset with normalized weights
- `nutrition_scores.csv`: Products with nutrition scores and confidence metrics
- `consolidated_weights.csv`: Combined weight data from all sources
- `processed_data.csv`: Cleaned and processed product data
- `food_group_summary.csv`: Weight statistics by food group
- `mw_fps_matches.csv`: Matches between McCance & Widdowson and Food Portion Sizes data
- `mw_fvs_matches.csv`: Matches between McCance & Widdowson and Fruit & Vegetable Survey data
- `weight_extraction_errors.csv`: Log of failed weight extraction patterns
- `training_data.csv`: Training data for the matching algorithm

### Input Data Requirements

For nutrition scoring, the system expects these columns (per 100g/ml):

**Required for Traffic Light scoring:**
- `Fat_g`, `SatFat_g`, `Sugars_g`, `Salt_g` (or `Sodium_mg`)

**Required for Nutri-Score:**
- `Energy_kcal` (or `Energy_kJ`), `Sugars_g`, `SatFat_g`, `Salt_g` (or `Sodium_mg`)
- `Fiber_g`, `Protein_g`, `FVN_percent` (fruits/vegetables/nuts percentage)

**For volume→weight conversion:**
- `Super_Category`, `Food_Category` (for density lookup)

**Confidence and provenance tracking:**
- `Serving_Weight_g` (for per-serving calculations)
- Various source tracking columns are added automatically

## Contributing

To improve the matching algorithm:

1. Add new data sources to the system
2. Run the matching process to generate new matches
3. Review matches for accuracy
4. Train the model using the new data
5. Evaluate performance to track improvements

## Project Structure

```
shelfscale/
  ├── api.py                  # FastAPI REST API
  ├── main.py                 # CLI entry point and data integration
  ├── config.py               # Configuration and paths
  ├── data_processing/        # Data cleaning and transformation
  │   ├── densities.csv       # Density mappings for volume conversion
  │   ├── weight_extraction.py # Enhanced weight extraction with volume support
  │   └── ...                 # Other processing modules
  ├── data_sourcing/          # Data acquisition components
  ├── matching/               # Matching algorithms and ML
  ├── scoring/                # Nutrition scoring systems
  │   ├── traffic_lights.py   # UK Traffic Light implementation
  │   ├── nutri_score.py      # Nutri-Score implementation
  │   └── hfss.py             # HFSS model (stub)
  ├── utils/                  # Utility functions including learning
  └── visualization/          # Data visualization and dashboard
      └── dashboard.py        # Enhanced Dash dashboard with upload
```

## Technical Details

### Machine Learning Implementation

The matching algorithm uses:

1. TF-IDF vectorization for initial text similarity
2. Multiple fuzzy matching metrics (ratio, partial ratio, token sort)
3. Random Forest classifier for learning from match features
4. Ensemble approach that falls back to weighted averaging when model isn't available

### Volume→Weight Conversion

The system uses category-specific density mappings:

1. **Density database**: CSV file with Super_Category, Food_Category, and density values
2. **Lookup hierarchy**: Specific category → Super category → fallback (1.0 g/ml)
3. **Data sources**: USDA, UK FSA, FAO, and food science literature
4. **Unit support**: ml, l, cl, dl, cups, tbsp, tsp with automatic conversion

### Nutrition Scoring Implementation

**Traffic Light System:**
- Official UK Food Standards Agency thresholds
- Separate thresholds for foods vs. beverages
- Per 100g/ml and per serving calculations
- Color coding: green (low), amber (medium), red (high)

**Nutri-Score Algorithm:**
- Implements official Santé Publique France algorithm
- Negative points (A): energy, sugars, saturated fat, sodium (0-10 each)
- Positive points (C): fruits/veg/nuts, fiber, protein (0-5 each)
- Final score = A - C, mapped to grades A-E
- Special rules for beverages and protein exception

### Continuous Learning

The system improves over time through:

1. Learning weights for different matching features
2. Storing feature importance in a persistent model
3. Tracking performance metrics to measure improvement
4. Incorporating verified matches into the training data
5. Logging extraction errors for pattern analysis
6. Confidence scoring for weights and nutrition scores
