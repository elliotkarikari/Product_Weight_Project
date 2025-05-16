# ShelfScale: Food Product Weight Analysis

ShelfScale is a comprehensive tool for analyzing and predicting food product weights by leveraging data from multiple sources and machine learning techniques.

## Features

- **Multi-source data integration**: Combines data from McCance & Widdowson's food composition tables, Food Portion Sizes PDFs, Fruit and Vegetable Survey data, and the Open Food Facts API.
- **Machine learning matching**: Uses advanced text matching algorithms with machine learning to match food products across different datasets.
- **Self-improving algorithm**: The matching algorithm learns from existing matches to improve accuracy over time.
- **Weight prediction**: Predicts product weights for new items based on similar products in the database.
- **Comprehensive data processing**: Cleans, transforms, and normalizes weight data from various formats.
- **Food categorization**: Categorizes products into food groups for better analysis.
- **Enhanced weight extraction**: Advanced pattern recognition for extracting weight information from text descriptions in various formats.

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
4. **Weight prediction**: For products with missing weights using:
   - Group-based prediction using food category averages
   - Similarity-based matching based on product names
5. **Confidence scoring**: Indicates reliability of extracted and predicted weights

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

## Usage

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

### Outputs

The system generates several output files:

- `consolidated_weights.csv`: Combined weight data from all sources
- `processed_data.csv`: Cleaned and processed product data
- `food_group_summary.csv`: Weight statistics by food group
- `mw_fps_matches.csv`: Matches between McCance & Widdowson and Food Portion Sizes data
- `mw_fvs_matches.csv`: Matches between McCance & Widdowson and Fruit & Vegetable Survey data
- `training_data.csv`: Training data for the matching algorithm

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
  ├── data_processing/        # Data cleaning and transformation
  ├── data_sourcing/          # Data acquisition components
  ├── matching/               # Matching algorithms and ML
  ├── utils/                  # Utility functions including learning
  └── visualization/          # Data visualization
```

## Technical Details

### Machine Learning Implementation

The matching algorithm uses:

1. TF-IDF vectorization for initial text similarity
2. Multiple fuzzy matching metrics (ratio, partial ratio, token sort)
3. Random Forest classifier for learning from match features
4. Ensemble approach that falls back to weighted averaging when model isn't available

### Continuous Learning

The system improves over time through:

1. Learning weights for different matching features
2. Storing feature importance in a persistent model
3. Tracking performance metrics to measure improvement
4. Incorporating verified matches into the training data
