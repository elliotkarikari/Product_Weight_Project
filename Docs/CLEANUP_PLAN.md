# Codebase Cleanup Plan

## Current Issues Identified

### 🗑️ Redundant/Obsolete Files
1. **Duplicate databases**: Multiple comprehensive_food_database files with timestamps
2. **Old test files**: quick_test.py (replaced by enhanced system)
3. **Demo outputs**: Multiple timestamped output files in various folders
4. **Temporary files**: Files in cache/, converted_data/ with timestamps
5. **Old documentation**: LLM_MATCHING_GUIDE.md (superseded by enhanced system)

### 📁 Poor Organization
1. **Root directory clutter**: Too many scripts in project root
2. **Mixed concerns**: Development files mixed with production files
3. **Inconsistent naming**: Some files follow conventions, others don't
4. **Missing structure**: No clear separation of tools vs core system

## 🎯 Target Structure

```
Product_Weight_Project/
├── README.md                          # Updated main documentation
├── requirements.txt                   # Dependencies
├── setup.py                          # Installation
├── .env                              # Environment variables
├── .gitignore                        # Git ignore rules
│
├── shelfscale/                       # Core system (KEEP AS IS - WORKING)
│   ├── __init__.py
│   ├── main.py
│   ├── api.py
│   ├── matching/                     # Enhanced matching system
│   ├── ml/                          # LLM and ML components
│   ├── data_processing/             # Data processing
│   └── ...
│
├── tools/                           # Retail-focused tools
│   ├── curation/                    # McCance & Widdowson curation
│   │   ├── llm_enhanced_product_curation.py
│   │   ├── ENHANCED_CURATION_RULES.md
│   │   └── CURATION_COMPARISON.md
│   ├── matching/                    # Enhanced food matching
│   │   ├── enhanced_food_matching.py
│   │   ├── demo_enhanced_matching.py
│   │   └── test_enhanced_matching.py
│   └── data_conversion/             # Data format converters
│       ├── data_converter.py
│       └── interactive_matcher.py
│
├── examples/                        # Usage examples and demos
│   ├── basic_matching_demo.py
│   ├── curation_demo.py
│   └── api_setup_guide.py
│
├── tests/                          # Test suite (CLEAN UP)
│   ├── __init__.py
│   ├── test_enhanced_system.py      # Main system tests
│   ├── test_curation.py            # Curation tests
│   └── test_matching.py            # Matching tests
│
├── data/                           # Data organization
│   ├── raw/                        # Original datasets (from Data/Raw Data/)
│   ├── processed/                  # Processed datasets (from Data/Processed/)
│   └── outputs/                    # System outputs
│       ├── curated_datasets/       # Curated McCance & Widdowson
│       ├── matched_products/       # Product matching results
│       └── reports/                # Analysis reports
│
├── docs/                          # Documentation
│   ├── ENHANCED_CURATION_GUIDE.md  # Curation system guide
│   ├── MATCHING_SYSTEM_GUIDE.md    # Matching system guide
│   ├── API_REFERENCE.md            # API documentation
│   └── DEVELOPMENT_GUIDE.md        # Development setup
│
└── legacy/                        # Legacy files (for reference)
    ├── jupyter_notebooks/          # Original analysis notebooks
    ├── old_tools/                  # Superseded tools
    └── archive/                    # Old outputs and experiments
```

## 🧹 Cleanup Actions

### Files to REMOVE
- comprehensive_food_database_*.csv (keep latest only)
- quick_test.py (replaced by enhanced system)
- curated_mccance_widdowson_*.csv (move to outputs)
- curation_report_*.md (move to outputs)
- All timestamped files in llm_matching_results/
- All timestamped files in llm_testing_results/
- cache/ folder contents
- converted_data/ folder (move useful files to outputs)

### Files to MOVE
- **To tools/curation/**: llm_enhanced_product_curation.py, ENHANCED_CURATION_RULES.md, CURATION_COMPARISON.md
- **To tools/matching/**: enhanced_food_matching.py, demo_enhanced_matching.py, test_enhanced_matching.py
- **To tools/data_conversion/**: data_converter.py, interactive_matcher.py
- **To examples/**: demos/ folder contents (renamed appropriately)
- **To data/**: Data/ folder (reorganized)
- **To legacy/**: Jupter_notebooks/ (renamed Jupyter_notebooks)
- **To tests/**: Consolidate test files, remove redundant ones

### Files to UPDATE
- README.md (completely rewrite for enhanced system)
- requirements.txt (ensure all dependencies listed)
- setup.py (update for new structure)

## 🎯 Benefits of New Structure

1. **Clear Separation**: Tools vs core system vs examples
2. **Enhanced Focus**: Everything built on retail-focused system
3. **Better Organization**: Logical folder structure
4. **Easier Maintenance**: Related files grouped together
5. **Professional Structure**: Standard Python project layout
6. **Future-Proof**: Easy to add new tools and features