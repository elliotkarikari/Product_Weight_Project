# McCance & Widdowson Curation Workflow

A comprehensive, intelligent workflow for processing McCance & Widdowson food composition data through automated curation, analysis, and continuous improvement.

## 🎯 Overview

This workflow combines the power of LLM (Large Language Model) intelligence with rule-based processing to create high-quality, curated food composition datasets. It includes automated pattern recognition, rule improvement, and continuous learning capabilities.

## 🏗️ Architecture

```
McCance_and_Widdowson_Workflow/
├── core_components/          # Main pipeline orchestration
│   └── automated_pipeline.py # Complete workflow automation
├── data_processing/          # Data curation components
│   ├── hybrid_curator.py     # Combines LLM + rules
│   ├── enhanced_rules_curator.py # Advanced rule-based curation
│   └── llm_retail_curator.py # LLM-powered curation
├── analysis/                 # Reasoning analysis
│   └── llm_reasoning_analyzer.py # Pattern extraction & insights
├── utils/                    # Utilities
│   └── rule_updater.py       # Automated rule improvements
├── config/                   # Configuration
│   ├── config_manager.py     # Base configuration
│   └── workflow_config.py    # Workflow-specific settings
├── output/                   # Generated outputs
│   ├── pipeline_runs/        # Timestamped workflow runs
│   ├── curation_output/      # Curated datasets
│   ├── analysis_output/      # Analysis results
│   └── rule_backups/         # Rule backup files
└── run_workflow.py          # Main entry point
```

## 🚀 Quick Start

### 1. Demo Mode (Recommended for first time)
```bash
cd McCance_and_Widdowson_Workflow
python run_workflow.py --demo
```

### 2. Production Mode
```bash
python run_workflow.py --production --method hybrid --target-size 1000
```

### 3. Interactive Mode
```bash
python run_workflow.py --interactive
```

### 4. Status Check
```bash
python run_workflow.py --status
```

## 🧠 Curation Methods

### Auto (Recommended)
- Automatically selects the best method based on available resources
- Falls back gracefully if LLM is unavailable

### Hybrid (Balanced)
- Uses rules for initial filtering
- Applies LLM for uncertain cases
- Cost-effective with high accuracy
- **Estimated cost**: ~$0.75 for full dataset

### LLM Only (Most Accurate)
- Uses LLM for all decisions
- Highest accuracy but higher cost
- **Estimated cost**: ~$2.25 for full dataset

### Rules Only (Fastest)
- Pure rule-based processing
- No API costs, very fast
- Good accuracy for most cases

## 📊 Workflow Stages

### 1. Data Loading
- Loads McCance & Widdowson food composition data
- Standardizes column names and formats
- Validates data integrity

### 2. Hybrid Curation
- **Rules Analysis**: Fast initial assessment of all products
- **Uncertainty Detection**: Identifies borderline cases
- **LLM Verification**: Applies LLM to uncertain products only
- **Decision Fusion**: Combines insights from both methods

### 3. Reasoning Analysis
- **Disagreement Analysis**: Identifies where LLM and rules disagree
- **Pattern Extraction**: Discovers common reasoning patterns
- **Insight Generation**: Creates actionable improvement suggestions

### 4. Rule Updates
- **Automated Improvement**: Updates rules based on LLM insights
- **Backup Management**: Safely backs up existing rules
- **Pattern Integration**: Incorporates new exclusion/inclusion patterns

### 5. Continuous Learning
- **Performance Tracking**: Monitors agreement rates and accuracy
- **Iterative Improvement**: Each run improves the next
- **Quality Metrics**: Comprehensive quality assessment

## 📈 Performance Metrics

| Method | Cost | Time | Accuracy | Items (from 2889) |
|--------|------|------|----------|-------------------|
| LLM | ~$2.25 | 30-60 min | Highest | ~1000-1200 |
| Rules | $0 | 2-5 min | Very Good | ~900-1100 |
| Hybrid | ~$0.75 | 10-20 min | High | ~1000-1150 |

## 🛠️ Configuration

### Workflow Configuration
Edit `config/workflow_config.py` to customize:

```python
# Data processing settings
default_target_size = None  # Auto-size
default_curation_method = "auto"
enable_llm_fallback = True

# Analysis settings
disagreement_threshold = 0.3
pattern_frequency_threshold = 3

# LLM settings
llm_model = "gpt-3.5-turbo"
llm_temperature = 0.1
```

### Preset Configurations
```python
from config.workflow_config import configure_for_demo, configure_for_production

# Demo configuration
config = configure_for_demo(llm_enabled=True)

# Production configuration  
config = configure_for_production(target_size=1000)
```

## 📁 Output Structure

Each workflow run creates a timestamped directory within the workflow folder:
```
McCance_and_Widdowson_Workflow/output/pipeline_runs/run_YYYYMMDD_HHMMSS/
├── pipeline_results.json        # Complete results
├── summary_report.json         # Key metrics summary
├── summary_report.txt          # Human-readable summary
├── llm_reasoning_analysis.json # Detailed analysis
└── curation_output/            # Curated datasets
    ├── hybrid_curated_retail_dataset.csv
    ├── hybrid_curation_report.json
    └── hybrid_analyses.json
```

Additional output directories:
```
McCance_and_Widdowson_Workflow/output/
├── pipeline_runs/              # All workflow runs (timestamped)
├── curation_output/           # Latest curation results  
├── analysis_output/           # Latest analysis results
└── rule_backups/              # Rule backup files
```

## 🔧 Advanced Usage

### Programmatic Interface
```python
from McCance_and_Widdowson_Workflow import run_complete_workflow

# Run with custom settings
results = run_complete_workflow(
    target_size=500,
    update_rules=True,
    backup_existing=True
)
```

### Component-Level Access
```python
from McCance_and_Widdowson_Workflow import HybridCurator, LLMReasoningAnalyzer

# Use individual components
curator = HybridCurator()
curated_df, report = await curator.curate_dataset_hybrid(df, method="hybrid")

analyzer = LLMReasoningAnalyzer()
patterns = analyzer.extract_patterns(analyses)
```

## 📚 Key Features

### 🤖 Intelligent Curation

- **Hybrid Approach**: Combines rule-based speed with LLM accuracy
- **Uncertainty Detection**: Automatically identifies cases needing LLM review
- **Cost Optimization**: Uses LLM only when necessary

### 🔍 Advanced Analysis

- **Disagreement Tracking**: Monitors where methods disagree
- **Pattern Recognition**: Discovers new food item patterns
- **Confidence Scoring**: Provides confidence metrics for all decisions

### ⚡ Continuous Improvement

- **Automated Rule Updates**: Learns from LLM decisions
- **Pattern Integration**: Incorporates new patterns automatically
- **Performance Monitoring**: Tracks improvement over time

### 🛡️ Safety & Reliability

- **Backup Management**: Always backs up before changes
- **Graceful Fallbacks**: Continues working if components fail
- **Comprehensive Logging**: Detailed logs for debugging

## 🔬 Research Applications

### Academic Research

```python
from config.workflow_config import configure_for_research

# Configure for detailed analysis
config = configure_for_research(save_everything=True)
```

### Quality Assessment

- **Agreement Rate Tracking**: Monitor LLM vs rules agreement
- **Confidence Distribution**: Analyze decision confidence patterns
- **Category Coverage**: Ensure balanced representation

### Pattern Discovery

- **Homemade Detection**: Identify non-retail items
- **Preparation Methods**: Extract cooking/preparation patterns
- **Food Categories**: Discover emerging food categories

## 🎓 Best Practices

### For Demo/Testing

1. Start with `--demo` mode
2. Use small target sizes (50-100 items)
3. Enable all analysis features

### For Production

1. Use `--production` mode
2. Set appropriate target sizes
3. Enable rule updates for continuous improvement
4. Monitor logs and results

### For Research

1. Use hybrid or LLM methods
2. Save all intermediate results
3. Lower thresholds for pattern detection
4. Analyze disagreement patterns

## 🚨 Troubleshooting

### Common Issues

**LLM Not Available**

- Check OpenAI API key configuration
- Verify internet connection
- Use `--method rules` as fallback

**Data Source Not Found**

- Check `MCCANCE_WIDDOWSON_PATH` in config
- Verify file exists and is readable
- Check sheet name configuration

**Memory Issues**

- Use smaller target sizes
- Enable intermediate result clearing
- Increase system memory

### Getting Help

1. **Status Check**: `python run_workflow.py --status`
2. **Logs**: Check workflow output directory for logs
3. **Configuration**: Verify settings in `config/workflow_config.py`

## 📊 Quality Metrics

The workflow tracks several quality metrics:

- **Reduction Rate**: Percentage of data filtered out
- **Retail Confidence**: Average confidence in retail product identification
- **Representativeness**: How well the final dataset represents food categories
- **Agreement Rate**: How often LLM and rules agree
- **Category Diversity**: Number of food categories represented

## 🔄 Continuous Improvement

Each workflow run contributes to system improvement:

1. **Pattern Learning**: New patterns discovered are integrated
2. **Rule Enhancement**: Rules are updated based on LLM insights  
3. **Threshold Optimization**: Confidence thresholds are refined
4. **Quality Tracking**: Performance metrics guide improvements

## 📞 Support

For questions or issues:

1. Check the troubleshooting section above
2. Review workflow logs in the output directory
3. Examine configuration settings
4. Run status check for system health

---

**Version**: 1.0.0  
**Data Source**: McCance & Widdowson Food Composition Dataset  
**License**: Product Weight Project License 