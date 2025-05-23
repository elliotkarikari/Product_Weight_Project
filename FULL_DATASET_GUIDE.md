# Full Dataset Curation Guide

This guide explains how to run curation on the entire dataset (2889+ items) and use the different methods available.

## 🔧 Current Status

Your system successfully processed 50 items in demo mode:

- **Cost**: $0.039 for 50 items  
- **Estimated full cost**: ~$2.25 for 2889 items
- **Reduction**: 64% (50 → 18 items)
- **Time**: Very fast for demos

## 📊 Available Methods

### 1. **LLM-Only** (Most Thorough)

- Uses OpenAI GPT for all decisions
- **Cost**: ~$2-5 for full dataset
- **Accuracy**: Highest
- **Speed**: Slower (API calls)

### 2. **Rules-Only** (Fastest)

- Uses extracted LLM logic as rules
- **Cost**: $0 (no API calls)
- **Accuracy**: Very good
- **Speed**: Very fast

### 3. **Hybrid** (Balanced)

- Rules for confident decisions, LLM for uncertain cases
- **Cost**: ~$0.50-1.50 for full dataset
- **Accuracy**: High
- **Speed**: Fast

### 4. **Auto** (Recommended)

- Automatically chooses best method based on your configuration

## 🚀 Running on Full Dataset

### Option 1: Disable Demo Mode in .env

Edit your `.env` file and change:

```env
# Change this line:
LLM_DEMO_MODE=true

# To this:
LLM_DEMO_MODE=false
```

### Option 2: Use Configuration Override

Run with demo mode disabled:

```python
from shelfscale.config_manager import get_config

config = get_config()
config.LLM_DEMO_MODE = False  # Override demo mode

# Then run your preferred curator
```

## 💰 Cost Management

### Before Running Full Dataset:

1. **Check your settings**:

   ```env
   MAX_ESTIMATED_COST=10.00        # Increase if needed
   COST_WARNING_THRESHOLD=5.00     # Warning level
   LLM_BATCH_SIZE=10              # Smaller = slower but safer
   RATE_LIMIT_DELAY=1.0           # Delay between API calls
   ```

2. **Monitor costs**:
   - Demo run: $0.039 for 50 items
   - Full dataset estimate: ~$2.25 (may vary)
   - Hybrid approach: ~$0.75 (rules + selective LLM)

## 🏃‍♂️ Quick Start Commands

### 1. LLM Full Dataset (Most Accurate)

```bash
cd Product_Weight_Project_Build
python -c "
from shelfscale.config_manager import get_config
config = get_config()
config.LLM_DEMO_MODE = False

from shelfscale.data_processing.llm_retail_curator import run_llm_curation_sync
curated_df, report = run_llm_curation_sync()
"
```

### 2. Rules Full Dataset (Fastest, Free)

```bash
cd Product_Weight_Project_Build
python shelfscale/data_processing/enhanced_rules_curator.py
```

### 3. Hybrid Full Dataset (Balanced)

```bash
cd Product_Weight_Project_Build
python shelfscale/data_processing/hybrid_curator.py
```

### 4. Interactive Hybrid Selector

```bash
cd Product_Weight_Project_Build
python -c "
from shelfscale.data_processing.hybrid_curator import run_hybrid_curation_sync

# Choose method interactively
print('Choose method:')
print('1. rules (free, fast)')
print('2. llm (thorough, ~$2.25)')  
print('3. hybrid (balanced, ~$0.75)')
print('4. auto (smart choice)')

choice = input('Enter choice (1-4): ')
method_map = {'1': 'rules', '2': 'llm', '3': 'hybrid', '4': 'auto'}
method = method_map.get(choice, 'auto')

# Disable demo mode
from shelfscale.config_manager import get_config
config = get_config()
config.LLM_DEMO_MODE = False

curated_df, report = run_hybrid_curation_sync(method=method)
"
```

## 📈 Performance Comparison

Based on demo results, here's what to expect:

| Method | Cost | Time | Accuracy | Items (from 2889) |
|--------|------|------|----------|-------------------|
| LLM | ~$2.25 | 30-60 min | Highest | ~1000-1200 |
| Rules | $0 | 2-5 min | Very Good | ~900-1100 |
| Hybrid | ~$0.75 | 10-20 min | High | ~1000-1150 |

## 🎯 Recommended Approaches

### For Production Use:

1. **Start with Rules** - Fast, free, good results
2. **Validate with LLM** - Run LLM on a sample to compare
3. **Use Hybrid** - Best of both worlds

### For Research/Analysis:

1. **Use LLM** - Most thorough analysis
2. **Document decisions** - LLM provides reasoning for each item

### For Regular Updates:

1. **Use Hybrid** - Balanced approach
2. **Cache results** - Avoid re-analyzing same items

## 🛠️ Advanced Configuration

### Optimize for Speed:

```env
LLM_BATCH_SIZE=20              # Larger batches
RATE_LIMIT_DELAY=0.5           # Faster rate
LLM_CACHE_ENABLED=true         # Enable caching
```

### Optimize for Cost:

```env
LLM_BATCH_SIZE=5               # Smaller batches
RATE_LIMIT_DELAY=2.0           # Slower rate
MAX_ESTIMATED_COST=3.00        # Lower cost limit
OPENAI_MODEL=gpt-3.5-turbo     # Cheaper model
```

### Optimize for Quality:

```env
OPENAI_MODEL=gpt-4             # Better model (more expensive)
OPENAI_TEMPERATURE=0.1         # More consistent
LLM_BATCH_SIZE=5               # Smaller batches for attention
```

## 📊 Understanding Results

### Output Files:

- **Dataset**: `*_curated_retail_dataset.csv` - Final curated items
- **Report**: `*_curation_report.json` - Detailed metrics and steps  
- **Analyses**: `*_analyses.json` - Individual item decisions

### Key Metrics:

- **Reduction Rate**: % of items removed
- **Retail Confidence**: How certain we are items are retail
- **Representativeness**: How typical/average items are
- **Category Diversity**: Number of different food categories

## 🚨 Troubleshooting

### Cost Limit Exceeded:

```md
🛑 Cost limit exceeded: $5.50 > $5.00
```

**Solution**: Increase `MAX_ESTIMATED_COST` in `.env`

### Model Not Found:

```md
OpenAI API error 404: model not found
```

**Solution**: Change `OPENAI_MODEL` to `gpt-3.5-turbo` in `.env`

### Rate Limited:

```md
API error 429, retrying...
```

**Solution**: Increase `RATE_LIMIT_DELAY` in `.env`

### Out of Memory

```md
Memory error processing large dataset
```

**Solution**: Reduce `LLM_BATCH_SIZE` or use `rules` method

## 🎉 Next Steps

1. **Choose your method** based on your priorities (cost/speed/accuracy)
2. **Configure settings** in your `.env` file
3. **Run full dataset** using one of the commands above
4. **Review results** in the output files
5. **Iterate and improve** based on your specific needs

The enhanced rules system now incorporates all the logic the LLM uses, so you have a robust fallback that doesn't require API calls while still having the option to use LLM when you need maximum accuracy! 