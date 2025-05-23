# Environment Configuration Setup

This guide explains how to set up your environment variables and API keys for the Product Weight Project.

## Quick Start

1. **Run the setup script:**

   ```bash
   python setup_environment.py
   ```

2. **Follow the prompts to:**
   - Create your `.env` file
   - Add your OpenAI API key
   - Install dependencies
   - Create necessary directories

## Manual Setup

### 1. Create .env File

Copy the template and create your own `.env` file:

```bash
# Copy the template
cp config_template.env .env

# Edit the .env file with your settings
notepad .env  # Windows
nano .env     # Linux/Mac
```

### 2. Required Configuration

**OpenAI API Key (Required for LLM features):**

```env
OPENAI_API_KEY=sk-your-actual-api-key-here
```

Get your API key from: "https://platform.openai.com/api-keys"

### 3. Optional Configuration

**LLM Settings:**

```env
OPENAI_MODEL=gpt-3.5-turbo          # or gpt-4
LLM_DEMO_MODE=true                  # Use small dataset for testing
LLM_DEMO_SIZE=50                    # Number of items in demo mode
LLM_BATCH_SIZE=10                   # Items to process per batch
```

**Cost Management:**

```env
MAX_ESTIMATED_COST=5.00             # Stop if cost exceeds this
COST_WARNING_THRESHOLD=2.00         # Warn when cost reaches this
```

**Rate Limiting:**

```env
RATE_LIMIT_DELAY=1.0                # Seconds between API calls
MAX_RETRIES=3                       # Retry failed API calls
```

## Configuration Options

### Core Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | - | Your OpenAI API key (required) |
| `OPENAI_MODEL` | `gpt-3.5-turbo` | GPT model to use |
| `OPENAI_MAX_TOKENS` | `300` | Max tokens per API call |
| `OPENAI_TEMPERATURE` | `0.3` | Model temperature (0-1) |

### LLM Curation Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_DEMO_MODE` | `true` | Limit dataset size for testing |
| `LLM_DEMO_SIZE` | `50` | Number of items in demo mode |
| `LLM_BATCH_SIZE` | `10` | Items processed per batch |
| `LLM_CACHE_ENABLED` | `true` | Cache LLM analyses |

### Cost & Rate Management

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_ESTIMATED_COST` | `5.00` | Maximum cost limit ($) |
| `COST_WARNING_THRESHOLD` | `2.00` | Cost warning threshold ($) |
| `RATE_LIMIT_DELAY` | `1.0` | Delay between API calls (seconds) |
| `MAX_RETRIES` | `3` | API call retry attempts |

### File Paths

| Variable | Default | Description |
|----------|---------|-------------|
| `OUTPUT_DIR` | `output/ai_curation` | Output directory |
| `CACHE_DIR` | `output/ai_curation` | Cache directory |
| `LOG_LEVEL` | `INFO` | Logging level |

## Using the Configuration

### In Python Code

```python
from shelfscale.config_manager import get_config

# Get configuration instance
config = get_config()

# Use configuration values
api_key = config.OPENAI_API_KEY
model = config.OPENAI_MODEL
demo_mode = config.LLM_DEMO_MODE

# Get grouped settings
openai_config = config.get_openai_config()
llm_settings = config.get_llm_settings()
```

### Configuration Validation

```python
from shelfscale.config_manager import get_config

config = get_config()

# Check if OpenAI is properly configured
if config.validate_openai_config():
    print("✅ OpenAI API key is configured")
else:
    print("❌ OpenAI API key missing or invalid")

# Print configuration summary
config.print_config_summary()
```

## Environment Files Explained

### `.env` (Your actual configuration)

- Contains your real API keys and settings
- **Never commit this to git**
- Listed in `.gitignore`

### `config_template.env` (Template)

- Template with example values
- Safe to commit to git
- Copy this to create your `.env` file

## Security Best Practices

1. **Never commit your `.env` file** to version control
2. **Use environment variables** in production
3. **Rotate API keys** regularly
4. **Set cost limits** to prevent unexpected charges
5. **Use demo mode** when testing

## Cost Management

The system includes several cost protection features:

- **Demo Mode**: Limits dataset size for testing
- **Cost Tracking**: Estimates API costs in real-time
- **Cost Limits**: Stops processing if costs exceed limits
- **Batch Processing**: Controls API call frequency
- **Caching**: Avoids re-analyzing the same items

## Troubleshooting

### Missing API Key

```md
❌ OpenAI API key not configured. Please set OPENAI_API_KEY in your .env file.
```

**Solution**: Add your API key to the `.env` file

### Import Error for dotenv

```bash
python-dotenv not installed, using system environment variables only
```

**Solution**: Install python-dotenv:

```bash
pip install python-dotenv>=0.19.0
```

### API Rate Limits

```md
API error 429, retrying...
```

**Solution**: Increase `RATE_LIMIT_DELAY` in your `.env` file

### Cost Limit Exceeded

```md
🛑 Cost limit exceeded: $5.50 > $5.00
```

**Solution**: Increase `MAX_ESTIMATED_COST` or use demo mode

## Example Complete .env File

```env
# OpenAI Configuration
OPENAI_API_KEY=sk-your-actual-api-key-here
OPENAI_MODEL=gpt-3.5-turbo
OPENAI_MAX_TOKENS=300
OPENAI_TEMPERATURE=0.3

# LLM Curation Settings
LLM_BATCH_SIZE=10
LLM_DEMO_MODE=true
LLM_DEMO_SIZE=50
LLM_CACHE_ENABLED=true

# Cost Management
MAX_ESTIMATED_COST=5.00
COST_WARNING_THRESHOLD=2.00

# Rate Limiting
RATE_LIMIT_DELAY=1.0
MAX_RETRIES=3

# File Paths
OUTPUT_DIR=output/ai_curation
CACHE_DIR=output/ai_curation
LOG_LEVEL=INFO
```

## Support

If you encounter issues:

1. Run the setup script: `python setup_environment.py`
2. Check the configuration: `config.print_config_summary()`
3. Verify dependencies: Install missing packages from `requirements_llm.txt`
4. Review this README for common solutions.
