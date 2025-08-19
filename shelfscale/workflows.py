"""
ShelfScale LLM-Enhanced Retail Workflows

This module extends the existing ShelfScale system with LLM-driven workflows
for creating retail-optimized food databases from raw data sources.

Integrates with:
- shelfscale.main for core processing
- shelfscale.matching for LLM-enhanced matching
- tools.curation for retail product curation
- tools.matching for cross-dataset matching
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import asyncio
import logging
import argparse
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict

# Import existing ShelfScale components
import shelfscale.config as config
from shelfscale.main import process_weight_info
from shelfscale.matching.algorithm import FoodMatcher
from shelfscale.data_processing.cleaner import DataCleaner
from shelfscale.utils.helpers import load_data, save_data
from shelfscale.utils.learning import create_weight_predictions

# Import enhanced retail tools
try:
    import sys
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    from tools.curation.llm_enhanced_product_curation import LLMProductCurator
    from tools.matching.enhanced_food_matching import EnhancedFoodMatcher
    from shelfscale.ml.llm_matcher import LLMMatcher
except ImportError as e:
    logging.warning(f"Enhanced retail tools not available: {e}")
    LLMProductCurator = None
    EnhancedFoodMatcher = None
    LLMMatcher = None

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class WorkflowConfig:
    """Configuration for retail workflow"""
    raw_data_path: str = "data/raw"
    output_path: str = "data/outputs/workflow"
    enable_llm_curation: bool = True
    enable_cross_dataset_matching: bool = True
    target_retail_products: int = 500
    min_retail_relevance: float = 0.6
    
    # Learning parameters for iterative improvement
    skip_rows: int = 0
    sheet_selection_strategy: str = 'default'  # 'default', 'skip_first', 'largest_sheet'
    data_filtering_strictness: str = 'medium'  # 'low', 'medium', 'high'
    header_detection_enabled: bool = True
    
class RetailWorkflowOrchestrator:
    """
    LLM-Enhanced Retail Workflow Orchestrator
    
    Extends existing ShelfScale functionality with retail-focused processing:
    1. Raw data analysis and prioritization
    2. LLM-driven retail relevance assessment  
    3. Cross-dataset matching with size preservation
    4. Comprehensive database consolidation
    5. Quality validation and optimization
    """
    
    def __init__(self, config: WorkflowConfig = None, api_key: str = None):
        """Initialize the retail workflow orchestrator"""
        self.config = config or WorkflowConfig()
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        
        # Initialize components if available
        self.llm_available = LLMMatcher is not None and self.api_key
        self.curator = None
        self.enhanced_matcher = None
        
        if self.llm_available:
            try:
                self.curator = LLMProductCurator(api_key=self.api_key)
                self.enhanced_matcher = EnhancedFoodMatcher(api_key=self.api_key)
                logger.info("🤖 LLM-enhanced components initialized")
            except Exception as e:
                logger.warning(f"LLM components initialization failed: {e}")
                self.llm_available = False
        
        # Core ShelfScale components (always available)
        self.food_matcher = FoodMatcher()
        self.data_cleaner = DataCleaner()
        
        # Setup paths
        self.raw_data_path = Path(self.config.raw_data_path)
        self.output_path = Path(self.config.output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🏪 Retail Workflow Orchestrator initialized (LLM: {self.llm_available})")
    
    async def run_retail_workflow(self) -> Dict[str, Any]:
        """
        Run the complete retail-focused workflow
        
        Returns:
            Dict containing workflow results and metrics
        """
        logger.info("🚀 Starting LLM-Enhanced Retail Workflow")
        
        workflow_start = datetime.now()
        results = {
            'start_time': workflow_start,
            'steps_completed': [],
            'outputs': {},
            'metrics': {},
            'llm_enhanced': self.llm_available
        }
        
        try:
            # Step 1: Discover and analyze raw data sources
            sources = await self.discover_data_sources()
            results['steps_completed'].append('data_discovery')
            results['metrics']['sources_found'] = len(sources)
            
            # Step 2: Load and process prioritized sources
            processed_data = await self.process_prioritized_sources(sources)
            results['steps_completed'].append('source_processing')
            results['metrics']['datasets_processed'] = len(processed_data)
            
            # Step 3: Apply retail curation if LLM available
            if self.llm_available and self.config.enable_llm_curation:
                curated_data = await self.apply_retail_curation(processed_data)
                results['steps_completed'].append('llm_curation')
                results['metrics']['products_curated'] = sum(len(df) for df in curated_data.values())
            else:
                curated_data = processed_data
                results['steps_completed'].append('basic_curation')
            
            # Step 4: Cross-dataset matching
            if len(curated_data) > 1 and self.config.enable_cross_dataset_matching:
                matched_data = await self.apply_cross_dataset_matching(curated_data)
                results['steps_completed'].append('cross_dataset_matching')
                results['metrics']['matches_found'] = len(matched_data) if isinstance(matched_data, pd.DataFrame) else 0
            else:
                # Single dataset - merge all available data
                matched_data = pd.concat(list(curated_data.values()), ignore_index=True)
                results['steps_completed'].append('data_concatenation')
            
            # Step 5: Apply core ShelfScale processing
            final_database = await self.apply_core_processing(matched_data)
            results['steps_completed'].append('core_processing')
            results['metrics']['final_products'] = len(final_database)
            
            # Step 6: Generate outputs
            output_files = await self.generate_outputs(final_database)
            results['steps_completed'].append('output_generation')
            results['outputs'] = output_files
            
            # Calculate final metrics
            results['end_time'] = datetime.now()
            results['total_duration'] = (results['end_time'] - workflow_start).total_seconds()
            results['success'] = True
            
            logger.info(f"✅ Retail workflow completed successfully in {results['total_duration']:.1f}s")
            return results
            
        except Exception as e:
            logger.error(f"❌ Retail workflow failed: {e}")
            results['error'] = str(e)
            results['success'] = False
            return results
    
    async def discover_data_sources(self) -> List[Dict[str, Any]]:
        """Discover and analyze data sources in raw data directory"""
        logger.info("📂 Discovering data sources...")
        
        sources = []
        
        # Find all data files
        for pattern in ['*.csv', '*.xlsx', '*.xls']:
            for file_path in self.raw_data_path.glob(pattern):
                source_info = {
                    'file_path': file_path,
                    'file_name': file_path.name,
                    'file_type': file_path.suffix[1:],
                    'file_size': file_path.stat().st_size,
                    'estimated_rows': 0,
                    'data_type': self.guess_data_type(file_path.name),
                    'retail_relevance': self.estimate_retail_relevance(file_path.name),
                    'processing_priority': 1
                }
                
                # Quick row estimation
                try:
                    if source_info['file_type'] == 'csv':
                        with open(file_path, 'r') as f:
                            source_info['estimated_rows'] = sum(1 for _ in f) - 1  # Exclude header
                    else:
                        # For Excel files, we'll get actual count during processing
                        source_info['estimated_rows'] = 'unknown'
                except:
                    source_info['estimated_rows'] = 'unknown'
                
                sources.append(source_info)
                logger.info(f"   📊 {file_path.name}: {source_info['data_type']} (relevance: {source_info['retail_relevance']:.2f})")
        
        # Sort by retail relevance and processing priority
        sources.sort(key=lambda x: (-x['retail_relevance'], x['processing_priority']))
        
        logger.info(f"✅ Discovered {len(sources)} data sources")
        return sources
    
    def guess_data_type(self, filename: str) -> str:
        """Guess data type from filename"""
        filename_lower = filename.lower()
        
        if any(term in filename_lower for term in ['mccance', 'widdowson', 'composition']):
            return 'food_composition'
        elif 'portion' in filename_lower or 'size' in filename_lower:
            return 'portion_sizes'
        elif 'survey' in filename_lower or 'sample' in filename_lower:
            return 'nutrition_survey'
        elif 'label' in filename_lower:
            return 'labelling_data'
        else:
            return 'unknown'
    
    def estimate_retail_relevance(self, filename: str) -> float:
        """Estimate retail relevance from filename"""
        filename_lower = filename.lower()
        
        # High relevance
        if any(term in filename_lower for term in ['mccance', 'widdowson', 'composition', 'nutrition']):
            return 0.9
        
        # Medium-high relevance
        if any(term in filename_lower for term in ['portion', 'sizes', 'labelling']):
            return 0.8
        
        # Medium relevance
        if any(term in filename_lower for term in ['survey', 'sample', 'fruit', 'vegetable']):
            return 0.6
        
        return 0.5
    
    async def process_prioritized_sources(self, sources: List[Dict[str, Any]]) -> Dict[str, pd.DataFrame]:
        """Process data sources in priority order"""
        logger.info("🔄 Processing prioritized data sources...")
        
        processed_data = {}
        
        for source in sources:
            if source['retail_relevance'] < self.config.min_retail_relevance:
                logger.info(f"   ⏭️ Skipping {source['file_name']} (low relevance: {source['retail_relevance']:.2f})")
                continue
            
            logger.info(f"   📊 Processing: {source['file_name']}")
            
            try:
                # Load the data using existing ShelfScale functionality with learning adjustments
                if source['file_type'] in ['xlsx', 'xls']:
                    # Apply learning for Excel files
                    read_kwargs = {}
                    if self.config.skip_rows > 0:
                        read_kwargs['skiprows'] = self.config.skip_rows
                        logger.info(f"      🧠 Learning: Skipping {self.config.skip_rows} rows")
                    
                    # Try to load the largest data sheet (not metadata)
                    if self.config.sheet_selection_strategy == 'largest_sheet':
                        excel_file = pd.ExcelFile(source['file_path'])
                        # Find sheet with most rows
                        largest_sheet = None
                        max_rows = 0
                        for sheet_name in excel_file.sheet_names:
                            try:
                                temp_df = pd.read_excel(source['file_path'], sheet_name=sheet_name, nrows=1)
                                sheet_rows = len(pd.read_excel(source['file_path'], sheet_name=sheet_name))
                                if sheet_rows > max_rows:
                                    max_rows = sheet_rows
                                    largest_sheet = sheet_name
                            except:
                                continue
                        
                        if largest_sheet:
                            read_kwargs['sheet_name'] = largest_sheet
                            logger.info(f"      🧠 Learning: Using largest sheet '{largest_sheet}' ({max_rows} rows)")
                    
                    df = pd.read_excel(source['file_path'], **read_kwargs)
                elif source['file_type'] == 'csv':
                    read_kwargs = {}
                    if self.config.skip_rows > 0:
                        read_kwargs['skiprows'] = self.config.skip_rows
                    df = pd.read_csv(source['file_path'], **read_kwargs)
                else:
                    logger.warning(f"      ⚠️ Unsupported file type: {source['file_type']}")
                    continue
                
                logger.info(f"      📈 Loaded {len(df)} records with {len(df.columns)} columns")
                
                # Apply learning-enhanced data cleaning
                df_cleaned = self.data_cleaner.clean(df)
                logger.info(f"      🧹 Cleaned data: {len(df_cleaned)} records")
                
                # Apply header contamination detection if enabled
                if self.config.header_detection_enabled:
                    original_count = len(df_cleaned)
                    df_cleaned = self._remove_header_contamination(df_cleaned)
                    if len(df_cleaned) < original_count:
                        removed = original_count - len(df_cleaned)
                        logger.info(f"      🧠 Learning: Removed {removed} header/metadata rows")
                
                # Store with descriptive name
                dataset_name = f"{source['data_type']}_{source['file_path'].stem}"
                processed_data[dataset_name] = df_cleaned
                
            except Exception as e:
                logger.error(f"      ❌ Failed to process {source['file_name']}: {e}")
        
        logger.info(f"✅ Processed {len(processed_data)} datasets")
        return processed_data
    
    async def apply_retail_curation(self, processed_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Apply LLM-driven retail curation to food composition data"""
        logger.info("🧠 Applying LLM-driven retail curation...")
        
        curated_data = {}
        
        for dataset_name, df in processed_data.items():
            if 'food_composition' in dataset_name and self.curator:
                logger.info(f"   🎯 Curating {dataset_name} for retail relevance...")
                
                try:
                    # Apply LLM curation
                    target_size = min(self.config.target_retail_products, len(df))
                    curated_df = await self.curator.curate_dataset(df, target_size=target_size)
                    
                    logger.info(f"      ✅ Curated from {len(df)} to {len(curated_df)} retail products")
                    curated_data[dataset_name] = curated_df
                    
                except Exception as e:
                    logger.warning(f"      ⚠️ LLM curation failed: {e}, using original data")
                    curated_data[dataset_name] = df
            else:
                # For non-food-composition data or when LLM unavailable, use as-is
                curated_data[dataset_name] = df
        
        logger.info("✅ Retail curation completed")
        return curated_data
    
    async def apply_cross_dataset_matching(self, curated_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Apply cross-dataset matching with size preservation"""
        logger.info("🔗 Applying cross-dataset matching...")
        
        if not self.enhanced_matcher:
            logger.info("   ⚠️ Enhanced matcher not available, using basic concatenation")
            return pd.concat(list(curated_data.values()), ignore_index=True)
        
        try:
            # Prepare temporary files for enhanced matching
            temp_files = {}
            for dataset_name, df in curated_data.items():
                temp_path = self.output_path / f"temp_{dataset_name}.csv"
                df.to_csv(temp_path, index=False)
                temp_files[dataset_name] = str(temp_path)
            
            # Apply enhanced matching
            await self.enhanced_matcher.load_datasets(temp_files)
            await self.enhanced_matcher.find_cross_dataset_matches()
            
            # Generate comprehensive database
            comprehensive_db = self.enhanced_matcher.create_comprehensive_database()
            
            logger.info(f"   ✅ Found {len(self.enhanced_matcher.matches)} matches")
            logger.info(f"   📊 Created comprehensive database with {len(comprehensive_db)} products")
            
            # Clean up temporary files
            for temp_path in temp_files.values():
                Path(temp_path).unlink(missing_ok=True)
            
            return comprehensive_db
            
        except Exception as e:
            logger.error(f"   ❌ Cross-dataset matching failed: {e}")
            logger.info("   🔄 Falling back to basic concatenation")
            return pd.concat(list(curated_data.values()), ignore_index=True)
    
    async def apply_core_processing(self, matched_data: pd.DataFrame) -> pd.DataFrame:
        """Apply core ShelfScale processing to the matched data"""
        logger.info("⚙️ Applying core ShelfScale processing...")
        
        # Apply weight processing using existing ShelfScale functionality
        weight_cols = self.detect_weight_columns(matched_data)
        if weight_cols:
            logger.info(f"   🏋️ Processing weights from columns: {weight_cols}")
            matched_data = process_weight_info(matched_data, weight_cols)
        
        # Apply nutrition scoring using existing ShelfScale functionality
        logger.info("   📊 Adding nutrition scores...")
        try:
            from shelfscale.main import apply_nutrition_scoring
            matched_data = apply_nutrition_scoring(matched_data, 'all')
        except Exception as e:
            logger.warning(f"   ⚠️ Nutrition scoring failed: {e}")
        
        # Add workflow metadata
        matched_data['workflow_processed'] = datetime.now().isoformat()
        matched_data['retail_optimized'] = True
        matched_data['llm_enhanced'] = self.llm_available
        
        logger.info(f"✅ Core processing completed: {len(matched_data)} products")
        return matched_data
    
    def detect_weight_columns(self, df: pd.DataFrame) -> List[str]:
        """Detect columns that likely contain weight information"""
        weight_cols = []
        for col in df.columns:
            col_lower = col.lower()
            if any(term in col_lower for term in ['weight', 'size', 'portion', 'pack', 'gram', 'kg', 'ml', 'litre']):
                weight_cols.append(col)
        return weight_cols
    
    async def generate_outputs(self, final_database: pd.DataFrame) -> Dict[str, str]:
        """Generate all output files and reports"""
        logger.info("📁 Generating outputs...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        outputs = {}
        
        # Main retail database
        main_output = self.output_path / f"retail_food_database_{timestamp}.csv"
        final_database.to_csv(main_output, index=False)
        outputs['main_database'] = str(main_output)
        
        # Quality report
        quality_report = self.generate_quality_report(final_database)
        report_file = self.output_path / f"quality_report_{timestamp}.md"
        with open(report_file, 'w') as f:
            f.write(quality_report)
        outputs['quality_report'] = str(report_file)
        
        # Summary statistics
        summary_stats = self.generate_summary_statistics(final_database)
        stats_file = self.output_path / f"summary_statistics_{timestamp}.json"
        with open(stats_file, 'w') as f:
            import json
            json.dump(summary_stats, f, indent=2, default=str)
        outputs['summary_statistics'] = str(stats_file)
        
        logger.info(f"✅ Generated {len(outputs)} output files")
        return outputs
    
    def _remove_header_contamination(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove rows that appear to be headers or metadata instead of food products"""
        if 'Food_Name' not in df.columns:
            return df
        
        # Patterns that indicate header/metadata contamination
        contamination_patterns = [
            'Notes on tables', 'Factors', 'Proximates', 'Inorganics', 'Vitamins',
            'Vitamin Fractions', 'fatty acids per 100g', 'Monounsaturated', 
            'Saturated fatty acids', 'per 100g food', 'List of tables',
            'Table', 'Sheet', 'Index', 'Contents', 'Introduction',
            'Methodology', 'References', 'Appendix'
        ]
        
        # Create mask to identify contaminated rows
        contamination_mask = pd.Series([False] * len(df))
        
        for pattern in contamination_patterns:
            mask = df['Food_Name'].str.contains(pattern, case=False, na=False)
            contamination_mask = contamination_mask | mask
        
        # Also remove rows where Food_Name starts with numbers followed by periods (like "1.1", "1.2")
        # which are often table of contents entries
        number_pattern_mask = df['Food_Name'].str.match(r'^\d+\.\d+', na=False)
        contamination_mask = contamination_mask | number_pattern_mask
        
        # Filter out contaminated rows
        clean_df = df[~contamination_mask].copy()
        
        return clean_df
    
    def generate_quality_report(self, df: pd.DataFrame) -> str:
        """Generate quality assessment report"""
        total_products = len(df)
        total_columns = len(df.columns)
        missing_percentage = (df.isnull().sum().sum() / (total_products * total_columns)) * 100
        
        # Analyze retail relevance if available
        retail_analysis = ""
        if 'retail_relevance_score' in df.columns:
            high_relevance = len(df[df['retail_relevance_score'] >= 0.8])
            retail_analysis = f"""
## Retail Relevance Analysis

- **High relevance products (≥0.8)**: {high_relevance} ({high_relevance/total_products*100:.1f}%)
- **Average relevance score**: {df['retail_relevance_score'].mean():.3f}
"""
        
        return f"""# Retail Food Database Quality Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Database Overview

- **Total Products**: {total_products:,}
- **Total Columns**: {total_columns}
- **Missing Data**: {missing_percentage:.2f}%
- **LLM Enhanced**: {self.llm_available}

{retail_analysis}

## Data Quality Metrics

- **Duplicate Records**: {df.duplicated().sum()}
- **Complete Records**: {len(df.dropna())} ({len(df.dropna())/total_products*100:.1f}%)

## Column Analysis

"""
    
    def generate_summary_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary statistics"""
        return {
            'total_products': len(df),
            'total_columns': len(df.columns),
            'missing_data_percentage': float((df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100),
            'llm_enhanced': self.llm_available,
            'workflow_timestamp': datetime.now().isoformat(),
            'columns': list(df.columns),
            'data_types': {col: str(dtype) for col, dtype in df.dtypes.items()}
        }

# Integration with existing main.py
def add_retail_workflow_args(parser: argparse.ArgumentParser):
    """Add retail workflow arguments to existing argument parser"""
    parser.add_argument('--retail-workflow', action='store_true',
                       help='Run LLM-enhanced retail workflow')
    parser.add_argument('--target-products', type=int, default=500,
                       help='Target number of retail products')
    parser.add_argument('--min-relevance', type=float, default=0.6,
                       help='Minimum retail relevance score')
    parser.add_argument('--disable-llm-curation', action='store_true',
                       help='Disable LLM-driven curation')
    parser.add_argument('--disable-cross-matching', action='store_true',
                       help='Disable cross-dataset matching')

async def run_retail_workflow(args):
    """Run the retail workflow with command line arguments"""
    config_obj = WorkflowConfig(
        target_retail_products=args.target_products,
        min_retail_relevance=args.min_relevance,
        enable_llm_curation=not args.disable_llm_curation,
        enable_cross_dataset_matching=not args.disable_cross_matching
    )
    
    orchestrator = RetailWorkflowOrchestrator(config=config_obj)
    results = await orchestrator.run_retail_workflow()
    
    print(f"\n🎉 Retail Workflow Results:")
    print(f"   • Success: {results['success']}")
    print(f"   • Duration: {results.get('total_duration', 0):.1f}s")
    print(f"   • Final products: {results.get('metrics', {}).get('final_products', 0)}")
    print(f"   • LLM enhanced: {results['llm_enhanced']}")
    
    if results['outputs']:
        print(f"   • Output files:")
        for output_type, file_path in results['outputs'].items():
            print(f"     - {output_type}: {file_path}")
    
    return results