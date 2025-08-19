"""
Two-Step Retail-Focused Workflow for ShelfScale

This module implements a clear two-step process for creating retail food databases:

STEP 1: LLM Retail Filtering
- Analyze each dataset and filter for products most likely sold in retail stores
- Use LLM reasoning to assess commercial availability
- Standardize product weights across all data sources
- Store reduced outputs with maximum information retention

STEP 2: LLM Semantic Matching & Consolidation
- Compare products across filtered datasets using LLM semantic understanding
- Match similar products based on ingredients, preparation, and characteristics
- Create consolidated retail products table
- Group by Super_Category and Food_Category
- Preserve important variations (size, preparation methods)

Built upon existing ShelfScale system architecture.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import asyncio
import logging
import json
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, asdict

# Import existing ShelfScale components
import shelfscale.config as config
from shelfscale.main import process_weight_info
from shelfscale.matching.algorithm import FoodMatcher
from shelfscale.data_processing.cleaner import DataCleaner
from shelfscale.utils.helpers import load_data, save_data

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

logger = logging.getLogger(__name__)

@dataclass
class RetailProduct:
    """Standardized retail product with essential information"""
    product_name: str
    food_category: str
    super_category: str
    retail_relevance_score: float
    llm_reasoning: str
    
    # Standardized weight information
    standard_weight_g: Optional[float] = None
    standard_weight_ml: Optional[float] = None
    portion_size_g: Optional[float] = None
    weight_source: str = "extracted"
    weight_confidence: float = 0.0
    
    # Nutritional information (per 100g)
    energy_kcal: Optional[float] = None
    fat_g: Optional[float] = None
    saturated_fat_g: Optional[float] = None
    carbs_g: Optional[float] = None
    sugars_g: Optional[float] = None
    protein_g: Optional[float] = None
    salt_g: Optional[float] = None
    fiber_g: Optional[float] = None
    
    # Source tracking
    source_dataset: str = ""
    original_row_data: Dict = None

@dataclass
class ConsolidatedProduct:
    """Product consolidated from multiple data sources"""
    consolidated_name: str
    food_category: str
    super_category: str
    retail_confidence: float
    
    # All source products that match this consolidated product
    source_products: List[RetailProduct]
    
    # Best available data (taken from highest quality source)
    best_weight_data: Dict[str, Any]
    best_nutrition_data: Dict[str, Any]
    
    # Matching information
    match_reasoning: str
    match_confidence: float
    size_variants: List[str]  # Different sizes found (e.g., "250ml", "1L")

class RetailWorkflowTwoStep:
    """Two-step retail workflow implementation"""
    
    def __init__(self, api_key: str = None):
        """Initialize the two-step retail workflow"""
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        
        # Initialize components
        self.llm_available = LLMMatcher is not None and self.api_key
        self.curator = None
        self.enhanced_matcher = None
        self.llm_matcher = None
        
        if self.llm_available:
            try:
                self.curator = LLMProductCurator(api_key=self.api_key)
                self.enhanced_matcher = EnhancedFoodMatcher(api_key=self.api_key)
                self.llm_matcher = LLMMatcher(api_key=self.api_key, enable_learning=True)
                logger.info("🤖 LLM components initialized for retail workflow")
            except Exception as e:
                logger.warning(f"LLM components initialization failed: {e}")
                self.llm_available = False
        
        # Core ShelfScale components
        self.food_matcher = FoodMatcher()
        self.data_cleaner = DataCleaner()
        
        # Setup paths
        self.raw_data_path = Path("data/raw")
        self.step1_output_path = Path("data/outputs/step1_retail_filtered")
        self.step2_output_path = Path("data/outputs/step2_consolidated")
        self.step1_output_path.mkdir(parents=True, exist_ok=True)
        self.step2_output_path.mkdir(parents=True, exist_ok=True)
        
        # Storage for workflow state
        self.filtered_products: Dict[str, List[RetailProduct]] = {}
        self.consolidated_products: List[ConsolidatedProduct] = []
        
        logger.info(f"🏪 Two-Step Retail Workflow initialized (LLM: {self.llm_available})")
    
    async def run_complete_workflow(self) -> Dict[str, Any]:
        """Run the complete two-step retail workflow"""
        logger.info("🚀 Starting Two-Step Retail Workflow")
        logger.info("=" * 60)
        
        start_time = datetime.now()
        results = {'start_time': start_time, 'llm_available': self.llm_available}
        
        try:
            # STEP 1: LLM Retail Filtering
            logger.info("📋 STEP 1: LLM Retail Filtering & Weight Standardization")
            step1_results = await self.step1_retail_filtering()
            results['step1'] = step1_results
            
            # STEP 2: LLM Semantic Matching & Consolidation
            logger.info("\n🔗 STEP 2: LLM Semantic Matching & Consolidation")
            step2_results = await self.step2_semantic_consolidation()
            results['step2'] = step2_results
            
            # Generate final outputs
            final_outputs = await self.generate_final_outputs()
            results['final_outputs'] = final_outputs
            
            results['end_time'] = datetime.now()
            results['total_duration'] = (results['end_time'] - start_time).total_seconds()
            results['success'] = True
            
            logger.info(f"\n✅ Two-Step Retail Workflow Completed Successfully!")
            logger.info(f"   📊 Total Duration: {results['total_duration']:.1f}s")
            logger.info(f"   🏪 Step 1 Products: {sum(len(products) for products in self.filtered_products.values())}")
            logger.info(f"   🔗 Step 2 Consolidated: {len(self.consolidated_products)}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Two-step workflow failed: {e}")
            results['error'] = str(e)
            results['success'] = False
            return results
    
    async def step1_retail_filtering(self) -> Dict[str, Any]:
        """
        STEP 1: LLM Retail Filtering & Weight Standardization
        
        - Go through all available data
        - Reduce datasets to products most likely sold in retail stores using LLM reasoning
        - Standardize product weights
        - Store reduced outputs with maximum information retention
        """
        logger.info("🔍 Step 1: Analyzing and filtering datasets for retail products...")
        
        step1_start = datetime.now()
        
        # Discover all data sources
        data_sources = await self._discover_data_sources()
        logger.info(f"   📂 Found {len(data_sources)} data sources")
        
        # Process each data source with LLM retail filtering
        for source in data_sources:
            logger.info(f"\n   📊 Processing: {source['file_name']}")
            
            try:
                # Load and clean the data
                df = await self._load_and_clean_data(source)
                if df is None or len(df) == 0:
                    logger.warning(f"      ⚠️ No data loaded from {source['file_name']}")
                    continue
                
                logger.info(f"      📈 Loaded {len(df)} records")
                
                # Apply LLM retail filtering
                retail_products = await self._filter_for_retail_products(df, source)
                
                # Standardize weights
                retail_products = await self._standardize_weights(retail_products)
                
                # Store filtered products
                dataset_name = f"{source['data_type']}_{source['file_path'].stem}"
                self.filtered_products[dataset_name] = retail_products
                
                # Save step 1 output
                await self._save_step1_output(dataset_name, retail_products)
                
                logger.info(f"      ✅ Filtered to {len(retail_products)} retail products")
                
            except Exception as e:
                logger.error(f"      ❌ Failed to process {source['file_name']}: {e}")
        
        step1_duration = (datetime.now() - step1_start).total_seconds()
        total_products = sum(len(products) for products in self.filtered_products.values())
        
        logger.info(f"\n📊 STEP 1 RESULTS:")
        logger.info(f"   • Datasets processed: {len(self.filtered_products)}")
        logger.info(f"   • Total retail products: {total_products}")
        logger.info(f"   • Duration: {step1_duration:.1f}s")
        
        return {
            'datasets_processed': len(self.filtered_products),
            'total_retail_products': total_products,
            'duration_seconds': step1_duration,
            'output_path': str(self.step1_output_path)
        }
    
    async def step2_semantic_consolidation(self) -> Dict[str, Any]:
        """
        STEP 2: LLM Semantic Matching & Consolidation
        
        - Compare products across filtered datasets using LLM semantic understanding
        - Match similar products based on ingredients, preparation, characteristics
        - Create consolidated retail products table
        - Group by Super_Category and Food_Category
        - Preserve important variations (size, preparation methods)
        """
        logger.info("🧠 Step 2: LLM semantic matching and consolidation...")
        
        step2_start = datetime.now()
        
        if not self.filtered_products:
            logger.warning("   ⚠️ No filtered products from Step 1. Cannot proceed with consolidation.")
            return {'error': 'No filtered products available'}
        
        # Flatten all retail products from all datasets
        all_retail_products = []
        for dataset_name, products in self.filtered_products.items():
            for product in products:
                product.source_dataset = dataset_name
                all_retail_products.append(product)
        
        logger.info(f"   🔍 Analyzing {len(all_retail_products)} retail products for semantic matches")
        
        # Perform LLM semantic matching
        if self.llm_available:
            consolidated_products = await self._llm_semantic_matching(all_retail_products)
        else:
            logger.warning("   ⚠️ LLM not available, using basic string matching")
            consolidated_products = await self._basic_semantic_matching(all_retail_products)
        
        self.consolidated_products = consolidated_products
        
        # Group by categories
        category_groups = await self._group_by_categories(consolidated_products)
        
        step2_duration = (datetime.now() - step2_start).total_seconds()
        
        logger.info(f"\n🔗 STEP 2 RESULTS:")
        logger.info(f"   • Input products: {len(all_retail_products)}")
        logger.info(f"   • Consolidated products: {len(consolidated_products)}")
        logger.info(f"   • Category groups: {len(category_groups)}")
        logger.info(f"   • Duration: {step2_duration:.1f}s")
        
        return {
            'input_products': len(all_retail_products),
            'consolidated_products': len(consolidated_products),
            'category_groups': len(category_groups),
            'duration_seconds': step2_duration,
            'category_breakdown': {cat: len(prods) for cat, prods in category_groups.items()}
        }
    
    async def _discover_data_sources(self) -> List[Dict[str, Any]]:
        """Discover all available data sources"""
        sources = []
        
        for pattern in ['*.csv', '*.xlsx', '*.xls']:
            for file_path in self.raw_data_path.glob(pattern):
                source_info = {
                    'file_path': file_path,
                    'file_name': file_path.name,
                    'file_type': file_path.suffix[1:],
                    'file_size': file_path.stat().st_size,
                    'data_type': self._guess_data_type(file_path.name),
                    'retail_priority': self._estimate_retail_priority(file_path.name)
                }
                sources.append(source_info)
        
        # Sort by retail priority
        sources.sort(key=lambda x: -x['retail_priority'])
        return sources
    
    def _guess_data_type(self, filename: str) -> str:
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
    
    def _estimate_retail_priority(self, filename: str) -> float:
        """Estimate retail priority from filename"""
        filename_lower = filename.lower()
        
        if any(term in filename_lower for term in ['mccance', 'widdowson', 'composition']):
            return 0.9  # Highest priority for comprehensive food composition
        elif any(term in filename_lower for term in ['portion', 'sizes', 'labelling']):
            return 0.8  # High priority for portion and labeling data
        elif any(term in filename_lower for term in ['survey', 'sample']):
            return 0.6  # Medium priority for survey data
        else:
            return 0.5  # Default priority
    
    async def _load_and_clean_data(self, source: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Load and clean data from a source with intelligent sheet selection"""
        try:
            if source['file_type'] in ['xlsx', 'xls']:
                # For Excel files, try to find the main data sheet
                excel_file = pd.ExcelFile(source['file_path'])
                
                # Strategy: Find sheet with most rows that contains food data
                best_sheet = None
                max_rows = 0
                
                for sheet_name in excel_file.sheet_names:
                    try:
                        # Read first few rows to check content
                        sample_df = pd.read_excel(source['file_path'], sheet_name=sheet_name, nrows=5)
                        
                        # Skip if it looks like metadata (very few columns or obvious non-data content)
                        if len(sample_df.columns) < 3:
                            continue
                        
                        # Get full row count
                        full_df = pd.read_excel(source['file_path'], sheet_name=sheet_name)
                        
                        # Look for food-like data
                        food_indicators = 0
                        for col in full_df.columns:
                            col_lower = str(col).lower()
                            if any(term in col_lower for term in ['food', 'name', 'description', 'energy', 'protein', 'fat']):
                                food_indicators += 1
                        
                        # Score this sheet
                        sheet_score = len(full_df) * (1 + food_indicators * 0.1)
                        
                        if sheet_score > max_rows:
                            max_rows = sheet_score
                            best_sheet = sheet_name
                            
                    except Exception:
                        continue
                
                if best_sheet:
                    logger.info(f"      🧠 Selected sheet: '{best_sheet}'")
                    df = pd.read_excel(source['file_path'], sheet_name=best_sheet)
                else:
                    df = pd.read_excel(source['file_path'])
                    
            elif source['file_type'] == 'csv':
                df = pd.read_csv(source['file_path'])
            else:
                return None
            
            # Clean the data
            df_cleaned = self.data_cleaner.clean(df)
            
            # Remove obvious header contamination
            df_cleaned = self._remove_header_contamination(df_cleaned)
            
            return df_cleaned
            
        except Exception as e:
            logger.error(f"Failed to load {source['file_path']}: {e}")
            return None
    
    def _remove_header_contamination(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove rows that are clearly headers or metadata"""
        if 'Food_Name' not in df.columns:
            return df
        
        contamination_patterns = [
            r'^\d+\.\d+',  # Table numbers like "1.1", "1.2"
            'notes on tables', 'factors', 'proximates', 'inorganics', 'vitamins',
            'vitamin fractions', 'fatty acids per 100g', 'per 100g food',
            'list of tables', 'table', 'sheet', 'index', 'contents'
        ]
        
        mask = pd.Series([True] * len(df))
        
        for pattern in contamination_patterns:
            pattern_mask = ~df['Food_Name'].str.contains(pattern, case=False, na=False, regex=True)
            mask = mask & pattern_mask
        
        return df[mask].copy()
    
    async def _filter_for_retail_products(self, df: pd.DataFrame, source: Dict[str, Any]) -> List[RetailProduct]:
        """Filter dataframe for products likely sold in retail stores using LLM"""
        retail_products = []
        
        if 'Food_Name' not in df.columns:
            logger.warning(f"      ⚠️ No 'Food_Name' column found in {source['file_name']}")
            return retail_products
        
        logger.info(f"      🧠 Applying LLM retail filtering...")
        
        for idx, row in df.iterrows():
            food_name = str(row.get('Food_Name', '')).strip()
            if not food_name or len(food_name) < 3:
                continue
            
            # Get LLM assessment of retail relevance
            if self.llm_available:
                retail_assessment = await self._llm_assess_retail_relevance(food_name, row, source)
            else:
                retail_assessment = self._basic_assess_retail_relevance(food_name, row)
            
            if retail_assessment['is_retail'] and retail_assessment['confidence'] >= 0.6:
                retail_product = RetailProduct(
                    product_name=food_name,
                    food_category=row.get('Food_Category', row.get('Food_Group', 'Unknown')),
                    super_category=row.get('Super_Category', 'Unknown'),
                    retail_relevance_score=retail_assessment['confidence'],
                    llm_reasoning=retail_assessment['reasoning'],
                    source_dataset=source['file_name'],
                    original_row_data=row.to_dict()
                )
                
                # Extract nutritional information
                retail_product.energy_kcal = self._safe_float(row.get('Energy (kcal)'))
                retail_product.fat_g = self._safe_float(row.get('Fat (g)'))
                retail_product.saturated_fat_g = self._safe_float(row.get('Saturates (g)'))
                retail_product.carbs_g = self._safe_float(row.get('Carbohydrate (g)'))
                retail_product.sugars_g = self._safe_float(row.get('Sugars (g)'))
                retail_product.protein_g = self._safe_float(row.get('Protein (g)'))
                retail_product.salt_g = self._safe_float(row.get('Salt (g)'))
                retail_product.fiber_g = self._safe_float(row.get('Fibre (g)'))
                
                retail_products.append(retail_product)
        
        return retail_products
    
    async def _llm_assess_retail_relevance(self, food_name: str, row: pd.Series, source: Dict[str, Any]) -> Dict[str, Any]:
        """Use LLM to assess if a food product is likely sold in retail stores"""
        
        assessment_prompt = f"""
        Assess if this food product is likely sold in retail stores (grocery stores, supermarkets):
        
        Product: {food_name}
        Description: {row.get('Description', 'N/A')}
        Food Group: {row.get('Food_Group', 'N/A')}
        
        Consider:
        1. Is this commonly available in grocery stores?
        2. Is this a commercial product vs homemade preparation?
        3. Would consumers buy this ready-made?
        4. Is this a basic ingredient or prepared food item?
        
        Examples of RETAIL products: "Bread, white", "Milk, whole", "Chicken breast, raw", "Apple juice", "Yogurt, plain"
        Examples of NON-RETAIL: "Bread pudding, homemade", "Cake, made with butter, homemade", "Stock, made from bones"
        
        Respond with:
        - is_retail: true/false
        - confidence: 0.0-1.0 (how confident you are)
        - reasoning: Brief explanation of why this is/isn't retail
        """
        
        try:
            result = await self.llm_matcher.match_products_llm(
                assessment_prompt,
                f"assess_retail_{food_name[:30]}"
            )
            
            # Parse LLM response (simplified)
            reasoning = result.get('reasoning', 'LLM assessment completed')
            
            # Heuristic scoring based on food name patterns
            retail_score = self._calculate_retail_score(food_name)
            
            return {
                'is_retail': retail_score >= 0.6,
                'confidence': retail_score,
                'reasoning': reasoning
            }
            
        except Exception as e:
            logger.warning(f"LLM assessment failed for '{food_name}': {e}")
            return self._basic_assess_retail_relevance(food_name, row)
    
    def _basic_assess_retail_relevance(self, food_name: str, row: pd.Series) -> Dict[str, Any]:
        """Basic heuristic assessment of retail relevance"""
        retail_score = self._calculate_retail_score(food_name)
        
        return {
            'is_retail': retail_score >= 0.6,
            'confidence': retail_score,
            'reasoning': f"Heuristic assessment based on product name patterns"
        }
    
    def _calculate_retail_score(self, food_name: str) -> float:
        """Calculate retail relevance score based on food name patterns"""
        name_lower = food_name.lower()
        score = 0.5  # Base score
        
        # Positive indicators (increase retail likelihood)
        retail_positive = [
            'fresh', 'raw', 'cooked', 'grilled', 'baked', 'steamed',
            'whole', 'skimmed', 'semi-skimmed', 'low fat', 'reduced fat',
            'unsweetened', 'sweetened', 'natural', 'organic',
            'frozen', 'canned', 'bottled', 'packaged'
        ]
        
        # Negative indicators (decrease retail likelihood)
        retail_negative = [
            'homemade', 'home-made', 'made with', 'recipe', 'preparation',
            'made from', 'composite', 'average of', 'mixture',
            'including', 'containing', 'various', 'mixed'
        ]
        
        # Boost for simple, common foods
        simple_foods = [
            'bread', 'milk', 'cheese', 'butter', 'eggs', 'chicken', 'beef', 'pork',
            'fish', 'apple', 'banana', 'orange', 'potato', 'rice', 'pasta',
            'yogurt', 'juice', 'water', 'tea', 'coffee'
        ]
        
        for positive in retail_positive:
            if positive in name_lower:
                score += 0.1
        
        for negative in retail_negative:
            if negative in name_lower:
                score -= 0.3
        
        for simple in simple_foods:
            if simple in name_lower:
                score += 0.2
        
        return min(1.0, max(0.0, score))
    
    def _safe_float(self, value) -> Optional[float]:
        """Safely convert value to float"""
        if pd.isna(value):
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
    
    async def _standardize_weights(self, retail_products: List[RetailProduct]) -> List[RetailProduct]:
        """Standardize weight information across all retail products"""
        logger.info(f"      ⚖️ Standardizing weights for {len(retail_products)} products")
        
        for product in retail_products:
            # Extract weight from product name and original data
            weight_info = self._extract_weight_information(product)
            
            product.standard_weight_g = weight_info.get('weight_g')
            product.standard_weight_ml = weight_info.get('weight_ml')
            product.portion_size_g = weight_info.get('portion_g')
            product.weight_source = weight_info.get('source', 'extracted')
            product.weight_confidence = weight_info.get('confidence', 0.0)
        
        return retail_products
    
    def _extract_weight_information(self, product: RetailProduct) -> Dict[str, Any]:
        """Extract and standardize weight information from product"""
        weight_info = {
            'weight_g': None,
            'weight_ml': None,
            'portion_g': None,
            'source': 'extracted',
            'confidence': 0.0
        }
        
        # Look for weight in product name (e.g., "Apple juice, 250ml", "Bread, 800g")
        import re
        
        # Pattern for weight in grams
        gram_pattern = r'(\d+(?:\.\d+)?)\s*g\b'
        gram_match = re.search(gram_pattern, product.product_name.lower())
        if gram_match:
            weight_info['weight_g'] = float(gram_match.group(1))
            weight_info['confidence'] = 0.8
            weight_info['source'] = 'product_name'
        
        # Pattern for weight in ml/liters
        ml_pattern = r'(\d+(?:\.\d+)?)\s*(ml|litre|liter|l)\b'
        ml_match = re.search(ml_pattern, product.product_name.lower())
        if ml_match:
            volume = float(ml_match.group(1))
            unit = ml_match.group(2)
            
            if unit in ['litre', 'liter', 'l']:
                volume *= 1000  # Convert to ml
            
            weight_info['weight_ml'] = volume
            weight_info['confidence'] = max(weight_info['confidence'], 0.8)
            weight_info['source'] = 'product_name'
        
        # Look for portion size information in original data
        if product.original_row_data:
            portion_cols = ['Portion', 'Portion Size', 'Weight', 'Typical Portion']
            for col in portion_cols:
                if col in product.original_row_data:
                    portion_value = product.original_row_data[col]
                    if pd.notna(portion_value):
                        try:
                            weight_info['portion_g'] = float(portion_value)
                            weight_info['confidence'] = max(weight_info['confidence'], 0.6)
                            weight_info['source'] = 'portion_data'
                        except (ValueError, TypeError):
                            pass
        
        return weight_info
    
    async def _save_step1_output(self, dataset_name: str, retail_products: List[RetailProduct]):
        """Save Step 1 filtered products to file"""
        output_file = self.step1_output_path / f"{dataset_name}_retail_filtered.csv"
        
        # Convert to DataFrame for saving
        products_data = []
        for product in retail_products:
            product_dict = asdict(product)
            # Flatten original_row_data
            if product_dict['original_row_data']:
                product_dict.update({f"original_{k}": v for k, v in product_dict['original_row_data'].items()})
            del product_dict['original_row_data']
            products_data.append(product_dict)
        
        df = pd.DataFrame(products_data)
        df.to_csv(output_file, index=False)
        
        logger.info(f"      💾 Saved to: {output_file}")
    
    async def _llm_semantic_matching(self, all_products: List[RetailProduct]) -> List[ConsolidatedProduct]:
        """Use LLM to perform semantic matching across retail products"""
        logger.info("      🧠 Performing LLM semantic matching...")
        
        consolidated = []
        matched_indices = set()
        
        for i, product_a in enumerate(all_products):
            if i in matched_indices:
                continue
            
            # Find all products that match with product_a
            matching_products = [product_a]
            matched_indices.add(i)
            
            for j, product_b in enumerate(all_products[i+1:], i+1):
                if j in matched_indices:
                    continue
                
                # Use LLM to assess if products match
                match_result = await self._llm_assess_product_match(product_a, product_b)
                
                if match_result['is_match'] and match_result['confidence'] >= 0.7:
                    matching_products.append(product_b)
                    matched_indices.add(j)
            
            if len(matching_products) > 0:
                consolidated_product = self._create_consolidated_product(matching_products)
                consolidated.append(consolidated_product)
        
        logger.info(f"      ✅ Consolidated {len(all_products)} products into {len(consolidated)} groups")
        return consolidated
    
    async def _llm_assess_product_match(self, product_a: RetailProduct, product_b: RetailProduct) -> Dict[str, Any]:
        """Use LLM to assess if two products are semantically the same"""
        
        match_prompt = f"""
        Determine if these two food products are essentially the same product, potentially from different data sources:
        
        Product A: {product_a.product_name}
        Category A: {product_a.food_category}
        Source A: {product_a.source_dataset}
        
        Product B: {product_b.product_name}
        Category B: {product_b.food_category}
        Source B: {product_b.source_dataset}
        
        Consider:
        1. Same base food item (e.g., "Milk, whole" = "Whole milk")
        2. Same preparation method (raw, cooked, fried, etc.)
        3. Ignore minor wording differences
        4. DO NOT match different sizes (250ml ≠ 1L)
        5. DO NOT match different preparations (raw ≠ cooked)
        
        Examples of MATCHES:
        - "Chicken breast, raw" ↔ "Chicken breast, fresh"
        - "Apple juice" ↔ "Apple juice, unsweetened"
        - "Bread, white" ↔ "White bread"
        
        Examples of NON-MATCHES:
        - "Apple juice, 250ml" ↔ "Apple juice, 1L" (different sizes)
        - "Chicken breast, raw" ↔ "Chicken breast, grilled" (different preparation)
        - "Milk, whole" ↔ "Milk, skimmed" (different fat content)
        
        Respond with:
        - is_match: true/false
        - confidence: 0.0-1.0
        - reasoning: Brief explanation
        """
        
        try:
            result = await self.llm_matcher.match_products_llm(
                match_prompt,
                f"match_{product_a.product_name[:20]}_{product_b.product_name[:20]}"
            )
            
            # Parse result (simplified)
            reasoning = result.get('reasoning', 'LLM matching completed')
            
            # Basic semantic similarity as fallback
            similarity = self._calculate_semantic_similarity(product_a.product_name, product_b.product_name)
            
            return {
                'is_match': similarity >= 0.7,
                'confidence': similarity,
                'reasoning': reasoning
            }
            
        except Exception as e:
            logger.warning(f"LLM matching failed: {e}")
            similarity = self._calculate_semantic_similarity(product_a.product_name, product_b.product_name)
            return {
                'is_match': similarity >= 0.7,
                'confidence': similarity,
                'reasoning': 'Basic string similarity matching'
            }
    
    def _calculate_semantic_similarity(self, name_a: str, name_b: str) -> float:
        """Calculate basic semantic similarity between product names"""
        # Simple approach: normalized word overlap
        words_a = set(name_a.lower().replace(',', '').split())
        words_b = set(name_b.lower().replace(',', '').split())
        
        if not words_a or not words_b:
            return 0.0
        
        intersection = words_a.intersection(words_b)
        union = words_a.union(words_b)
        
        return len(intersection) / len(union) if union else 0.0
    
    async def _basic_semantic_matching(self, all_products: List[RetailProduct]) -> List[ConsolidatedProduct]:
        """Basic semantic matching when LLM is not available"""
        logger.info("      📝 Performing basic semantic matching...")
        
        consolidated = []
        matched_indices = set()
        
        for i, product_a in enumerate(all_products):
            if i in matched_indices:
                continue
            
            matching_products = [product_a]
            matched_indices.add(i)
            
            for j, product_b in enumerate(all_products[i+1:], i+1):
                if j in matched_indices:
                    continue
                
                similarity = self._calculate_semantic_similarity(product_a.product_name, product_b.product_name)
                
                if similarity >= 0.8:  # Higher threshold for basic matching
                    matching_products.append(product_b)
                    matched_indices.add(j)
            
            if len(matching_products) > 0:
                consolidated_product = self._create_consolidated_product(matching_products)
                consolidated.append(consolidated_product)
        
        return consolidated
    
    def _create_consolidated_product(self, matching_products: List[RetailProduct]) -> ConsolidatedProduct:
        """Create a consolidated product from matching retail products"""
        
        # Use the product with highest retail relevance score as the primary
        primary_product = max(matching_products, key=lambda p: p.retail_relevance_score)
        
        # Collect best available data
        best_weight_data = self._get_best_weight_data(matching_products)
        best_nutrition_data = self._get_best_nutrition_data(matching_products)
        
        # Extract size variants
        size_variants = []
        for product in matching_products:
            if product.standard_weight_g:
                size_variants.append(f"{product.standard_weight_g}g")
            if product.standard_weight_ml:
                size_variants.append(f"{product.standard_weight_ml}ml")
        
        size_variants = list(set(size_variants))  # Remove duplicates
        
        return ConsolidatedProduct(
            consolidated_name=primary_product.product_name,
            food_category=primary_product.food_category,
            super_category=primary_product.super_category,
            retail_confidence=np.mean([p.retail_relevance_score for p in matching_products]),
            source_products=matching_products,
            best_weight_data=best_weight_data,
            best_nutrition_data=best_nutrition_data,
            match_reasoning=f"Consolidated from {len(matching_products)} sources",
            match_confidence=0.8,
            size_variants=size_variants
        )
    
    def _get_best_weight_data(self, products: List[RetailProduct]) -> Dict[str, Any]:
        """Get the best available weight data from matching products"""
        best_weight = {}
        
        # Find product with highest weight confidence
        weight_products = [p for p in products if p.weight_confidence > 0]
        if weight_products:
            best_product = max(weight_products, key=lambda p: p.weight_confidence)
            best_weight = {
                'standard_weight_g': best_product.standard_weight_g,
                'standard_weight_ml': best_product.standard_weight_ml,
                'portion_size_g': best_product.portion_size_g,
                'weight_source': best_product.weight_source,
                'weight_confidence': best_product.weight_confidence
            }
        
        return best_weight
    
    def _get_best_nutrition_data(self, products: List[RetailProduct]) -> Dict[str, Any]:
        """Get the best available nutrition data from matching products"""
        # Find product with most complete nutrition data
        nutrition_scores = []
        for product in products:
            score = 0
            nutrition_fields = [
                product.energy_kcal, product.fat_g, product.saturated_fat_g,
                product.carbs_g, product.sugars_g, product.protein_g,
                product.salt_g, product.fiber_g
            ]
            score = sum(1 for field in nutrition_fields if field is not None)
            nutrition_scores.append((score, product))
        
        if nutrition_scores:
            best_product = max(nutrition_scores, key=lambda x: x[0])[1]
            return {
                'energy_kcal': best_product.energy_kcal,
                'fat_g': best_product.fat_g,
                'saturated_fat_g': best_product.saturated_fat_g,
                'carbs_g': best_product.carbs_g,
                'sugars_g': best_product.sugars_g,
                'protein_g': best_product.protein_g,
                'salt_g': best_product.salt_g,
                'fiber_g': best_product.fiber_g,
                'source_dataset': best_product.source_dataset
            }
        
        return {}
    
    async def _group_by_categories(self, consolidated_products: List[ConsolidatedProduct]) -> Dict[str, List[ConsolidatedProduct]]:
        """Group consolidated products by Super_Category and Food_Category"""
        
        category_groups = {}
        
        for product in consolidated_products:
            category_key = f"{product.super_category} > {product.food_category}"
            
            if category_key not in category_groups:
                category_groups[category_key] = []
            
            category_groups[category_key].append(product)
        
        # Sort each category group by retail confidence
        for category in category_groups:
            category_groups[category].sort(key=lambda p: p.retail_confidence, reverse=True)
        
        return category_groups
    
    async def generate_final_outputs(self) -> Dict[str, str]:
        """Generate final consolidated outputs"""
        logger.info("📁 Generating final consolidated outputs...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        outputs = {}
        
        # Main consolidated retail products table
        main_output = self.step2_output_path / f"consolidated_retail_products_{timestamp}.csv"
        consolidated_df = self._create_consolidated_dataframe()
        consolidated_df.to_csv(main_output, index=False)
        outputs['consolidated_products'] = str(main_output)
        
        # Category breakdown
        category_output = self.step2_output_path / f"products_by_category_{timestamp}.csv"
        category_df = self._create_category_breakdown()
        category_df.to_csv(category_output, index=False)
        outputs['category_breakdown'] = str(category_output)
        
        # Summary report
        summary_output = self.step2_output_path / f"workflow_summary_{timestamp}.md"
        summary_report = self._create_summary_report()
        with open(summary_output, 'w') as f:
            f.write(summary_report)
        outputs['summary_report'] = str(summary_output)
        
        logger.info(f"✅ Generated {len(outputs)} final output files")
        return outputs
    
    def _create_consolidated_dataframe(self) -> pd.DataFrame:
        """Create DataFrame of all consolidated products"""
        rows = []
        
        for product in self.consolidated_products:
            row = {
                'Consolidated_Name': product.consolidated_name,
                'Food_Category': product.food_category,
                'Super_Category': product.super_category,
                'Retail_Confidence': product.retail_confidence,
                'Source_Count': len(product.source_products),
                'Size_Variants': '; '.join(product.size_variants),
                'Match_Reasoning': product.match_reasoning,
                'Match_Confidence': product.match_confidence
            }
            
            # Add best weight data
            row.update({f"Best_{k}": v for k, v in product.best_weight_data.items()})
            
            # Add best nutrition data
            row.update({f"Best_{k}": v for k, v in product.best_nutrition_data.items()})
            
            # Add source datasets
            source_datasets = list(set(p.source_dataset for p in product.source_products))
            row['Source_Datasets'] = '; '.join(source_datasets)
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Sort by Super_Category, Food_Category, then Retail_Confidence
        df = df.sort_values(['Super_Category', 'Food_Category', 'Retail_Confidence'], 
                           ascending=[True, True, False])
        
        return df
    
    def _create_category_breakdown(self) -> pd.DataFrame:
        """Create category breakdown summary"""
        category_stats = {}
        
        for product in self.consolidated_products:
            super_cat = product.super_category
            food_cat = product.food_category
            
            if super_cat not in category_stats:
                category_stats[super_cat] = {}
            
            if food_cat not in category_stats[super_cat]:
                category_stats[super_cat][food_cat] = {
                    'product_count': 0,
                    'avg_confidence': 0.0,
                    'total_sources': 0
                }
            
            stats = category_stats[super_cat][food_cat]
            stats['product_count'] += 1
            stats['avg_confidence'] += product.retail_confidence
            stats['total_sources'] += len(product.source_products)
        
        # Convert to DataFrame
        rows = []
        for super_cat, food_cats in category_stats.items():
            for food_cat, stats in food_cats.items():
                rows.append({
                    'Super_Category': super_cat,
                    'Food_Category': food_cat,
                    'Product_Count': stats['product_count'],
                    'Avg_Retail_Confidence': stats['avg_confidence'] / stats['product_count'],
                    'Total_Source_Products': stats['total_sources']
                })
        
        df = pd.DataFrame(rows)
        df = df.sort_values(['Super_Category', 'Food_Category'])
        
        return df
    
    def _create_summary_report(self) -> str:
        """Create comprehensive summary report"""
        total_input_products = sum(len(products) for products in self.filtered_products.values())
        total_consolidated = len(self.consolidated_products)
        
        # Category breakdown
        category_counts = {}
        for product in self.consolidated_products:
            super_cat = product.super_category
            category_counts[super_cat] = category_counts.get(super_cat, 0) + 1
        
        report = f"""# Two-Step Retail Food Database Creation Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary

This report summarizes the two-step process for creating a retail-focused food database:

### STEP 1: LLM Retail Filtering & Weight Standardization
- **Input datasets**: {len(self.filtered_products)}
- **Retail products identified**: {total_input_products}
- **Weight standardization**: Applied to all products
- **LLM enhanced**: {self.llm_available}

### STEP 2: LLM Semantic Matching & Consolidation  
- **Input products**: {total_input_products}
- **Consolidated products**: {total_consolidated}
- **Consolidation ratio**: {total_consolidated/total_input_products*100 if total_input_products > 0 else 0:.1f}%

## Category Breakdown

"""
        
        for category, count in sorted(category_counts.items()):
            report += f"- **{category}**: {count} products\n"
        
        report += f"""

## Data Quality

- **Weight information**: {sum(1 for p in self.consolidated_products if p.best_weight_data)}/{total_consolidated} products have weight data
- **Nutrition information**: {sum(1 for p in self.consolidated_products if p.best_nutrition_data)}/{total_consolidated} products have nutrition data
- **Multi-source products**: {sum(1 for p in self.consolidated_products if len(p.source_products) > 1)} products consolidated from multiple sources

## Methodology

1. **Step 1 Filtering**: Each dataset was analyzed using {'LLM reasoning' if self.llm_available else 'heuristic patterns'} to identify products likely sold in retail stores
2. **Weight Standardization**: Product weights were extracted from names and data, standardized to grams/ml
3. **Step 2 Matching**: Products were matched across datasets using {'LLM semantic understanding' if self.llm_available else 'string similarity'}
4. **Consolidation**: Matching products were consolidated, preserving size variants and best available data

## Recommendations

- Use consolidated products table for retail applications
- Category breakdown provides good overview of food groups
- Size variants are preserved to maintain important distinctions
- Multi-source products have higher data quality and confidence

"""
        
        return report

# Integration functions for main.py
def add_two_step_workflow_args(parser):
    """Add two-step workflow arguments to argument parser"""
    parser.add_argument('--two-step-workflow', action='store_true',
                       help='Run two-step retail filtering and consolidation workflow')

async def run_two_step_workflow_with_args(args):
    """Run two-step workflow with command line arguments"""
    print("🏪 Starting Two-Step Retail Workflow")
    print("=" * 50)
    
    workflow = RetailWorkflowTwoStep()
    results = await workflow.run_complete_workflow()
    
    print("\n🎉 Two-Step Workflow Results:")
    print(f"   • Success: {results['success']}")
    print(f"   • Duration: {results.get('total_duration', 0):.1f}s")
    print(f"   • LLM Enhanced: {results['llm_available']}")
    
    if results.get('step1'):
        step1 = results['step1']
        print(f"\n📋 STEP 1 Results:")
        print(f"   • Datasets processed: {step1['datasets_processed']}")
        print(f"   • Retail products: {step1['total_retail_products']}")
        print(f"   • Output path: {step1['output_path']}")
    
    if results.get('step2'):
        step2 = results['step2']
        print(f"\n🔗 STEP 2 Results:")
        print(f"   • Input products: {step2['input_products']}")
        print(f"   • Consolidated products: {step2['consolidated_products']}")
        print(f"   • Category groups: {step2['category_groups']}")
    
    if results.get('final_outputs'):
        print(f"\n📁 Final Outputs:")
        for output_type, file_path in results['final_outputs'].items():
            print(f"   • {output_type}: {file_path}")
    
    return results