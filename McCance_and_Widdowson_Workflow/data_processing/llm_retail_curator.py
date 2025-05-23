"""
LLM-Powered Retail Dataset Curator
Uses OpenAI's GPT models for intelligent food product analysis and curation
"""

import pandas as pd
import numpy as np
import json
import os
import logging
import time
from typing import Dict, List, Tuple, Optional, Any, Set
from collections import Counter, defaultdict
import asyncio
import aiohttp
from dataclasses import dataclass
import sys
sys.path.append('.')

from shelfscale.config_manager import get_config, ConfigManager

logger = logging.getLogger(__name__)


@dataclass
class ProductAnalysis:
    """Results of LLM analysis for a food product"""
    food_code: str
    food_name: str
    is_retail_product: bool
    retail_confidence: float
    representativeness_score: float
    product_category: str
    reasoning: str
    suggested_group: Optional[str] = None


class LLMRetailCurator:
    """
    LLM-powered retail dataset curator using OpenAI's GPT models
    """
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        # Use provided config or get global config
        self.config = config_manager or get_config()
        
        # Setup configuration
        self.api_key = self.config.OPENAI_API_KEY
        self.model = self.config.OPENAI_MODEL
        self.max_tokens = self.config.OPENAI_MAX_TOKENS
        self.temperature = self.config.OPENAI_TEMPERATURE
        
        # LLM settings
        self.batch_size = self.config.LLM_BATCH_SIZE
        self.rate_limit_delay = self.config.RATE_LIMIT_DELAY
        self.max_retries = self.config.MAX_RETRIES
        
        # Cost management
        self.max_cost = self.config.MAX_ESTIMATED_COST
        self.cost_warning = self.config.COST_WARNING_THRESHOLD
        
        # Initialize tracking
        self.analysis_cache = {}
        self.cost_tracker = {'total_tokens': 0, 'estimated_cost': 0.0}
        
        # File paths
        cache_dir = self.config.CACHE_DIR
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_file = os.path.join(cache_dir, "llm_analysis_cache.json")
        
        # Load or create analysis cache if enabled
        if self.config.LLM_CACHE_ENABLED:
            self._load_analysis_cache()
        
        # Validate configuration
        if not self.config.validate_openai_config():
            logger.warning("OpenAI API key not properly configured")
    
    async def curate_intelligent_dataset(self, df: pd.DataFrame, target_size: Optional[int] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Main method: Use LLM to intelligently curate the retail dataset
        """
        print("🧠 LLM-POWERED RETAIL DATASET CURATION STARTING")
        print("=" * 60)
        
        # Print configuration summary
        self.config.print_config_summary()
        
        # Check if we should use demo mode
        if self.config.LLM_DEMO_MODE and len(df) > self.config.LLM_DEMO_SIZE:
            print(f"⚠️  Demo mode: Using first {self.config.LLM_DEMO_SIZE} items (full dataset has {len(df)} items)")
            df = df.head(self.config.LLM_DEMO_SIZE)
        
        curation_report = {
            'original_size': len(df),
            'llm_model_used': self.model,
            'demo_mode': self.config.LLM_DEMO_MODE,
            'steps': [],
            'cost_analysis': {},
            'final_metrics': {},
            'configuration': {
                'batch_size': self.batch_size,
                'max_tokens': self.max_tokens,
                'temperature': self.temperature,
                'cache_enabled': self.config.LLM_CACHE_ENABLED
            }
        }
        
        # Validate API key before starting
        if not self.config.validate_openai_config():
            print("❌ OpenAI API key not configured. Please set OPENAI_API_KEY in your .env file.")
            return pd.DataFrame(), curation_report
        
        # Step 1: LLM Analysis of all products
        print("🤖 Running LLM analysis on all products...")
        product_analyses = await self._analyze_all_products(df)
        curation_report['steps'].append({
            'step': 'llm_product_analysis',
            'products_analyzed': len(product_analyses),
            'cache_hits': len(df) - len(product_analyses) if self.config.LLM_CACHE_ENABLED else 0
        })
        
        # Check cost limits
        if self.cost_tracker['estimated_cost'] > self.cost_warning:
            print(f"⚠️  Cost warning: Estimated cost ${self.cost_tracker['estimated_cost']:.3f} exceeds threshold ${self.cost_warning}")
        
        if self.cost_tracker['estimated_cost'] > self.max_cost:
            print(f"🛑 Cost limit exceeded: ${self.cost_tracker['estimated_cost']:.3f} > ${self.max_cost}")
            print("Stopping curation to prevent excessive costs.")
            return pd.DataFrame(), curation_report
        
        # Continue with remaining steps...
        # Step 2: Filter retail products using LLM decisions
        print("🛒 Filtering retail products based on LLM analysis...")
        retail_df = self._filter_retail_products(df, product_analyses)
        curation_report['steps'].append({
            'step': 'llm_retail_filtering',
            'input_size': len(df),
            'output_size': len(retail_df),
            'removal_rate': (len(df) - len(retail_df)) / len(df) * 100 if len(df) > 0 else 0
        })
        
        # Step 3: LLM-based intelligent grouping
        print("🔗 Creating intelligent product groups...")
        product_groups = await self._create_intelligent_groups(retail_df, product_analyses)
        curation_report['steps'].append({
            'step': 'llm_intelligent_grouping',
            'groups_created': len(product_groups),
            'avg_group_size': np.mean([len(group) for group in product_groups.values()]) if product_groups else 0
        })
        
        # Step 4: Select best representatives using LLM scoring
        print("🎯 Selecting best representatives...")
        representative_df = await self._select_best_representatives(retail_df, product_groups, product_analyses, target_size)
        curation_report['steps'].append({
            'step': 'llm_representative_selection',
            'input_size': len(retail_df),
            'output_size': len(representative_df),
            'coverage_ratio': len(representative_df) / len(retail_df) if len(retail_df) > 0 else 0
        })
        
        # Step 5: Generate LLM recommendations for improvement
        print("💡 Generating LLM recommendations...")
        recommendations = await self._generate_recommendations(df, representative_df, product_analyses)
        curation_report['recommendations'] = recommendations
        
        # Step 6: Calculate final metrics
        final_metrics = self._calculate_curation_metrics(df, representative_df, product_analyses)
        curation_report['final_metrics'] = final_metrics
        curation_report['cost_analysis'] = self.cost_tracker.copy()
        
        # Save results
        self._save_curation_results(representative_df, curation_report, product_analyses)
        
        print(f"\n✅ LLM CURATION COMPLETE")
        print(f"📊 Original: {len(df):,} → Final: {len(representative_df):,} items")
        print(f"🎯 Reduction: {(1 - len(representative_df)/len(df))*100:.1f}%" if len(df) > 0 else "🎯 Reduction: N/A")
        print(f"💰 Estimated Cost: ${self.cost_tracker['estimated_cost']:.3f}")
        print(f"🔤 Total Tokens: {self.cost_tracker['total_tokens']:,}")
        
        return representative_df, curation_report
    
    async def _analyze_all_products(self, df: pd.DataFrame) -> List[ProductAnalysis]:
        """Analyze all products using LLM, with caching"""
        analyses = []
        products_to_analyze = []
        
        # Check cache first
        for idx, row in df.iterrows():
            cache_key = f"{row.get('Food_Code', idx)}_{row.get('Food_Name', '')}"
            if cache_key in self.analysis_cache:
                analyses.append(ProductAnalysis(**self.analysis_cache[cache_key]))
            else:
                products_to_analyze.append((idx, row))
        
        if products_to_analyze:
            print(f"  🆕 Analyzing {len(products_to_analyze)} new products with LLM...")
            
            # Process in batches
            for i in range(0, len(products_to_analyze), self.batch_size):
                batch = products_to_analyze[i:i + self.batch_size]
                batch_analyses = await self._analyze_product_batch(batch)
                analyses.extend(batch_analyses)
                
                # Rate limiting
                if i + self.batch_size < len(products_to_analyze):
                    await asyncio.sleep(self.rate_limit_delay)  # Use configured rate limit delay
        
        return analyses
    
    async def _analyze_product_batch(self, product_batch: List[Tuple[int, pd.Series]]) -> List[ProductAnalysis]:
        """Analyze a batch of products using LLM"""
        batch_analyses = []
        
        for idx, product in product_batch:
            try:
                analysis = await self._analyze_single_product(product)
                batch_analyses.append(analysis)
                
                # Cache the result
                cache_key = f"{product.get('Food_Code', idx)}_{product.get('Food_Name', '')}"
                self.analysis_cache[cache_key] = analysis.__dict__
                
            except Exception as e:
                logger.error(f"Error analyzing product {product.get('Food_Name', '')}: {e}")
                # Create fallback analysis
                fallback = ProductAnalysis(
                    food_code=str(product.get('Food_Code', idx)),
                    food_name=str(product.get('Food_Name', '')),
                    is_retail_product=True,  # Conservative default
                    retail_confidence=0.5,
                    representativeness_score=0.5,
                    product_category="Unknown",
                    reasoning="LLM analysis failed - using fallback"
                )
                batch_analyses.append(fallback)
        
        # Save cache after each batch
        self._save_analysis_cache()
        
        return batch_analyses
    
    async def _analyze_single_product(self, product: pd.Series) -> ProductAnalysis:
        """Analyze a single product using LLM"""
        food_name = str(product.get('Food_Name', ''))
        description = str(product.get('Description', ''))
        group = str(product.get('Group', ''))
        food_code = str(product.get('Food_Code', ''))
        
        prompt = f"""
Analyze this food product for retail grocery store suitability:

Product: {food_name}
Description: {description}
Food Group: {group}

Please evaluate:
1. Is this a retail grocery product (vs. restaurant/homemade/recipe)?
2. How representative is this of the average product in its category?
3. What product category does this belong to?

Respond in this exact JSON format:
{{
    "is_retail_product": true/false,
    "retail_confidence": 0.0-1.0,
    "representativeness_score": 0.0-1.0,
    "product_category": "category_name",
    "reasoning": "brief explanation"
}}

Consider retail products as: fresh produce, packaged foods, canned goods, frozen items, dairy, meat, etc. that consumers buy in grocery stores.
Exclude: homemade items, restaurant dishes, takeaway foods, very specific recipe preparations.
For representativeness: prefer "average", "typical", "standard" preparations over "premium", "organic", "branded" or very specific variations.
"""
        
        try:
            response = await self._call_openai_api(prompt)
            
            # Clean the response - remove markdown formatting if present
            cleaned_response = response.strip()
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response[7:]  # Remove ```json
            if cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[:-3]  # Remove ```
            cleaned_response = cleaned_response.strip()
            
            # If response is empty after cleaning
            if not cleaned_response:
                logger.warning(f"Empty response after cleaning for {food_name}")
                raise ValueError("Empty response after cleaning")
            
            # Try to find JSON in the response if it's mixed with other text
            if not cleaned_response.startswith('{'):
                import re
                json_match = re.search(r'\{.*\}', cleaned_response, re.DOTALL)
                if json_match:
                    cleaned_response = json_match.group(0)
                else:
                    logger.error(f"No JSON found in response for {food_name}: {response}")
                    raise ValueError("No valid JSON found in response")
            
            # Parse JSON response with error handling
            try:
                result = json.loads(cleaned_response)
            except json.JSONDecodeError as json_error:
                logger.error(f"JSON decode error for {food_name}: {json_error}")
                logger.error(f"Response content: {repr(response)}")
                
                # Try to fix common JSON issues
                try:
                    import re
                    # Remove trailing commas and fix quotes
                    fixed_response = re.sub(r',\s*}', '}', cleaned_response)
                    fixed_response = re.sub(r',\s*]', ']', fixed_response)
                    result = json.loads(fixed_response)
                    logger.info(f"Successfully parsed JSON after fixing for {food_name}")
                except json.JSONDecodeError:
                    # If still failing, create a default response
                    logger.warning(f"Could not parse JSON for {food_name}, creating default analysis")
                    result = {
                        "is_retail_product": True,
                        "retail_confidence": 0.5,
                        "representativeness_score": 0.5,
                        "product_category": "Unknown",
                        "reasoning": f"JSON parsing failed for: {food_name}"
                    }
            
            return ProductAnalysis(
                food_code=food_code,
                food_name=food_name,
                is_retail_product=result.get('is_retail_product', True),
                retail_confidence=float(result.get('retail_confidence', 0.5)),
                representativeness_score=float(result.get('representativeness_score', 0.5)),
                product_category=result.get('product_category', 'Unknown'),
                reasoning=result.get('reasoning', 'No reasoning provided')
            )
            
        except Exception as e:
            logger.error(f"LLM API call failed for {food_name}: {e}")
            # Return a conservative default analysis instead of crashing
            return ProductAnalysis(
                food_code=food_code,
                food_name=food_name,
                is_retail_product=True,  # Conservative default
                retail_confidence=0.5,
                representativeness_score=0.5,
                product_category="Unknown",
                reasoning=f"API error: {str(e)}"
            )
    
    async def _call_openai_api(self, prompt: str, max_tokens: int = None) -> str:
        """Make API call to OpenAI"""
        if max_tokens is None:
            max_tokens = self.max_tokens
            
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        data = {
            'model': self.model,
            'messages': [
                {'role': 'system', 'content': 'You are an expert food analyst specializing in retail grocery products. Always respond with valid JSON.'},
                {'role': 'user', 'content': prompt}
            ],
            'max_tokens': max_tokens,
            'temperature': self.temperature
        }
        
        for attempt in range(self.max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            if attempt < self.max_retries - 1:
                                logger.warning(f"API error {response.status}, retrying... ({attempt + 1}/{self.max_retries})")
                                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                                continue
                            else:
                                raise Exception(f"OpenAI API error {response.status}: {error_text}")
                        
                        result = await response.json()
                        
                        # Validate response structure and content
                        if 'choices' not in result or not result['choices']:
                            logger.error(f"Invalid API response structure: {result}")
                            if attempt < self.max_retries - 1:
                                logger.warning(f"Retrying due to invalid response structure... ({attempt + 1}/{self.max_retries})")
                                await asyncio.sleep(2 ** attempt)
                                continue
                            else:
                                raise Exception("Invalid API response structure")
                        
                        content = result['choices'][0]['message']['content']
                        
                        # Check for empty content
                        if not content or content.strip() == "":
                            logger.warning("Empty response from API")
                            if attempt < self.max_retries - 1:
                                logger.warning(f"Retrying due to empty response... ({attempt + 1}/{self.max_retries})")
                                await asyncio.sleep(2 ** attempt)
                                continue
                            else:
                                raise Exception("Received empty response from API")
                        
                        # Track usage and costs
                        usage = result.get('usage', {})
                        total_tokens = usage.get('total_tokens', 0)
                        self.cost_tracker['total_tokens'] += total_tokens
                        
                        # Estimate cost (GPT-4.1 Mini pricing: $0.40 per 1M input tokens, $1.60 per 1M output tokens)
                        input_tokens = usage.get('prompt_tokens', total_tokens * 0.8)
                        output_tokens = usage.get('completion_tokens', total_tokens * 0.2)
                        cost_per_input_token = 0.40 / 1_000_000
                        cost_per_output_token = 1.60 / 1_000_000
                        self.cost_tracker['estimated_cost'] += (input_tokens * cost_per_input_token) + (output_tokens * cost_per_output_token)
                        
                        return content.strip()
                        
            except Exception as e:
                if attempt < self.max_retries - 1:
                    logger.warning(f"API call failed, retrying... ({attempt + 1}/{self.max_retries}): {e}")
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise e
    
    def _filter_retail_products(self, df: pd.DataFrame, analyses: List[ProductAnalysis]) -> pd.DataFrame:
        """Filter products based on LLM retail analysis"""
        analysis_dict = {analysis.food_code: analysis for analysis in analyses}
        
        retail_indices = []
        for idx, row in df.iterrows():
            food_code = str(row.get('Food_Code', idx))
            analysis = analysis_dict.get(food_code)
            
            if analysis and analysis.is_retail_product and analysis.retail_confidence > 0.6:
                retail_indices.append(idx)
        
        return df.loc[retail_indices].copy()
    
    async def _create_intelligent_groups(self, df: pd.DataFrame, analyses: List[ProductAnalysis]) -> Dict[str, List[int]]:
        """Create intelligent product groups using LLM understanding"""
        if df.empty:
            return {}
        
        # Get product categories from LLM analyses
        analysis_dict = {analysis.food_code: analysis for analysis in analyses}
        
        # Group by LLM-determined categories
        category_groups = defaultdict(list)
        
        for idx, row in df.iterrows():
            food_code = str(row.get('Food_Code', idx))
            analysis = analysis_dict.get(food_code)
            
            if analysis:
                category = analysis.product_category
                category_groups[category].append(idx)
            else:
                category_groups['Unknown'].append(idx)
        
        # For large categories, create sub-groups using LLM
        final_groups = {}
        group_id = 0
        
        for category, indices in category_groups.items():
            if len(indices) <= 5:  # Small groups stay as-is
                final_groups[f"{category}_{group_id}"] = indices
                group_id += 1
            else:
                # Split large categories into sub-groups using LLM
                sub_groups = await self._create_sub_groups(df.loc[indices], category)
                for sub_group_name, sub_indices in sub_groups.items():
                    final_groups[f"{category}_{sub_group_name}_{group_id}"] = sub_indices
                    group_id += 1
        
        return final_groups
    
    async def _create_sub_groups(self, category_df: pd.DataFrame, category_name: str) -> Dict[str, List[int]]:
        """Create sub-groups within a category using LLM"""
        if len(category_df) <= 5:
            return {"main": category_df.index.tolist()}
        
        # Create a summary of products for LLM grouping
        product_list = []
        for idx, row in category_df.iterrows():
            product_list.append(f"{idx}: {row.get('Food_Name', '')}")
        
        products_text = "\n".join(product_list[:20])  # Limit for token management
        
        prompt = f"""
Group these {category_name} products into 3-5 logical sub-groups based on similarity:

{products_text}

Create sub-groups like: "fresh_fruits", "dried_fruits", "fruit_juices", etc.
Respond in JSON format:
{{
    "sub_group_1_name": [index1, index2, ...],
    "sub_group_2_name": [index3, index4, ...],
    ...
}}
"""
        
        try:
            response = await self._call_openai_api(prompt, max_tokens=500)
            sub_groups = json.loads(response)
            
            # Validate and convert to proper format
            validated_groups = {}
            for group_name, indices in sub_groups.items():
                valid_indices = [idx for idx in indices if idx in category_df.index]
                if valid_indices:
                    validated_groups[group_name] = valid_indices
            
            return validated_groups
            
        except Exception as e:
            logger.error(f"Sub-grouping failed for {category_name}: {e}")
            # Fallback: simple alphabetical grouping
            return {"main": category_df.index.tolist()}
    
    async def _select_best_representatives(self, df: pd.DataFrame, product_groups: Dict[str, List[int]], 
                                        analyses: List[ProductAnalysis], target_size: Optional[int] = None) -> pd.DataFrame:
        """Select best representatives using LLM scoring"""
        analysis_dict = {analysis.food_code: analysis for analysis in analyses}
        
        representatives = []
        
        for group_name, group_indices in product_groups.items():
            if not group_indices:
                continue
            
            group_df = df.loc[group_indices]
            
            if len(group_df) == 1:
                representatives.extend(group_indices)
            else:
                # Select best representative(s) based on LLM representativeness scores
                best_representative = await self._select_group_representative(group_df, analysis_dict)
                if best_representative is not None:
                    representatives.append(best_representative)
        
        representative_df = df.loc[representatives].copy() if representatives else pd.DataFrame(columns=df.columns)
        
        # Apply target size if specified
        if target_size and len(representative_df) > target_size:
            representative_df = self._reduce_to_target_size(representative_df, analysis_dict, target_size)
        
        return representative_df
    
    async def _select_group_representative(self, group_df: pd.DataFrame, analysis_dict: Dict[str, ProductAnalysis]) -> Optional[int]:
        """Select the best representative from a group"""
        if group_df.empty:
            return None
        
        best_idx = None
        best_score = -1
        
        for idx, row in group_df.iterrows():
            food_code = str(row.get('Food_Code', idx))
            analysis = analysis_dict.get(food_code)
            
            if analysis:
                # Combined score: representativeness + retail confidence
                score = (analysis.representativeness_score * 0.7) + (analysis.retail_confidence * 0.3)
                
                if score > best_score:
                    best_score = score
                    best_idx = idx
        
        return best_idx
    
    def _reduce_to_target_size(self, df: pd.DataFrame, analysis_dict: Dict[str, ProductAnalysis], target_size: int) -> pd.DataFrame:
        """Reduce dataset to target size while maintaining diversity"""
        if len(df) <= target_size:
            return df
        
        # Score each product
        scores = []
        for idx, row in df.iterrows():
            food_code = str(row.get('Food_Code', idx))
            analysis = analysis_dict.get(food_code)
            
            if analysis:
                # Combined score with diversity bonus
                base_score = (analysis.representativeness_score * 0.6) + (analysis.retail_confidence * 0.4)
                
                # Diversity bonus for underrepresented categories
                category_count = sum(1 for _, other_row in df.iterrows() 
                                   if analysis_dict.get(str(other_row.get('Food_Code', '')), 
                                                       ProductAnalysis('', '', False, 0, 0, '', '')).product_category == analysis.product_category)
                diversity_bonus = 0.2 / max(category_count, 1)
                
                final_score = base_score + diversity_bonus
                scores.append((idx, final_score))
        
        # Select top scoring products
        scores.sort(key=lambda x: x[1], reverse=True)
        selected_indices = [idx for idx, _ in scores[:target_size]]
        
        return df.loc[selected_indices]
    
    async def _generate_recommendations(self, original_df: pd.DataFrame, final_df: pd.DataFrame, 
                                      analyses: List[ProductAnalysis]) -> List[str]:
        """Generate LLM-powered recommendations for improvement"""
        reduction_rate = (len(original_df) - len(final_df)) / len(original_df) * 100
        
        # Analyze what was removed
        final_codes = set(str(row.get('Food_Code', idx)) for idx, row in final_df.iterrows())
        removed_analyses = [a for a in analyses if a.food_code not in final_codes]
        
        # Get categories and patterns
        removed_categories = Counter([a.product_category for a in removed_analyses])
        kept_categories = Counter([a.product_category for a in analyses if a.food_code in final_codes])
        
        prompt = f"""
Analyze this retail dataset curation result and provide recommendations:

Original dataset: {len(original_df)} items
Final dataset: {len(final_df)} items  
Reduction rate: {reduction_rate:.1f}%

Top removed categories: {dict(removed_categories.most_common(5))}
Top kept categories: {dict(kept_categories.most_common(5))}

Please provide 3-5 specific recommendations for improving the curation process.
Focus on: balance, representativeness, retail relevance, and completeness.

Respond as a simple list of recommendations (no JSON).
"""
        
        try:
            response = await self._call_openai_api(prompt, max_tokens=400)
            # Split into individual recommendations
            recommendations = [rec.strip() for rec in response.split('\n') if rec.strip() and not rec.strip().startswith('```')]
            return recommendations[:5]  # Limit to 5 recommendations
        except Exception as e:
            logger.error(f"Failed to generate recommendations: {e}")
            return ["Review filtering criteria for better balance", "Consider including more diverse product categories"]
    
    def _calculate_curation_metrics(self, original_df: pd.DataFrame, final_df: pd.DataFrame, 
                                   analyses: List[ProductAnalysis]) -> Dict[str, Any]:
        """Calculate comprehensive curation metrics"""
        if original_df.empty or final_df.empty:
            return {'error': 'Empty datasets'}
        
        # Get analyses for final dataset
        final_codes = set(str(row.get('Food_Code', idx)) for idx, row in final_df.iterrows())
        final_analyses = [a for a in analyses if a.food_code in final_codes]
        
        metrics = {
            'reduction_rate': (len(original_df) - len(final_df)) / len(original_df) * 100,
            'avg_retail_confidence': np.mean([a.retail_confidence for a in final_analyses]) if final_analyses else 0,
            'avg_representativeness': np.mean([a.representativeness_score for a in final_analyses]) if final_analyses else 0,
            'category_diversity': len(set(a.product_category for a in final_analyses)),
            'retail_purity': sum(1 for a in final_analyses if a.is_retail_product) / len(final_analyses) if final_analyses else 0,
        }
        
        # Overall quality score
        metrics['llm_quality_score'] = np.mean([
            metrics['avg_retail_confidence'],
            metrics['avg_representativeness'],
            min(metrics['category_diversity'] / 10, 1.0),  # Normalize diversity
            metrics['retail_purity']
        ])
        
        return metrics
    
    def _save_curation_results(self, curated_df: pd.DataFrame, report: Dict[str, Any], 
                              analyses: List[ProductAnalysis]):
        """Save all curation results"""
        output_dir = self.config.OUTPUT_DIR
        os.makedirs(output_dir, exist_ok=True)
        
        # Save curated dataset
        curated_df.to_csv(os.path.join(output_dir, "llm_curated_retail_dataset.csv"), index=False)
        
        # Save detailed report
        with open(os.path.join(output_dir, "llm_curation_report.json"), 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Save LLM analyses
        analyses_data = [analysis.__dict__ for analysis in analyses]
        with open(os.path.join(output_dir, "llm_product_analyses.json"), 'w') as f:
            json.dump(analyses_data, f, indent=2)
        
        print(f"\n📁 LLM CURATION RESULTS SAVED:")
        print(f"  📊 Dataset: {os.path.join(output_dir, 'llm_curated_retail_dataset.csv')}")
        print(f"  📋 Report: {os.path.join(output_dir, 'llm_curation_report.json')}")
        print(f"  🧠 Analyses: {os.path.join(output_dir, 'llm_product_analyses.json')}")
    
    def _load_analysis_cache(self):
        """Load analysis cache from file"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    self.analysis_cache = json.load(f)
                print(f"📚 Loaded {len(self.analysis_cache)} cached analyses")
            except Exception as e:
                logger.error(f"Failed to load analysis cache: {e}")
                self.analysis_cache = {}
        else:
            self.analysis_cache = {}
    
    def _save_analysis_cache(self):
        """Save analysis cache to file"""
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.analysis_cache, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save analysis cache: {e}")


async def run_llm_curation(api_key: str = None, target_size: Optional[int] = None, model: str = "gpt-3.5-turbo"):
    """Run LLM-powered retail dataset curation"""
    if not api_key:
        api_key = input("Please enter your OpenAI API key: ").strip()
        if not api_key:
            print("❌ No API key provided. Cannot run LLM curation.")
            return None, None
    
    print("🧠 STARTING LLM-POWERED RETAIL CURATION")
    print("=" * 60)
    
    # Get configuration
    config = get_config()
    
    # Load data
    print("📥 Loading McCance & Widdowson data...")
    df = pd.read_excel(
        config.MCCANCE_WIDDOWSON_PATH,
        sheet_name=config.MW_SHEET_NAME_FOR_MAIN_PY
    )
    
    # Standardize column names
    column_mapping = {
        'Food Code': 'Food_Code',
        'Food Name': 'Food_Name',
        'Group': 'Group',
        'Description': 'Description'
    }
    df = df.rename(columns=column_mapping)
    
    # For demo/testing, limit to first 50 items to control costs
    if len(df) > 50:
        print(f"⚠️  Demo mode: Using first 50 items (full dataset has {len(df)} items)")
        df = df.head(50)
    
    # Initialize LLM curator
    curator = LLMRetailCurator(api_key=api_key, model=model)
    
    # Run LLM curation
    curated_df, curation_report = await curator.curate_intelligent_dataset(df, target_size)
    
    return curated_df, curation_report


def run_llm_curation_sync(api_key: str = None, target_size: Optional[int] = None):
    """Synchronous wrapper for LLM curation"""
    return asyncio.run(run_llm_curation(api_key, target_size))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Check if API key is available
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key or api_key == 'your-api-key-here':
        print("🔑 Please set your OpenAI API key:")
        print("   Option 1: Set OPENAI_API_KEY environment variable")
        print("   Option 2: Enter it when prompted")
        print("\n💰 Note: This will make API calls to OpenAI (estimated cost: $1-5 for full dataset)")
        
        if input("\nContinue? (y/n): ").lower().startswith('y'):
            curated_df, report = run_llm_curation_sync()
        else:
            print("❌ LLM curation cancelled")
    else:
        curated_df, report = run_llm_curation_sync(api_key) 