"""
Hybrid Retail Dataset Curator
Combines LLM and rule-based approaches for flexible curation
"""

import pandas as pd
import numpy as np
import json
import os
import logging
import asyncio
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import sys
sys.path.append('.')

from shelfscale.config_manager import get_config
from shelfscale.data_processing.llm_retail_curator import LLMRetailCurator, ProductAnalysis
from shelfscale.data_processing.enhanced_rules_curator import EnhancedRulesCurator, RuleBasedAnalysis

logger = logging.getLogger(__name__)


@dataclass
class HybridAnalysis:
    """Combined analysis from both LLM and rules"""
    food_code: str
    food_name: str
    llm_analysis: Optional[ProductAnalysis]
    rules_analysis: Optional[RuleBasedAnalysis]
    final_decision: Dict[str, Any]
    method_used: str


class HybridCurator:
    """
    Hybrid curator that can use LLM, rules, or both
    """
    
    def __init__(self, config_manager=None):
        self.config = config_manager or get_config()
        
        # Initialize both curators
        self.llm_curator = None
        self.rules_curator = EnhancedRulesCurator(config_manager)
        
        # Only initialize LLM curator if API key is available
        if self.config.validate_openai_config():
            self.llm_curator = LLMRetailCurator(config_manager)
        
        self.hybrid_stats = {
            'llm_used': 0,
            'rules_used': 0,
            'hybrid_used': 0,
            'agreements': 0,
            'disagreements': 0
        }
    
    async def curate_dataset_hybrid(
        self, 
        df: pd.DataFrame, 
        method: str = "auto",  # "llm", "rules", "hybrid", "auto"
        target_size: Optional[int] = None,
        llm_fallback: bool = True
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Main hybrid curation method
        
        Args:
            df: Input dataframe
            method: "llm", "rules", "hybrid", or "auto"
            target_size: Target size for final dataset
            llm_fallback: Use LLM as fallback if rules are uncertain
        """
        print("🔄 HYBRID RETAIL DATASET CURATION STARTING")
        print("=" * 60)
        
        # Determine actual method to use
        actual_method = self._determine_method(method)
        print(f"🎯 Using method: {actual_method}")
        
        curation_report = {
            'original_size': len(df),
            'method_requested': method,
            'method_used': actual_method,
            'hybrid_stats': {},
            'steps': [],
            'final_metrics': {}
        }
        
        # Run curation based on method
        if actual_method == "llm":
            return await self._curate_llm_only(df, target_size, curation_report)
        elif actual_method == "rules":
            return self._curate_rules_only(df, target_size, curation_report)
        elif actual_method == "hybrid":
            return await self._curate_hybrid(df, target_size, curation_report, llm_fallback)
        else:
            raise ValueError(f"Unknown method: {actual_method}")
    
    def _determine_method(self, requested_method: str) -> str:
        """Determine the actual method to use based on availability"""
        
        if requested_method == "auto":
            # Auto-select based on availability and configuration
            if self.llm_curator and not self.config.LLM_DEMO_MODE:
                return "llm"  # Use LLM for full dataset
            elif self.llm_curator and self.config.LLM_DEMO_MODE:
                return "hybrid"  # Use hybrid for demos (rules + selective LLM)
            else:
                return "rules"  # Fallback to rules if no LLM available
        
        elif requested_method == "llm":
            if not self.llm_curator:
                print("⚠️  LLM not available, falling back to rules")
                return "rules"
            return "llm"
        
        elif requested_method == "rules":
            return "rules"
        
        elif requested_method == "hybrid":
            if not self.llm_curator:
                print("⚠️  LLM not available for hybrid mode, using rules only")
                return "rules"
            return "hybrid"
        
        else:
            raise ValueError(f"Unknown method: {requested_method}")
    
    async def _curate_llm_only(self, df: pd.DataFrame, target_size: Optional[int], report: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Run LLM-only curation"""
        print("🧠 Running LLM-only curation...")
        
        result_df, llm_report = await self.llm_curator.curate_intelligent_dataset(df, target_size)
        
        # Merge reports
        report.update(llm_report)
        report['method_used'] = 'llm'
        
        return result_df, report
    
    def _curate_rules_only(self, df: pd.DataFrame, target_size: Optional[int], report: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Run rules-only curation"""
        print("🔧 Running rules-only curation...")
        
        result_df, rules_report = self.rules_curator.curate_dataset(df, target_size)
        
        # Merge reports
        report.update(rules_report)
        report['method_used'] = 'rules'
        
        return result_df, report
    
    async def _curate_hybrid(self, df: pd.DataFrame, target_size: Optional[int], report: Dict[str, Any], llm_fallback: bool) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Run hybrid curation combining both methods"""
        print("🔄 Running hybrid curation (rules + selective LLM)...")
        
        # Step 1: Run rules analysis on all products
        print("📋 Step 1: Analyzing all products with rules...")
        rules_analyses = []
        for idx, row in df.iterrows():
            analysis = self.rules_curator.analyze_product(row)
            rules_analyses.append(analysis)
        
        # Step 2: Identify uncertain cases for LLM verification
        uncertain_products = []
        confident_analyses = []
        
        for idx, analysis in enumerate(rules_analyses):
            # Consider uncertain if confidence is moderate or if it's a borderline case
            is_uncertain = (
                0.4 < analysis.retail_confidence < 0.8 or  # Moderate confidence
                0.3 < analysis.representativeness_score < 0.7 or  # Moderate representativeness
                analysis.product_category == "Unknown"  # Unknown category
            )
            
            if is_uncertain and llm_fallback:
                uncertain_products.append((idx, df.iloc[idx]))
            else:
                confident_analyses.append((idx, analysis))
        
        print(f"  🔍 Found {len(uncertain_products)} uncertain products for LLM verification")
        print(f"  ✅ {len(confident_analyses)} products confident with rules")
        
        # Step 3: Run LLM on uncertain products only
        llm_analyses = {}
        if uncertain_products and self.llm_curator:
            print("🧠 Step 2: LLM verification of uncertain products...")
            
            # Create a small dataframe for uncertain products
            uncertain_df = pd.DataFrame([product for _, product in uncertain_products])
            
            # Run LLM analysis
            try:
                llm_product_analyses = await self.llm_curator._analyze_all_products(uncertain_df)
                for i, (original_idx, _) in enumerate(uncertain_products):
                    if i < len(llm_product_analyses):
                        llm_analyses[original_idx] = llm_product_analyses[i]
                        self.hybrid_stats['llm_used'] += 1
            except Exception as e:
                logger.error(f"LLM analysis failed for uncertain products: {e}")
                print("⚠️  LLM analysis failed, using rules for uncertain products")
        
        # Step 4: Create hybrid analyses
        print("🔄 Step 3: Combining analyses...")
        hybrid_analyses = []
        
        for idx in range(len(df)):
            rules_analysis = rules_analyses[idx]
            llm_analysis = llm_analyses.get(idx)
            
            if llm_analysis:
                # Combine LLM and rules
                final_analysis = self._combine_analyses(rules_analysis, llm_analysis)
                method_used = "hybrid"
                self.hybrid_stats['hybrid_used'] += 1
                
                # Check agreement
                if (rules_analysis.is_retail_product == llm_analysis.is_retail_product and
                    abs(rules_analysis.retail_confidence - llm_analysis.retail_confidence) < 0.3):
                    self.hybrid_stats['agreements'] += 1
                else:
                    self.hybrid_stats['disagreements'] += 1
            else:
                # Use rules only
                final_analysis = self._rules_to_final_analysis(rules_analysis)
                method_used = "rules"
                self.hybrid_stats['rules_used'] += 1
            
            hybrid_analysis = HybridAnalysis(
                food_code=rules_analysis.food_code,
                food_name=rules_analysis.food_name,
                llm_analysis=llm_analysis,
                rules_analysis=rules_analysis,
                final_decision=final_analysis,
                method_used=method_used
            )
            hybrid_analyses.append(hybrid_analysis)
        
        # Step 5: Filter and select representatives
        print("🛒 Step 4: Filtering and selecting representatives...")
        final_df = self._select_final_representatives(df, hybrid_analyses, target_size)
        
        # Update report
        report['hybrid_stats'] = self.hybrid_stats.copy()
        report['steps'] = [
            {'step': 'rules_analysis', 'products_analyzed': len(rules_analyses)},
            {'step': 'llm_verification', 'uncertain_products': len(uncertain_products), 'llm_analyzed': len(llm_analyses)},
            {'step': 'hybrid_combination', 'final_products': len(final_df)}
        ]
        
        # Calculate metrics
        final_metrics = self._calculate_hybrid_metrics(df, final_df, hybrid_analyses)
        report['final_metrics'] = final_metrics
        
        # Save results
        self._save_hybrid_results(final_df, report, hybrid_analyses)
        
        print(f"\n✅ HYBRID CURATION COMPLETE")
        print(f"📊 Original: {len(df):,} → Final: {len(final_df):,} items")
        print(f"🎯 Reduction: {(1 - len(final_df)/len(df))*100:.1f}%" if len(df) > 0 else "🎯 Reduction: N/A")
        print(f"🔧 Rules used: {self.hybrid_stats['rules_used']}")
        print(f"🧠 LLM used: {self.hybrid_stats['llm_used']}")
        print(f"🔄 Hybrid decisions: {self.hybrid_stats['hybrid_used']}")
        if self.hybrid_stats['hybrid_used'] > 0:
            agreement_rate = self.hybrid_stats['agreements'] / (self.hybrid_stats['agreements'] + self.hybrid_stats['disagreements']) * 100
            print(f"🤝 Agreement rate: {agreement_rate:.1f}%")
        
        return final_df, report
    
    def _combine_analyses(self, rules_analysis: RuleBasedAnalysis, llm_analysis: ProductAnalysis) -> Dict[str, Any]:
        """Combine rules and LLM analyses into final decision"""
        
        # Weight the analyses (LLM gets slightly higher weight for retail decision)
        retail_confidence = (rules_analysis.retail_confidence * 0.4 + llm_analysis.retail_confidence * 0.6)
        representativeness = (rules_analysis.representativeness_score * 0.6 + llm_analysis.representativeness_score * 0.4)
        
        # Use LLM category if available, otherwise rules
        category = llm_analysis.product_category if llm_analysis.product_category != "Unknown" else rules_analysis.product_category
        
        return {
            'is_retail_product': retail_confidence > 0.6,
            'retail_confidence': retail_confidence,
            'representativeness_score': representativeness,
            'product_category': category,
            'reasoning': f"Hybrid: Rules({rules_analysis.retail_confidence:.2f}) + LLM({llm_analysis.retail_confidence:.2f}) = {retail_confidence:.2f}"
        }
    
    def _rules_to_final_analysis(self, rules_analysis: RuleBasedAnalysis) -> Dict[str, Any]:
        """Convert rules analysis to final analysis format"""
        return {
            'is_retail_product': rules_analysis.is_retail_product,
            'retail_confidence': rules_analysis.retail_confidence,
            'representativeness_score': rules_analysis.representativeness_score,
            'product_category': rules_analysis.product_category,
            'reasoning': f"Rules only: {rules_analysis.reasoning}"
        }
    
    def _select_final_representatives(self, df: pd.DataFrame, hybrid_analyses: List[HybridAnalysis], target_size: Optional[int]) -> pd.DataFrame:
        """Select final representatives from hybrid analyses"""
        
        # Filter retail products
        retail_indices = []
        for idx, analysis in enumerate(hybrid_analyses):
            if (analysis.final_decision['is_retail_product'] and 
                analysis.final_decision['retail_confidence'] > 0.6):
                retail_indices.append(idx)
        
        if not retail_indices:
            return pd.DataFrame(columns=df.columns)
        
        retail_df = df.iloc[retail_indices].copy()
        retail_analyses = [hybrid_analyses[i] for i in retail_indices]
        
        # Group by category and select representatives
        from collections import defaultdict
        category_groups = defaultdict(list)
        for i, analysis in enumerate(retail_analyses):
            category = analysis.final_decision['product_category']
            category_groups[category].append(i)
        
        # Select representatives from each category
        representative_indices = []
        for category, indices in category_groups.items():
            if not indices:
                continue
            
            # Sort by combined score
            scored_indices = []
            for i in indices:
                analysis = retail_analyses[i]
                score = (analysis.final_decision['retail_confidence'] * 0.4 + 
                        analysis.final_decision['representativeness_score'] * 0.6)
                scored_indices.append((i, score))
            
            scored_indices.sort(key=lambda x: x[1], reverse=True)
            
            # Select top representatives
            num_reps = min(3, max(1, len(scored_indices) // 3))
            representative_indices.extend([idx for idx, _ in scored_indices[:num_reps]])
        
        final_df = retail_df.iloc[representative_indices].copy()
        
        # Apply target size if specified
        if target_size and len(final_df) > target_size:
            # Score all and select top
            scores = []
            for idx, row in final_df.iterrows():
                analysis = next((a for a in retail_analyses if a.food_code == str(row.get('Food_Code', idx))), None)
                if analysis:
                    score = (analysis.final_decision['retail_confidence'] * 0.4 + 
                           analysis.final_decision['representativeness_score'] * 0.6)
                    scores.append((idx, score))
            
            scores.sort(key=lambda x: x[1], reverse=True)
            selected_indices = [idx for idx, _ in scores[:target_size]]
            final_df = final_df.loc[selected_indices]
        
        return final_df
    
    def _calculate_hybrid_metrics(self, original_df: pd.DataFrame, final_df: pd.DataFrame, analyses: List[HybridAnalysis]) -> Dict[str, Any]:
        """Calculate hybrid curation metrics"""
        if original_df.empty or final_df.empty:
            return {'error': 'Empty datasets'}
        
        final_codes = set(str(row.get('Food_Code', idx)) for idx, row in final_df.iterrows())
        final_analyses = [a for a in analyses if a.food_code in final_codes]
        
        metrics = {
            'reduction_rate': (len(original_df) - len(final_df)) / len(original_df) * 100,
            'avg_retail_confidence': np.mean([a.final_decision['retail_confidence'] for a in final_analyses]) if final_analyses else 0,
            'avg_representativeness': np.mean([a.final_decision['representativeness_score'] for a in final_analyses]) if final_analyses else 0,
            'category_diversity': len(set(a.final_decision['product_category'] for a in final_analyses)),
            'retail_purity': sum(1 for a in final_analyses if a.final_decision['is_retail_product']) / len(final_analyses) if final_analyses else 0,
            'method_distribution': {
                'rules_only': sum(1 for a in final_analyses if a.method_used == 'rules'),
                'llm_only': sum(1 for a in final_analyses if a.method_used == 'llm'),
                'hybrid': sum(1 for a in final_analyses if a.method_used == 'hybrid')
            }
        }
        
        metrics['hybrid_quality_score'] = np.mean([
            metrics['avg_retail_confidence'],
            metrics['avg_representativeness'],
            min(metrics['category_diversity'] / 10, 1.0),
            metrics['retail_purity']
        ])
        
        return metrics
    
    def _save_hybrid_results(self, curated_df: pd.DataFrame, report: Dict[str, Any], analyses: List[HybridAnalysis]):
        """Save hybrid curation results"""
        output_dir = self.config.OUTPUT_DIR.replace('ai_curation', 'hybrid_curation')
        os.makedirs(output_dir, exist_ok=True)
        
        # Save curated dataset
        curated_df.to_csv(os.path.join(output_dir, "hybrid_curated_retail_dataset.csv"), index=False)
        
        # Save detailed report
        with open(os.path.join(output_dir, "hybrid_curation_report.json"), 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Save hybrid analyses
        analyses_data = []
        for analysis in analyses:
            data = {
                'food_code': analysis.food_code,
                'food_name': analysis.food_name,
                'method_used': analysis.method_used,
                'final_decision': analysis.final_decision,
                'llm_analysis': analysis.llm_analysis.__dict__ if analysis.llm_analysis else None,
                'rules_analysis': analysis.rules_analysis.__dict__ if analysis.rules_analysis else None
            }
            analyses_data.append(data)
        
        with open(os.path.join(output_dir, "hybrid_analyses.json"), 'w') as f:
            json.dump(analyses_data, f, indent=2)
        
        print(f"\n📁 HYBRID CURATION RESULTS SAVED:")
        print(f"  📊 Dataset: {os.path.join(output_dir, 'hybrid_curated_retail_dataset.csv')}")
        print(f"  📋 Report: {os.path.join(output_dir, 'hybrid_curation_report.json')}")
        print(f"  🔄 Analyses: {os.path.join(output_dir, 'hybrid_analyses.json')}")


# Convenience functions
async def run_hybrid_curation(method: str = "auto", target_size: Optional[int] = None, llm_fallback: bool = True):
    """Run hybrid curation with specified method"""
    print("🔄 STARTING HYBRID RETAIL CURATION")
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
    
    # Initialize hybrid curator
    curator = HybridCurator()
    
    # Run curation
    curated_df, report = await curator.curate_dataset_hybrid(df, method, target_size, llm_fallback)
    
    return curated_df, report


def run_hybrid_curation_sync(method: str = "auto", target_size: Optional[int] = None, llm_fallback: bool = True):
    """Synchronous wrapper for hybrid curation"""
    return asyncio.run(run_hybrid_curation(method, target_size, llm_fallback))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("🔄 HYBRID CURATOR - Choose your method:")
    print("1. auto - Automatically choose best method")
    print("2. rules - Rules-based only (fast, no API costs)")
    print("3. llm - LLM-based only (thorough, API costs)")
    print("4. hybrid - Combine rules + selective LLM (balanced)")
    
    method_choice = input("Choose method (1-4) or enter method name: ").strip()
    
    method_map = {
        '1': 'auto',
        '2': 'rules', 
        '3': 'llm',
        '4': 'hybrid'
    }
    
    method = method_map.get(method_choice, method_choice)
    
    target_size_input = input("Target size (press Enter for auto): ").strip()
    target_size = int(target_size_input) if target_size_input else None
    
    curated_df, report = run_hybrid_curation_sync(method, target_size) 