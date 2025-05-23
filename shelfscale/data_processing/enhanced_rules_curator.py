"""
Enhanced Rules-Based Retail Dataset Curator
Incorporates LLM-derived logic for robust curation without API calls
"""

import pandas as pd
import numpy as np
import json
import os
import logging
import re
from typing import Dict, List, Tuple, Optional, Any, Set
from collections import Counter, defaultdict
from dataclasses import dataclass
import sys
sys.path.append('.')

from shelfscale.config_manager import get_config

logger = logging.getLogger(__name__)


@dataclass
class RuleBasedAnalysis:
    """Results of rule-based analysis for a food product"""
    food_code: str
    food_name: str
    is_retail_product: bool
    retail_confidence: float
    representativeness_score: float
    product_category: str
    reasoning: str
    rule_matches: List[str]


class EnhancedRulesCurator:
    """
    Enhanced rules-based curator incorporating LLM-derived logic
    """
    
    def __init__(self, config_manager=None):
        self.config = config_manager or get_config()
        
        # Initialize rule sets based on LLM learning
        self._load_retail_rules()
        self._load_representativeness_rules()
        self._load_category_rules()
        
        # Tracking
        self.analysis_cache = {}
        self.rule_stats = defaultdict(int)
    
    def _load_retail_rules(self):
        """Load rules for identifying retail products (from LLM insights)"""
        # Updated on 2025-05-23 04:16:43 based on LLM analysis

        # Updated on 2025-05-23 04:09:14 based on LLM analysis

        # Updated on 2025-05-23 04:00:45 based on LLM analysis

        # Updated on 2025-05-23 03:35:53 based on LLM analysis

        
        # INCLUDE rules - strong indicators of retail products
        self.retail_include_patterns = {
            'packaged_foods': [
                r'\b(canned|tinned|jarred|bottled|packaged|frozen)\b',
                r'\b(brand|branded)\b',
                r'\bin syrup\b', r'\bin brine\b', r'\bin sauce\b',
                r'\b(ready meal|convenience)\b'
            ],
            'fresh_produce': [
                r'\b(fresh|raw)\b',
                r'\b(apple|banana|carrot|potato|onion|tomato|lettuce)\b',
                r'\b(meat|beef|chicken|pork|fish|salmon)\b',
                r'\b(milk|cheese|yogurt|butter)\b'
            ],
            'basic_ingredients': [
                r'\b(flour|sugar|salt|pepper|oil|vinegar)\b',
                r'\b(rice|pasta|bread|cereal)\b',
                r'\b(egg|eggs)\b'
            ],
            'retail_descriptors': [
                r'\b(supermarket|grocery|retail|commercial)\b',
                r'\b(standard|typical|average|common)\b',
                r'\b(lean|extra lean|semi-skimmed|full fat)\b'
            ],
            # Updated based on LLM analysis - these were being over-excluded
            'cooking_methods': [
                r'\b(fried|grilled|roasted|baked|steamed)\b',
                r'\bin batter\b',
                r'\bwith vegetable\b',
                r'\bsunflower oil\b',
                r'\bcorn oil\b'
                r'\bfried\b',
                r'\bhomemade\b',
                r'\bwith\b',
]
        }
        
        # EXCLUDE rules - strong indicators of non-retail products  
        self.retail_exclude_patterns = {
            'homemade': [
                r'\b(homemade|home-made|home made)\b',
                r'\b(recipe|cooked at home)\b',
                r'\b(baked at home|prepared at home)\b'
            ],
            'restaurant': [
                r'\b(restaurant|takeaway|take-away|fast food)\b',
                r'\b(pub food|café|coffee shop)\b',
                r'\b(served|dining)\b'
            ],
            'specific_preparations': [
                r'\b(fried in|grilled with|marinated in)\b',
                r'\b(stuffed with|topped with|glazed with)\b',
                r'\b(recipe includes|cooked with)\b'
            ],
            'brand_specific': [
                r'\b[A-Z][a-z]+ brand\b',  # "Tesco brand", "Sainsbury brand"
                r'\b(premium|luxury|gourmet|artisan)\b',
                r'\b(organic|free-range|grass-fed)\b'
            ],
            # New patterns based on LLM analysis - strong indicators of non-retail
            'database_preparations': [
                r'\bboiled in\b.*\bwater\b',
                r'\bunsalted water\b',
                r'\bflesh only\b',
                r'\bweighed with\b',
                r'\bcalculated from\b',
                r'\bstewed without\b.*\bsugar\b',
                r'\bstewed with\b.*\bsugar\b',
                r'\bdrained\b.*\bfrom\b'
            ],
            'measurement_indicators': [
                r'\bonly,\s+weighed\b',
                r'\bonly,\s+peeled\b',
                r'\bonly,\s+flesh\b',
                r'\bmeat only\b',
                r'\bskin,\s+weighed\b'
            ],
            'nan_entries': [
                r'^nan$',
                r'^\s*$',
                r'^null$'
                r'\bboiled\b',
                r'\bwater\b',
                r'\bunsalted\b',
                r'\bwith\b',
                r'\bflesh\b',
]
        }
    
    def _load_representativeness_rules(self):
        """Load rules for scoring representativeness"""
        
        # HIGH representativeness indicators
        self.high_representativeness = [
            r'\b(average|typical|standard|common|regular)\b',
            r'\b(medium|normal)\b',
            r'\b(lean|semi-skimmed)\b',
            r'^[A-Z][a-z]+ [a-z]+$',  # Simple two-word names like "Cheddar cheese"
        ]
        
        # MEDIUM representativeness indicators
        self.medium_representativeness = [
            r'\b(fresh|cooked|boiled|grilled|baked)\b',
            r'\b(with skin|without skin|skinless)\b',
            r'\b(white|brown|red)\b'
        ]
        
        # LOW representativeness indicators
        self.low_representativeness = [
            r'\b(premium|luxury|gourmet|artisan|specialty)\b',
            r'\b(organic|free-range|grass-fed|wild)\b',
            r'\b(flavoured|seasoned|marinated)\b',
            r'\b(exotic|imported|speciality)\b',
            r'\b(diet|low-fat|reduced|light)\b',
            r'\b(stuffed|filled|topped|glazed)\b'
        ]
    
    def _load_category_rules(self):
        """Load rules for categorizing products"""
        
        self.category_patterns = {
            'Meat & Poultry': [
                r'\b(beef|pork|lamb|chicken|turkey|duck|meat)\b',
                r'\b(steak|chop|mince|sausage|bacon|ham)\b',
                r'\b(liver|kidney|heart)\b'
            ],
            'Fish & Seafood': [
                r'\b(fish|salmon|cod|tuna|sardine|mackerel)\b',
                r'\b(prawn|shrimp|crab|lobster|shellfish)\b',
                r'\b(seafood|marine)\b',
                r'\b(frozen seafood|fresh seafood)\b'
            ],
            'Dairy & Eggs': [
                r'\b(milk|cheese|yogurt|yoghurt|butter|cream)\b',
                r'\b(egg|eggs|dairy)\b',
                r'\b(cheddar|brie|mozzarella)\b'
            ],
            'Fruits': [
                r'\b(apple|banana|orange|grape|berry|fruit)\b',
                r'\b(strawberry|blueberry|raspberry)\b',
                r'\b(citrus|tropical)\b',
                r'\b(canned fruit|dried fruit)\b'
            ],
            'Vegetables': [
                r'\b(potato|carrot|onion|tomato|lettuce|cabbage)\b',
                r'\b(broccoli|spinach|pepper|cucumber|vegetable)\b',
                r'\b(root|leafy|green)\b',
                r'\b(fresh vegetables|frozen vegetables)\b'
            ],
            # Updated categories based on LLM analysis top categories
            'Fresh Produce': [
                r'\b(fresh|raw)\b.*\b(fruit|vegetable|produce)\b',
                r'\b(fresh produce|farm fresh)\b',
                r'\b(fresh|raw)\b.*\b(apple|banana|carrot|potato|onion|tomato)\b'
            ],
            'Prepared Meals': [
                r'\b(prepared meal|ready meal|convenience food)\b',
                r'\b(meal|dish|casserole|curry|stir-fry)\b',
                r'\b(prepared foods|cooked meal)\b'
            ],
            'Grains & Cereals': [
                r'\b(bread|rice|pasta|cereal|grain)\b',
                r'\b(wheat|oat|barley|quinoa)\b',
                r'\b(flour|noodle)\b',
                r'\b(breakfast cereals|breakfast cereal)\b'
            ],
            'Beverages': [
                r'\b(juice|drink|beverage|water|tea|coffee)\b',
                r'\b(cola|soda|milk drink)\b',
                r'\b(smoothie|shake)\b'
            ],
            'Snacks & Confectionery': [
                r'\b(biscuit|cookie|cake|chocolate|sweet)\b',
                r'\b(crisp|chip|snack|candy)\b',
                r'\b(confectionery|dessert|desserts)\b'
            ],
            'Oils & Fats': [
                r'\b(oil|fat|butter|margarine)\b',
                r'\b(olive oil|sunflower|coconut)\b'
            ],
            'Condiments & Seasonings': [
                r'\b(sauce|dressing|vinegar|mustard)\b',
                r'\b(salt|pepper|herb|spice)\b',
                r'\b(ketchup|mayonnaise)\b'
            ],
            'Bakery': [
                r'\b(bread|bun|roll|pastry|bakery)\b',
                r'\b(cake|muffin|croissant|bagel)\b',
                r'\b(loaf|slice|baked goods)\b'
            ]
        }
    
    def analyze_product(self, product: pd.Series) -> RuleBasedAnalysis:
        """Analyze a single product using enhanced rules"""
        food_code = str(product.get('Food_Code', ''))
        food_name = str(product.get('Food_Name', ''))
        description = str(product.get('Description', ''))
        group = str(product.get('Group', ''))
        
        # Combine text for analysis
        full_text = f"{food_name} {description} {group}".lower()
        
        # Analyze retail suitability
        is_retail, retail_confidence, retail_reasoning = self._analyze_retail_suitability(full_text)
        
        # Analyze representativeness
        repr_score, repr_reasoning = self._analyze_representativeness(full_text)
        
        # Determine category
        category, category_reasoning = self._determine_category(full_text)
        
        # Combine reasoning
        combined_reasoning = f"Retail: {retail_reasoning}; Representativeness: {repr_reasoning}; Category: {category_reasoning}"
        
        return RuleBasedAnalysis(
            food_code=food_code,
            food_name=food_name,
            is_retail_product=is_retail,
            retail_confidence=retail_confidence,
            representativeness_score=repr_score,
            product_category=category,
            reasoning=combined_reasoning,
            rule_matches=[]  # Could be populated with specific rule matches
        )
    
    def _analyze_retail_suitability(self, text: str) -> Tuple[bool, float, str]:
        """Analyze if product is suitable for retail"""
        include_score = 0
        exclude_score = 0
        matched_rules = []
        
        # Check include patterns
        for category, patterns in self.retail_include_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    include_score += 1
                    matched_rules.append(f"Include-{category}")
                    self.rule_stats[f"include_{category}"] += 1
        
        # Check exclude patterns
        for category, patterns in self.retail_exclude_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    exclude_score += 1
                    matched_rules.append(f"Exclude-{category}")
                    self.rule_stats[f"exclude_{category}"] += 1
        
        # Calculate retail probability
        if exclude_score > 0:
            # Strong exclusion indicators
            retail_confidence = max(0.1, 0.7 - (exclude_score * 0.2))
            is_retail = retail_confidence > 0.5
            reasoning = f"Exclude rules matched ({exclude_score}), confidence reduced"
        elif include_score > 0:
            # Positive indicators
            retail_confidence = min(0.9, 0.6 + (include_score * 0.1))
            is_retail = True
            reasoning = f"Include rules matched ({include_score}), likely retail"
        else:
            # No clear indicators - use conservative approach
            retail_confidence = 0.5
            is_retail = True  # Default to include unless excluded
            reasoning = "No clear indicators, defaulting to retail"
        
        return is_retail, retail_confidence, reasoning
    
    def _analyze_representativeness(self, text: str) -> Tuple[float, str]:
        """Analyze how representative the product is"""
        high_score = 0
        medium_score = 0
        low_score = 0
        
        # Check representativeness patterns
        for pattern in self.high_representativeness:
            if re.search(pattern, text, re.IGNORECASE):
                high_score += 1
                self.rule_stats["high_repr"] += 1
        
        for pattern in self.medium_representativeness:
            if re.search(pattern, text, re.IGNORECASE):
                medium_score += 1
                self.rule_stats["medium_repr"] += 1
        
        for pattern in self.low_representativeness:
            if re.search(pattern, text, re.IGNORECASE):
                low_score += 1
                self.rule_stats["low_repr"] += 1
        
        # Calculate representativeness score (0-1)
        if low_score > 0:
            # Specialty/premium indicators reduce representativeness
            score = max(0.2, 0.6 - (low_score * 0.15))
            reasoning = f"Low repr indicators ({low_score}), specialty product"
        elif high_score > 0:
            # Standard/typical indicators increase representativeness
            score = min(0.9, 0.7 + (high_score * 0.1))
            reasoning = f"High repr indicators ({high_score}), typical product"
        elif medium_score > 0:
            # Medium indicators
            score = 0.6 + (medium_score * 0.05)
            reasoning = f"Medium repr indicators ({medium_score}), moderately typical"
        else:
            # No clear indicators
            score = 0.5
            reasoning = "No clear representativeness indicators"
        
        return score, reasoning
    
    def _determine_category(self, text: str) -> Tuple[str, str]:
        """Determine product category"""
        category_scores = {}
        
        for category, patterns in self.category_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    score += 1
                    self.rule_stats[f"category_{category}"] += 1
            
            if score > 0:
                category_scores[category] = score
        
        if category_scores:
            # Return category with highest score
            best_category = max(category_scores.keys(), key=lambda k: category_scores[k])
            reasoning = f"Matched {category_scores[best_category]} patterns for {best_category}"
            return best_category, reasoning
        else:
            return "Unknown", "No category patterns matched"
    
    def curate_dataset(self, df: pd.DataFrame, target_size: Optional[int] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Main curation method using enhanced rules"""
        print("🔧 ENHANCED RULES-BASED RETAIL CURATION STARTING")
        print("=" * 60)
        
        curation_report = {
            'original_size': len(df),
            'method': 'enhanced_rules',
            'steps': [],
            'rule_statistics': {},
            'final_metrics': {}
        }
        
        # Step 1: Analyze all products
        print("📋 Analyzing all products with enhanced rules...")
        analyses = []
        for idx, row in df.iterrows():
            analysis = self.analyze_product(row)
            analyses.append(analysis)
        
        curation_report['steps'].append({
            'step': 'rule_based_analysis',
            'products_analyzed': len(analyses)
        })
        
        # Step 2: Filter retail products
        print("🛒 Filtering retail products...")
        retail_analyses = [a for a in analyses if a.is_retail_product and a.retail_confidence > 0.6]
        retail_df = df.loc[[analyses.index(a) for a in retail_analyses]].copy()
        
        curation_report['steps'].append({
            'step': 'retail_filtering',
            'input_size': len(df),
            'output_size': len(retail_df),
            'removal_rate': (len(df) - len(retail_df)) / len(df) * 100 if len(df) > 0 else 0
        })
        
        # Step 3: Group by category and select representatives
        print("🔗 Grouping by category...")
        category_groups = defaultdict(list)
        for i, analysis in enumerate(retail_analyses):
            category_groups[analysis.product_category].append(i)
        
        # Step 4: Select best representatives
        print("🎯 Selecting best representatives...")
        representative_indices = []
        for category, indices in category_groups.items():
            if not indices:
                continue
            
            # Sort by combined score (retail confidence + representativeness)
            category_analyses = [retail_analyses[i] for i in indices]
            scored_analyses = [
                (i, a.retail_confidence * 0.4 + a.representativeness_score * 0.6)
                for i, a in zip(indices, category_analyses)
            ]
            scored_analyses.sort(key=lambda x: x[1], reverse=True)
            
            # Select top representatives (at least 1, max 3 per category)
            num_representatives = min(3, max(1, len(scored_analyses) // 3))
            representative_indices.extend([idx for idx, _ in scored_analyses[:num_representatives]])
        
        representative_df = retail_df.iloc[representative_indices].copy()
        
        # Apply target size if specified
        if target_size and len(representative_df) > target_size:
            representative_df = self._reduce_to_target_size(representative_df, analyses, target_size)
        
        curation_report['steps'].append({
            'step': 'representative_selection',
            'input_size': len(retail_df),
            'output_size': len(representative_df),
            'coverage_ratio': len(representative_df) / len(retail_df) if len(retail_df) > 0 else 0
        })
        
        # Calculate final metrics
        final_metrics = self._calculate_metrics(df, representative_df, analyses)
        curation_report['final_metrics'] = final_metrics
        curation_report['rule_statistics'] = dict(self.rule_stats)
        
        # Save results
        self._save_results(representative_df, curation_report, analyses)
        
        print(f"\n✅ ENHANCED RULES CURATION COMPLETE")
        print(f"📊 Original: {len(df):,} → Final: {len(representative_df):,} items")
        print(f"🎯 Reduction: {(1 - len(representative_df)/len(df))*100:.1f}%" if len(df) > 0 else "🎯 Reduction: N/A")
        print(f"🔧 Rules Applied: {len(self.rule_stats)} different rule types")
        
        return representative_df, curation_report
    
    def _reduce_to_target_size(self, df: pd.DataFrame, analyses: List[RuleBasedAnalysis], target_size: int) -> pd.DataFrame:
        """Reduce dataset to target size while maintaining diversity"""
        if len(df) <= target_size:
            return df
        
        # Score each product and maintain category diversity
        category_counts = Counter([a.product_category for a in analyses])
        
        scores = []
        for idx, row in df.iterrows():
            # Find corresponding analysis
            analysis = next((a for a in analyses if a.food_code == str(row.get('Food_Code', idx))), None)
            if analysis:
                base_score = analysis.retail_confidence * 0.4 + analysis.representativeness_score * 0.6
                # Diversity bonus for underrepresented categories
                diversity_bonus = 0.2 / max(category_counts[analysis.product_category], 1)
                final_score = base_score + diversity_bonus
                scores.append((idx, final_score))
        
        # Select top scoring products
        scores.sort(key=lambda x: x[1], reverse=True)
        selected_indices = [idx for idx, _ in scores[:target_size]]
        
        return df.loc[selected_indices]
    
    def _calculate_metrics(self, original_df: pd.DataFrame, final_df: pd.DataFrame, analyses: List[RuleBasedAnalysis]) -> Dict[str, Any]:
        """Calculate curation metrics"""
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
        metrics['rules_quality_score'] = np.mean([
            metrics['avg_retail_confidence'],
            metrics['avg_representativeness'],
            min(metrics['category_diversity'] / 10, 1.0),
            metrics['retail_purity']
        ])
        
        return metrics
    
    def _save_results(self, curated_df: pd.DataFrame, report: Dict[str, Any], analyses: List[RuleBasedAnalysis]):
        """Save curation results"""
        output_dir = self.config.OUTPUT_DIR.replace('ai_curation', 'rules_curation')
        os.makedirs(output_dir, exist_ok=True)
        
        # Save curated dataset
        curated_df.to_csv(os.path.join(output_dir, "rules_curated_retail_dataset.csv"), index=False)
        
        # Save detailed report
        with open(os.path.join(output_dir, "rules_curation_report.json"), 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Save rule analyses
        analyses_data = [analysis.__dict__ for analysis in analyses]
        with open(os.path.join(output_dir, "rules_product_analyses.json"), 'w') as f:
            json.dump(analyses_data, f, indent=2)
        
        print(f"\n📁 RULES CURATION RESULTS SAVED:")
        print(f"  📊 Dataset: {os.path.join(output_dir, 'rules_curated_retail_dataset.csv')}")
        print(f"  📋 Report: {os.path.join(output_dir, 'rules_curation_report.json')}")
        print(f"  🔧 Analyses: {os.path.join(output_dir, 'rules_product_analyses.json')}")


def run_enhanced_rules_curation(target_size: Optional[int] = None):
    """Run enhanced rules-based curation"""
    print("🔧 STARTING ENHANCED RULES-BASED CURATION")
    print("=" * 60)
    
    # Load data
    print("📥 Loading McCance & Widdowson data...")
    from shelfscale.config_manager import get_config
    df = pd.read_excel(
        get_config.MCCANCE_WIDDOWSON_PATH,
        sheet_name=get_config.MW_SHEET_NAME_FOR_MAIN_PY
    )
    
    # Standardize column names
    column_mapping = {
        'Food Code': 'Food_Code',
        'Food Name': 'Food_Name',
        'Group': 'Group',
        'Description': 'Description'
    }
    df = df.rename(columns=column_mapping)
    
    # Initialize curator
    curator = EnhancedRulesCurator()
    
    # Run curation
    curated_df, report = curator.curate_dataset(df, target_size)
    
    return curated_df, report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    curated_df, report = run_enhanced_rules_curation() 