#!/usr/bin/env python3
"""
Analyze LLM reasoning patterns from hybrid curation results
Extract insights to improve the rules-based curator
"""

import json
import pandas as pd
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any
import re
import os

def load_hybrid_analyses(file_path: str) -> List[Dict]:
    """Load the hybrid analyses JSON file"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading file: {e}")
        return []

def analyze_disagreements(analyses: List[Dict]) -> Dict[str, Any]:
    """Analyze cases where LLM and rules disagreed"""
    disagreements = []
    
    for analysis in analyses:
        if analysis.get('method_used') == 'hybrid':
            llm_data = analysis.get('llm_analysis', {})
            rules_data = analysis.get('rules_analysis', {})
            
            if llm_data and rules_data:
                llm_retail = llm_data.get('is_retail_product', True)
                rules_retail = rules_data.get('is_retail_product', True)
                
                # Check for significant disagreement
                if llm_retail != rules_retail:
                    disagreements.append({
                        'food_name': analysis.get('food_name', ''),
                        'food_code': analysis.get('food_code', ''),
                        'llm_decision': llm_retail,
                        'rules_decision': rules_retail,
                        'llm_confidence': llm_data.get('retail_confidence', 0),
                        'rules_confidence': rules_data.get('retail_confidence', 0),
                        'llm_reasoning': llm_data.get('reasoning', ''),
                        'rules_reasoning': rules_data.get('reasoning', ''),
                        'llm_category': llm_data.get('product_category', ''),
                        'rules_category': rules_data.get('product_category', '')
                    })
    
    return {
        'total_disagreements': len(disagreements),
        'disagreements': disagreements
    }

def extract_llm_patterns(analyses: List[Dict]) -> Dict[str, Any]:
    """Extract common patterns from LLM reasoning"""
    
    # Patterns for exclusion (non-retail)
    exclude_patterns = Counter()
    include_patterns = Counter()
    
    # Category patterns
    llm_categories = Counter()
    
    # Reasoning phrases
    exclude_phrases = Counter()
    include_phrases = Counter()
    
    for analysis in analyses:
        llm_data = analysis.get('llm_analysis', {})
        if not llm_data:
            continue
            
        food_name = analysis.get('food_name', '').lower()
        reasoning = llm_data.get('reasoning', '').lower()
        is_retail = llm_data.get('is_retail_product', True)
        category = llm_data.get('product_category', '')
        
        # Collect categories
        if category:
            llm_categories[category] += 1
        
        # Extract patterns from food names
        words = food_name.split()
        
        if not is_retail:
            # Extract exclude patterns
            for word in words:
                if len(word) > 3:  # Ignore short words
                    exclude_patterns[word] += 1
            
            # Extract exclude phrases from reasoning
            exclude_phrases[reasoning] += 1
        else:
            # Extract include patterns
            for word in words:
                if len(word) > 3:
                    include_patterns[word] += 1
            
            include_phrases[reasoning] += 1
    
    return {
        'exclude_patterns': dict(exclude_patterns.most_common(50)),
        'include_patterns': dict(include_patterns.most_common(50)),
        'llm_categories': dict(llm_categories.most_common(20)),
        'exclude_reasoning': dict(exclude_phrases.most_common(20)),
        'include_reasoning': dict(include_phrases.most_common(20))
    }

def analyze_homemade_detection(analyses: List[Dict]) -> Dict[str, List[str]]:
    """Analyze how LLM detects homemade/non-retail items"""
    
    homemade_indicators = []
    restaurant_indicators = []
    takeaway_indicators = []
    recipe_indicators = []
    
    for analysis in analyses:
        llm_data = analysis.get('llm_analysis', {})
        if not llm_data or llm_data.get('is_retail_product', True):
            continue
            
        food_name = analysis.get('food_name', '').lower()
        reasoning = llm_data.get('reasoning', '').lower()
        
        # Check for specific non-retail indicators
        if any(word in food_name for word in ['homemade', 'home-made', 'home made']):
            homemade_indicators.append(food_name)
        elif any(word in food_name for word in ['restaurant', 'takeaway', 'take-away', 'take away']):
            if 'takeaway' in food_name or 'take-away' in food_name or 'take away' in food_name:
                takeaway_indicators.append(food_name)
            else:
                restaurant_indicators.append(food_name)
        elif any(word in food_name for word in ['recipe', 'cooked', 'prepared', 'stewed', 'fried']):
            recipe_indicators.append(food_name)
    
    return {
        'homemade_items': homemade_indicators[:20],
        'restaurant_items': restaurant_indicators[:20], 
        'takeaway_items': takeaway_indicators[:20],
        'recipe_items': recipe_indicators[:20]
    }

def generate_rule_improvements(patterns: Dict[str, Any], disagreements: Dict[str, Any]) -> Dict[str, List[str]]:
    """Generate specific improvements for the rules curator"""
    
    improvements = {
        'new_exclude_patterns': [],
        'new_include_patterns': [],
        'category_improvements': [],
        'confidence_adjustments': []
    }
    
    # Analyze disagreements where LLM said non-retail but rules said retail
    false_positives = [d for d in disagreements['disagreements'] 
                      if not d['llm_decision'] and d['rules_decision']]
    
    # Extract common words from false positives
    false_positive_words = Counter()
    for fp in false_positives:
        words = fp['food_name'].lower().split()
        for word in words:
            if len(word) > 3:
                false_positive_words[word] += 1
    
    # Suggest new exclude patterns
    for word, count in false_positive_words.most_common(10):
        if count >= 3:  # Appears in at least 3 disagreements
            improvements['new_exclude_patterns'].append(word)
    
    # Analyze cases where rules said non-retail but LLM said retail
    false_negatives = [d for d in disagreements['disagreements'] 
                      if d['llm_decision'] and not d['rules_decision']]
    
    false_negative_words = Counter()
    for fn in false_negatives:
        words = fn['food_name'].lower().split()
        for word in words:
            if len(word) > 3:
                false_negative_words[word] += 1
    
    # Suggest patterns that might be over-excluding
    for word, count in false_negative_words.most_common(10):
        if count >= 3:
            improvements['new_include_patterns'].append(word)
    
    return improvements

def main():
    """Main analysis function"""
    
    print("🔍 ANALYZING LLM REASONING PATTERNS")
    print("=" * 50)
    
    # Load the data
    file_path = "D:\LIDA\Product_Weight_Project\output\hybrid_curation\hybrid_analyses.json"
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return
    
    print("📥 Loading hybrid analyses...")
    analyses = load_hybrid_analyses(file_path)
    print(f"   Loaded {len(analyses)} analyses")
    
    # Analyze disagreements
    print("\n🔍 Analyzing LLM vs Rules disagreements...")
    disagreements = analyze_disagreements(analyses)
    print(f"   Found {disagreements['total_disagreements']} disagreements")
    
    # Extract patterns
    print("\n🧠 Extracting LLM reasoning patterns...")
    patterns = extract_llm_patterns(analyses)
    
    # Analyze homemade detection
    print("\n🏠 Analyzing homemade/non-retail detection...")
    homemade_analysis = analyze_homemade_detection(analyses)
    
    # Generate improvements
    print("\n💡 Generating rule improvements...")
    improvements = generate_rule_improvements(patterns, disagreements)
    
    # Create comprehensive report
    report = {
        'summary': {
            'total_analyses': len(analyses),
            'total_disagreements': disagreements['total_disagreements'],
            'disagreement_rate': disagreements['total_disagreements'] / len(analyses) * 100 if analyses else 0
        },
        'disagreements': disagreements,
        'llm_patterns': patterns,
        'homemade_analysis': homemade_analysis,
        'suggested_improvements': improvements
    }
    
    # Save detailed report
    output_file = "D:\LIDA\Product_Weight_Project\output\hybrid_curation\llm_reasoning_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📊 ANALYSIS COMPLETE")
    print(f"   📁 Detailed report saved: {output_file}")
    
    # Print key insights
    print(f"\n🔑 KEY INSIGHTS:")
    print(f"   📈 Disagreement rate: {report['summary']['disagreement_rate']:.1f}%")
    print(f"   🏠 Homemade items found: {len(homemade_analysis['homemade_items'])}")
    print(f"   🍽️ Restaurant items found: {len(homemade_analysis['restaurant_items'])}")
    print(f"   🥡 Takeaway items found: {len(homemade_analysis['takeaway_items'])}")
    
    print(f"\n💡 SUGGESTED RULE IMPROVEMENTS:")
    print(f"   ➕ New exclude patterns: {improvements['new_exclude_patterns'][:5]}")
    print(f"   ➖ Over-excluding patterns: {improvements['new_include_patterns'][:5]}")
    
    # Show top LLM categories
    print(f"\n📂 TOP LLM CATEGORIES:")
    for cat, count in list(patterns['llm_categories'].items())[:10]:
        print(f"   {cat}: {count}")
    
    return report

if __name__ == "__main__":
    report = main() 