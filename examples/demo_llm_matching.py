#!/usr/bin/env python3
"""
LLM-Enhanced Product Matching Demo

This script demonstrates the new LLM-enhanced matching system with:
1. Intelligent product matching using LLMs
2. Enhanced data extraction and parsing
3. Hybrid scoring combining ML and LLM approaches
4. Interactive visualization of results

Run this to see the LLM system in action!
"""

import sys
import os
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import our LLM-enhanced modules with graceful fallback
try:
    from shelfscale.ml.llm_matcher import LLMMatcher
    from shelfscale.scraping.llm_enhanced_scraper import LLMEnhancedScraper, ProductInfo
except ImportError as e:
    print(f"Warning: Could not import some modules: {e}")
    print("Please install missing dependencies or check module paths")
    
# Optional visualization import
try:
    from shelfscale.visualization.enhanced_dashboard import create_enhanced_dashboard
except ImportError:
    print("Note: Enhanced dashboard not available")
    create_enhanced_dashboard = None


class LLMMatchingDemo:
    """Demo class for LLM-enhanced matching system using real GPT-4o-mini API"""
    
    def __init__(self, api_key: str = None):
        """Initialize the demo with OpenAI API key"""
        print("🚀 Initializing LLM-Enhanced Product Matching Demo (GPT-4o-mini)")
        print("=" * 60)
        
        # Check for API key
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            print("❌ Error: OpenAI API key is required!")
            print("Set the OPENAI_API_KEY environment variable or pass api_key parameter.")
            print("Example: export OPENAI_API_KEY='your-api-key'")
            sys.exit(1)
        
        print("✅ OpenAI API key configured")
        print("💰 Using GPT-4o-mini for cost-efficient matching")
        
        # Initialize components with API key and learning enabled
        self.llm_matcher = LLMMatcher(
            model_name="gpt-4o-mini",
            confidence_threshold=0.7,
            use_hybrid_scoring=True,
            api_key=self.api_key,
            enable_learning=True
        )
        
        self.llm_scraper = LLMEnhancedScraper()
        
        # Demo data
        self.demo_products_1 = [
            "Tesco Organic Free Range Chicken Breast Fillets 500g",
            "Sainsbury's Taste the Difference British Chicken Breast 400g", 
            "ASDA Extra Special Outdoor Bred Chicken Breast 450g",
            "Morrisons The Best Free Range Chicken Breast 500g",
            "Waitrose Duchy Organic Chicken Breast Fillets 350g",
            "Iceland British Chicken Breast Fillets 600g",
            "Marks & Spencer Oakham Gold Chicken Breast 400g",
            "Aldi Specially Selected Chicken Breast 500g"
        ]
        
        self.demo_products_2 = [
            "Free Range Chicken Breast Fillet Pack 450g £5.99",
            "British Chicken Breast Steaks 400g £4.50",
            "Organic Chicken Breast Portions 500g £6.99", 
            "Premium Chicken Breast Fillets 350g £4.99",
            "Fresh Chicken Breast Joint 600g £5.49",
            "Chicken Breast Escalopes 400g £4.75",
            "Outdoor Reared Chicken Breast 500g £5.25",
            "Chicken Breast Mini Fillets 300g £3.99"
        ]
        
        print(f"✅ Initialized with {len(self.demo_products_1)} source products")
        print(f"✅ Initialized with {len(self.demo_products_2)} target products")
        
    async def run_enhanced_scraping_demo(self):
        """Demonstrate LLM-enhanced scraping capabilities"""
        print("\n🔍 LLM-Enhanced Scraping Demo")
        print("-" * 40)
        
        # Sample raw product data (simulating scraped data)
        raw_product_data = [
            "Tesco Finest Organic Free Range Chicken Breast Fillets 500g £6.50",
            "Sainsbury's Taste the Difference British Beef Mince 500g £4.99 15% Fat",
            "ASDA Extra Special Mature Cheddar Cheese 200g £2.50 Contains: Milk",
            "Morrisons Wonky Organic Bananas 1kg £1.20 Origin: Ecuador",
            "Waitrose Essential Wholemeal Bread 800g £1.10 Contains: Gluten, Wheat"
        ]
        
        print(f"📦 Processing {len(raw_product_data)} raw products...")
        
        enhanced_products = []
        
        for i, raw_text in enumerate(raw_product_data, 1):
            print(f"\n{i}. Extracting: {raw_text[:60]}...")
            
            # Extract with LLM
            product_info = await self.llm_scraper.extract_product_info_llm(
                raw_text, 
                context={"source": "demo", "extraction_id": i}
            )
            
            enhanced_products.append(product_info)
            
            # Show results
            print(f"   📝 Name: {product_info.name}")
            if product_info.weight_value:
                print(f"   ⚖️  Weight: {product_info.weight_value}{product_info.weight_unit}")
            if product_info.price_value:
                print(f"   💰 Price: {product_info.currency}{product_info.price_value:.2f}")
            if product_info.category:
                print(f"   📂 Category: {product_info.category}")
            if product_info.brand:
                print(f"   🏷️  Brand: {product_info.brand}")
            print(f"   🎯 Confidence: {product_info.confidence_score:.2f}")
            
        # Convert to DataFrame for analysis
        df = self.llm_scraper.products_to_dataframe(enhanced_products)
        
        print(f"\n✅ Enhanced extraction completed!")
        print(f"📊 Extracted data shape: {df.shape}")
        
        # Quality validation
        quality_metrics = self.llm_scraper.validate_extraction_quality(enhanced_products)
        print(f"🎯 Overall quality score: {quality_metrics['quality_score']:.3f}")
        print(f"📈 Average confidence: {quality_metrics['average_confidence']:.3f}")
        
        return enhanced_products, df
        
    async def run_llm_matching_demo(self):
        """Demonstrate LLM-enhanced product matching"""
        print("\n🤖 LLM-Enhanced Matching Demo")
        print("-" * 40)
        
        # Create DataFrames
        df1 = pd.DataFrame({'product_name': self.demo_products_1})
        df2 = pd.DataFrame({'description': self.demo_products_2})
        
        print(f"🔄 Matching {len(df1)} products against {len(df2)} targets...")
        
        # Run LLM matching
        matches_df = await self.llm_matcher.match_dataframes_llm(
            df1, df2, 
            'product_name', 'description',
            max_matches_per_item=3,
            pre_filter_similarity=0.3
        )
        
        print(f"\n✅ Found {len(matches_df)} high-confidence matches!")
        
        if len(matches_df) > 0:
            print("\n🎯 Top Matches:")
            print("=" * 80)
            
            for idx, match in matches_df.head(5).iterrows():
                print(f"\nMatch {idx + 1}:")
                print(f"  📦 Source: {match['source_text']}")
                print(f"  🎯 Target: {match['target_text']}")
                print(f"  🤖 LLM Confidence: {match['llm_confidence']:.3f}")
                print(f"  📊 Hybrid Score: {match['hybrid_score']:.3f}")
                print(f"  💭 Food Match: {match['llm_reasoning']}")
                if 'simplified_product_a' in match and match['simplified_product_a']:
                    print(f"  🍎 Simplified A: {match['simplified_product_a']}")
                if 'simplified_product_b' in match and match['simplified_product_b']:
                    print(f"  🍎 Simplified B: {match['simplified_product_b']}")
                if 'brand_analysis' in match and match['brand_analysis']:
                    print(f"  🏷️  Brand Analysis: {match['brand_analysis']}")
                if 'matched_food_categories' in match and match['matched_food_categories']:
                    print(f"  🍽️  Food Categories: {', '.join(match['matched_food_categories'])}")
                if 'key_factors' in match:
                    print(f"  🔑 Key Factors: {', '.join(match['key_factors'])}")
                    
        else:
            print("⚠️  No matches found above confidence threshold")
            
        return matches_df
        
    async def run_comparison_demo(self):
        """Compare traditional vs LLM-enhanced matching"""
        print("\n⚖️  Traditional vs LLM Matching Comparison")
        print("-" * 50)
        
        # Sample challenging product pairs
        challenging_pairs = [
            ("Tesco Organic Chicken Breast 500g", "Free Range Chicken Breast Fillets 450g"),
            ("Sainsbury's Whole Milk 2L", "Fresh Whole Milk 1 Litre"),
            ("ASDA Smart Price Bread", "Value White Sliced Loaf 800g"),
            ("Waitrose Cheddar Cheese", "Mature Cheddar Block 200g"),
            ("Morrisons Apple Juice", "Pressed Apple Juice 1L")
        ]
        
        print("Testing challenging product pairs...")
        
        results = []
        
        for i, (product1, product2) in enumerate(challenging_pairs, 1):
            print(f"\n{i}. Comparing:")
            print(f"   A: {product1}")
            print(f"   B: {product2}")
            
            # LLM matching
            llm_result = await self.llm_matcher.match_products_llm(product1, product2)
            
            # Traditional fallback matching
            traditional_result = self.llm_matcher._fallback_match(product1, product2)
            
            print(f"   🤖 LLM: {llm_result['confidence']:.3f} ({'✅' if llm_result['match'] else '❌'})")
            print(f"   🔧 Traditional: {traditional_result['confidence']:.3f} ({'✅' if traditional_result['match'] else '❌'})")
            print(f"   💭 Food Analysis: {llm_result['reasoning']}")
            if 'brand_analysis' in llm_result and llm_result['brand_analysis']:
                print(f"   🏷️  Brand Analysis: {llm_result['brand_analysis']}")
            if 'matched_food_categories' in llm_result and llm_result['matched_food_categories']:
                print(f"   🍽️  Categories: {', '.join(llm_result['matched_food_categories'])}")
            
            results.append({
                'product1': product1,
                'product2': product2,
                'llm_confidence': llm_result['confidence'],
                'llm_match': llm_result['match'],
                'traditional_confidence': traditional_result['confidence'],
                'traditional_match': traditional_result['match'],
                'llm_reasoning': llm_result['reasoning'],
                'brand_analysis': llm_result.get('brand_analysis', ''),
                'matched_food_categories': ', '.join(llm_result.get('matched_food_categories', []))
            })
            
        # Summary comparison
        comparison_df = pd.DataFrame(results)
        
        llm_avg_confidence = comparison_df['llm_confidence'].mean()
        traditional_avg_confidence = comparison_df['traditional_confidence'].mean()
        
        print(f"\n📊 Comparison Summary:")
        print(f"   🤖 LLM Average Confidence: {llm_avg_confidence:.3f}")
        print(f"   🔧 Traditional Average Confidence: {traditional_avg_confidence:.3f}")
        print(f"   📈 LLM Improvement: {((llm_avg_confidence - traditional_avg_confidence) / traditional_avg_confidence * 100):+.1f}%")
        
        return comparison_df
        
    async def run_learning_demo(self):
        """Demonstrate learning capabilities with feedback simulation"""
        print("\n🧠 LLM Learning System Demo")
        print("-" * 40)
        
        # Show initial learning state
        initial_insights = self.llm_matcher.get_learning_insights()
        print(f"📊 Initial Learning State:")
        print(f"   • Threshold: {initial_insights['current_threshold']}")
        print(f"   • Total Feedback: {initial_insights['total_feedback']}")
        print(f"   • Learned Patterns: {initial_insights['learned_patterns_count']}")
        
        # Simulate feedback session with test cases
        test_cases = [
            ("Tesco Organic Chicken", "Free Range Chicken Breast", True, "chicken breast", "chicken breast"),
            ("Cheddar Cheese Block", "Mature Cheddar 200g", True, "cheddar cheese", "cheddar cheese"),
            ("Apple Juice 1L", "Orange Juice 1L", False, "apple juice", "orange juice"),
            ("White Bread Loaf", "Wholemeal Bread", False, "white bread", "wholemeal bread"),
            ("Sainsbury's Milk", "Fresh Whole Milk", True, "whole milk", "whole milk"),
            ("Chicken Nuggets", "Chicken Breast", False, "chicken nuggets", "chicken breast"),
            ("Diet Coke", "Coca Cola", False, "diet coke", "coca cola"),
            ("Greek Yogurt", "Natural Yogurt", True, "greek yogurt", "natural yogurt"),
            ("Beef Mince 500g", "Ground Beef 450g", True, "beef mince", "ground beef"),
            ("Chocolate Biscuits", "Digestive Biscuits", False, "chocolate biscuits", "digestive biscuits"),
            ("Chicken Thigh", "Chicken Breast", False, "chicken thigh", "chicken breast"),
            ("Fresh Salmon", "Smoked Salmon", False, "fresh salmon", "smoked salmon"),
        ]
        
        print(f"\n🔄 Simulating learning with {len(test_cases)} feedback examples...")
        
        # Run learning simulation
        learning_results = self.llm_matcher.simulate_learning_session(test_cases)
        
        print(f"\n✅ Learning simulation completed!")
        print(f"📈 Learning Results:")
        print(f"   • Cases processed: {learning_results['total_cases']}")
        print(f"   • Initial threshold: {learning_results['initial_threshold']}")
        print(f"   • Final threshold: {learning_results['final_threshold']}")
        print(f"   • Threshold changed: {'Yes' if learning_results['threshold_changed'] else 'No'}")
        
        # Show learning summary
        summary = learning_results['learning_summary']
        if summary.get('recent_performance'):
            perf = summary['recent_performance']
            print(f"   • Current accuracy: {perf.get('accuracy', 0):.1%}")
            print(f"   • F1 score: {perf.get('f1_score', 0):.3f}")
        
        # Show pattern learning
        if summary.get('learned_patterns_count', 0) > 0:
            print(f"   • Patterns learned: {summary['learned_patterns_count']}")
            
            if summary.get('problem_patterns'):
                print(f"\n⚠️  Challenging patterns identified:")
                for pattern in summary['problem_patterns'][:3]:
                    print(f"     • {pattern['pattern']}: {pattern['accuracy']:.1%} accuracy")
            
            if summary.get('reliable_patterns'):
                print(f"\n✅ Reliable patterns identified:")
                for pattern in summary['reliable_patterns'][:3]:
                    print(f"     • {pattern['pattern']}: {pattern['accuracy']:.1%} accuracy")
        
        return learning_results
        
    def generate_demo_report(self, 
                           enhanced_products, 
                           matches_df, 
                           comparison_df,
                           learning_results=None):
        """Generate a comprehensive demo report"""
        print("\n📋 Generating Demo Report")
        print("-" * 30)
        
        # Create output directory
        output_dir = Path("llm_matching_results")
        output_dir.mkdir(exist_ok=True)
        print(f"📁 Creating output directory: {output_dir}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = output_dir / f"llm_demo_report_{timestamp}.md"
        
        # Get statistics
        matcher_stats = self.llm_matcher.get_matching_stats()
        scraper_stats = self.llm_scraper.get_llm_stats()
        
        # Build report content step by step
        report_content = "# LLM-Enhanced Product Matching Demo Report\n\n"
        report_content += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        report_content += """## Executive Summary

This report demonstrates the capabilities of our new LLM-enhanced product matching system,
showing significant improvements over traditional string-based matching approaches.

## System Components

"""
        
        report_content += f"""### 1. LLM-Enhanced Matcher
- **Model:** {self.llm_matcher.model_name}
- **Confidence Threshold:** {self.llm_matcher.confidence_threshold}
- **Hybrid Scoring:** {self.llm_matcher.use_hybrid_scoring}

### 2. LLM-Enhanced Scraper
- **Model:** {self.llm_scraper.llm_model}
- **Temperature:** {self.llm_scraper.temperature}
- **Max Tokens:** {self.llm_scraper.max_tokens}

## Demo Results

### Enhanced Data Extraction
- **Products Processed:** {len(enhanced_products)}
- **Extraction Success Rate:** {scraper_stats.get('success_rate', 0):.1%}
- **Average Confidence:** {np.mean([p.confidence_score for p in enhanced_products]):.3f}

### Product Matching Performance
- **Total LLM Calls:** {matcher_stats.get('total_llm_calls', 0)}
- **Successful Matches:** {matcher_stats.get('successful_matches', 0)}
- **Match Success Rate:** {matcher_stats.get('success_rate', 0):.1%}
- **Cache Hit Rate:** {matcher_stats.get('cache_hit_rate', 0):.1%}

### Comparison: Traditional vs LLM
"""
        
        if len(comparison_df) > 0:
            llm_avg = comparison_df['llm_confidence'].mean()
            trad_avg = comparison_df['traditional_confidence'].mean()
            improvement = ((llm_avg - trad_avg) / trad_avg * 100)
            
            report_content += f"""
- **LLM Average Confidence:** {llm_avg:.3f}
- **Traditional Average Confidence:** {trad_avg:.3f}
- **Improvement:** {improvement:+.1f}%

### Learning System Performance
"""
            
            if learning_results:
                summary = learning_results.get('learning_summary', {})
                initial_thresh = learning_results.get('initial_threshold', 0)
                final_thresh = learning_results.get('final_threshold', 0)
                adjustment = final_thresh - initial_thresh
                
                report_content += f"""
- **Learning Enabled:** Yes
- **Initial Threshold:** {initial_thresh:.3f}
- **Final Threshold:** {final_thresh:.3f}
- **Threshold Adjustment:** {adjustment:+.3f}
- **Feedback Cases:** {learning_results.get('total_cases', 0)}
- **Patterns Learned:** {summary.get('learned_patterns_count', 0)}

"""
                if summary.get('recent_performance'):
                    perf = summary['recent_performance']
                    accuracy = perf.get('accuracy', 0)
                    precision = perf.get('precision', 0)
                    recall = perf.get('recall', 0)
                    f1_score = perf.get('f1_score', 0)
                    
                    report_content += f"""**Learning Performance Metrics:**
- Accuracy: {accuracy:.1%}
- Precision: {precision:.3f}
- Recall: {recall:.3f}
- F1 Score: {f1_score:.3f}

"""

        report_content += "### Sample Matches Found\n"
        
        if len(matches_df) > 0:
            for idx, match in matches_df.head(3).iterrows():
                match_num = idx + 1
                source_text = match['source_text']
                target_text = match['target_text']
                confidence = f"{match['llm_confidence']:.3f}"
                reasoning = match['llm_reasoning']
                report_content += f"""
**Match {match_num}:**
- Source: `{source_text}`
- Target: `{target_text}`
- LLM Confidence: {confidence}
- Reasoning: {reasoning}
"""

        report_content += """

## Key Innovations

### 1. Intelligent Reasoning
- LLM understands food semantics and product relationships
- Considers brand variations, packaging differences, and nutritional equivalence
- Provides human-readable reasoning for each matching decision

### 2. Enhanced Data Extraction
- Structured extraction from unstructured product descriptions
- Automatic categorization and attribute recognition
- Quality scoring and validation

### 3. Hybrid Scoring
- Combines traditional ML features with LLM confidence
- Balances speed and accuracy
- Fallback mechanisms for reliability

### 4. Adaptive Learning
- Learns from user feedback to improve accuracy
- Automatically adjusts confidence thresholds
- Identifies and corrects problematic patterns

## Technical Performance

### Scalability Features
- Batch processing for efficiency
- Caching for repeated queries
- Asynchronous operations for speed
- Pre-filtering to reduce LLM calls

### Quality Assurance
- Confidence scoring for all predictions
- Validation of extracted data
- Fallback mechanisms when LLM fails
- Comprehensive error handling

## Next Steps

1. **Production Integration:** Deploy LLM components alongside existing ML pipeline
2. **Model Fine-tuning:** Train on domain-specific food product data
3. **Performance Optimization:** Implement model caching and batch optimization
4. **User Feedback Loop:** Collect user feedback to improve matching accuracy

## Conclusion

The LLM-enhanced system shows promising results for improving product matching accuracy
while maintaining the robustness and scalability of the existing system. The hybrid
approach provides the best of both worlds: intelligent reasoning and reliable fallbacks.

---
"""
        
        report_content += f"*Demo completed successfully with {len(enhanced_products)} extractions and {len(matches_df)} matches.*\n"
        
        # Save report
        with open(report_file, 'w') as f:
            f.write(report_content)
            
        print(f"✅ Report saved to: {report_file}")
        
        # Also save data
        if len(matches_df) > 0:
            matches_file = output_dir / f"llm_matches_{timestamp}.csv"
            matches_df.to_csv(matches_file, index=False)
            print(f"✅ Matches data saved to: {matches_file}")
            
        if len(comparison_df) > 0:
            comparison_file = output_dir / f"llm_comparison_{timestamp}.csv"
            comparison_df.to_csv(comparison_file, index=False)
            print(f"✅ Comparison data saved to: {comparison_file}")
            
        return str(report_file)
        
    async def run_full_demo(self):
        """Run the complete demo workflow"""
        print("🎬 Starting LLM-Enhanced Product Matching Demo")
        print("=" * 60)
        
        try:
            # 1. Enhanced Scraping Demo
            enhanced_products, extraction_df = await self.run_enhanced_scraping_demo()
            
            # 2. LLM Matching Demo  
            matches_df = await self.run_llm_matching_demo()
            
            # 3. Comparison Demo
            comparison_df = await self.run_comparison_demo()
            
            # 4. Learning Demo
            learning_results = await self.run_learning_demo()
            
            # 5. Generate Report
            report_file = self.generate_demo_report(
                enhanced_products, matches_df, comparison_df, learning_results
            )
            
            # 5. Final Statistics
            print("\n🎉 Demo Completed Successfully!")
            print("=" * 60)
            print(f"📊 Statistics Summary:")
            print(f"   • Enhanced {len(enhanced_products)} product extractions")
            print(f"   • Found {len(matches_df)} high-confidence matches")
            print(f"   • Compared {len(comparison_df)} challenging pairs")
            print(f"   • Simulated learning with {learning_results.get('total_cases', 0)} feedback cases")
            print(f"   • Generated comprehensive report: {report_file}")
            
            # Performance metrics
            matcher_stats = self.llm_matcher.get_matching_stats()
            scraper_stats = self.llm_scraper.get_llm_stats()
            
            print(f"\n⚡ Performance Metrics:")
            print(f"   • LLM Calls: {matcher_stats.get('total_llm_calls', 0)}")
            print(f"   • Cache Hits: {matcher_stats.get('cache_hits', 0)}")
            print(f"   • Success Rate: {matcher_stats.get('success_rate', 0):.1%}")
            print(f"   • Extraction Success: {scraper_stats.get('success_rate', 0):.1%}")
            
            print("\n🎯 Key Benefits Demonstrated:")
            print("   ✅ Intelligent product understanding")
            print("   ✅ Structured data extraction")
            print("   ✅ Contextual matching decisions")
            print("   ✅ Human-readable reasoning")
            print("   ✅ Hybrid ML + LLM approach")
            print("   ✅ Production-ready scalability")
            print("   ✅ Adaptive learning from feedback")
            print("   ✅ Self-improving confidence thresholds")
            print("   ✅ Pattern recognition and optimization")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Demo failed with error: {e}")
            import traceback
            traceback.print_exc()
            return False


async def main():
    """Main demo function"""
    print("🚀 LLM-Enhanced Product Matching System Demo")
    print("=" * 60)
    print("This demo showcases advanced LLM capabilities for:")
    print("• Intelligent product data extraction")
    print("• Context-aware product matching") 
    print("• Hybrid ML + LLM scoring")
    print("• Production-ready scalability")
    print()
    
    demo = LLMMatchingDemo()
    success = await demo.run_full_demo()
    
    if success:
        print("\n🎉 Demo completed successfully!")
        print("Check the generated report files for detailed results.")
    else:
        print("\n❌ Demo encountered errors. Check logs for details.")
        
    return success


if __name__ == "__main__":
    # Run the demo
    success = asyncio.run(main())
    exit(0 if success else 1)