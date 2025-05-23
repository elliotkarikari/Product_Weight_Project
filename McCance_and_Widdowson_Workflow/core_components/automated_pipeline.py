"""
Automated Curation Pipeline
Runs LLM curation → Reasoning analysis → Rule updates in sequence
"""

import os
import sys
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add to path for imports
sys.path.append('.')

from shelfscale.config_manager import get_config
from shelfscale.data_processing.hybrid_curator import run_hybrid_curation_sync
from analyze_llm_reasoning import (
    load_hybrid_analyses, 
    analyze_disagreements, 
    extract_llm_patterns,
    analyze_homemade_detection,
    generate_rule_improvements
)
from rule_updater import RuleUpdater

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('automated_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AutomatedCurationPipeline:
    """
    Automated pipeline for intelligent curation with continuous improvement
    """
    
    def __init__(self, config_manager=None):
        self.config = config_manager or get_config()
        self.base_output_dir = Path("output")
        self.pipeline_dir = self.base_output_dir / "pipeline_runs"
        self.pipeline_dir.mkdir(exist_ok=True, parents=True)
        
        # Create timestamped run directory
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.pipeline_dir / f"run_{self.timestamp}"
        self.run_dir.mkdir(exist_ok=True)
        
        logger.info(f"🚀 Initialized pipeline run: {self.run_dir}")
    
    def run_full_pipeline(self, target_size: Optional[int] = None, 
                         update_rules: bool = True, 
                         backup_existing: bool = True) -> Dict[str, Any]:
        """
        Run the complete automated curation pipeline
        
        Args:
            target_size: Target dataset size (None for automatic)
            update_rules: Whether to update rules based on LLM analysis
            backup_existing: Whether to backup existing rules before updating
            
        Returns:
            Pipeline results summary
        """
        
        logger.info("🔄 Starting Automated Curation Pipeline")
        logger.info("=" * 60)
        
        pipeline_results = {
            "timestamp": self.timestamp,
            "run_directory": str(self.run_dir),
            "stages": {},
            "overall_success": False
        }
        
        try:
            # Stage 1: Run Hybrid Curation
            logger.info("🧠 STAGE 1: Running Hybrid Curation")
            curation_results = self._run_curation_stage(target_size)
            pipeline_results["stages"]["curation"] = curation_results
            
            # Stage 2: Analyze LLM Reasoning
            logger.info("🔍 STAGE 2: Analyzing LLM Reasoning")
            analysis_results = self._run_analysis_stage()
            pipeline_results["stages"]["analysis"] = analysis_results
            
            # Stage 3: Update Rules (if requested)
            if update_rules:
                logger.info("⚡ STAGE 3: Updating Rules")
                update_results = self._run_rule_update_stage(backup_existing)
                pipeline_results["stages"]["rule_updates"] = update_results
            else:
                logger.info("⏭️  STAGE 3: Skipping rule updates (disabled)")
                pipeline_results["stages"]["rule_updates"] = {"skipped": True}
            
            # Stage 4: Generate Summary Report
            logger.info("📊 STAGE 4: Generating Summary Report")
            summary_results = self._generate_summary_report(pipeline_results)
            pipeline_results["stages"]["summary"] = summary_results
            
            pipeline_results["overall_success"] = True
            logger.info("✅ Pipeline completed successfully!")
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {str(e)}")
            pipeline_results["error"] = str(e)
            pipeline_results["overall_success"] = False
            raise
        
        finally:
            # Save pipeline results
            results_file = self.run_dir / "pipeline_results.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(pipeline_results, f, indent=2, default=str)
            logger.info(f"📁 Pipeline results saved: {results_file}")
        
        return pipeline_results
    
    def _run_curation_stage(self, target_size: Optional[int]) -> Dict[str, Any]:
        """Run the hybrid curation stage"""
        
        logger.info("  🔹 Running hybrid curation...")
        
        try:
            # Run hybrid curation
            curated_df, hybrid_report = run_hybrid_curation_sync(method="auto", target_size=target_size)
            
            # Extract results for reporting
            hybrid_results = {
                "original_size": hybrid_report.get("original_size", 0),
                "final_size": len(curated_df) if curated_df is not None else 0,
                "reduction_percentage": hybrid_report.get("final_metrics", {}).get("reduction_rate", 0),
                "method_used": hybrid_report.get("method_used", "unknown"),
                "llm_decisions": hybrid_report.get("hybrid_stats", {}).get("llm_used", 0),
                "rules_decisions": hybrid_report.get("hybrid_stats", {}).get("rules_used", 0),
                "agreement_rate": 0
            }
            
            # Calculate agreement rate if available
            stats = hybrid_report.get("hybrid_stats", {})
            if stats.get("agreements", 0) + stats.get("disagreements", 0) > 0:
                total_comparisons = stats["agreements"] + stats["disagreements"]
                hybrid_results["agreement_rate"] = stats["agreements"] / total_comparisons * 100
            
            # Copy results to run directory
            curation_output_dir = self.run_dir / "curation_output"
            curation_output_dir.mkdir(exist_ok=True)
            
            # Copy main outputs
            hybrid_dir = Path("output/hybrid_curation")
            if hybrid_dir.exists():
                for file_path in hybrid_dir.glob("*"):
                    if file_path.is_file():
                        shutil.copy2(file_path, curation_output_dir / file_path.name)
            
            logger.info(f"  ✅ Curation completed: {hybrid_results['final_size']} items")
            
            return {
                "success": True,
                "results": hybrid_results,
                "output_directory": str(curation_output_dir)
            }
            
        except Exception as e:
            logger.error(f"  ❌ Curation failed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _run_analysis_stage(self) -> Dict[str, Any]:
        """Run the LLM reasoning analysis stage"""
        
        logger.info("  🔹 Analyzing LLM reasoning patterns...")
        
        try:
            # Load hybrid analyses
            analyses_file = "output/hybrid_curation/hybrid_analyses.json"
            if not os.path.exists(analyses_file):
                raise FileNotFoundError(f"Analyses file not found: {analyses_file}")
            
            analyses = load_hybrid_analyses(analyses_file)
            logger.info(f"  📥 Loaded {len(analyses)} analyses")
            
            # Run all analysis functions
            disagreements = analyze_disagreements(analyses)
            patterns = extract_llm_patterns(analyses)
            homemade_analysis = analyze_homemade_detection(analyses)
            improvements = generate_rule_improvements(patterns, disagreements)
            
            # Combine results
            analysis_results = {
                "summary": {
                    "total_analyses": len(analyses),
                    "total_disagreements": disagreements["total_disagreements"],
                    "disagreement_rate": disagreements["total_disagreements"] / len(analyses) * 100
                },
                "disagreements": disagreements,
                "patterns": patterns,
                "homemade_analysis": homemade_analysis,
                "suggested_improvements": improvements
            }
            
            # Save detailed analysis to run directory
            analysis_output_file = self.run_dir / "llm_reasoning_analysis.json"
            with open(analysis_output_file, 'w', encoding='utf-8') as f:
                json.dump(analysis_results, f, indent=2, default=str)
            
            logger.info(f"  ✅ Analysis completed: {disagreements['total_disagreements']} disagreements found")
            logger.info(f"  📊 Disagreement rate: {analysis_results['summary']['disagreement_rate']:.1f}%")
            
            return {
                "success": True,
                "results": analysis_results,
                "output_file": str(analysis_output_file)
            }
            
        except Exception as e:
            logger.error(f"  ❌ Analysis failed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _run_rule_update_stage(self, backup_existing: bool = True) -> Dict[str, Any]:
        """Update rules based on LLM analysis"""
        
        logger.info("  🔹 Updating rules based on LLM insights...")
        
        try:
            # Load the latest analysis results
            analysis_file = self.run_dir / "llm_reasoning_analysis.json"
            if not analysis_file.exists():
                raise FileNotFoundError("Analysis results not found for rule updates")
            
            with open(analysis_file, 'r') as f:
                analysis_results = json.load(f)
            
            # Create rule updater and apply updates
            rules_file_path = "Product_Weight_Project_Build/shelfscale/data_processing/enhanced_rules_curator.py"
            if not os.path.exists(rules_file_path):
                rules_file_path = "shelfscale/data_processing/enhanced_rules_curator.py"
            
            rule_updater = RuleUpdater(rules_file_path)
            update_results = rule_updater.update_rules_from_analysis(
                analysis_results, 
                backup=backup_existing
            )
            
            # Copy backup to run directory if created
            if update_results.get("backup_info", {}).get("backup_file"):
                backup_source = Path(update_results["backup_info"]["backup_file"])
                if backup_source.exists():
                    backup_dest = self.run_dir / backup_source.name
                    shutil.copy2(backup_source, backup_dest)
                    update_results["backup_info"]["run_backup_file"] = str(backup_dest)
            
            if update_results["success"]:
                logger.info(f"  ✅ Rules updated successfully")
                logger.info(f"  📝 Exclude patterns added: {update_results['updates_applied']['new_exclude_patterns']}")
                logger.info(f"  📝 Include patterns added: {update_results['updates_applied']['new_include_patterns']}")
                
                # Log specific patterns added
                details = update_results.get("details", {})
                if details.get("new_exclude_patterns"):
                    logger.info(f"  🔍 New exclude patterns: {details['new_exclude_patterns'][:3]}")
                if details.get("new_include_patterns"):
                    logger.info(f"  ➕ New include patterns: {details['new_include_patterns'][:3]}")
            else:
                logger.error(f"  ❌ Rule update failed: {update_results.get('error', 'Unknown error')}")
            
            return {
                "success": update_results["success"],
                "update_results": update_results
            }
            
        except Exception as e:
            logger.error(f"  ❌ Rule update failed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _generate_summary_report(self, pipeline_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a comprehensive summary report"""
        
        logger.info("  🔹 Generating summary report...")
        
        try:
            # Extract key metrics
            curation = pipeline_results["stages"].get("curation", {})
            analysis = pipeline_results["stages"].get("analysis", {})
            
            summary = {
                "pipeline_success": pipeline_results["overall_success"],
                "timestamp": self.timestamp,
                "run_directory": str(self.run_dir),
                "stages_completed": len([s for s in pipeline_results["stages"].values() if s.get("success", False)]),
                "total_stages": len(pipeline_results["stages"])
            }
            
            # Add curation metrics if available
            if curation.get("success") and "results" in curation:
                curation_results = curation["results"]
                summary["curation_metrics"] = {
                    "original_size": curation_results.get("original_size"),
                    "final_size": curation_results.get("final_size"),
                    "reduction_percentage": curation_results.get("reduction_percentage"),
                    "llm_decisions": curation_results.get("llm_decisions"),
                    "rules_decisions": curation_results.get("rules_decisions")
                }
            
            # Add analysis metrics if available
            if analysis.get("success") and "results" in analysis:
                analysis_results = analysis["results"]["summary"]
                summary["analysis_metrics"] = {
                    "total_disagreements": analysis_results.get("total_disagreements"),
                    "disagreement_rate": analysis_results.get("disagreement_rate"),
                    "total_analyses": analysis_results.get("total_analyses")
                }
            
            # Save summary report
            summary_file = self.run_dir / "summary_report.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, default=str)
            
            # Create human-readable summary
            readable_summary = self._create_readable_summary(summary)
            readable_file = self.run_dir / "summary_report.txt"
            with open(readable_file, 'w', encoding='utf-8') as f:
                f.write(readable_summary)
            
            logger.info("  ✅ Summary report generated")
            
            return {
                "success": True,
                "summary": summary,
                "summary_file": str(summary_file),
                "readable_file": str(readable_file)
            }
            
        except Exception as e:
            logger.error(f"  ❌ Summary generation failed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _create_readable_summary(self, summary: Dict[str, Any]) -> str:
        """Create a human-readable summary report"""
        
        report = f"""
🤖 AUTOMATED CURATION PIPELINE SUMMARY
============================================
📅 Run Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
📁 Run ID: {self.timestamp}
📂 Output Directory: {self.run_dir}

🎯 OVERALL RESULTS
------------------
✅ Pipeline Success: {'Yes' if summary['pipeline_success'] else 'No'}
🔄 Stages Completed: {summary['stages_completed']}/{summary['total_stages']}

"""
        
        if "curation_metrics" in summary:
            cm = summary["curation_metrics"]
            report += f"""📊 CURATION RESULTS
-------------------
📈 Original Dataset: {cm.get('original_size', 'N/A')} items
📉 Final Dataset: {cm.get('final_size', 'N/A')} items  
🎯 Reduction: {cm.get('reduction_percentage', 'N/A'):.1f}%
🧠 LLM Decisions: {cm.get('llm_decisions', 'N/A')}
📋 Rules Decisions: {cm.get('rules_decisions', 'N/A')}

"""
        
        if "analysis_metrics" in summary:
            am = summary["analysis_metrics"]
            report += f"""🔍 ANALYSIS RESULTS
-------------------
🔄 Total Analyses: {am.get('total_analyses', 'N/A')}
⚖️ Disagreements: {am.get('total_disagreements', 'N/A')}
📊 Disagreement Rate: {am.get('disagreement_rate', 'N/A'):.1f}%

"""
        
        report += f"""📁 OUTPUT FILES
---------------
All results saved to: {self.run_dir}
- pipeline_results.json (Complete results)
- summary_report.json (Summary metrics)
- llm_reasoning_analysis.json (Detailed analysis)
- curation_output/ (Curated datasets)

"""
        
        return report


def run_automated_pipeline(target_size: Optional[int] = None, 
                          update_rules: bool = True,
                          backup_existing: bool = True) -> Dict[str, Any]:
    """
    Run the complete automated curation pipeline
    
    Args:
        target_size: Target dataset size (None for automatic)
        update_rules: Whether to update rules based on LLM analysis
        backup_existing: Whether to backup existing rules before updating
        
    Returns:
        Pipeline results summary
    """
    
    pipeline = AutomatedCurationPipeline()
    return pipeline.run_full_pipeline(
        target_size=target_size,
        update_rules=update_rules,
        backup_existing=backup_existing
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run automated curation pipeline")
    parser.add_argument("--target-size", type=int, help="Target dataset size")
    parser.add_argument("--no-rule-updates", action="store_true", 
                       help="Skip rule updates")
    parser.add_argument("--no-backup", action="store_true",
                       help="Skip backing up existing rules")
    
    args = parser.parse_args()
    
    try:
        results = run_automated_pipeline(
            target_size=args.target_size,
            update_rules=not args.no_rule_updates,
            backup_existing=not args.no_backup
        )
        
        if results["overall_success"]:
            print("\n🎉 Pipeline completed successfully!")
            print(f"📂 Results saved to: {results['run_directory']}")
        else:
            print("\n❌ Pipeline failed!")
            print(f"Error: {results.get('error', 'Unknown error')}")
            
    except KeyboardInterrupt:
        print("\n⏹️  Pipeline interrupted by user")
    except Exception as e:
        print(f"\n💥 Pipeline crashed: {str(e)}")
        raise 