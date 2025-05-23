#!/usr/bin/env python3
"""
McCance & Widdowson Workflow Runner
===================================

Main entry point for running the complete McCance & Widdowson data curation workflow.
This script provides both command-line and programmatic interfaces for the workflow.

Usage:
    python run_workflow.py --method auto --target-size 1000
    python run_workflow.py --demo
    python run_workflow.py --production
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Any

# Add the parent directory to sys.path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config.workflow_config import get_workflow_config, configure_for_demo, configure_for_production
from core_components.automated_pipeline import run_automated_pipeline


def setup_logging(config):
    """Setup logging for the workflow"""
    log_level = getattr(logging, config.log_level.upper())
    
    handlers = [logging.StreamHandler()]
    
    if config.save_logs_to_file:
        log_file = config.workflow_output_dir / f"{config.log_file_prefix}_run.log"
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def print_workflow_banner():
    """Print the workflow banner"""
    print("""
🍎 MCCANCE & WIDDOWSON CURATION WORKFLOW
========================================
Intelligent food composition data curation with continuous improvement

Components:
• Automated Pipeline Orchestration
• Hybrid LLM + Rules Curation
• LLM Reasoning Pattern Analysis  
• Automated Rule Improvement
• Continuous Learning Loop

Data Source: McCance & Widdowson Food Composition Dataset
Version: 1.0.0
""")


def run_demo_workflow(target_size: Optional[int] = 50) -> Dict[str, Any]:
    """Run a demo version of the workflow"""
    print("🎯 DEMO MODE: Running with limited dataset")
    print("=" * 50)
    
    # Configure for demo
    config = configure_for_demo(llm_enabled=True)
    if target_size:
        config.default_target_size = target_size
    
    setup_logging(config)
    
    return run_automated_pipeline(
        target_size=config.default_target_size,
        update_rules=config.enable_automatic_rule_updates,
        backup_existing=config.backup_rules_before_update
    )


def run_production_workflow(target_size: Optional[int] = None, 
                           method: str = "auto") -> Dict[str, Any]:
    """Run the production version of the workflow"""
    print("🚀 PRODUCTION MODE: Running full workflow")
    print("=" * 50)
    
    # Configure for production
    config = configure_for_production(target_size=target_size)
    config.default_curation_method = method
    
    setup_logging(config)
    
    return run_automated_pipeline(
        target_size=config.default_target_size,
        update_rules=config.enable_automatic_rule_updates,
        backup_existing=config.backup_rules_before_update
    )


def run_interactive_workflow():
    """Run the workflow with interactive configuration"""
    print("🔧 INTERACTIVE MODE: Configure your workflow")
    print("=" * 50)
    
    # Get user preferences
    print("\n📋 Curation Method:")
    print("1. auto - Automatically choose best method")
    print("2. rules - Rules-based only (fast, free)")
    print("3. llm - LLM-based only (thorough, API costs)")
    print("4. hybrid - Combine rules + selective LLM (balanced)")
    
    method_choice = input("Choose method (1-4): ").strip()
    method_map = {'1': 'auto', '2': 'rules', '3': 'llm', '4': 'hybrid'}
    method = method_map.get(method_choice, 'auto')
    
    target_size_input = input("\n🎯 Target dataset size (press Enter for auto): ").strip()
    target_size = int(target_size_input) if target_size_input else None
    
    update_rules_input = input("\n⚡ Update rules automatically? (Y/n): ").strip().lower()
    update_rules = update_rules_input != 'n'
    
    backup_input = input("💾 Backup existing rules? (Y/n): ").strip().lower()  
    backup_existing = backup_input != 'n'
    
    # Configure and run
    config = get_workflow_config()
    config.default_curation_method = method
    config.default_target_size = target_size
    config.enable_automatic_rule_updates = update_rules
    config.backup_rules_before_update = backup_existing
    
    setup_logging(config)
    
    print(f"\n🔄 Running workflow with:")
    print(f"  Method: {method}")
    print(f"  Target size: {target_size or 'auto'}")
    print(f"  Update rules: {update_rules}")
    print(f"  Backup rules: {backup_existing}")
    
    return run_automated_pipeline(
        target_size=target_size,
        update_rules=update_rules,
        backup_existing=backup_existing
    )


def get_workflow_status() -> Dict[str, Any]:
    """Get the status of the workflow system"""
    config = get_workflow_config()
    
    # Check if LLM is available
    llm_available = config.is_llm_available()
    
    # Check data source
    data_path = config.get_data_source_path()
    data_available = Path(data_path).exists() if data_path else False
    
    # Check output directories
    output_dirs = {
        name: path.exists() for name, path in {
            "workflow_output": config.workflow_output_dir,
            "pipeline_runs": config.pipeline_runs_dir,
            "curation_output": config.curation_output_dir,
            "analysis_output": config.analysis_output_dir
        }.items()
    }
    
    return {
        "workflow_ready": data_available,
        "llm_available": llm_available,
        "data_source": {
            "path": data_path,
            "available": data_available
        },
        "output_directories": output_dirs,
        "configuration": config.to_dict()
    }


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="McCance & Widdowson Curation Workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_workflow.py --demo
  python run_workflow.py --production --method hybrid --target-size 1000
  python run_workflow.py --interactive
  python run_workflow.py --status
        """
    )
    
    # Workflow modes
    parser.add_argument("--demo", action="store_true",
                       help="Run in demo mode with limited dataset")
    parser.add_argument("--production", action="store_true",
                       help="Run in production mode with full dataset")
    parser.add_argument("--interactive", action="store_true",
                       help="Run in interactive mode with user configuration")
    parser.add_argument("--status", action="store_true",
                       help="Show workflow system status")
    
    # Configuration options
    parser.add_argument("--method", choices=["auto", "rules", "llm", "hybrid"],
                       default="auto", help="Curation method to use")
    parser.add_argument("--target-size", type=int,
                       help="Target size for final dataset")
    parser.add_argument("--no-rule-updates", action="store_true",
                       help="Skip automatic rule updates")
    parser.add_argument("--no-backup", action="store_true",
                       help="Skip backing up existing rules")
    
    args = parser.parse_args()
    
    # Show banner
    print_workflow_banner()
    
    try:
        if args.status:
            # Show status
            status = get_workflow_status()
            print("📊 WORKFLOW STATUS")
            print("=" * 30)
            print(f"✅ Workflow Ready: {'Yes' if status['workflow_ready'] else 'No'}")
            print(f"🧠 LLM Available: {'Yes' if status['llm_available'] else 'No'}")
            print(f"📁 Data Source: {status['data_source']['path']}")
            print(f"📂 Data Available: {'Yes' if status['data_source']['available'] else 'No'}")
            
            print("\n📁 Output Directories:")
            for name, exists in status['output_directories'].items():
                print(f"  {name}: {'✅' if exists else '❌'}")
            
            return
        
        elif args.demo:
            # Run demo
            results = run_demo_workflow(args.target_size or 50)
            
        elif args.production:
            # Run production
            results = run_production_workflow(args.target_size, args.method)
            
        elif args.interactive:
            # Run interactive
            results = run_interactive_workflow()
            
        else:
            # Default: run with command line arguments
            config = get_workflow_config()
            config.default_curation_method = args.method
            if args.target_size:
                config.default_target_size = args.target_size
            
            setup_logging(config)
            
            results = run_automated_pipeline(
                target_size=args.target_size,
                update_rules=not args.no_rule_updates,
                backup_existing=not args.no_backup
            )
        
        # Show results summary
        if results and results.get("overall_success"):
            print("\n🎉 WORKFLOW COMPLETED SUCCESSFULLY!")
            print("=" * 40)
            print(f"📂 Results directory: {results.get('run_directory', 'N/A')}")
            
            # Show key metrics
            curation = results.get("stages", {}).get("curation", {})
            if curation.get("success") and "results" in curation:
                metrics = curation["results"]
                print(f"📊 Original size: {metrics.get('original_size', 'N/A')}")
                print(f"📉 Final size: {metrics.get('final_size', 'N/A')}")
                print(f"🎯 Reduction: {metrics.get('reduction_percentage', 'N/A'):.1f}%")
        else:
            print("\n❌ WORKFLOW FAILED!")
            if results:
                print(f"Error: {results.get('error', 'Unknown error')}")
    
    except KeyboardInterrupt:
        print("\n⏹️  Workflow interrupted by user")
    except Exception as e:
        print(f"\n💥 Workflow crashed: {str(e)}")
        raise


if __name__ == "__main__":
    main() 