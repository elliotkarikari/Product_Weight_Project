"""
McCance & Widdowson Workflow Package
====================================

This package contains the complete workflow for processing McCance & Widdowson food composition data
through intelligent curation, analysis, and rule improvement cycles.

Core Components:
- Automated Curation Pipeline: Orchestrates the entire workflow
- Hybrid Curator: Combines LLM and rule-based curation
- LLM Reasoning Analysis: Analyzes LLM decisions for patterns
- Rule Updater: Automatically improves rules based on LLM insights

Workflow Architecture:
1. Data Loading: McCance & Widdowson data
2. Hybrid Curation: Rules + selective LLM verification
3. Reasoning Analysis: Extract patterns from LLM decisions
4. Rule Updates: Automatically improve curation rules
5. Continuous Improvement: Iterative enhancement cycle
"""

__version__ = "1.0.0"
__author__ = "Product Weight Project Team"

# Core workflow components
from .core_components.automated_pipeline import AutomatedCurationPipeline, run_automated_pipeline
from .data_processing.hybrid_curator import HybridCurator, run_hybrid_curation_sync
from .analysis.llm_reasoning_analyzer import (
    LLMReasoningAnalyzer,
    load_hybrid_analyses,
    analyze_disagreements,
    extract_llm_patterns
)
from .utils.rule_updater import RuleUpdater

# Configuration
from .config.workflow_config import WorkflowConfig

__all__ = [
    "AutomatedCurationPipeline",
    "run_automated_pipeline", 
    "HybridCurator",
    "run_hybrid_curation_sync",
    "LLMReasoningAnalyzer",
    "RuleUpdater",
    "WorkflowConfig"
]


def run_complete_workflow(target_size=None, update_rules=True, backup_existing=True):
    """
    Run the complete McCance & Widdowson workflow
    
    Args:
        target_size: Target dataset size (None for automatic)
        update_rules: Whether to update rules based on analysis
        backup_existing: Whether to backup existing rules
        
    Returns:
        Complete workflow results
    """
    print("🍎 MCCANCE & WIDDOWSON COMPLETE WORKFLOW")
    print("=" * 60)
    
    return run_automated_pipeline(
        target_size=target_size,
        update_rules=update_rules,
        backup_existing=backup_existing
    )


def get_workflow_info():
    """Get information about the McCance & Widdowson workflow"""
    return {
        "name": "McCance & Widdowson Curation Workflow",
        "version": __version__,
        "description": "Intelligent food composition data curation with continuous improvement",
        "components": [
            "Automated Pipeline Orchestration",
            "Hybrid LLM + Rules Curation", 
            "LLM Reasoning Pattern Analysis",
            "Automated Rule Improvement",
            "Continuous Learning Loop"
        ],
        "data_source": "McCance & Widdowson Food Composition Dataset",
        "output_formats": ["CSV datasets", "JSON reports", "Analysis summaries"]
    } 