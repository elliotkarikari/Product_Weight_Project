"""
McCance & Widdowson Workflow Configuration
==========================================

Centralized configuration for the complete McCance & Widdowson data curation workflow.
This includes settings for data loading, curation methods, analysis parameters, and output formats.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
from .config_manager import get_config


@dataclass
class WorkflowConfig:
    """
    Configuration class for McCance & Widdowson workflow
    """
    
    # Base paths
    workflow_root: str = "McCance_and_Widdowson_Workflow"
    data_source: str = "McCance & Widdowson Food Composition Dataset"
    
    # Data processing settings
    default_target_size: Optional[int] = None
    default_curation_method: str = "auto"  # "auto", "llm", "rules", "hybrid"
    enable_llm_fallback: bool = True
    
    # Analysis settings
    disagreement_threshold: float = 0.3  # Confidence difference threshold
    pattern_frequency_threshold: int = 3  # Minimum occurrences for pattern recognition
    max_exclude_patterns: int = 10  # Maximum new exclude patterns to add
    max_include_patterns: int = 5   # Maximum new include patterns to add
    
    # Rule update settings
    enable_automatic_rule_updates: bool = True
    backup_rules_before_update: bool = True
    rule_update_confidence_threshold: float = 0.7
    
    # LLM settings
    llm_model: str = "gpt-3.5-turbo"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 500
    llm_timeout_seconds: int = 30
    
    # Output settings
    save_detailed_analyses: bool = True
    save_intermediate_results: bool = True
    generate_summary_reports: bool = True
    
    # Logging settings
    log_level: str = "INFO"
    save_logs_to_file: bool = True
    log_file_prefix: str = "mw_workflow"
    
    def __post_init__(self):
        """Initialize derived settings"""
        self.base_config = get_config()
        
        # Set up output directories within the workflow folder
        workflow_dir = Path(__file__).parent.parent  # McCance_and_Widdowson_Workflow directory
        self.workflow_output_dir = workflow_dir / "output"
        self.pipeline_runs_dir = self.workflow_output_dir / "pipeline_runs"
        self.curation_output_dir = self.workflow_output_dir / "curation_output"
        self.analysis_output_dir = self.workflow_output_dir / "analysis_output"
        self.rule_backup_dir = self.workflow_output_dir / "rule_backups"
        
        # Create directories
        for directory in [self.workflow_output_dir, self.pipeline_runs_dir, 
                         self.curation_output_dir, self.analysis_output_dir, self.rule_backup_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def get_data_source_path(self) -> str:
        """Get the path to the McCance & Widdowson data source"""
        return self.base_config.MCCANCE_WIDDOWSON_PATH
    
    def get_sheet_name(self) -> str:
        """Get the sheet name for McCance & Widdowson data"""
        return self.base_config.MW_SHEET_NAME_FOR_MAIN_PY
    
    def is_llm_available(self) -> bool:
        """Check if LLM functionality is available"""
        return self.base_config.validate_openai_config()
    
    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration"""
        return {
            "model": self.llm_model,
            "temperature": self.llm_temperature,
            "max_tokens": self.llm_max_tokens,
            "timeout": self.llm_timeout_seconds
        }
    
    def get_curation_config(self) -> Dict[str, Any]:
        """Get curation configuration"""
        return {
            "method": self.default_curation_method,
            "target_size": self.default_target_size,
            "llm_fallback": self.enable_llm_fallback,
            "llm_available": self.is_llm_available()
        }
    
    def get_analysis_config(self) -> Dict[str, Any]:
        """Get analysis configuration"""
        return {
            "disagreement_threshold": self.disagreement_threshold,
            "pattern_frequency_threshold": self.pattern_frequency_threshold,
            "save_detailed": self.save_detailed_analyses
        }
    
    def get_rule_update_config(self) -> Dict[str, Any]:
        """Get rule update configuration"""
        return {
            "enabled": self.enable_automatic_rule_updates,
            "backup": self.backup_rules_before_update,
            "confidence_threshold": self.rule_update_confidence_threshold,
            "max_exclude_patterns": self.max_exclude_patterns,
            "max_include_patterns": self.max_include_patterns
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "workflow_info": {
                "root": self.workflow_root,
                "data_source": self.data_source,
                "version": "1.0.0"
            },
            "data_processing": self.get_curation_config(),
            "llm_settings": self.get_llm_config(),
            "analysis": self.get_analysis_config(),
            "rule_updates": self.get_rule_update_config(),
            "output": {
                "workflow_output_dir": str(self.workflow_output_dir),
                "save_detailed_analyses": self.save_detailed_analyses,
                "save_intermediate_results": self.save_intermediate_results,
                "generate_summary_reports": self.generate_summary_reports
            },
            "logging": {
                "level": self.log_level,
                "save_to_file": self.save_logs_to_file,
                "file_prefix": self.log_file_prefix
            }
        }


# Global workflow configuration instance
_workflow_config = None


def get_workflow_config() -> WorkflowConfig:
    """Get the global workflow configuration instance"""
    global _workflow_config
    if _workflow_config is None:
        _workflow_config = WorkflowConfig()
    return _workflow_config


def update_workflow_config(**kwargs) -> WorkflowConfig:
    """Update workflow configuration with new values"""
    global _workflow_config
    config = get_workflow_config()
    
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown configuration parameter: {key}")
    
    return config


def get_workflow_paths() -> Dict[str, Path]:
    """Get all workflow-related paths"""
    config = get_workflow_config()
    return {
        "workflow_output": config.workflow_output_dir,
        "pipeline_runs": config.pipeline_runs_dir,
        "curation_output": config.curation_output_dir,
        "analysis_output": config.analysis_output_dir,
        "rule_backups": config.rule_backup_dir
    }


# Convenience functions for specific configurations
def configure_for_demo(llm_enabled: bool = True):
    """Configure workflow for demo purposes"""
    return update_workflow_config(
        default_target_size=50,
        default_curation_method="hybrid" if llm_enabled else "rules",
        enable_llm_fallback=llm_enabled,
        save_detailed_analyses=True,
        generate_summary_reports=True
    )


def configure_for_production(target_size: Optional[int] = None):
    """Configure workflow for production use"""
    return update_workflow_config(
        default_target_size=target_size,
        default_curation_method="auto",
        enable_llm_fallback=True,
        enable_automatic_rule_updates=True,
        save_detailed_analyses=True,
        save_intermediate_results=True,
        generate_summary_reports=True
    )


def configure_for_research(save_everything: bool = True):
    """Configure workflow for research purposes"""
    return update_workflow_config(
        default_curation_method="hybrid",
        enable_llm_fallback=True,
        save_detailed_analyses=save_everything,
        save_intermediate_results=save_everything,
        generate_summary_reports=True,
        disagreement_threshold=0.2,  # More sensitive to disagreements
        pattern_frequency_threshold=2  # Lower threshold for pattern detection
    ) 