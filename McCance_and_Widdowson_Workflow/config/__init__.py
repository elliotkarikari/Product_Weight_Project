"""
Config Module
============

Contains configuration management for the McCance & Widdowson workflow.
"""

from .config_manager import get_config
from .workflow_config import (
    WorkflowConfig,
    get_workflow_config,
    update_workflow_config,
    configure_for_demo,
    configure_for_production,
    configure_for_research
)

__all__ = [
    "get_config",
    "WorkflowConfig",
    "get_workflow_config", 
    "update_workflow_config",
    "configure_for_demo",
    "configure_for_production", 
    "configure_for_research"
] 