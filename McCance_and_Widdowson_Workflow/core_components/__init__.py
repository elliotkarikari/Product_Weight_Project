"""
Core Components Module
=====================

Contains the main pipeline orchestration components for the McCance & Widdowson workflow.
"""

from .automated_pipeline import AutomatedCurationPipeline, run_automated_pipeline

__all__ = ["AutomatedCurationPipeline", "run_automated_pipeline"] 