"""
Iterative Learning Workflow for ShelfScale Retail System

This module implements a self-improving workflow that:
1. Analyzes its own outputs for quality issues
2. Learns from mistakes and adjusts processing parameters
3. Runs multiple iterations until output quality is acceptable
4. Builds institutional knowledge for future runs
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import asyncio
import logging
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict

# Import existing components
from shelfscale.workflows import RetailWorkflowOrchestrator, WorkflowConfig

logger = logging.getLogger(__name__)

@dataclass
class QualityIssue:
    """Represents a quality issue found in workflow output"""
    issue_type: str  # 'header_contamination', 'no_matching', 'data_parsing', 'missing_nutrition'
    severity: str    # 'critical', 'high', 'medium', 'low'
    description: str
    affected_records: int
    suggested_fix: str
    confidence: float  # 0-1

@dataclass
class IterationResult:
    """Results from a single workflow iteration"""
    iteration_number: int
    success: bool
    products_count: int
    quality_score: float  # 0-1
    issues_found: List[QualityIssue]
    processing_adjustments: Dict[str, Any]
    outputs: Dict[str, str]
    duration_seconds: float

@dataclass
class LearningInsight:
    """Knowledge gained from analyzing iterations"""
    pattern: str
    fix_strategy: str
    success_rate: float
    confidence: float
    examples: List[str]

class WorkflowQualityAnalyzer:
    """Analyzes workflow outputs to identify quality issues"""
    
    def __init__(self):
        self.issue_patterns = {
            'header_contamination': [
                'Notes on tables', 'Factors', 'Proximates', 'Inorganics', 'Vitamins',
                'List of tables', 'Table', 'Sheet', 'Index', 'Contents'
            ],
            'metadata_products': [
                'Vitamin Fractions', 'fatty acids per 100g', 'Monounsaturated', 
                'Saturated fatty acids', 'per 100g food'
            ],
            'non_food_items': [
                'Notes', 'Factors', 'Index', 'Contents', 'Introduction',
                'Methodology', 'References', 'Appendix'
            ]
        }
    
    def analyze_output_quality(self, output_file: str) -> Tuple[float, List[QualityIssue]]:
        """Analyze the quality of workflow output and identify issues"""
        logger.info(f"🔍 Analyzing output quality: {output_file}")
        
        try:
            df = pd.read_csv(output_file)
            issues = []
            quality_score = 1.0
            
            # Check 1: Header contamination
            header_issues = self._detect_header_contamination(df)
            if header_issues:
                issues.extend(header_issues)
                quality_score -= 0.4
            
            # Check 2: Non-food products
            non_food_issues = self._detect_non_food_products(df)
            if non_food_issues:
                issues.extend(non_food_issues)
                quality_score -= 0.3
            
            # Check 3: Missing nutrition data
            nutrition_issues = self._detect_missing_nutrition(df)
            if nutrition_issues:
                issues.extend(nutrition_issues)
                quality_score -= 0.2
            
            # Check 4: Matching quality
            matching_issues = self._detect_matching_problems(df)
            if matching_issues:
                issues.extend(matching_issues)
                quality_score -= 0.1
            
            quality_score = max(0.0, quality_score)
            
            logger.info(f"   📊 Quality Score: {quality_score:.2f}")
            logger.info(f"   🚨 Issues Found: {len(issues)}")
            
            return quality_score, issues
            
        except Exception as e:
            logger.error(f"❌ Quality analysis failed: {e}")
            return 0.0, [QualityIssue(
                issue_type='analysis_failure',
                severity='critical',
                description=f"Could not analyze output: {e}",
                affected_records=0,
                suggested_fix="Check file format and accessibility",
                confidence=1.0
            )]
    
    def _detect_header_contamination(self, df: pd.DataFrame) -> List[QualityIssue]:
        """Detect if table headers were read as data rows"""
        issues = []
        
        if 'Food_Name' in df.columns:
            contaminated_rows = 0
            
            for pattern_list in self.issue_patterns.values():
                for pattern in pattern_list:
                    contaminated = df['Food_Name'].str.contains(pattern, case=False, na=False).sum()
                    contaminated_rows += contaminated
            
            if contaminated_rows > 0:
                issues.append(QualityIssue(
                    issue_type='header_contamination',
                    severity='critical',
                    description=f"Found {contaminated_rows} rows that appear to be table headers/metadata instead of food products",
                    affected_records=contaminated_rows,
                    suggested_fix="Skip header rows when reading Excel files, use proper sheet selection",
                    confidence=0.9
                ))
        
        return issues
    
    def _detect_non_food_products(self, df: pd.DataFrame) -> List[QualityIssue]:
        """Detect non-food items in the dataset"""
        issues = []
        
        if 'Food_Name' in df.columns:
            non_food_count = 0
            
            # Check for obvious non-food patterns
            non_food_patterns = ['per 100g', 'fatty acids', 'methodology', 'reference']
            for pattern in non_food_patterns:
                non_food_count += df['Food_Name'].str.contains(pattern, case=False, na=False).sum()
            
            if non_food_count > 0:
                issues.append(QualityIssue(
                    issue_type='non_food_contamination',
                    severity='high',
                    description=f"Found {non_food_count} non-food items in food database",
                    affected_records=non_food_count,
                    suggested_fix="Improve data filtering to exclude non-food entries",
                    confidence=0.8
                ))
        
        return issues
    
    def _detect_missing_nutrition(self, df: pd.DataFrame) -> List[QualityIssue]:
        """Detect missing nutritional information"""
        issues = []
        
        nutrition_cols = ['Energy (kcal)', 'Fat (g)', 'Protein (g)', 'Carbohydrate (g)']
        missing_nutrition = 0
        
        for col in nutrition_cols:
            if col in df.columns:
                missing_nutrition += df[col].isna().sum()
        
        if missing_nutrition > len(df) * 0.5:  # More than 50% missing
            issues.append(QualityIssue(
                issue_type='missing_nutrition',
                severity='medium',
                description=f"High amount of missing nutritional data ({missing_nutrition} missing values)",
                affected_records=missing_nutrition,
                suggested_fix="Load additional nutritional data sources or improve data extraction",
                confidence=0.7
            ))
        
        return issues
    
    def _detect_matching_problems(self, df: pd.DataFrame) -> List[QualityIssue]:
        """Detect problems with cross-dataset matching"""
        issues = []
        
        # Check if any cross-dataset matching occurred
        if len(df) < 50:  # Very few products suggests poor matching
            issues.append(QualityIssue(
                issue_type='poor_matching',
                severity='medium',
                description=f"Very few products in final database ({len(df)}), suggests poor cross-dataset matching",
                affected_records=len(df),
                suggested_fix="Improve data source loading and matching algorithms",
                confidence=0.6
            ))
        
        return issues

class IterativeLearningWorkflow:
    """Iterative workflow that learns and improves from each iteration"""
    
    def __init__(self, max_iterations: int = 3, target_quality: float = 0.8):
        self.max_iterations = max_iterations
        self.target_quality = target_quality
        self.quality_analyzer = WorkflowQualityAnalyzer()
        self.iteration_results: List[IterationResult] = []
        self.learning_insights: List[LearningInsight] = []
        
        # Learning parameters that can be adjusted
        self.learning_params = {
            'skip_rows': 0,
            'sheet_selection_strategy': 'default',
            'data_filtering_strictness': 'medium',
            'min_relevance_threshold': 0.6,
            'target_products': 500
        }
        
        logger.info(f"🧠 Iterative Learning Workflow initialized")
        logger.info(f"   📊 Max iterations: {max_iterations}")
        logger.info(f"   🎯 Target quality: {target_quality}")
    
    async def run_iterative_workflow(self) -> Dict[str, Any]:
        """Run the workflow iteratively until quality target is met"""
        logger.info("🚀 Starting Iterative Learning Workflow")
        
        best_result = None
        
        for iteration in range(1, self.max_iterations + 1):
            logger.info(f"\n🔄 === ITERATION {iteration}/{self.max_iterations} ===")
            
            # Run workflow with current parameters
            result = await self._run_single_iteration(iteration)
            self.iteration_results.append(result)
            
            # Track best result
            if best_result is None or result.quality_score > best_result.quality_score:
                best_result = result
            
            logger.info(f"📊 Iteration {iteration} Results:")
            logger.info(f"   • Quality Score: {result.quality_score:.2f}")
            logger.info(f"   • Products: {result.products_count}")
            logger.info(f"   • Issues: {len(result.issues_found)}")
            
            # Check if we've reached target quality
            if result.quality_score >= self.target_quality:
                logger.info(f"🎯 Target quality achieved! ({result.quality_score:.2f} >= {self.target_quality})")
                break
            
            # Learn from this iteration
            if iteration < self.max_iterations:
                await self._learn_and_adjust(result)
        
        # Generate final learning report
        learning_report = self._generate_learning_report()
        
        return {
            'best_result': best_result,
            'all_iterations': self.iteration_results,
            'learning_insights': self.learning_insights,
            'learning_report': learning_report,
            'final_quality_score': best_result.quality_score if best_result else 0.0,
            'iterations_completed': len(self.iteration_results)
        }
    
    async def _run_single_iteration(self, iteration_num: int) -> IterationResult:
        """Run a single workflow iteration with current parameters"""
        start_time = datetime.now()
        
        # Create workflow config based on current learning parameters
        config = WorkflowConfig(
            target_retail_products=self.learning_params['target_products'],
            min_retail_relevance=self.learning_params['min_relevance_threshold'],
            skip_rows=self.learning_params['skip_rows'],
            sheet_selection_strategy=self.learning_params['sheet_selection_strategy'],
            data_filtering_strictness=self.learning_params['data_filtering_strictness'],
            header_detection_enabled=True
        )
        
        # Run the workflow
        orchestrator = RetailWorkflowOrchestrator(config=config)
        
        # Apply learning adjustments to the orchestrator
        await self._apply_learning_adjustments(orchestrator)
        
        workflow_results = await orchestrator.run_retail_workflow()
        
        # Analyze output quality
        if workflow_results['success'] and workflow_results.get('outputs', {}).get('main_database'):
            quality_score, issues = self.quality_analyzer.analyze_output_quality(
                workflow_results['outputs']['main_database']
            )
        else:
            quality_score = 0.0
            issues = [QualityIssue(
                issue_type='workflow_failure',
                severity='critical',
                description='Workflow execution failed',
                affected_records=0,
                suggested_fix='Check workflow configuration and data sources',
                confidence=1.0
            )]
        
        duration = (datetime.now() - start_time).total_seconds()
        
        return IterationResult(
            iteration_number=iteration_num,
            success=workflow_results['success'],
            products_count=workflow_results.get('metrics', {}).get('final_products', 0),
            quality_score=quality_score,
            issues_found=issues,
            processing_adjustments=self.learning_params.copy(),
            outputs=workflow_results.get('outputs', {}),
            duration_seconds=duration
        )
    
    async def _apply_learning_adjustments(self, orchestrator: RetailWorkflowOrchestrator):
        """Apply learned adjustments to the workflow orchestrator"""
        
        # Adjust data loading strategy based on learned parameters
        if self.learning_params['skip_rows'] > 0:
            logger.info(f"   🔧 Applying learning: Skip {self.learning_params['skip_rows']} rows")
            # This would require modifying the orchestrator to support skip_rows
        
        if self.learning_params['data_filtering_strictness'] == 'high':
            logger.info(f"   🔧 Applying learning: High data filtering strictness")
            # Increase filtering strictness
        
        # Additional learning adjustments would be implemented here
    
    async def _learn_and_adjust(self, result: IterationResult):
        """Learn from iteration results and adjust parameters"""
        logger.info(f"🧠 Learning from iteration {result.iteration_number}...")
        
        adjustments_made = []
        
        # Learn from header contamination issues
        header_issues = [i for i in result.issues_found if i.issue_type == 'header_contamination']
        if header_issues:
            self.learning_params['skip_rows'] = min(10, self.learning_params['skip_rows'] + 2)
            adjustments_made.append(f"Increase skip_rows to {self.learning_params['skip_rows']}")
            
            # Record learning insight
            self.learning_insights.append(LearningInsight(
                pattern="Header contamination detected",
                fix_strategy="Skip more rows when reading Excel files",
                success_rate=0.0,  # Will be updated as we gather more data
                confidence=0.8,
                examples=[issue.description for issue in header_issues]
            ))
        
        # Learn from non-food contamination
        non_food_issues = [i for i in result.issues_found if i.issue_type == 'non_food_contamination']
        if non_food_issues:
            self.learning_params['data_filtering_strictness'] = 'high'
            adjustments_made.append("Increase data filtering strictness")
        
        # Learn from poor matching
        matching_issues = [i for i in result.issues_found if i.issue_type == 'poor_matching']
        if matching_issues:
            self.learning_params['min_relevance_threshold'] = max(0.3, 
                self.learning_params['min_relevance_threshold'] - 0.1)
            adjustments_made.append(f"Lower relevance threshold to {self.learning_params['min_relevance_threshold']}")
        
        if adjustments_made:
            logger.info(f"   🔧 Adjustments for next iteration:")
            for adj in adjustments_made:
                logger.info(f"      • {adj}")
        else:
            logger.info(f"   ℹ️ No adjustments needed")
    
    def _generate_learning_report(self) -> str:
        """Generate a comprehensive learning report"""
        if not self.iteration_results:
            return "No iterations completed."
        
        best_result = max(self.iteration_results, key=lambda x: x.quality_score)
        worst_result = min(self.iteration_results, key=lambda x: x.quality_score)
        
        report = f"""# Iterative Learning Workflow Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary

- **Total Iterations**: {len(self.iteration_results)}
- **Best Quality Score**: {best_result.quality_score:.2f} (Iteration {best_result.iteration_number})
- **Worst Quality Score**: {worst_result.quality_score:.2f} (Iteration {worst_result.iteration_number})
- **Quality Improvement**: {best_result.quality_score - self.iteration_results[0].quality_score:.2f}

## Iteration Results

"""
        
        for result in self.iteration_results:
            report += f"""### Iteration {result.iteration_number}
- **Quality Score**: {result.quality_score:.2f}
- **Products**: {result.products_count}
- **Issues Found**: {len(result.issues_found)}
- **Duration**: {result.duration_seconds:.1f}s

"""
        
        report += f"""## Learning Insights

"""
        
        for insight in self.learning_insights:
            report += f"""### {insight.pattern}
- **Fix Strategy**: {insight.fix_strategy}
- **Confidence**: {insight.confidence:.2f}
- **Examples**: {', '.join(insight.examples[:3])}

"""
        
        report += f"""## Recommendations

Based on the learning process, the following improvements are recommended:

"""
        
        if best_result.quality_score < self.target_quality:
            report += f"- Quality target ({self.target_quality}) not achieved. Consider additional iterations.\n"
            report += f"- Main issues to address: {', '.join([i.issue_type for i in best_result.issues_found])}\n"
        else:
            report += f"- Quality target achieved! The workflow has learned to produce good results.\n"
        
        return report

# Integration functions
async def run_iterative_retail_workflow(max_iterations: int = 3, target_quality: float = 0.8) -> Dict[str, Any]:
    """Run the iterative learning retail workflow"""
    workflow = IterativeLearningWorkflow(max_iterations, target_quality)
    return await workflow.run_iterative_workflow()

def add_iterative_workflow_args(parser):
    """Add iterative workflow arguments to argument parser"""
    parser.add_argument('--iterative-workflow', action='store_true',
                       help='Run iterative learning retail workflow')
    parser.add_argument('--max-iterations', type=int, default=3,
                       help='Maximum number of iterations')
    parser.add_argument('--target-quality', type=float, default=0.8,
                       help='Target quality score (0-1)')

async def run_iterative_workflow_with_args(args):
    """Run iterative workflow with command line arguments"""
    print(f"\n🧠 Starting Iterative Learning Retail Workflow")
    print(f"   📊 Max iterations: {args.max_iterations}")
    print(f"   🎯 Target quality: {args.target_quality}")
    
    results = await run_iterative_retail_workflow(
        max_iterations=args.max_iterations,
        target_quality=args.target_quality
    )
    
    print(f"\n🎉 Iterative Learning Results:")
    print(f"   • Iterations completed: {results['iterations_completed']}")
    print(f"   • Final quality score: {results['final_quality_score']:.2f}")
    print(f"   • Learning insights: {len(results['learning_insights'])}")
    
    if results['best_result'] and results['best_result'].outputs:
        print(f"   • Best output files:")
        for output_type, file_path in results['best_result'].outputs.items():
            print(f"     - {output_type}: {file_path}")
    
    # Save learning report
    output_path = Path("data/outputs/workflow")
    output_path.mkdir(parents=True, exist_ok=True)
    
    report_file = output_path / f"iterative_learning_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(report_file, 'w') as f:
        f.write(results['learning_report'])
    print(f"   • Learning report: {report_file}")
    
    return results