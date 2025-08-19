"""
Model training and evaluation framework for ShelfScale

Provides comprehensive training pipeline with validation, hyperparameter tuning,
and performance evaluation capabilities.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

from .matching_engine import AIMatchingEngine
from ..utils.logging_config import get_logger, monitor_performance

logger = get_logger(__name__)


class ModelTrainer:
    """
    Comprehensive model training and evaluation system
    """
    
    def __init__(self, output_dir: str = "model_outputs"):
        """
        Initialize model trainer
        
        Args:
            output_dir: Directory for saving outputs
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Training history
        self.training_history = []
        
    @monitor_performance("model_training")
    def train_and_evaluate(self, df1: pd.DataFrame, df2: pd.DataFrame,
                          text_col1: str, text_col2: str,
                          category_col1: str = None, category_col2: str = None,
                          ground_truth_matches: List[Tuple[int, int]] = None,
                          test_size: float = 0.2,
                          hyperparameter_tuning: bool = True) -> Dict[str, Any]:
        """
        Complete training and evaluation pipeline
        
        Args:
            df1: First dataframe
            df2: Second dataframe
            text_col1: Text column in df1
            text_col2: Text column in df2
            category_col1: Category column in df1
            category_col2: Category column in df2
            ground_truth_matches: List of (idx1, idx2) tuples for known matches
            test_size: Fraction for test set
            hyperparameter_tuning: Whether to perform hyperparameter tuning
            
        Returns:
            Comprehensive evaluation results
        """
        logger.info("Starting comprehensive model training and evaluation")
        
        # Initialize matching engine
        matching_engine = AIMatchingEngine()
        
        # Prepare training data
        logger.info("Preparing training data...")
        feature_df, labels = matching_engine.prepare_training_data(
            df1, df2, text_col1, text_col2, category_col1, category_col2
        )
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            feature_df, labels, test_size=test_size, 
            stratify=labels, random_state=42
        )
        
        logger.info(f"Training set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        
        # Train models
        logger.info("Training models...")
        training_results = matching_engine.train(X_train, y_train)
        
        # Evaluate on test set
        logger.info("Evaluating on test set...")
        test_results = matching_engine.evaluate_model(X_test, y_test)
        
        # Hyperparameter tuning
        tuning_results = {}
        if hyperparameter_tuning:
            logger.info("Performing hyperparameter tuning...")
            tuning_results = self._hyperparameter_tuning(X_train, y_train)
            
        # Feature importance analysis
        feature_importance = self._analyze_feature_importance(matching_engine, training_results)
        
        # Generate visualizations
        visualization_paths = self._create_visualizations(
            matching_engine, X_test, y_test, test_results
        )
        
        # Comprehensive results
        results = {
            'timestamp': datetime.now().isoformat(),
            'training_results': training_results,
            'test_results': test_results,
            'tuning_results': tuning_results,
            'feature_importance': feature_importance,
            'visualization_paths': visualization_paths,
            'data_info': {
                'train_size': len(X_train),
                'test_size': len(X_test),
                'feature_count': len(feature_df.columns),
                'positive_rate': np.mean(labels)
            },
            'matching_engine': matching_engine
        }
        
        # Save results
        self._save_results(results)
        
        # Store in history
        self.training_history.append(results)
        
        logger.info("Training and evaluation completed successfully")
        return results
        
    def _hyperparameter_tuning(self, X_train: pd.DataFrame, y_train: np.ndarray) -> Dict[str, Any]:
        """Perform hyperparameter tuning"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        
        # Parameter grids for different models
        param_grids = {
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        }
        
        tuning_results = {}
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train.values)
        
        for model_name, param_grid in param_grids.items():
            logger.info(f"Tuning hyperparameters for {model_name}")
            
            if model_name == 'random_forest':
                base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
            else:
                continue
                
            # Grid search with cross-validation
            grid_search = GridSearchCV(
                base_model, param_grid,
                cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
                scoring='f1',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_scaled, y_train)
            
            tuning_results[model_name] = {
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'cv_results': grid_search.cv_results_
            }
            
            logger.info(f"Best {model_name} score: {grid_search.best_score_:.3f}")
            logger.info(f"Best {model_name} params: {grid_search.best_params_}")
            
        return tuning_results
        
    def _analyze_feature_importance(self, matching_engine: AIMatchingEngine, 
                                   training_results: Dict[str, Any]) -> Dict[str, List]:
        """Analyze feature importance across models"""
        feature_importance = {}
        
        for model_name, results in training_results.items():
            if 'feature_importance' in results:
                feature_importance[model_name] = results['feature_importance']
                
        return feature_importance
        
    def _create_visualizations(self, matching_engine: AIMatchingEngine,
                             X_test: pd.DataFrame, y_test: np.ndarray,
                             test_results: Dict[str, Any]) -> List[str]:
        """Create evaluation visualizations"""
        visualization_paths = []
        
        try:
            # 1. Confusion matrices
            for model_name, results in test_results.items():
                if 'confusion_matrix' in results:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    cm = np.array(results['confusion_matrix'])
                    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues')
                    ax.set_title(f'Confusion Matrix - {model_name}')
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('Actual')
                    
                    path = os.path.join(self.output_dir, f'confusion_matrix_{model_name}.png')
                    plt.savefig(path, dpi=300, bbox_inches='tight')
                    plt.close()
                    visualization_paths.append(path)
                    
            # 2. Feature importance plot
            if hasattr(matching_engine, 'feature_names') and matching_engine.feature_names:
                for model_name, model in matching_engine.models.items():
                    if hasattr(model, 'feature_importances_'):
                        importances = model.feature_importances_
                        indices = np.argsort(importances)[::-1][:20]  # Top 20
                        
                        fig, ax = plt.subplots(figsize=(12, 8))
                        feature_names = np.array(matching_engine.feature_names)
                        ax.barh(range(len(indices)), importances[indices])
                        ax.set_yticks(range(len(indices)))
                        ax.set_yticklabels(feature_names[indices])
                        ax.set_xlabel('Feature Importance')
                        ax.set_title(f'Top Feature Importances - {model_name}')
                        ax.invert_yaxis()
                        
                        path = os.path.join(self.output_dir, f'feature_importance_{model_name}.png')
                        plt.savefig(path, dpi=300, bbox_inches='tight')
                        plt.close()
                        visualization_paths.append(path)
                        
            # 3. Model comparison
            if len(test_results) > 1:
                metrics = ['accuracy', 'precision', 'recall', 'f1']
                model_names = list(test_results.keys())
                
                metric_values = {}
                for metric in metrics:
                    metric_values[metric] = [test_results[model][metric] for model in model_names]
                    
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                axes = axes.ravel()
                
                for i, metric in enumerate(metrics):
                    axes[i].bar(model_names, metric_values[metric])
                    axes[i].set_title(f'{metric.capitalize()} Comparison')
                    axes[i].set_ylabel(metric.capitalize())
                    axes[i].tick_params(axis='x', rotation=45)
                    
                plt.tight_layout()
                path = os.path.join(self.output_dir, 'model_comparison.png')
                plt.savefig(path, dpi=300, bbox_inches='tight')
                plt.close()
                visualization_paths.append(path)
                
        except Exception as e:
            logger.warning(f"Error creating visualizations: {e}")
            
        return visualization_paths
        
    def _save_results(self, results: Dict[str, Any]):
        """Save training results"""
        import json
        
        # Create serializable results (remove non-serializable objects)
        serializable_results = {}
        for key, value in results.items():
            if key != 'matching_engine':  # Skip the engine object
                try:
                    json.dumps(value)  # Test serializability
                    serializable_results[key] = value
                except (TypeError, ValueError):
                    serializable_results[key] = str(value)
                    
        # Save as JSON
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = os.path.join(self.output_dir, f'training_results_{timestamp}.json')
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
            
        logger.info(f"Results saved to {results_path}")
        
    def compare_with_baseline(self, results: Dict[str, Any], 
                            baseline_accuracy: float = 0.1898) -> Dict[str, Any]:
        """
        Compare new results with baseline (original 18.98% accuracy)
        
        Args:
            results: Training results
            baseline_accuracy: Baseline accuracy to compare against
            
        Returns:
            Comparison metrics
        """
        comparison = {}
        
        for model_name, model_results in results['test_results'].items():
            model_accuracy = model_results['accuracy']
            improvement = (model_accuracy - baseline_accuracy) / baseline_accuracy
            
            comparison[model_name] = {
                'baseline_accuracy': baseline_accuracy,
                'new_accuracy': model_accuracy,
                'improvement_ratio': improvement,
                'improvement_percentage': improvement * 100,
                'meets_target': model_accuracy >= 0.75  # 75% target
            }
            
        return comparison
        
    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of all training sessions"""
        if not self.training_history:
            return {"message": "No training sessions completed"}
            
        summary = {
            'total_sessions': len(self.training_history),
            'best_accuracy': 0,
            'best_f1': 0,
            'latest_session': None
        }
        
        for session in self.training_history:
            if session['test_results']:
                for model_name, results in session['test_results'].items():
                    if results['accuracy'] > summary['best_accuracy']:
                        summary['best_accuracy'] = results['accuracy']
                        summary['best_model'] = model_name
                        
                    if results['f1'] > summary['best_f1']:
                        summary['best_f1'] = results['f1']
                        
        summary['latest_session'] = self.training_history[-1]['timestamp']
        
        return summary