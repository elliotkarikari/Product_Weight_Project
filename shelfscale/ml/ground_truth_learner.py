"""
Ground Truth Learning System for ShelfScale

This module uses existing matched data from the output folder to continuously
improve the matching system through active learning and feedback loops.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
from datetime import datetime

from .matching_engine import AIMatchingEngine
from .text_preprocessor import EnhancedTextPreprocessor
from ..utils.logging_config import get_logger, monitor_performance

logger = get_logger(__name__)


class GroundTruthLearner:
    """
    Active learning system that uses existing ground truth data to improve
    the matching algorithm over time
    """
    
    def __init__(self, output_dir: str = "output", 
                 feedback_threshold: float = 0.8,
                 learning_rate: float = 0.1):
        """
        Initialize ground truth learner
        
        Args:
            output_dir: Directory containing ground truth data
            feedback_threshold: Confidence threshold for accepting automatic feedback
            learning_rate: Rate of model adaptation
        """
        self.output_dir = Path(output_dir)
        self.feedback_threshold = feedback_threshold
        self.learning_rate = learning_rate
        
        # Initialize components
        self.matching_engine = None
        self.preprocessor = EnhancedTextPreprocessor()
        
        # Learning statistics
        self.learning_stats = {
            'ground_truth_loaded': 0,
            'positive_examples': 0,
            'negative_examples': 0,
            'model_updates': 0,
            'performance_improvements': 0
        }
        
    @monitor_performance("ground_truth_loading")
    def load_existing_ground_truth(self) -> Dict[str, pd.DataFrame]:
        """
        Load all existing ground truth data from output directory
        
        Returns:
            Dictionary of ground truth datasets
        """
        logger.info("Loading existing ground truth data")
        
        ground_truth_data = {}
        
        # Expected ground truth files
        gt_files = {
            'mw_fps_matches': 'mw_fps_matches.csv',
            'mw_fvs_matches': 'mw_fvs_matches.csv',
            'consolidated_weights': 'consolidated_weights.csv',
            'processed_data': 'processed_data.csv'
        }
        
        for name, filename in gt_files.items():
            file_path = self.output_dir / filename
            
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path)
                    ground_truth_data[name] = df
                    logger.info(f"Loaded {name}: {len(df)} records")
                    self.learning_stats['ground_truth_loaded'] += len(df)
                    
                except Exception as e:
                    logger.error(f"Error loading {filename}: {e}")
            else:
                logger.warning(f"Ground truth file not found: {filename}")
                
        return ground_truth_data
        
    def analyze_ground_truth_quality(self, gt_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Analyze the quality of ground truth data
        
        Args:
            gt_data: Ground truth datasets
            
        Returns:
            Quality analysis report
        """
        logger.info("Analyzing ground truth data quality")
        
        analysis = {
            'dataset_sizes': {},
            'similarity_distributions': {},
            'match_rates': {},
            'data_completeness': {},
            'quality_score': 0.0
        }
        
        for name, df in gt_data.items():
            if df.empty:
                continue
                
            analysis['dataset_sizes'][name] = len(df)
            
            # Analyze similarity scores if available
            if 'Similarity_Score' in df.columns:
                sim_scores = pd.to_numeric(df['Similarity_Score'], errors='coerce')
                analysis['similarity_distributions'][name] = {
                    'mean': sim_scores.mean(),
                    'std': sim_scores.std(),
                    'high_confidence': (sim_scores >= 0.8).sum(),
                    'medium_confidence': ((sim_scores >= 0.5) & (sim_scores < 0.8)).sum(),
                    'low_confidence': (sim_scores < 0.5).sum()
                }
                
                # Calculate match rate
                analysis['match_rates'][name] = (sim_scores >= 0.5).mean()
                
            # Analyze data completeness
            completeness = {}
            for col in df.columns:
                completeness[col] = 1 - (df[col].isna().sum() / len(df))
            analysis['data_completeness'][name] = completeness
            
        # Calculate overall quality score
        total_records = sum(analysis['dataset_sizes'].values())
        avg_match_rate = np.mean(list(analysis['match_rates'].values())) if analysis['match_rates'] else 0
        
        analysis['quality_score'] = min(1.0, (total_records / 1000) * 0.5 + avg_match_rate * 0.5)
        
        logger.info(f"Ground truth quality score: {analysis['quality_score']:.3f}")
        return analysis
        
    @monitor_performance("training_data_preparation")
    def prepare_training_data_from_ground_truth(self, gt_data: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Prepare training data from ground truth matches
        
        Args:
            gt_data: Ground truth datasets
            
        Returns:
            Tuple of (features_df, labels_array)
        """
        logger.info("Preparing training data from ground truth")
        
        training_pairs = []
        labels = []
        
        # Process M&W to FPS matches
        if 'mw_fps_matches' in gt_data:
            mw_fps_df = gt_data['mw_fps_matches']
            
            for _, row in mw_fps_df.iterrows():
                source_name = row.get('Food Name_source', '')
                target_name = row.get('Food_Name_target', '')
                similarity = row.get('Similarity_Score', 0)
                
                if pd.notna(source_name) and pd.notna(target_name):
                    training_pairs.append((str(source_name), str(target_name)))
                    
                    # Convert similarity to binary label
                    label = 1 if similarity >= 0.5 else 0
                    labels.append(label)
                    
        # Process M&W to FVS matches
        if 'mw_fvs_matches' in gt_data:
            mw_fvs_df = gt_data['mw_fvs_matches']
            
            for _, row in mw_fvs_df.iterrows():
                source_name = row.get('Food Name_source', '')
                target_name = row.get('Food_Name_target', '')
                similarity = row.get('Similarity_Score', 0)
                
                if pd.notna(source_name) and pd.notna(target_name):
                    training_pairs.append((str(source_name), str(target_name)))
                    
                    label = 1 if similarity >= 0.5 else 0
                    labels.append(label)
                    
        # Generate negative examples
        negative_pairs = self._generate_negative_examples(training_pairs, ratio=0.3)
        training_pairs.extend(negative_pairs)
        labels.extend([0] * len(negative_pairs))
        
        # Extract features
        if training_pairs:
            from .feature_extractor import FeatureExtractor
            feature_extractor = FeatureExtractor(use_semantic_features=False)
            
            # Fit feature extractor
            all_texts = [pair[0] for pair in training_pairs] + [pair[1] for pair in training_pairs]
            feature_extractor.fit(all_texts)
            
            # Create feature matrix
            features_df = feature_extractor.create_feature_matrix(training_pairs)
            labels_array = np.array(labels)
            
            # Update statistics
            self.learning_stats['positive_examples'] = sum(labels)
            self.learning_stats['negative_examples'] = len(labels) - sum(labels)
            
            logger.info(f"Prepared {len(training_pairs)} training examples:")
            logger.info(f"  Positive: {sum(labels)} ({sum(labels)/len(labels)*100:.1f}%)")
            logger.info(f"  Negative: {len(labels) - sum(labels)} ({(len(labels) - sum(labels))/len(labels)*100:.1f}%)")
            
            return features_df, labels_array
            
        else:
            logger.warning("No valid training pairs found in ground truth data")
            return pd.DataFrame(), np.array([])
            
    def _generate_negative_examples(self, positive_pairs: List[Tuple[str, str]], 
                                   ratio: float = 0.3) -> List[Tuple[str, str]]:
        """
        Generate negative examples by randomly pairing dissimilar items
        
        Args:
            positive_pairs: List of positive training pairs
            ratio: Ratio of negative to positive examples
            
        Returns:
            List of negative example pairs
        """
        if not positive_pairs:
            return []
            
        # Extract unique items
        all_items = list(set([item for pair in positive_pairs for item in pair]))
        
        negative_count = int(len(positive_pairs) * ratio)
        negative_pairs = []
        
        # Create random pairs that are unlikely to match
        np.random.seed(42)  # For reproducibility
        
        for _ in range(negative_count):
            item1, item2 = np.random.choice(all_items, 2, replace=False)
            
            # Simple heuristic: if items share no common words, likely negative
            words1 = set(item1.lower().split())
            words2 = set(item2.lower().split())
            
            if len(words1 & words2) == 0:  # No common words
                negative_pairs.append((item1, item2))
                
        logger.info(f"Generated {len(negative_pairs)} negative examples")
        return negative_pairs
        
    @monitor_performance("model_improvement")
    def improve_matching_model(self, features_df: pd.DataFrame, 
                              labels: np.ndarray,
                              validation_split: float = 0.2) -> Dict[str, Any]:
        """
        Use ground truth data to improve the matching model
        
        Args:
            features_df: Feature matrix
            labels: Binary labels
            validation_split: Fraction for validation
            
        Returns:
            Improvement results
        """
        logger.info("Improving matching model with ground truth data")
        
        if features_df.empty or len(labels) == 0:
            logger.warning("No training data available for improvement")
            return {'success': False, 'reason': 'No training data'}
            
        # Initialize or get existing matching engine
        if self.matching_engine is None:
            self.matching_engine = AIMatchingEngine(use_ensemble=True)
            
        # Train the model
        training_results = self.matching_engine.train(features_df, labels)
        
        # Evaluate improvement
        evaluation_results = self.matching_engine.evaluate_model(features_df, labels)
        
        improvement_results = {
            'success': True,
            'training_samples': len(features_df),
            'positive_rate': np.mean(labels),
            'training_results': training_results,
            'evaluation_results': evaluation_results,
            'timestamp': datetime.now().isoformat()
        }
        
        # Update statistics
        self.learning_stats['model_updates'] += 1
        
        # Check for performance improvement
        avg_f1 = np.mean([r.get('f1', 0) for r in evaluation_results.values()])
        if avg_f1 > 0.7:  # Threshold for good performance
            self.learning_stats['performance_improvements'] += 1
            
        logger.info(f"Model improvement completed. Average F1: {avg_f1:.3f}")
        return improvement_results
        
    def create_feedback_loop(self, new_matches: pd.DataFrame,
                           confidence_threshold: float = None) -> Dict[str, Any]:
        """
        Create feedback loop for continuous learning from new matches
        
        Args:
            new_matches: New match results to learn from
            confidence_threshold: Threshold for accepting automatic feedback
            
        Returns:
            Feedback processing results
        """
        if confidence_threshold is None:
            confidence_threshold = self.feedback_threshold
            
        logger.info(f"Processing feedback from {len(new_matches)} new matches")
        
        feedback_data = {
            'high_confidence_matches': [],
            'uncertain_matches': [],
            'suggested_improvements': []
        }
        
        for _, match in new_matches.iterrows():
            confidence = match.get('confidence', 0)
            
            if confidence >= confidence_threshold:
                # High confidence - use for automatic learning
                feedback_data['high_confidence_matches'].append({
                    'text1': match.get('text1', ''),
                    'text2': match.get('text2', ''),
                    'confidence': confidence,
                    'auto_label': 1
                })
                
            elif 0.3 <= confidence < confidence_threshold:
                # Uncertain - flag for human review
                feedback_data['uncertain_matches'].append({
                    'text1': match.get('text1', ''),
                    'text2': match.get('text2', ''),
                    'confidence': confidence,
                    'needs_review': True
                })
                
        # Generate improvement suggestions
        if feedback_data['uncertain_matches']:
            feedback_data['suggested_improvements'] = [
                "Consider manual review of uncertain matches to improve training data",
                "Add category-specific matching rules for better accuracy",
                "Collect more ground truth data for underrepresented categories"
            ]
            
        logger.info(f"Feedback processed: {len(feedback_data['high_confidence_matches'])} high confidence, "
                   f"{len(feedback_data['uncertain_matches'])} uncertain")
        
        return feedback_data
        
    def save_learning_progress(self, filename: str = None):
        """Save learning progress and statistics"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"learning_progress_{timestamp}.json"
            
        progress_data = {
            'timestamp': datetime.now().isoformat(),
            'learning_stats': self.learning_stats,
            'model_info': {
                'is_trained': self.matching_engine.is_trained if self.matching_engine else False,
                'confidence_threshold': self.feedback_threshold
            }
        }
        
        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            json.dump(progress_data, f, indent=2)
            
        logger.info(f"Learning progress saved to {output_path}")
        
    def get_learning_summary(self) -> Dict[str, Any]:
        """Get summary of learning progress"""
        summary = {
            'learning_stats': self.learning_stats.copy(),
            'ground_truth_quality': 'Good' if self.learning_stats['ground_truth_loaded'] > 100 else 'Limited',
            'model_performance': 'Improved' if self.learning_stats['performance_improvements'] > 0 else 'Baseline',
            'recommendations': []
        }
        
        # Generate recommendations
        if self.learning_stats['ground_truth_loaded'] < 100:
            summary['recommendations'].append("Collect more ground truth data for better learning")
            
        if self.learning_stats['positive_examples'] < 50:
            summary['recommendations'].append("Increase positive examples for better model training")
            
        if self.learning_stats['model_updates'] == 0:
            summary['recommendations'].append("Initialize model training with available ground truth")
            
        return summary