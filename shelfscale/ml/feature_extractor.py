"""
Feature extraction for AI-powered food product matching

This module creates numerical features from food product data for machine learning
algorithms, including semantic similarity features, text statistics, and more.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from fuzzywuzzy import fuzz
from Levenshtein import distance as levenshtein_distance

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Advanced feature extraction for food product matching
    
    Creates multiple types of features:
    1. Text similarity features (TF-IDF, fuzzy matching, edit distances)
    2. Structural features (length, word count, etc.)
    3. Semantic features (when sentence transformers are available)
    4. Category-specific features
    5. Brand matching features
    """
    
    def __init__(self, use_semantic_features: bool = True, 
                 max_features: int = 5000,
                 ngram_range: Tuple[int, int] = (1, 3)):
        """
        Initialize feature extractor
        
        Args:
            use_semantic_features: Whether to use semantic embeddings (requires sentence-transformers)
            max_features: Maximum features for TF-IDF vectorizer
            ngram_range: N-gram range for TF-IDF
        """
        self.use_semantic_features = use_semantic_features
        self.max_features = max_features
        self.ngram_range = ngram_range
        
        # Initialize vectorizers
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english',
            lowercase=True,
            token_pattern=r'\b\w+\b'
        )
        
        self.count_vectorizer = CountVectorizer(
            max_features=max_features//2,
            ngram_range=(1, 2),
            stop_words='english',
            lowercase=True
        )
        
        # Try to initialize semantic model
        self.semantic_model = None
        if use_semantic_features:
            self._initialize_semantic_model()
            
        # Fitted status
        self.is_fitted = False
        
    def _initialize_semantic_model(self):
        """Initialize sentence transformer model if available"""
        try:
            from sentence_transformers import SentenceTransformer
            self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Semantic model initialized successfully")
        except ImportError:
            logger.warning("sentence-transformers not available. Semantic features disabled.")
            self.use_semantic_features = False
        except Exception as e:
            logger.warning(f"Failed to initialize semantic model: {e}")
            self.use_semantic_features = False
            
    def fit(self, texts: List[str]):
        """
        Fit the feature extractors on a corpus of texts
        
        Args:
            texts: List of text strings to fit on
        """
        logger.info(f"Fitting feature extractors on {len(texts)} texts")
        
        # Clean texts
        clean_texts = [text if isinstance(text, str) else "" for text in texts]
        clean_texts = [text for text in clean_texts if text.strip()]
        
        if not clean_texts:
            raise ValueError("No valid texts provided for fitting")
            
        # Fit vectorizers
        self.tfidf_vectorizer.fit(clean_texts)
        self.count_vectorizer.fit(clean_texts)
        
        self.is_fitted = True
        logger.info("Feature extractors fitted successfully")
        
    def extract_similarity_features(self, text1: str, text2: str) -> Dict[str, float]:
        """
        Extract similarity features between two texts
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Dictionary of similarity features
        """
        if not isinstance(text1, str) or not isinstance(text2, str):
            return self._empty_similarity_features()
            
        features = {}
        
        # Fuzzy matching features
        features['fuzz_ratio'] = fuzz.ratio(text1, text2) / 100.0
        features['fuzz_partial_ratio'] = fuzz.partial_ratio(text1, text2) / 100.0
        features['fuzz_token_sort_ratio'] = fuzz.token_sort_ratio(text1, text2) / 100.0
        features['fuzz_token_set_ratio'] = fuzz.token_set_ratio(text1, text2) / 100.0
        
        # Edit distance features
        max_len = max(len(text1), len(text2))
        if max_len > 0:
            features['levenshtein_similarity'] = 1 - (levenshtein_distance(text1, text2) / max_len)
        else:
            features['levenshtein_similarity'] = 1.0
            
        # Jaccard similarity (word level)
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        features['jaccard_similarity'] = intersection / union if union > 0 else 0.0
        
        # Character n-gram similarities
        features['char_2gram_sim'] = self._char_ngram_similarity(text1, text2, 2)
        features['char_3gram_sim'] = self._char_ngram_similarity(text1, text2, 3)
        
        # Length ratio
        len1, len2 = len(text1), len(text2)
        features['length_ratio'] = min(len1, len2) / max(len1, len2) if max(len1, len2) > 0 else 1.0
        
        # Word count ratio
        words1_count = len(text1.split())
        words2_count = len(text2.split())
        features['word_count_ratio'] = min(words1_count, words2_count) / max(words1_count, words2_count) if max(words1_count, words2_count) > 0 else 1.0
        
        # Common prefix/suffix length
        features['common_prefix_ratio'] = self._common_prefix_ratio(text1, text2)
        features['common_suffix_ratio'] = self._common_suffix_ratio(text1, text2)
        
        # TF-IDF cosine similarity (if fitted)
        if self.is_fitted:
            try:
                tfidf1 = self.tfidf_vectorizer.transform([text1])
                tfidf2 = self.tfidf_vectorizer.transform([text2])
                features['tfidf_cosine_sim'] = cosine_similarity(tfidf1, tfidf2)[0, 0]
            except:
                features['tfidf_cosine_sim'] = 0.0
        else:
            features['tfidf_cosine_sim'] = 0.0
            
        # Semantic similarity (if available)
        if self.use_semantic_features and self.semantic_model:
            try:
                embeddings = self.semantic_model.encode([text1, text2])
                semantic_sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0, 0]
                features['semantic_similarity'] = float(semantic_sim)
            except Exception as e:
                logger.warning(f"Semantic similarity calculation failed: {e}")
                features['semantic_similarity'] = 0.0
        else:
            features['semantic_similarity'] = 0.0
            
        return features
        
    def extract_structural_features(self, text1: str, text2: str, 
                                   features1: Dict = None, 
                                   features2: Dict = None) -> Dict[str, float]:
        """
        Extract structural features comparing two texts
        
        Args:
            text1: First text
            text2: Second text  
            features1: Optional preprocessed features for text1
            features2: Optional preprocessed features for text2
            
        Returns:
            Dictionary of structural features
        """
        structural_features = {}
        
        # Use provided features or extract basic ones
        if features1 is None:
            features1 = self._basic_text_features(text1)
        if features2 is None:
            features2 = self._basic_text_features(text2)
            
        # Feature differences and ratios
        for feat_name in ['word_count', 'char_count', 'complexity_score']:
            if feat_name in features1 and feat_name in features2:
                val1, val2 = features1[feat_name], features2[feat_name]
                
                # Ratio
                max_val = max(val1, val2)
                min_val = min(val1, val2)
                structural_features[f'{feat_name}_ratio'] = min_val / max_val if max_val > 0 else 1.0
                
                # Absolute difference (normalized)
                structural_features[f'{feat_name}_diff'] = abs(val1 - val2) / max(max_val, 1)
                
        # Binary feature matches
        binary_features = ['has_numbers', 'has_weights', 'has_brands']
        for feat_name in binary_features:
            if feat_name in features1 and feat_name in features2:
                match = int(features1[feat_name] == features2[feat_name])
                structural_features[f'{feat_name}_match'] = float(match)
                
        # Category matching
        if 'category' in features1 and 'category' in features2:
            cat_match = int(features1['category'] == features2['category'])
            structural_features['category_match'] = float(cat_match)
            
        # Processing type matching
        if 'processing_type' in features1 and 'processing_type' in features2:
            proc_match = int(features1['processing_type'] == features2['processing_type'])
            structural_features['processing_type_match'] = float(proc_match)
            
        # Brand overlap
        if 'brands' in features1 and 'brands' in features2:
            brands1 = set(features1['brands']) if features1['brands'] else set()
            brands2 = set(features2['brands']) if features2['brands'] else set()
            
            if brands1 or brands2:
                intersection = len(brands1.intersection(brands2))
                union = len(brands1.union(brands2))
                structural_features['brand_jaccard'] = intersection / union if union > 0 else 0.0
                structural_features['brand_overlap'] = float(intersection > 0)
            else:
                structural_features['brand_jaccard'] = 0.0
                structural_features['brand_overlap'] = 0.0
                
        # Weight similarity
        if 'weights' in features1 and 'weights' in features2:
            weights1 = features1['weights'] if features1['weights'] else []
            weights2 = features2['weights'] if features2['weights'] else []
            
            structural_features['weight_similarity'] = self._weight_similarity(weights1, weights2)
            
        return structural_features
        
    def extract_all_features(self, text1: str, text2: str,
                           features1: Dict = None,
                           features2: Dict = None) -> Dict[str, float]:
        """
        Extract all features for a pair of texts
        
        Args:
            text1: First text
            text2: Second text
            features1: Optional preprocessed features for text1
            features2: Optional preprocessed features for text2
            
        Returns:
            Dictionary containing all extracted features
        """
        all_features = {}
        
        # Similarity features
        similarity_features = self.extract_similarity_features(text1, text2)
        all_features.update(similarity_features)
        
        # Structural features
        structural_features = self.extract_structural_features(text1, text2, features1, features2)
        all_features.update(structural_features)
        
        # Composite features
        composite_features = self._create_composite_features(all_features)
        all_features.update(composite_features)
        
        return all_features
        
    def create_feature_matrix(self, text_pairs: List[Tuple[str, str]],
                            feature_pairs: List[Tuple[Dict, Dict]] = None) -> pd.DataFrame:
        """
        Create feature matrix for multiple text pairs
        
        Args:
            text_pairs: List of (text1, text2) tuples
            feature_pairs: Optional list of (features1, features2) tuples
            
        Returns:
            DataFrame with features for each pair
        """
        logger.info(f"Creating feature matrix for {len(text_pairs)} text pairs")
        
        feature_list = []
        
        for i, (text1, text2) in enumerate(text_pairs):
            features1, features2 = None, None
            if feature_pairs and i < len(feature_pairs):
                features1, features2 = feature_pairs[i]
                
            features = self.extract_all_features(text1, text2, features1, features2)
            feature_list.append(features)
            
        feature_df = pd.DataFrame(feature_list)
        
        # Fill any missing values
        feature_df = feature_df.fillna(0.0)
        
        logger.info(f"Feature matrix created with shape {feature_df.shape}")
        return feature_df
        
    def _char_ngram_similarity(self, text1: str, text2: str, n: int) -> float:
        """Calculate character n-gram similarity"""
        ngrams1 = set([text1[i:i+n] for i in range(len(text1)-n+1)])
        ngrams2 = set([text2[i:i+n] for i in range(len(text2)-n+1)])
        
        if not ngrams1 and not ngrams2:
            return 1.0
        if not ngrams1 or not ngrams2:
            return 0.0
            
        intersection = len(ngrams1.intersection(ngrams2))
        union = len(ngrams1.union(ngrams2))
        
        return intersection / union if union > 0 else 0.0
        
    def _common_prefix_ratio(self, text1: str, text2: str) -> float:
        """Calculate ratio of common prefix length to shorter text"""
        prefix_len = 0
        min_len = min(len(text1), len(text2))
        
        for i in range(min_len):
            if text1[i] == text2[i]:
                prefix_len += 1
            else:
                break
                
        return prefix_len / min_len if min_len > 0 else 1.0
        
    def _common_suffix_ratio(self, text1: str, text2: str) -> float:
        """Calculate ratio of common suffix length to shorter text"""
        suffix_len = 0
        min_len = min(len(text1), len(text2))
        
        for i in range(1, min_len + 1):
            if text1[-i] == text2[-i]:
                suffix_len += 1
            else:
                break
                
        return suffix_len / min_len if min_len > 0 else 1.0
        
    def _basic_text_features(self, text: str) -> Dict[str, Union[int, float, bool]]:
        """Extract basic features from a single text"""
        if not isinstance(text, str):
            text = ""
            
        return {
            'word_count': len(text.split()),
            'char_count': len(text),
            'has_numbers': bool(re.search(r'\d', text)),
            'complexity_score': min(len(text) / 50, 1.0)  # Simple complexity metric
        }
        
    def _weight_similarity(self, weights1: List[Dict], weights2: List[Dict]) -> float:
        """Calculate similarity between weight specifications"""
        if not weights1 and not weights2:
            return 1.0
        if not weights1 or not weights2:
            return 0.0
            
        # Find closest weight matches
        max_similarity = 0.0
        
        for w1 in weights1:
            for w2 in weights2:
                if w1['unit'] == w2['unit']:
                    # Same unit - compare values
                    val_ratio = min(w1['value'], w2['value']) / max(w1['value'], w2['value'])
                    similarity = val_ratio
                else:
                    # Different units - lower similarity
                    similarity = 0.3
                    
                max_similarity = max(max_similarity, similarity)
                
        return max_similarity
        
    def _create_composite_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """Create composite features from basic features"""
        composite = {}
        
        # Average fuzzy score
        fuzzy_features = [f for f in features.keys() if f.startswith('fuzz_')]
        if fuzzy_features:
            composite['avg_fuzzy_score'] = np.mean([features[f] for f in fuzzy_features])
            
        # Text similarity composite (combining multiple similarity metrics)
        text_sim_features = ['fuzz_token_set_ratio', 'jaccard_similarity', 'tfidf_cosine_sim']
        available_features = [f for f in text_sim_features if f in features]
        if available_features:
            composite['text_similarity_composite'] = np.mean([features[f] for f in available_features])
            
        # Structural similarity composite
        struct_sim_features = [f for f in features.keys() if f.endswith('_ratio') or f.endswith('_match')]
        if struct_sim_features:
            composite['structural_similarity_composite'] = np.mean([features[f] for f in struct_sim_features])
            
        # Overall confidence score
        key_features = ['fuzz_token_set_ratio', 'jaccard_similarity', 'semantic_similarity', 'tfidf_cosine_sim']
        available_key_features = [f for f in key_features if f in features]
        if available_key_features:
            # Weighted average with higher weight on semantic similarity
            weights = {'semantic_similarity': 0.4, 'fuzz_token_set_ratio': 0.25, 
                      'jaccard_similarity': 0.2, 'tfidf_cosine_sim': 0.15}
            
            weighted_sum = sum(features[f] * weights.get(f, 1.0) for f in available_key_features)
            total_weight = sum(weights.get(f, 1.0) for f in available_key_features)
            
            composite['overall_confidence'] = weighted_sum / total_weight if total_weight > 0 else 0.0
            
        return composite
        
    def _empty_similarity_features(self) -> Dict[str, float]:
        """Return empty similarity features dictionary"""
        return {
            'fuzz_ratio': 0.0,
            'fuzz_partial_ratio': 0.0,
            'fuzz_token_sort_ratio': 0.0,
            'fuzz_token_set_ratio': 0.0,
            'levenshtein_similarity': 0.0,
            'jaccard_similarity': 0.0,
            'char_2gram_sim': 0.0,
            'char_3gram_sim': 0.0,
            'length_ratio': 0.0,
            'word_count_ratio': 0.0,
            'common_prefix_ratio': 0.0,
            'common_suffix_ratio': 0.0,
            'tfidf_cosine_sim': 0.0,
            'semantic_similarity': 0.0
        }