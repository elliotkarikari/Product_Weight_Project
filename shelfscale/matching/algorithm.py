"""
Food matching algorithms to match products across different datasets
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz


def preprocess_text(text: str) -> str:
    """
    Preprocess text for matching by lowercasing and removing punctuation
    
    Args:
        text: Input text
        
    Returns:
        Preprocessed text
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    # Convert to lowercase and remove extra whitespace
    processed = text.lower().strip()
    
    # Replace punctuation with spaces
    for char in ",.;:!?/\\()[]{}<>\"'":
        processed = processed.replace(char, ' ')
    
    # Replace multiple spaces with a single space
    processed = ' '.join(processed.split())
    
    return processed


class FoodMatcher:
    """Food matching algorithm to find matches between food datasets"""
    
    def __init__(self, similarity_threshold: float = 0.6):
        """
        Initialize the food matcher
        
        Args:
            similarity_threshold: Threshold for considering a match
        """
        self.similarity_threshold = similarity_threshold
        self.vectorizer = None
    
    def _compute_tfidf_similarity(self, source_df: pd.DataFrame, 
                                 target_df: pd.DataFrame,
                                 source_col: str,
                                 target_col: str) -> pd.DataFrame:
        """
        Compute TF-IDF based cosine similarity between two datasets
        
        Args:
            source_df: Source DataFrame
            target_df: Target DataFrame
            source_col: Column name in source DataFrame for matching
            target_col: Column name in target DataFrame for matching
            
        Returns:
            DataFrame with match results
        """
        # Get text columns and preprocess
        source_texts = source_df[source_col].apply(preprocess_text)
        target_texts = target_df[target_col].apply(preprocess_text)
        
        # Fit and transform the vectorizer on combined texts
        all_texts = pd.concat([source_texts, target_texts])
        self.vectorizer = TfidfVectorizer(analyzer='word', 
                                         ngram_range=(1, 2),
                                         min_df=2,
                                         stop_words='english')
        
        # Create TF-IDF matrix
        tfidf_matrix = self.vectorizer.fit_transform(all_texts)
        
        # Split matrices back to source and target
        source_tfidf = tfidf_matrix[:len(source_texts)]
        target_tfidf = tfidf_matrix[len(source_texts):]
        
        # Compute cosine similarity
        similarity_matrix = cosine_similarity(source_tfidf, target_tfidf)
        
        return similarity_matrix
    
    def match_datasets(self, 
                      source_df: pd.DataFrame, 
                      target_df: pd.DataFrame,
                      source_col: str,
                      target_col: str,
                      additional_match_cols: List[Tuple[str, str]] = None) -> pd.DataFrame:
        """
        Match items between source and target DataFrames
        
        Args:
            source_df: Source DataFrame
            target_df: Target DataFrame
            source_col: Primary column name in source DataFrame for matching
            target_col: Primary column name in target DataFrame for matching
            additional_match_cols: Additional column pairs for enhancing matches [(source_col, target_col), ...]
            
        Returns:
            DataFrame with match results
        """
        # Calculate primary similarity
        similarity_matrix = self._compute_tfidf_similarity(
            source_df, target_df, source_col, target_col
        )
        
        # Initialize results DataFrame
        matches = []
        
        # For each source item, find the best match
        for i, source_item in source_df.iterrows():
            # Get top match index and score
            best_match_idx = np.argmax(similarity_matrix[i])
            similarity_score = similarity_matrix[i, best_match_idx]
            
            # Add additional similarity if specified
            if additional_match_cols:
                for src_col, tgt_col in additional_match_cols:
                    if src_col in source_df.columns and tgt_col in target_df.columns:
                        src_val = str(source_item[src_col]) if not pd.isna(source_item[src_col]) else ""
                        tgt_val = str(target_df.iloc[best_match_idx][tgt_col]) if not pd.isna(target_df.iloc[best_match_idx][tgt_col]) else ""
                        
                        # Use fuzzy ratio for additional columns
                        if src_val and tgt_val:
                            additional_score = fuzz.ratio(src_val.lower(), tgt_val.lower()) / 100.0
                            # Weighted average with main similarity
                            similarity_score = (similarity_score * 0.7) + (additional_score * 0.3)
            
            # Only include matches above threshold
            if similarity_score >= self.similarity_threshold:
                target_item = target_df.iloc[best_match_idx]
                
                match = {
                    'Source_Index': i,
                    'Target_Index': best_match_idx,
                    'Source_Item': source_item[source_col],
                    'Target_Item': target_item[target_col],
                    'Similarity_Score': similarity_score
                }
                matches.append(match)
                
        # Create results DataFrame
        match_df = pd.DataFrame(matches)
        
        # Sort by similarity score in descending order
        if not match_df.empty:
            match_df = match_df.sort_values('Similarity_Score', ascending=False)
        
        return match_df
    
    def merge_matched_datasets(self, 
                              source_df: pd.DataFrame, 
                              target_df: pd.DataFrame,
                              match_df: pd.DataFrame,
                              source_cols: List[str] = None,
                              target_cols: List[str] = None,
                              merged_cols: Dict[str, str] = None) -> pd.DataFrame:
        """
        Merge source and target DataFrames based on matching results
        
        Args:
            source_df: Source DataFrame
            target_df: Target DataFrame
            match_df: Matching results DataFrame from match_datasets
            source_cols: Source columns to include in result (None=all)
            target_cols: Target columns to include in result (None=all)
            merged_cols: Dictionary to rename columns {'old_name': 'new_name'}
            
        Returns:
            Merged DataFrame with data from both sources
        """
        if match_df.empty:
            return pd.DataFrame()
        
        # Select relevant columns from source and target
        if source_cols:
            source_subset = source_df.loc[match_df['Source_Index'], source_cols].reset_index(drop=True)
        else:
            source_subset = source_df.loc[match_df['Source_Index']].reset_index(drop=True)
            
        if target_cols:
            target_subset = target_df.loc[match_df['Target_Index'], target_cols].reset_index(drop=True)
        else:
            target_subset = target_df.loc[match_df['Target_Index']].reset_index(drop=True)
        
        # Add _source and _target suffixes to avoid column name conflicts
        source_subset = source_subset.add_suffix('_source')
        target_subset = target_subset.add_suffix('_target')
        
        # Merge DataFrames
        merged_df = pd.concat([source_subset, target_subset, match_df['Similarity_Score']], axis=1)
        
        # Rename columns if specified
        if merged_cols:
            merged_df = merged_df.rename(columns=merged_cols)
        
        return merged_df 