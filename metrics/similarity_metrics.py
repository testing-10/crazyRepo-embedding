"""
Similarity metrics for embedding evaluation.

Provides lightweight, reusable similarity calculations with proper edge case handling.
"""

import numpy as np
from typing import List, Union, Optional
from scipy.stats import pearsonr, spearmanr
import warnings

class SimilarityMetrics:
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score (-1 to 1)
            
        Raises:
            ValueError: If vectors have different dimensions or are empty
        """
        if len(vec1) == 0 or len(vec2) == 0:
            raise ValueError("Vectors cannot be empty")
        
        if len(vec1) != len(vec2):
            raise ValueError("Vectors must have the same dimension")
        
        # Handle zero vectors
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return np.dot(vec1, vec2) / (norm1 * norm2)

    def euclidean_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate Euclidean distance between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Euclidean distance (>= 0)
        """
        if len(vec1) == 0 or len(vec2) == 0:
            raise ValueError("Vectors cannot be empty")
        
        if len(vec1) != len(vec2):
            raise ValueError("Vectors must have the same dimension")
        
        return np.linalg.norm(vec1 - vec2)

    def manhattan_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate Manhattan (L1) distance between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Manhattan distance (>= 0)
        """
        if len(vec1) == 0 or len(vec2) == 0:
            raise ValueError("Vectors cannot be empty")
        
        if len(vec1) != len(vec2):
            raise ValueError("Vectors must have the same dimension")
        
        return np.sum(np.abs(vec1 - vec2))

    def pearson_correlation(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate Pearson correlation coefficient between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Pearson correlation (-1 to 1)
        """
        if len(vec1) == 0 or len(vec2) == 0:
            raise ValueError("Vectors cannot be empty")
        
        if len(vec1) != len(vec2):
            raise ValueError("Vectors must have the same dimension")
        
        if len(vec1) < 2:
            return 0.0
        
        # Handle constant vectors
        if np.std(vec1) == 0 or np.std(vec2) == 0:
            return 0.0
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            corr, _ = pearsonr(vec1, vec2)
            return corr if not np.isnan(corr) else 0.0

    def spearman_correlation(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate Spearman rank correlation between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Spearman correlation (-1 to 1)
        """
        if len(vec1) == 0 or len(vec2) == 0:
            raise ValueError("Vectors cannot be empty")
        
        if len(vec1) != len(vec2):
            raise ValueError("Vectors must have the same dimension")
        
        if len(vec1) < 2:
            return 0.0
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            corr, _ = spearmanr(vec1, vec2)
            return corr if not np.isnan(corr) else 0.0

    def semantic_similarity_score(
        embeddings1: List[np.ndarray], 
        embeddings2: List[np.ndarray],
        similarity_func: str = "cosine"
    ) -> List[float]:
        """
        Calculate semantic similarity scores between two lists of embeddings.
        
        Args:
            embeddings1: First list of embeddings
            embeddings2: Second list of embeddings
            similarity_func: Similarity function to use ("cosine", "euclidean", "manhattan")
            
        Returns:
            List of similarity scores
        """
        if len(embeddings1) != len(embeddings2):
            raise ValueError("Embedding lists must have the same length")
        
        if len(embeddings1) == 0:
            return []
        
        similarity_functions = {
            "cosine": cosine_similarity,
            "euclidean": lambda x, y: 1.0 / (1.0 + euclidean_distance(x, y)),
            "manhattan": lambda x, y: 1.0 / (1.0 + manhattan_distance(x, y))
        }
        
        if similarity_func not in similarity_functions:
            raise ValueError(f"Unknown similarity function: {similarity_func}")
        
        func = similarity_functions[similarity_func]
        scores = []
        
        for emb1, emb2 in zip(embeddings1, embeddings2):
            try:
                score = func(emb1, emb2)
                scores.append(score)
            except Exception as e:
                # Handle individual calculation errors gracefully
                scores.append(0.0)
        
        return scores

    def batch_cosine_similarity(
        embeddings: np.ndarray, 
        query_embedding: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Calculate cosine similarity for a batch of embeddings.
        
        Args:
            embeddings: Matrix of embeddings (n_samples, n_features)
            query_embedding: Single query embedding. If None, calculates pairwise similarities
            
        Returns:
            Array of similarity scores
        """
        if embeddings.size == 0:
            return np.array([])
        
        if query_embedding is not None:
            # Query vs all embeddings
            if len(query_embedding) != embeddings.shape[1]:
                raise ValueError("Query embedding dimension mismatch")
            
            # Normalize embeddings
            embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
            query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
            
            return np.dot(embeddings_norm, query_norm)
        else:
            # Pairwise similarities
            embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
            return np.dot(embeddings_norm, embeddings_norm.T)
