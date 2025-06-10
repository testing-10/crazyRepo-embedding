"""
Vector Operations - Mathematical operations for embedding vectors
"""

import numpy as np
from typing import List, Union, Tuple, Optional, Dict, Any
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings


class VectorOperations:
    """
    Utility class for vector operations commonly used in embedding evaluation.
    Provides similarity calculations, clustering, dimensionality reduction, and statistics.
    """
    
    @staticmethod
    def normalize_vector(vector: Union[List[float], np.ndarray]) -> np.ndarray:
        """
        Normalize a vector to unit length.
        
        Args:
            vector: Input vector
            
        Returns:
            Normalized vector as numpy array
            
        Raises:
            ValueError: If vector is zero or invalid
        """
        vector = np.array(vector, dtype=np.float32)
        
        if vector.size == 0:
            raise ValueError("Cannot normalize empty vector")
        
        norm = np.linalg.norm(vector)
        if norm == 0:
            raise ValueError("Cannot normalize zero vector")
        
        return vector / norm
    
    @staticmethod
    def cosine_similarity(vector1: Union[List[float], np.ndarray],
                         vector2: Union[List[float], np.ndarray]) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vector1: First vector
            vector2: Second vector
            
        Returns:
            Cosine similarity score (-1 to 1)
            
        Raises:
            ValueError: If vectors have different dimensions or are invalid
        """
        v1 = np.array(vector1, dtype=np.float32)
        v2 = np.array(vector2, dtype=np.float32)
        
        if v1.shape != v2.shape:
            raise ValueError(f"Vector dimensions don't match: {v1.shape} vs {v2.shape}")
        
        if v1.size == 0:
            raise ValueError("Cannot calculate similarity for empty vectors")
        
        # Handle zero vectors
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(v1, v2) / (norm1 * norm2))
    
    @staticmethod
    def euclidean_distance(vector1: Union[List[float], np.ndarray],
                          vector2: Union[List[float], np.ndarray]) -> float:
        """
        Calculate Euclidean distance between two vectors.
        
        Args:
            vector1: First vector
            vector2: Second vector
            
        Returns:
            Euclidean distance
            
        Raises:
            ValueError: If vectors have different dimensions
        """
        v1 = np.array(vector1, dtype=np.float32)
        v2 = np.array(vector2, dtype=np.float32)
        
        if v1.shape != v2.shape:
            raise ValueError(f"Vector dimensions don't match: {v1.shape} vs {v2.shape}")
        
        return float(np.linalg.norm(v1 - v2))
    
    @staticmethod
    def manhattan_distance(vector1: Union[List[float], np.ndarray],
                          vector2: Union[List[float], np.ndarray]) -> float:
        """
        Calculate Manhattan (L1) distance between two vectors.
        
        Args:
            vector1: First vector
            vector2: Second vector
            
        Returns:
            Manhattan distance
            
        Raises:
            ValueError: If vectors have different dimensions
        """
        v1 = np.array(vector1, dtype=np.float32)
        v2 = np.array(vector2, dtype=np.float32)
        
        if v1.shape != v2.shape:
            raise ValueError(f"Vector dimensions don't match: {v1.shape} vs {v2.shape}")
        
        return float(np.sum(np.abs(v1 - v2)))
    
    @staticmethod
    def dot_product(vector1: Union[List[float], np.ndarray],
                   vector2: Union[List[float], np.ndarray]) -> float:
        """
        Calculate dot product between two vectors.
        
        Args:
            vector1: First vector
            vector2: Second vector
            
        Returns:
            Dot product
            
        Raises:
            ValueError: If vectors have different dimensions
        """
        v1 = np.array(vector1, dtype=np.float32)
        v2 = np.array(vector2, dtype=np.float32)
        
        if v1.shape != v2.shape:
            raise ValueError(f"Vector dimensions don't match: {v1.shape} vs {v2.shape}")
        
        return float(np.dot(v1, v2))
    
    @staticmethod
    def batch_cosine_similarity(vectors1: Union[List[List[float]], np.ndarray],
                               vectors2: Union[List[List[float]], np.ndarray]) -> np.ndarray:
        """
        Calculate cosine similarity between two sets of vectors (batch operation).
        
        Args:
            vectors1: First set of vectors (n_samples_1, n_features)
            vectors2: Second set of vectors (n_samples_2, n_features)
            
        Returns:
            Similarity matrix (n_samples_1, n_samples_2)
            
        Raises:
            ValueError: If vector dimensions don't match
        """
        v1 = np.array(vectors1, dtype=np.float32)
        v2 = np.array(vectors2, dtype=np.float32)
        
        if v1.ndim == 1:
            v1 = v1.reshape(1, -1)
        if v2.ndim == 1:
            v2 = v2.reshape(1, -1)
        
        if v1.shape[1] != v2.shape[1]:
            raise ValueError(f"Vector dimensions don't match: {v1.shape[1]} vs {v2.shape[1]}")
        
        return cosine_similarity(v1, v2)
    
    @staticmethod
    def batch_euclidean_distance(vectors1: Union[List[List[float]], np.ndarray],
                                vectors2: Union[List[List[float]], np.ndarray]) -> np.ndarray:
        """
        Calculate Euclidean distance between two sets of vectors (batch operation).
        
        Args:
            vectors1: First set of vectors (n_samples_1, n_features)
            vectors2: Second set of vectors (n_samples_2, n_features)
            
        Returns:
            Distance matrix (n_samples_1, n_samples_2)
            
        Raises:
            ValueError: If vector dimensions don't match
        """
        v1 = np.array(vectors1, dtype=np.float32)
        v2 = np.array(vectors2, dtype=np.float32)
        
        if v1.ndim == 1:
            v1 = v1.reshape(1, -1)
        if v2.ndim == 1:
            v2 = v2.reshape(1, -1)
        
        if v1.shape[1] != v2.shape[1]:
            raise ValueError(f"Vector dimensions don't match: {v1.shape[1]} vs {v2.shape[1]}")
        
        return euclidean_distances(v1, v2)
    
    @staticmethod
    def find_most_similar(query_vector: Union[List[float], np.ndarray],
                         candidate_vectors: Union[List[List[float]], np.ndarray],
                         top_k: int = 5,
                         metric: str = 'cosine') -> List[Tuple[int, float]]:
        """
        Find most similar vectors to a query vector.
        
        Args:
            query_vector: Query vector
            candidate_vectors: Set of candidate vectors
            top_k: Number of top results to return
            metric: Similarity metric ('cosine', 'euclidean', 'manhattan')
            
        Returns:
            List of (index, score) tuples sorted by similarity
            
        Raises:
            ValueError: If metric is not supported or vectors are invalid
        """
        query = np.array(query_vector, dtype=np.float32)
        candidates = np.array(candidate_vectors, dtype=np.float32)
        
        if candidates.ndim == 1:
            candidates = candidates.reshape(1, -1)
        
        if query.shape[0] != candidates.shape[1]:
            raise ValueError(f"Query vector dimension {query.shape[0]} doesn't match candidate dimension {candidates.shape[1]}")
        
        if metric == 'cosine':
            similarities = cosine_similarity(query.reshape(1, -1), candidates)[0]
            # Higher is better for cosine similarity
            indices = np.argsort(similarities)[::-1][:top_k]
            return [(int(idx), float(similarities[idx])) for idx in indices]
        
        elif metric == 'euclidean':
            distances = euclidean_distances(query.reshape(1, -1), candidates)[0]
            # Lower is better for distance
            indices = np.argsort(distances)[:top_k]
            return [(int(idx), float(distances[idx])) for idx in indices]
        
        elif metric == 'manhattan':
            distances = [VectorOperations.manhattan_distance(query, candidate) 
                        for candidate in candidates]
            # Lower is better for distance
            indices = np.argsort(distances)[:top_k]
            return [(int(idx), float(distances[idx])) for idx in indices]
        
        else:
            raise ValueError(f"Unsupported metric: {metric}")
    
    @staticmethod
    def cluster_vectors(vectors: Union[List[List[float]], np.ndarray],
                       n_clusters: int,
                       random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Cluster vectors using K-means.
        
        Args:
            vectors: Input vectors
            n_clusters: Number of clusters
            random_state: Random state for reproducibility
            
        Returns:
            Tuple of (cluster_labels, cluster_centers, inertia)
            
        Raises:
            ValueError: If number of clusters is invalid
        """
        vectors_array = np.array(vectors, dtype=np.float32)
        
        if vectors_array.ndim == 1:
            vectors_array = vectors_array.reshape(1, -1)
        
        if n_clusters <= 0 or n_clusters > len(vectors_array):
            raise ValueError(f"Invalid number of clusters: {n_clusters}")
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
            labels = kmeans.fit_predict(vectors_array)
        
        return labels, kmeans.cluster_centers_, kmeans.inertia_
    
    @staticmethod
    def reduce_dimensionality(vectors: Union[List[List[float]], np.ndarray],
                             n_components: int,
                             method: str = 'pca') -> Tuple[np.ndarray, Any]:
        """
        Reduce dimensionality of vectors.
        
        Args:
            vectors: Input vectors
            n_components: Target number of dimensions
            method: Dimensionality reduction method ('pca')
            
        Returns:
            Tuple of (reduced_vectors, fitted_model)
            
        Raises:
            ValueError: If method is not supported or parameters are invalid
        """
        vectors_array = np.array(vectors, dtype=np.float32)
        
        if vectors_array.ndim == 1:
            vectors_array = vectors_array.reshape(1, -1)
        
        if n_components <= 0 or n_components > vectors_array.shape[1]:
            raise ValueError(f"Invalid number of components: {n_components}")
        
        if method == 'pca':
            pca = PCA(n_components=n_components, random_state=42)
            reduced_vectors = pca.fit_transform(vectors_array)
            return reduced_vectors, pca
        else:
            raise ValueError(f"Unsupported dimensionality reduction method: {method}")
    
    @staticmethod
    def calculate_vector_stats(vectors: Union[List[List[float]], np.ndarray]) -> Dict[str, Any]:
        """
        Calculate statistics for a set of vectors.
        
        Args:
            vectors: Input vectors
            
        Returns:
            Dictionary containing vector statistics
        """
        vectors_array = np.array(vectors, dtype=np.float32)
        
        if vectors_array.ndim == 1:
            vectors_array = vectors_array.reshape(1, -1)
        
        stats = {
            'n_vectors': vectors_array.shape[0],
            'n_dimensions': vectors_array.shape[1],
            'mean_vector': np.mean(vectors_array, axis=0),
            'std_vector': np.std(vectors_array, axis=0),
            'min_values': np.min(vectors_array, axis=0),
            'max_values': np.max(vectors_array, axis=0),
            'mean_magnitude': np.mean([np.linalg.norm(v) for v in vectors_array]),
            'std_magnitude': np.std([np.linalg.norm(v) for v in vectors_array])
        }
        
        # Calculate pairwise similarities if not too many vectors
        if vectors_array.shape[0] <= 1000:
            similarities = cosine_similarity(vectors_array)
            # Get upper triangle (excluding diagonal)
            upper_triangle = similarities[np.triu_indices_from(similarities, k=1)]
            
            stats.update({
                'mean_pairwise_similarity': float(np.mean(upper_triangle)),
                'std_pairwise_similarity': float(np.std(upper_triangle)),
                'min_pairwise_similarity': float(np.min(upper_triangle)),
                'max_pairwise_similarity': float(np.max(upper_triangle))
            })
        
        return stats
    
    @staticmethod
    def calculate_correlation(scores1: Union[List[float], np.ndarray],
                            scores2: Union[List[float], np.ndarray],
                            method: str = 'pearson') -> Tuple[float, float]:
        """
        Calculate correlation between two sets of scores.
        
        Args:
            scores1: First set of scores
            scores2: Second set of scores
            method: Correlation method ('pearson', 'spearman')
            
        Returns:
            Tuple of (correlation_coefficient, p_value)
            
        Raises:
            ValueError: If method is not supported or scores have different lengths
        """
        s1 = np.array(scores1, dtype=np.float32)
        s2 = np.array(scores2, dtype=np.float32)
        
        if len(s1) != len(s2):
            raise ValueError(f"Score arrays have different lengths: {len(s1)} vs {len(s2)}")
        
        if len(s1) < 2:
            raise ValueError("Need at least 2 data points for correlation")
        
        if method == 'pearson':
            corr, p_value = pearsonr(s1, s2)
        elif method == 'spearman':
            corr, p_value = spearmanr(s1, s2)
        else:
            raise ValueError(f"Unsupported correlation method: {method}")
        
        return float(corr), float(p_value)
    
    @staticmethod
    def is_valid_vector(vector: Union[List[float], np.ndarray]) -> bool:
        """
        Check if a vector is valid (no NaN, inf values).
        
        Args:
            vector: Vector to validate
            
        Returns:
            True if vector is valid, False otherwise
        """
        try:
            v = np.array(vector, dtype=np.float32)
            return not (np.any(np.isnan(v)) or np.any(np.isinf(v)))
        except (ValueError, TypeError):
            return False
    
    @staticmethod
    def filter_valid_vectors(vectors: Union[List[List[float]], np.ndarray]) -> Tuple[np.ndarray, List[int]]:
        """
        Filter out invalid vectors (containing NaN or inf values).
        
        Args:
            vectors: Input vectors
            
        Returns:
            Tuple of (valid_vectors, valid_indices)
        """
        vectors_array = np.array(vectors, dtype=np.float32)
        
        if vectors_array.ndim == 1:
            vectors_array = vectors_array.reshape(1, -1)
        
        valid_mask = ~(np.any(np.isnan(vectors_array), axis=1) | 
                      np.any(np.isinf(vectors_array), axis=1))
        
        valid_indices = np.where(valid_mask)[0].tolist()
        valid_vectors = vectors_array[valid_mask]
        
        return valid_vectors, valid_indices
    
    @staticmethod
    def interpolate_vectors(vector1: Union[List[float], np.ndarray],
                           vector2: Union[List[float], np.ndarray],
                           alpha: float = 0.5) -> np.ndarray:
        """
        Interpolate between two vectors.
        
        Args:
            vector1: First vector
            vector2: Second vector
            alpha: Interpolation factor (0.0 = vector1, 1.0 = vector2)
            
        Returns:
            Interpolated vector
            
        Raises:
            ValueError: If vectors have different dimensions or alpha is invalid
        """
        v1 = np.array(vector1, dtype=np.float32)
        v2 = np.array(vector2, dtype=np.float32)
        
        if v1.shape != v2.shape:
            raise ValueError(f"Vector dimensions don't match: {v1.shape} vs {v2.shape}")
        
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"Alpha must be between 0.0 and 1.0, got {alpha}")
        
        return (1 - alpha) * v1 + alpha * v2
    
    @staticmethod
    def vector_magnitude(vector: Union[List[float], np.ndarray]) -> float:
        """
        Calculate the magnitude (L2 norm) of a vector.
        
        Args:
            vector: Input vector
            
        Returns:
            Vector magnitude
        """
        v = np.array(vector, dtype=np.float32)
        return float(np.linalg.norm(v))
    
    @staticmethod
    def angle_between_vectors(vector1: Union[List[float], np.ndarray],
                             vector2: Union[List[float], np.ndarray],
                             degrees: bool = False) -> float:
        """
        Calculate angle between two vectors.
        
        Args:
            vector1: First vector
            vector2: Second vector
            degrees: Return angle in degrees instead of radians
            
        Returns:
            Angle between vectors
            
        Raises:
            ValueError: If vectors have different dimensions or are zero vectors
        """
        v1 = np.array(vector1, dtype=np.float32)
        v2 = np.array(vector2, dtype=np.float32)
        
        if v1.shape != v2.shape:
            raise ValueError(f"Vector dimensions don't match: {v1.shape} vs {v2.shape}")
        
        # Calculate cosine similarity
        cos_sim = VectorOperations.cosine_similarity(v1, v2)
        
        # Clamp to valid range for arccos
        cos_sim = np.clip(cos_sim, -1.0, 1.0)
        
        angle = np.arccos(cos_sim)
        
        if degrees:
            angle = np.degrees(angle)
        
        return float(angle)
