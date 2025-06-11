"""
Clustering metrics for embedding evaluation.

Provides lightweight, reusable clustering evaluation metrics with proper edge case handling.
"""

import numpy as np
from typing import List, Union, Optional
from sklearn.metrics import (
    silhouette_score as sklearn_silhouette_score,
    calinski_harabasz_score as sklearn_calinski_harabasz_score,
    davies_bouldin_score as sklearn_davies_bouldin_score,
    adjusted_rand_score as sklearn_adjusted_rand_score,
    normalized_mutual_info_score as sklearn_normalized_mutual_info_score
)
from collections import Counter
import warnings

class ClusteringMetrics:
    def silhouette_score(embeddings: np.ndarray, labels: np.ndarray) -> float:
        """
        Calculate silhouette score for clustering quality.
        
        Args:
            embeddings: Array of embeddings (n_samples, n_features)
            labels: Cluster labels for each sample
            
        Returns:
            Silhouette score (-1 to 1, higher is better)
        """
        if len(embeddings) == 0 or len(labels) == 0:
            return 0.0
        
        if len(embeddings) != len(labels):
            raise ValueError("Embeddings and labels must have the same length")
        
        # Need at least 2 samples and 2 clusters
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2 or len(embeddings) < 2:
            return 0.0
        
        # Check if all samples are in one cluster
        if len(unique_labels) == 1:
            return 0.0
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return sklearn_silhouette_score(embeddings, labels)
        except Exception:
            return 0.0

    def calinski_harabasz_score(embeddings: np.ndarray, labels: np.ndarray) -> float:
        """
        Calculate Calinski-Harabasz score (variance ratio criterion).
        
        Args:
            embeddings: Array of embeddings (n_samples, n_features)
            labels: Cluster labels for each sample
            
        Returns:
            Calinski-Harabasz score (higher is better)
        """
        if len(embeddings) == 0 or len(labels) == 0:
            return 0.0
        
        if len(embeddings) != len(labels):
            raise ValueError("Embeddings and labels must have the same length")
        
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2 or len(embeddings) < 2:
            return 0.0
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return sklearn_calinski_harabasz_score(embeddings, labels)
        except Exception:
            return 0.0

    def davies_bouldin_score(embeddings: np.ndarray, labels: np.ndarray) -> float:
        """
        Calculate Davies-Bouldin score for clustering quality.
        
        Args:
            embeddings: Array of embeddings (n_samples, n_features)
            labels: Cluster labels for each sample
            
        Returns:
            Davies-Bouldin score (lower is better)
        """
        if len(embeddings) == 0 or len(labels) == 0:
            return float('inf')
        
        if len(embeddings) != len(labels):
            raise ValueError("Embeddings and labels must have the same length")
        
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2 or len(embeddings) < 2:
            return float('inf')
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return sklearn_davies_bouldin_score(embeddings, labels)
        except Exception:
            return float('inf')

    def adjusted_rand_score(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
        """
        Calculate Adjusted Rand Index between true and predicted labels.
        
        Args:
            true_labels: Ground truth cluster labels
            pred_labels: Predicted cluster labels
            
        Returns:
            Adjusted Rand Index (-1 to 1, higher is better)
        """
        if len(true_labels) == 0 or len(pred_labels) == 0:
            return 0.0
        
        if len(true_labels) != len(pred_labels):
            raise ValueError("Label arrays must have the same length")
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return sklearn_adjusted_rand_score(true_labels, pred_labels)
        except Exception:
            return 0.0

    def normalized_mutual_info_score(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
        """
        Calculate Normalized Mutual Information between true and predicted labels.
        
        Args:
            true_labels: Ground truth cluster labels
            pred_labels: Predicted cluster labels
            
        Returns:
            Normalized Mutual Information (0 to 1, higher is better)
        """
        if len(true_labels) == 0 or len(pred_labels) == 0:
            return 0.0
        
        if len(true_labels) != len(pred_labels):
            raise ValueError("Label arrays must have the same length")
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return sklearn_normalized_mutual_info_score(true_labels, pred_labels)
        except Exception:
            return 0.0

    def clustering_purity(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
        """
        Calculate clustering purity score.
        
        Args:
            true_labels: Ground truth cluster labels
            pred_labels: Predicted cluster labels
            
        Returns:
            Purity score (0 to 1, higher is better)
        """
        if len(true_labels) == 0 or len(pred_labels) == 0:
            return 0.0
        
        if len(true_labels) != len(pred_labels):
            raise ValueError("Label arrays must have the same length")
        
        # Create contingency matrix
        contingency_matrix = {}
        for true_label, pred_label in zip(true_labels, pred_labels):
            if pred_label not in contingency_matrix:
                contingency_matrix[pred_label] = {}
            if true_label not in contingency_matrix[pred_label]:
                contingency_matrix[pred_label][true_label] = 0
            contingency_matrix[pred_label][true_label] += 1
        
        # Calculate purity
        total_correct = 0
        for pred_cluster in contingency_matrix:
            max_count = max(contingency_matrix[pred_cluster].values())
            total_correct += max_count
        
        return total_correct / len(true_labels) if len(true_labels) > 0 else 0.0

    def inertia_score(embeddings: np.ndarray, labels: np.ndarray, centroids: Optional[np.ndarray] = None) -> float:
        """
        Calculate within-cluster sum of squares (inertia).
        
        Args:
            embeddings: Array of embeddings (n_samples, n_features)
            labels: Cluster labels for each sample
            centroids: Cluster centroids. If None, calculated from data
            
        Returns:
            Inertia score (lower is better)
        """
        if len(embeddings) == 0 or len(labels) == 0:
            return float('inf')
        
        if len(embeddings) != len(labels):
            raise ValueError("Embeddings and labels must have the same length")
        
        unique_labels = np.unique(labels)
        if len(unique_labels) == 0:
            return float('inf')
        
        try:
            total_inertia = 0.0
            
            for label in unique_labels:
                cluster_points = embeddings[labels == label]
                if len(cluster_points) == 0:
                    continue
                
                if centroids is not None and label < len(centroids):
                    centroid = centroids[label]
                else:
                    centroid = np.mean(cluster_points, axis=0)
                
                # Calculate sum of squared distances to centroid
                distances = np.sum((cluster_points - centroid) ** 2, axis=1)
                total_inertia += np.sum(distances)
            
            return total_inertia
        except Exception:
            return float('inf')

    def cluster_separation_score(embeddings: np.ndarray, labels: np.ndarray) -> float:
        """
        Calculate average separation between cluster centroids.
        
        Args:
            embeddings: Array of embeddings (n_samples, n_features)
            labels: Cluster labels for each sample
            
        Returns:
            Separation score (higher is better)
        """
        if len(embeddings) == 0 or len(labels) == 0:
            return 0.0
        
        if len(embeddings) != len(labels):
            raise ValueError("Embeddings and labels must have the same length")
        
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            return 0.0
        
        try:
            # Calculate centroids
            centroids = []
            for label in unique_labels:
                cluster_points = embeddings[labels == label]
                if len(cluster_points) > 0:
                    centroids.append(np.mean(cluster_points, axis=0))
            
            if len(centroids) < 2:
                return 0.0
            
            # Calculate pairwise distances between centroids
            total_distance = 0.0
            count = 0
            
            for i in range(len(centroids)):
                for j in range(i + 1, len(centroids)):
                    distance = np.linalg.norm(centroids[i] - centroids[j])
                    total_distance += distance
                    count += 1
            
            return total_distance / count if count > 0 else 0.0
        except Exception:
            return 0.0
