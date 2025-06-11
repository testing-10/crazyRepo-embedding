"""
Clustering Evaluator

Evaluates embedding models on clustering tasks using various
clustering algorithms and evaluation metrics.
"""

import time
import numpy as np
from typing import Dict, List, Any, Optional
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    adjusted_rand_score, normalized_mutual_info_score, homogeneity_score,
    completeness_score, v_measure_score
)
from sklearn.preprocessing import StandardScaler

from evaluators.base_embedding_evaluator import BaseEmbeddingEvaluator, EvaluationResult
from datasets.dataset_loader import DatasetSample
from utils.vector_operations import VectorOperations

class ClusteringEvaluator(BaseEmbeddingEvaluator):
    """Evaluator for clustering tasks"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.vector_ops = VectorOperations()
        
        # Default configuration
        self.default_config = {
            'clustering_algorithms': ['kmeans', 'hierarchical'],
            'metrics': ['silhouette_score', 'calinski_harabasz', 'davies_bouldin'],
            'n_clusters_range': [2, 3, 4, 5, 8, 10],
            'auto_determine_clusters': True,
            'normalize_embeddings': True,
            'random_state': 42,
            'max_samples': 1000  # Limit for performance
        }
        
        # Merge with provided config
        self.config = {**self.default_config, **self.config}
    
    def evaluate(self, 
                 embeddings: List[np.ndarray], 
                 samples: List[DatasetSample],
                 model_name: str,
                 dataset_name: str,
                 **kwargs) -> EvaluationResult:
        """Evaluate clustering performance"""
        
        start_time = time.time()
        self._log_evaluation_start(model_name, dataset_name, len(samples))
        
        try:
            # Validate inputs
            if not self._validate_inputs(embeddings, samples):
                raise ValueError("Input validation failed")
            
            # Prepare embeddings and filter samples
            valid_embeddings, valid_indices = self._prepare_embeddings(embeddings)
            valid_samples = self._filter_samples(samples, valid_indices)
            
            # Limit samples for performance if needed
            if len(valid_embeddings) > self.config['max_samples']:
                indices = np.random.choice(len(valid_embeddings), self.config['max_samples'], replace=False)
                valid_embeddings = valid_embeddings[indices]
                valid_samples = [valid_samples[i] for i in indices]
                self.logger.info(f"Subsampled to {self.config['max_samples']} samples for clustering")
            
            # Normalize embeddings if configured
            if self.config['normalize_embeddings']:
                scaler = StandardScaler()
                valid_embeddings = scaler.fit_transform(valid_embeddings)
            
            # Extract true labels if available
            true_labels = self._extract_true_labels(valid_samples)
            
            # Perform clustering evaluation
            metrics = self._evaluate_clustering(valid_embeddings, true_labels)
            
            # Generate metadata
            metadata = self._generate_metadata(valid_embeddings, valid_samples, true_labels)
            
            execution_time = time.time() - start_time
            self._log_evaluation_end(metrics, execution_time)
            
            return self._create_result(
                task_type="clustering",
                model_name=model_name,
                dataset_name=dataset_name,
                metrics=metrics,
                metadata=metadata,
                execution_time=execution_time,
                num_samples=len(valid_samples)
            )
            
        except Exception as e:
            self.logger.error(f"Clustering evaluation failed: {e}")
            raise
    
    def _extract_true_labels(self, samples: List[DatasetSample]) -> Optional[np.ndarray]:
        """Extract true cluster labels from samples if available"""
        labels = []
        
        for sample in samples:
            if sample.metadata and 'cluster' in sample.metadata:
                labels.append(sample.metadata['cluster'])
            elif sample.metadata and 'domain' in sample.metadata:
                labels.append(sample.metadata['domain'])
            elif sample.metadata and 'category' in sample.metadata:
                labels.append(sample.metadata['category'])
            elif sample.label is not None:
                labels.append(sample.label)
            else:
                labels.append(None)
        
        # Convert to numeric labels if possible
        if any(label is not None for label in labels):
            unique_labels = list(set([l for l in labels if l is not None]))
            label_map = {label: i for i, label in enumerate(unique_labels)}
            
            numeric_labels = []
            for label in labels:
                if label is not None:
                    numeric_labels.append(label_map[label])
                else:
                    numeric_labels.append(-1)  # Unknown cluster
            
            return np.array(numeric_labels)
        
        return None
    
    def _evaluate_clustering(self, embeddings: np.ndarray, true_labels: Optional[np.ndarray]) -> Dict[str, float]:
        """Evaluate clustering performance with different algorithms and metrics"""
        metrics = {}
        
        # Determine optimal number of clusters
        if self.config['auto_determine_clusters'] and true_labels is not None:
            n_true_clusters = len(np.unique(true_labels[true_labels >= 0]))
            if n_true_clusters > 1:
                optimal_k = n_true_clusters
            else:
                optimal_k = self._find_optimal_clusters(embeddings)
        else:
            optimal_k = self._find_optimal_clusters(embeddings)
        
        metrics['optimal_n_clusters'] = optimal_k
        
        # Evaluate different clustering algorithms
        for algorithm in self.config['clustering_algorithms']:
            try:
                predicted_labels = self._perform_clustering(embeddings, algorithm, optimal_k)
                
                if predicted_labels is not None:
                    # Internal metrics (don't require true labels)
                    internal_metrics = self._calculate_internal_metrics(embeddings, predicted_labels, algorithm)
                    metrics.update(internal_metrics)
                    
                    # External metrics (require true labels)
                    if true_labels is not None:
                        external_metrics = self._calculate_external_metrics(true_labels, predicted_labels, algorithm)
                        metrics.update(external_metrics)
                
            except Exception as e:
                self.logger.warning(f"Error evaluating {algorithm} clustering: {e}")
        
        return metrics
    
    def _find_optimal_clusters(self, embeddings: np.ndarray) -> int:
        """Find optimal number of clusters using silhouette analysis"""
        best_score = -1
        optimal_k = 2
        
        for k in self.config['n_clusters_range']:
            if k >= len(embeddings):
                continue
                
            try:
                kmeans = KMeans(n_clusters=k, random_state=self.config['random_state'], n_init=10)
                labels = kmeans.fit_predict(embeddings)
                score = silhouette_score(embeddings, labels)
                
                if score > best_score:
                    best_score = score
                    optimal_k = k
                    
            except Exception as e:
                self.logger.warning(f"Error evaluating k={k}: {e}")
                continue
        
        return optimal_k
    
    def _perform_clustering(self, embeddings: np.ndarray, algorithm: str, n_clusters: int) -> Optional[np.ndarray]:
        """Perform clustering with specified algorithm"""
        try:
            if algorithm == 'kmeans':
                clusterer = KMeans(
                    n_clusters=n_clusters, 
                    random_state=self.config['random_state'],
                    n_init=10
                )
                labels = clusterer.fit_predict(embeddings)
                
            elif algorithm == 'hierarchical':
                clusterer = AgglomerativeClustering(n_clusters=n_clusters)
                labels = clusterer.fit_predict(embeddings)
                
            elif algorithm == 'dbscan':
                # DBSCAN doesn't require n_clusters, use eps parameter
                eps = self._estimate_dbscan_eps(embeddings)
                clusterer = DBSCAN(eps=eps, min_samples=2)
                labels = clusterer.fit_predict(embeddings)
                
            else:
                self.logger.warning(f"Unknown clustering algorithm: {algorithm}")
                return None
            
            return labels
            
        except Exception as e:
            self.logger.error(f"Clustering with {algorithm} failed: {e}")
            return None
    
    def _estimate_dbscan_eps(self, embeddings: np.ndarray) -> float:
        """Estimate eps parameter for DBSCAN"""
        from sklearn.neighbors import NearestNeighbors
        
        try:
            # Use k=4 as a heuristic
            k = min(4, len(embeddings) - 1)
            neighbors = NearestNeighbors(n_neighbors=k)
            neighbors.fit(embeddings)
            distances, _ = neighbors.kneighbors(embeddings)
            
            # Use 75th percentile of k-th nearest neighbor distances
            eps = np.percentile(distances[:, -1], 75)
            return eps
            
        except Exception:
            # Fallback to a reasonable default
            return 0.5
    
    def _calculate_internal_metrics(self, embeddings: np.ndarray, labels: np.ndarray, algorithm: str) -> Dict[str, float]:
        """Calculate internal clustering metrics"""
        metrics = {}
        prefix = f"{algorithm}_"
        
        try:
            # Only calculate if we have more than one cluster
            n_clusters = len(np.unique(labels[labels >= 0]))  # Exclude noise points (-1)
            
            if n_clusters > 1:
                # Silhouette Score
                if 'silhouette_score' in self.config['metrics']:
                    silhouette = silhouette_score(embeddings, labels)
                    metrics[f'{prefix}silhouette_score'] = float(silhouette)
                
                # Calinski-Harabasz Index
                if 'calinski_harabasz' in self.config['metrics']:
                    ch_score = calinski_harabasz_score(embeddings, labels)
                    metrics[f'{prefix}calinski_harabasz_score'] = float(ch_score)
                
                # Davies-Bouldin Index
                if 'davies_bouldin' in self.config['metrics']:
                    db_score = davies_bouldin_score(embeddings, labels)
                    metrics[f'{prefix}davies_bouldin_score'] = float(db_score)
            
            # Number of clusters found
            metrics[f'{prefix}n_clusters_found'] = n_clusters
            
            # Number of noise points (for DBSCAN)
            if algorithm == 'dbscan':
                n_noise = np.sum(labels == -1)
                metrics[f'{prefix}n_noise_points'] = int(n_noise)
                metrics[f'{prefix}noise_ratio'] = float(n_noise / len(labels))
            
        except Exception as e:
            self.logger.warning(f"Error calculating internal metrics for {algorithm}: {e}")
        
        return metrics
    
    def _calculate_external_metrics(self, true_labels: np.ndarray, predicted_labels: np.ndarray, algorithm: str) -> Dict[str, float]:
        """Calculate external clustering metrics (require ground truth)"""
        metrics = {}
        prefix = f"{algorithm}_"
        
        try:
            # Filter out unknown labels (-1) for fair comparison
            mask = true_labels >= 0
            if np.sum(mask) == 0:
                return metrics
            
            true_filtered = true_labels[mask]
            pred_filtered = predicted_labels[mask]
            
            # Adjusted Rand Index
            ari = adjusted_rand_score(true_filtered, pred_filtered)
            metrics[f'{prefix}adjusted_rand_index'] = float(ari)
            
            # Normalized Mutual Information
            nmi = normalized_mutual_info_score(true_filtered, pred_filtered)
            metrics[f'{prefix}normalized_mutual_info'] = float(nmi)
            
            # Homogeneity, Completeness, V-measure
            homogeneity = homogeneity_score(true_filtered, pred_filtered)
            completeness = completeness_score(true_filtered, pred_filtered)
            v_measure = v_measure_score(true_filtered, pred_filtered)
            
            metrics[f'{prefix}homogeneity_score'] = float(homogeneity)
            metrics[f'{prefix}completeness_score'] = float(completeness)
            metrics[f'{prefix}v_measure_score'] = float(v_measure)
            
        except Exception as e:
            self.logger.warning(f"Error calculating external metrics for {algorithm}: {e}")
        
        return metrics
    
    def _generate_metadata(self, embeddings: np.ndarray, samples: List[DatasetSample], 
                          true_labels: Optional[np.ndarray]) -> Dict[str, Any]:
        """Generate clustering evaluation metadata"""
        metadata = {
            'embedding_dimension': int(embeddings.shape[1]),
            'n_samples': len(embeddings),
            'algorithms_used': self.config['clustering_algorithms'],
            'config': self.config.copy()
        }
        
        if true_labels is not None:
            unique_labels = np.unique(true_labels[true_labels >= 0])
            metadata['true_n_clusters'] = len(unique_labels)
            metadata['has_ground_truth'] = True
        else:
            metadata['has_ground_truth'] = False
        
        # Domain distribution if available
        domains = [sample.metadata.get('domain') for sample in samples if sample.metadata]
        if domains:
            unique_domains = list(set([d for d in domains if d is not None]))
            metadata['domains'] = unique_domains
            metadata['domain_distribution'] = {
                domain: domains.count(domain) for domain in unique_domains
            }
        
        return metadata
    
    def get_required_metrics(self) -> List[str]:
        """Get list of metrics this evaluator provides"""
        return [
            'silhouette_score',
            'calinski_harabasz_score',
            'davies_bouldin_score',
            'adjusted_rand_index',
            'normalized_mutual_info',
            'homogeneity_score',
            'completeness_score',
            'v_measure_score'
        ]
