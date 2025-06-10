"""
Metrics module for embedding model evaluation.

This module provides lightweight, reusable metric calculations for:
- Similarity metrics (cosine, euclidean, etc.)
- Clustering metrics (silhouette, calinski-harabasz, etc.)
- Retrieval metrics (precision, recall, MAP, NDCG, etc.)
- Classification metrics (accuracy, F1, etc.)
- Efficiency metrics (latency, throughput, etc.)

All metrics handle edge cases and are independently testable.
"""

from .similarity_metrics import (
    cosine_similarity,
    euclidean_distance,
    manhattan_distance,
    pearson_correlation,
    spearman_correlation,
    semantic_similarity_score
)

from .clustering_metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    clustering_purity
)

from .retrieval_metrics import (
    precision_at_k,
    recall_at_k,
    f1_at_k,
    mean_average_precision,
    ndcg_at_k,
    reciprocal_rank,
    hit_rate_at_k
)

from .classification_metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix_metrics,
    classification_report_dict
)

from .efficiency_metrics import (
    calculate_latency,
    calculate_throughput,
    memory_usage,
    tokens_per_second,
    cost_per_embedding,
    efficiency_score
)

__all__ = [
    # Similarity metrics
    'cosine_similarity', 'euclidean_distance', 'manhattan_distance',
    'pearson_correlation', 'spearman_correlation', 'semantic_similarity_score',
    
    # Clustering metrics
    'silhouette_score', 'calinski_harabasz_score', 'davies_bouldin_score',
    'adjusted_rand_score', 'normalized_mutual_info_score', 'clustering_purity',
    
    # Retrieval metrics
    'precision_at_k', 'recall_at_k', 'f1_at_k', 'mean_average_precision',
    'ndcg_at_k', 'reciprocal_rank', 'hit_rate_at_k',
    
    # Classification metrics
    'accuracy_score', 'precision_score', 'recall_score', 'f1_score',
    'confusion_matrix_metrics', 'classification_report_dict',
    
    # Efficiency metrics
    'calculate_latency', 'calculate_throughput', 'memory_usage',
    'tokens_per_second', 'cost_per_embedding', 'efficiency_score'
]
