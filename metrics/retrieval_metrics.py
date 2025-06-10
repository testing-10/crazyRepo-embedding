"""
Retrieval metrics for embedding evaluation.

Provides lightweight, reusable retrieval evaluation metrics with proper edge case handling.
"""

import numpy as np
from typing import List, Union, Optional, Set
import warnings

def precision_at_k(retrieved: List[int], relevant: Set[int], k: int) -> float:
    """
    Calculate Precision@K for retrieval evaluation.
    
    Args:
        retrieved: List of retrieved item IDs (ordered by relevance)
        relevant: Set of relevant item IDs
        k: Number of top items to consider
        
    Returns:
        Precision@K score (0 to 1)
    """
    if k <= 0:
        return 0.0
    
    if len(retrieved) == 0:
        return 0.0
    
    if len(relevant) == 0:
        return 0.0
    
    # Take top k retrieved items
    top_k = retrieved[:k]
    
    # Count relevant items in top k
    relevant_in_top_k = sum(1 for item in top_k if item in relevant)
    
    return relevant_in_top_k / min(k, len(top_k))

def recall_at_k(retrieved: List[int], relevant: Set[int], k: int) -> float:
    """
    Calculate Recall@K for retrieval evaluation.
    
    Args:
        retrieved: List of retrieved item IDs (ordered by relevance)
        relevant: Set of relevant item IDs
        k: Number of top items to consider
        
    Returns:
        Recall@K score (0 to 1)
    """
    if k <= 0:
        return 0.0
    
    if len(relevant) == 0:
        return 0.0
    
    if len(retrieved) == 0:
        return 0.0
    
    # Take top k retrieved items
    top_k = retrieved[:k]
    
    # Count relevant items in top k
    relevant_in_top_k = sum(1 for item in top_k if item in relevant)
    
    return relevant_in_top_k / len(relevant)

def f1_at_k(retrieved: List[int], relevant: Set[int], k: int) -> float:
    """
    Calculate F1@K for retrieval evaluation.
    
    Args:
        retrieved: List of retrieved item IDs (ordered by relevance)
        relevant: Set of relevant item IDs
        k: Number of top items to consider
        
    Returns:
        F1@K score (0 to 1)
    """
    precision = precision_at_k(retrieved, relevant, k)
    recall = recall_at_k(retrieved, relevant, k)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * (precision * recall) / (precision + recall)

def mean_average_precision(retrieved_lists: List[List[int]], relevant_sets: List[Set[int]]) -> float:
    """
    Calculate Mean Average Precision (MAP) across multiple queries.
    
    Args:
        retrieved_lists: List of retrieved item lists for each query
        relevant_sets: List of relevant item sets for each query
        
    Returns:
        MAP score (0 to 1)
    """
    if len(retrieved_lists) == 0 or len(relevant_sets) == 0:
        return 0.0
    
    if len(retrieved_lists) != len(relevant_sets):
        raise ValueError("Retrieved lists and relevant sets must have the same length")
    
    total_ap = 0.0
    valid_queries = 0
    
    for retrieved, relevant in zip(retrieved_lists, relevant_sets):
        ap = average_precision(retrieved, relevant)
        if not np.isnan(ap):
            total_ap += ap
            valid_queries += 1
    
    return total_ap / valid_queries if valid_queries > 0 else 0.0

def average_precision(retrieved: List[int], relevant: Set[int]) -> float:
    """
    Calculate Average Precision for a single query.
    
    Args:
        retrieved: List of retrieved item IDs (ordered by relevance)
        relevant: Set of relevant item IDs
        
    Returns:
        Average Precision score (0 to 1)
    """
    if len(relevant) == 0:
        return 0.0
    
    if len(retrieved) == 0:
        return 0.0
    
    precision_sum = 0.0
    relevant_count = 0
    
    for i, item in enumerate(retrieved):
        if item in relevant:
            relevant_count += 1
            precision_at_i = relevant_count / (i + 1)
            precision_sum += precision_at_i
    
    return precision_sum / len(relevant) if len(relevant) > 0 else 0.0

def ndcg_at_k(retrieved: List[int], relevant_scores: dict, k: int) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain at K.
    
    Args:
        retrieved: List of retrieved item IDs (ordered by relevance)
        relevant_scores: Dictionary mapping item IDs to relevance scores
        k: Number of top items to consider
        
    Returns:
        NDCG@K score (0 to 1)
    """
    if k <= 0:
        return 0.0
    
    if len(retrieved) == 0 or len(relevant_scores) == 0:
        return 0.0
    
    # Calculate DCG@K
    dcg = 0.0
    for i, item in enumerate(retrieved[:k]):
        if item in relevant_scores:
            relevance = relevant_scores[item]
            dcg += relevance / np.log2(i + 2)  # i+2 because log2(1) = 0
    
    # Calculate IDCG@K (Ideal DCG)
    sorted_scores = sorted(relevant_scores.values(), reverse=True)
    idcg = 0.0
    for i, score in enumerate(sorted_scores[:k]):
        idcg += score / np.log2(i + 2)
    
    return dcg / idcg if idcg > 0 else 0.0

def reciprocal_rank(retrieved: List[int], relevant: Set[int]) -> float:
    """
    Calculate Reciprocal Rank (RR) for a single query.
    
    Args:
        retrieved: List of retrieved item IDs (ordered by relevance)
        relevant: Set of relevant item IDs
        
    Returns:
        Reciprocal Rank score (0 to 1)
    """
    if len(retrieved) == 0 or len(relevant) == 0:
        return 0.0
    
    for i, item in enumerate(retrieved):
        if item in relevant:
            return 1.0 / (i + 1)
    
    return 0.0

def mean_reciprocal_rank(retrieved_lists: List[List[int]], relevant_sets: List[Set[int]]) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR) across multiple queries.
    
    Args:
        retrieved_lists: List of retrieved item lists for each query
        relevant_sets: List of relevant item sets for each query
        
    Returns:
        MRR score (0 to 1)
    """
    if len(retrieved_lists) == 0 or len(relevant_sets) == 0:
        return 0.0
    
    if len(retrieved_lists) != len(relevant_sets):
        raise ValueError("Retrieved lists and relevant sets must have the same length")
    
    total_rr = 0.0
    for retrieved, relevant in zip(retrieved_lists, relevant_sets):
        total_rr += reciprocal_rank(retrieved, relevant)
    
    return total_rr / len(retrieved_lists)

def hit_rate_at_k(retrieved: List[int], relevant: Set[int], k: int) -> float:
    """
    Calculate Hit Rate@K (whether any relevant item is in top K).
    
    Args:
        retrieved: List of retrieved item IDs (ordered by relevance)
        relevant: Set of relevant item IDs
        k: Number of top items to consider
        
    Returns:
        Hit Rate@K score (0 or 1)
    """
    if k <= 0:
        return 0.0
    
    if len(retrieved) == 0 or len(relevant) == 0:
        return 0.0
    
    top_k = retrieved[:k]
    return 1.0 if any(item in relevant for item in top_k) else 0.0

def coverage_at_k(retrieved_lists: List[List[int]], all_items: Set[int], k: int) -> float:
    """
    Calculate Coverage@K (fraction of all items that appear in top K across queries).
    
    Args:
        retrieved_lists: List of retrieved item lists for each query
        all_items: Set of all possible item IDs
        k: Number of top items to consider
        
    Returns:
        Coverage@K score (0 to 1)
    """
    if k <= 0 or len(all_items) == 0:
        return 0.0
    
    if len(retrieved_lists) == 0:
        return 0.0
    
    covered_items = set()
    for retrieved in retrieved_lists:
        covered_items.update(retrieved[:k])
    
    return len(covered_items) / len(all_items)

def diversity_at_k(retrieved: List[int], item_features: dict, k: int) -> float:
    """
    Calculate diversity of retrieved items at K based on feature similarity.
    
    Args:
        retrieved: List of retrieved item IDs (ordered by relevance)
        item_features: Dictionary mapping item IDs to feature vectors
        k: Number of top items to consider
        
    Returns:
        Diversity score (0 to 1, higher is more diverse)
    """
    if k <= 0 or len(retrieved) == 0:
        return 0.0
    
    top_k = retrieved[:k]
    valid_items = [item for item in top_k if item in item_features]
    
    if len(valid_items) < 2:
        return 0.0
    
    # Calculate pairwise similarities
    total_similarity = 0.0
    pair_count = 0
    
    for i in range(len(valid_items)):
        for j in range(i + 1, len(valid_items)):
            item1, item2 = valid_items[i], valid_items[j]
            features1 = np.array(item_features[item1])
            features2 = np.array(item_features[item2])
            
            # Cosine similarity
            norm1 = np.linalg.norm(features1)
            norm2 = np.linalg.norm(features2)
            
            if norm1 > 0 and norm2 > 0:
                similarity = np.dot(features1, features2) / (norm1 * norm2)
                total_similarity += similarity
                pair_count += 1
    
    if pair_count == 0:
        return 0.0
    
    avg_similarity = total_similarity / pair_count
    return 1.0 - avg_similarity  # Higher diversity = lower similarity
