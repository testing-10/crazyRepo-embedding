"""
Embedding Evaluators

This module provides evaluation logic for different embedding tasks including
semantic similarity, clustering, retrieval, classification, efficiency, and robustness evaluation.
"""

from evaluators.base_embedding_evaluator import BaseEmbeddingEvaluator, EvaluationResult, EvaluatorRegistry
from evaluators.semantic_similarity_evaluator import SemanticSimilarityEvaluator
from evaluators.clustering_evaluator import ClusteringEvaluator
from evaluators.retrieval_evaluator import RetrievalEvaluator
from evaluators.classification_evaluator import ClassificationEvaluator
from evaluators.efficiency_evaluator import EfficiencyEvaluator
from evaluators.robustness_evaluator import RobustnessEvaluator
from evaluators.evaluator_factory import (
    EvaluatorFactory, EvaluationPipeline, EVALUATOR_FACTORY,
    register_evaluator, create_evaluator, list_evaluators, 
    get_evaluator_info, create_evaluation_pipeline, evaluate_embeddings
)

__all__ = [
    # Base classes
    'BaseEmbeddingEvaluator',
    'EvaluationResult',
    'EvaluatorRegistry',
    
    # Evaluator classes
    'SemanticSimilarityEvaluator',
    'ClusteringEvaluator',
    'RetrievalEvaluator',
    'ClassificationEvaluator',
    'EfficiencyEvaluator',
    'RobustnessEvaluator',
    
    # Factory and pipeline
    'EvaluatorFactory',
    'EvaluationPipeline',
    'EVALUATOR_FACTORY',
    
    # Convenience functions
    'register_evaluator',
    'create_evaluator',
    'list_evaluators',
    'get_evaluator_info',
    'create_evaluation_pipeline',
    'evaluate_embeddings'
]

# Create global evaluator registry (legacy support)
EVALUATOR_REGISTRY = EvaluatorRegistry()

# Register all evaluators in legacy registry
EVALUATOR_REGISTRY.register('similarity', SemanticSimilarityEvaluator)
EVALUATOR_REGISTRY.register('clustering', ClusteringEvaluator)
EVALUATOR_REGISTRY.register('retrieval', RetrievalEvaluator)
EVALUATOR_REGISTRY.register('classification', ClassificationEvaluator)
EVALUATOR_REGISTRY.register('efficiency', EfficiencyEvaluator)
EVALUATOR_REGISTRY.register('robustness', RobustnessEvaluator)

def get_evaluator(task_type: str, **kwargs):
    """Get an evaluator instance for the specified task type (legacy support)"""
    return EVALUATOR_REGISTRY.get_evaluator(task_type, **kwargs)
