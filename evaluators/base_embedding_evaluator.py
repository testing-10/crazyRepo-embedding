"""
Base Embedding Evaluator

Provides the abstract base class and registry for all embedding evaluators.
All evaluators must extend BaseEmbeddingEvaluator and implement the evaluate() method.
"""

import time
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np

from utils.embedding_logger import EmbeddingLogger
from utils.embedding_cost_tracker import CostTracker
from datasets.dataset_loader import DatasetSample

logger = EmbeddingLogger.get_logger(__name__)

@dataclass
class EvaluationResult:
    """Standard evaluation result structure"""
    task_type: str
    model_name: str
    dataset_name: str
    metrics: Dict[str, float]
    metadata: Dict[str, Any]
    execution_time: float
    num_samples: int
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'task_type': self.task_type,
            'model_name': self.model_name,
            'dataset_name': self.dataset_name,
            'metrics': self.metrics,
            'metadata': self.metadata,
            'execution_time': self.execution_time,
            'num_samples': self.num_samples,
            'timestamp': self.timestamp
        }

class BaseEmbeddingEvaluator(ABC):
    """Abstract base class for all embedding evaluators"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.cost_tracker = CostTracker()
        self.logger = EmbeddingLogger.get_logger(self.__class__.__name__)
        
    @abstractmethod
    def evaluate(self, 
                 embeddings: List[np.ndarray], 
                 samples: List[DatasetSample],
                 model_name: str,
                 dataset_name: str,
                 **kwargs) -> EvaluationResult:
        """
        Evaluate embeddings on the given samples
        
        Args:
            embeddings: List of embedding vectors
            samples: List of dataset samples
            model_name: Name of the embedding model
            dataset_name: Name of the dataset
            **kwargs: Additional evaluation parameters
            
        Returns:
            EvaluationResult with metrics and metadata
        """
        pass
    
    @abstractmethod
    def get_required_metrics(self) -> List[str]:
        """Get list of metrics this evaluator requires"""
        pass
    
    def _validate_inputs(self, 
                        embeddings: List[np.ndarray], 
                        samples: List[DatasetSample]) -> bool:
        """Validate input data consistency"""
        try:
            if len(embeddings) != len(samples):
                raise ValueError(f"Embeddings count ({len(embeddings)}) != samples count ({len(samples)})")
            
            if not embeddings:
                raise ValueError("No embeddings provided")
            
            # Check embedding dimensions consistency
            embedding_dims = [emb.shape[-1] for emb in embeddings if emb is not None]
            if len(set(embedding_dims)) > 1:
                self.logger.warning(f"Inconsistent embedding dimensions: {set(embedding_dims)}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Input validation failed: {e}")
            return False
    
    def _prepare_embeddings(self, embeddings: List[np.ndarray]) -> np.ndarray:
        """Prepare embeddings array, handling None values"""
        valid_embeddings = []
        valid_indices = []
        
        for i, emb in enumerate(embeddings):
            if emb is not None and emb.size > 0:
                valid_embeddings.append(emb)
                valid_indices.append(i)
            else:
                self.logger.warning(f"Invalid embedding at index {i}")
        
        if not valid_embeddings:
            raise ValueError("No valid embeddings found")
        
        return np.array(valid_embeddings), valid_indices
    
    def _filter_samples(self, samples: List[DatasetSample], valid_indices: List[int]) -> List[DatasetSample]:
        """Filter samples to match valid embeddings"""
        return [samples[i] for i in valid_indices]
    
    def _create_result(self,
                      task_type: str,
                      model_name: str,
                      dataset_name: str,
                      metrics: Dict[str, float],
                      metadata: Dict[str, Any],
                      execution_time: float,
                      num_samples: int) -> EvaluationResult:
        """Create standardized evaluation result"""
        from datetime import datetime
        
        return EvaluationResult(
            task_type=task_type,
            model_name=model_name,
            dataset_name=dataset_name,
            metrics=metrics,
            metadata=metadata,
            execution_time=execution_time,
            num_samples=num_samples,
            timestamp=datetime.now().isoformat()
        )
    
    def _log_evaluation_start(self, model_name: str, dataset_name: str, num_samples: int):
        """Log evaluation start"""
        self.logger.info(f"Starting {self.__class__.__name__} evaluation")
        self.logger.info(f"Model: {model_name}, Dataset: {dataset_name}, Samples: {num_samples}")
    
    def _log_evaluation_end(self, metrics: Dict[str, float], execution_time: float):
        """Log evaluation completion"""
        self.logger.info(f"Evaluation completed in {execution_time:.2f}s")
        for metric, value in metrics.items():
            self.logger.info(f"{metric}: {value:.4f}")

class EvaluatorRegistry:
    """Registry for managing embedding evaluators"""
    
    def __init__(self):
        self._evaluators: Dict[str, type] = {}
        
    def register(self, task_type: str, evaluator_class: type):
        """Register an evaluator for a task type"""
        if not issubclass(evaluator_class, BaseEmbeddingEvaluator):
            raise ValueError(f"Evaluator must inherit from BaseEmbeddingEvaluator")
            
        self._evaluators[task_type] = evaluator_class
        logger.info(f"Registered evaluator: {task_type} -> {evaluator_class.__name__}")
        
    def get_evaluator(self, task_type: str, **kwargs) -> BaseEmbeddingEvaluator:
        """Get an evaluator instance for the task type"""
        if task_type not in self._evaluators:
            raise ValueError(f"Unknown evaluator type: {task_type}")
            
        return self._evaluators[task_type](**kwargs)
    
    def list_evaluators(self) -> List[str]:
        """List all registered evaluator types"""
        return list(self._evaluators.keys())
    
    def is_registered(self, task_type: str) -> bool:
        """Check if evaluator is registered"""
        return task_type in self._evaluators

def evaluate_embeddings(task_type: str,
                       embeddings: List[np.ndarray],
                       samples: List[DatasetSample],
                       model_name: str,
                       dataset_name: str,
                       config: Optional[Dict[str, Any]] = None,
                       **kwargs) -> EvaluationResult:
    """
    Convenience function to evaluate embeddings
    
    Args:
        task_type: Type of evaluation (similarity, clustering, retrieval)
        embeddings: List of embedding vectors
        samples: List of dataset samples
        model_name: Name of the embedding model
        dataset_name: Name of the dataset
        config: Evaluator configuration
        **kwargs: Additional evaluation parameters
        
    Returns:
        EvaluationResult with metrics and metadata
    """
    from . import EVALUATOR_REGISTRY
    
    evaluator = EVALUATOR_REGISTRY.get_evaluator(task_type, config=config)
    return evaluator.evaluate(embeddings, samples, model_name, dataset_name, **kwargs)
