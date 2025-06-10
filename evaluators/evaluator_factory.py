"""
Evaluator Factory

Central factory for managing and creating embedding evaluators.
Provides a unified interface for evaluator registration, discovery, and instantiation.
"""

from typing import Dict, List, Any, Optional, Type, Union
import importlib
import inspect
from abc import ABC

from .base_embedding_evaluator import BaseEmbeddingEvaluator, EvaluationResult
from utils.embedding_logger import EmbeddingLogger

logger = EmbeddingLogger.get_logger(__name__)

class EvaluatorFactory:
    """Factory for creating and managing embedding evaluators"""
    
    def __init__(self):
        self._evaluators: Dict[str, Type[BaseEmbeddingEvaluator]] = {}
        self._evaluator_configs: Dict[str, Dict[str, Any]] = {}
        self._evaluator_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Auto-register built-in evaluators
        self._register_builtin_evaluators()
    
    def register_evaluator(self, 
                          name: str, 
                          evaluator_class: Type[BaseEmbeddingEvaluator],
                          config: Optional[Dict[str, Any]] = None,
                          metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Register an evaluator class
        
        Args:
            name: Unique name for the evaluator
            evaluator_class: Evaluator class that extends BaseEmbeddingEvaluator
            config: Default configuration for the evaluator
            metadata: Additional metadata about the evaluator
        """
        if not issubclass(evaluator_class, BaseEmbeddingEvaluator):
            raise ValueError(f"Evaluator class must inherit from BaseEmbeddingEvaluator")
        
        if name in self._evaluators:
            logger.warning(f"Overriding existing evaluator: {name}")
        
        self._evaluators[name] = evaluator_class
        self._evaluator_configs[name] = config or {}
        self._evaluator_metadata[name] = metadata or {}
        
        logger.info(f"Registered evaluator: {name} -> {evaluator_class.__name__}")
    
    def unregister_evaluator(self, name: str) -> None:
        """Unregister an evaluator"""
        if name in self._evaluators:
            del self._evaluators[name]
            self._evaluator_configs.pop(name, None)
            self._evaluator_metadata.pop(name, None)
            logger.info(f"Unregistered evaluator: {name}")
        else:
            logger.warning(f"Evaluator not found: {name}")
    
    def create_evaluator(self, 
                        name: str, 
                        config: Optional[Dict[str, Any]] = None) -> BaseEmbeddingEvaluator:
        """
        Create an evaluator instance
        
        Args:
            name: Name of the evaluator to create
            config: Configuration to override defaults
            
        Returns:
            Evaluator instance
        """
        if name not in self._evaluators:
            raise ValueError(f"Unknown evaluator: {name}. Available: {list(self._evaluators.keys())}")
        
        evaluator_class = self._evaluators[name]
        
        # Merge default config with provided config
        final_config = self._evaluator_configs[name].copy()
        if config:
            final_config.update(config)
        
        try:
            return evaluator_class(config=final_config)
        except Exception as e:
            logger.error(f"Failed to create evaluator {name}: {e}")
            raise
    
    def get_evaluator_info(self, name: str) -> Dict[str, Any]:
        """Get information about a registered evaluator"""
        if name not in self._evaluators:
            raise ValueError(f"Unknown evaluator: {name}")
        
        evaluator_class = self._evaluators[name]
        
        # Create temporary instance to get required metrics
        try:
            temp_instance = evaluator_class()
            required_metrics = temp_instance.get_required_metrics()
        except Exception:
            required_metrics = []
        
        return {
            'name': name,
            'class': evaluator_class.__name__,
            'module': evaluator_class.__module__,
            'docstring': evaluator_class.__doc__,
            'required_metrics': required_metrics,
            'default_config': self._evaluator_configs[name].copy(),
            'metadata': self._evaluator_metadata[name].copy()
        }
    
    def list_evaluators(self) -> List[str]:
        """List all registered evaluator names"""
        return list(self._evaluators.keys())
    
    def list_evaluator_details(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed information about all registered evaluators"""
        details = {}
        for name in self._evaluators.keys():
            try:
                details[name] = self.get_evaluator_info(name)
            except Exception as e:
                logger.warning(f"Could not get info for evaluator {name}: {e}")
                details[name] = {'error': str(e)}
        return details
    
    def is_registered(self, name: str) -> bool:
        """Check if an evaluator is registered"""
        return name in self._evaluators
    
    def get_evaluators_by_task(self, task_type: str) -> List[str]:
        """Get evaluators suitable for a specific task type"""
        suitable_evaluators = []
        
        task_mapping = {
            'similarity': ['similarity', 'semantic_similarity'],
            'clustering': ['clustering'],
            'retrieval': ['retrieval'],
            'classification': ['classification'],
            'efficiency': ['efficiency', 'performance'],
            'robustness': ['robustness', 'stability']
        }
        
        target_names = task_mapping.get(task_type.lower(), [task_type.lower()])
        
        for evaluator_name in self._evaluators.keys():
            if any(target in evaluator_name.lower() for target in target_names):
                suitable_evaluators.append(evaluator_name)
        
        return suitable_evaluators
    
    def validate_evaluator_config(self, name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize evaluator configuration"""
        if name not in self._evaluators:
            raise ValueError(f"Unknown evaluator: {name}")
        
        # Get default config
        default_config = self._evaluator_configs[name].copy()
        
        # Merge and validate
        validated_config = default_config.copy()
        validated_config.update(config)
        
        # Basic validation (can be extended)
        validation_results = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'config': validated_config
        }
        
        # Check for unknown keys (warning only)
        if default_config:
            unknown_keys = set(config.keys()) - set(default_config.keys())
            if unknown_keys:
                validation_results['warnings'].append(
                    f"Unknown configuration keys: {list(unknown_keys)}"
                )
        
        return validation_results
    
    def create_evaluation_pipeline(self, 
                                  evaluator_configs: List[Dict[str, Any]]) -> 'EvaluationPipeline':
        """Create an evaluation pipeline with multiple evaluators"""
        return EvaluationPipeline(self, evaluator_configs)
    
    def _register_builtin_evaluators(self) -> None:
        """Register built-in evaluators"""
        try:
            # Import and register semantic similarity evaluator
            from .semantic_similarity_evaluator import SemanticSimilarityEvaluator
            self.register_evaluator(
                'similarity',
                SemanticSimilarityEvaluator,
                metadata={'task_type': 'similarity', 'builtin': True}
            )
            
            # Import and register clustering evaluator
            from .clustering_evaluator import ClusteringEvaluator
            self.register_evaluator(
                'clustering',
                ClusteringEvaluator,
                metadata={'task_type': 'clustering', 'builtin': True}
            )
            
            # Import and register retrieval evaluator
            from .retrieval_evaluator import RetrievalEvaluator
            self.register_evaluator(
                'retrieval',
                RetrievalEvaluator,
                metadata={'task_type': 'retrieval', 'builtin': True}
            )
            
            # Import and register classification evaluator
            from .classification_evaluator import ClassificationEvaluator
            self.register_evaluator(
                'classification',
                ClassificationEvaluator,
                metadata={'task_type': 'classification', 'builtin': True}
            )
            
            # Import and register efficiency evaluator
            from .efficiency_evaluator import EfficiencyEvaluator
            self.register_evaluator(
                'efficiency',
                EfficiencyEvaluator,
                metadata={'task_type': 'efficiency', 'builtin': True}
            )
            
            # Import and register robustness evaluator
            from .robustness_evaluator import RobustnessEvaluator
            self.register_evaluator(
                'robustness',
                RobustnessEvaluator,
                metadata={'task_type': 'robustness', 'builtin': True}
            )
            
            logger.info("Successfully registered all built-in evaluators")
            
        except ImportError as e:
            logger.error(f"Failed to import built-in evaluators: {e}")
        except Exception as e:
            logger.error(f"Error registering built-in evaluators: {e}")

class EvaluationPipeline:
    """Pipeline for running multiple evaluators in sequence"""
    
    def __init__(self, factory: EvaluatorFactory, evaluator_configs: List[Dict[str, Any]]):
        self.factory = factory
        self.evaluator_configs = evaluator_configs
        self.evaluators = []
        
        # Create evaluator instances
        for config in evaluator_configs:
            if 'name' not in config:
                raise ValueError("Evaluator config must include 'name' field")
            
            evaluator_name = config['name']
            evaluator_config = config.get('config', {})
            
            evaluator = self.factory.create_evaluator(evaluator_name, evaluator_config)
            self.evaluators.append({
                'name': evaluator_name,
                'instance': evaluator,
                'config': config
            })
    
    def run_evaluation(self, 
                      embeddings: List,
                      samples: List,
                      model_name: str,
                      dataset_name: str,
                      **kwargs) -> Dict[str, EvaluationResult]:
        """Run all evaluators in the pipeline"""
        results = {}
        
        for evaluator_info in self.evaluators:
            evaluator_name = evaluator_info['name']
            evaluator = evaluator_info['instance']
            
            try:
                logger.info(f"Running evaluator: {evaluator_name}")
                
                result = evaluator.evaluate(
                    embeddings=embeddings,
                    samples=samples,
                    model_name=model_name,
                    dataset_name=dataset_name,
                    **kwargs
                )
                
                results[evaluator_name] = result
                logger.info(f"Completed evaluator: {evaluator_name}")
                
            except Exception as e:
                logger.error(f"Evaluator {evaluator_name} failed: {e}")
                # Continue with other evaluators
                results[evaluator_name] = None
        
        return results
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the evaluation pipeline"""
        return {
            'num_evaluators': len(self.evaluators),
            'evaluator_names': [e['name'] for e in self.evaluators],
            'evaluator_configs': self.evaluator_configs
        }

# Global factory instance
EVALUATOR_FACTORY = EvaluatorFactory()

# Convenience functions
def register_evaluator(name: str, 
                      evaluator_class: Type[BaseEmbeddingEvaluator],
                      config: Optional[Dict[str, Any]] = None,
                      metadata: Optional[Dict[str, Any]] = None) -> None:
    """Register an evaluator with the global factory"""
    EVALUATOR_FACTORY.register_evaluator(name, evaluator_class, config, metadata)

def create_evaluator(name: str, config: Optional[Dict[str, Any]] = None) -> BaseEmbeddingEvaluator:
    """Create an evaluator instance using the global factory"""
    return EVALUATOR_FACTORY.create_evaluator(name, config)

def list_evaluators() -> List[str]:
    """List all available evaluators"""
    return EVALUATOR_FACTORY.list_evaluators()

def get_evaluator_info(name: str) -> Dict[str, Any]:
    """Get information about an evaluator"""
    return EVALUATOR_FACTORY.get_evaluator_info(name)

def create_evaluation_pipeline(evaluator_configs: List[Dict[str, Any]]) -> EvaluationPipeline:
    """Create an evaluation pipeline"""
    return EVALUATOR_FACTORY.create_evaluation_pipeline(evaluator_configs)

def evaluate_embeddings(evaluator_name: str,
                       embeddings: List,
                       samples: List,
                       model_name: str,
                       dataset_name: str,
                       config: Optional[Dict[str, Any]] = None,
                       **kwargs) -> EvaluationResult:
    """
    Convenience function to evaluate embeddings with a single evaluator
    
    Args:
        evaluator_name: Name of the evaluator to use
        embeddings: List of embedding vectors
        samples: List of dataset samples
        model_name: Name of the embedding model
        dataset_name: Name of the dataset
        config: Evaluator configuration
        **kwargs: Additional evaluation parameters
        
    Returns:
        EvaluationResult
    """
    evaluator = create_evaluator(evaluator_name, config)
    return evaluator.evaluate(embeddings, samples, model_name, dataset_name, **kwargs)
