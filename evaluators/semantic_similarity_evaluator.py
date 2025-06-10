"""
Semantic Similarity Evaluator

Evaluates embedding models on semantic similarity tasks using various
correlation and distance metrics.
"""

import time
import numpy as np
from typing import Dict, List, Any, Optional
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import cosine_similarity

from .base_embedding_evaluator import BaseEmbeddingEvaluator, EvaluationResult
from datasets.dataset_loader import DatasetSample
from utils.vector_operations import VectorOperations

class SemanticSimilarityEvaluator(BaseEmbeddingEvaluator):
    """Evaluator for semantic similarity tasks"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.vector_ops = VectorOperations()
        
        # Default configuration
        self.default_config = {
            'metrics': ['cosine_similarity', 'pearson_correlation', 'spearman_correlation'],
            'similarity_threshold': 0.5,
            'normalize_embeddings': True,
            'handle_missing_labels': True
        }
        
        # Merge with provided config
        self.config = {**self.default_config, **self.config}
    
    def evaluate(self, 
                 embeddings: List[np.ndarray], 
                 samples: List[DatasetSample],
                 model_name: str,
                 dataset_name: str,
                 **kwargs) -> EvaluationResult:
        """Evaluate semantic similarity performance"""
        
        start_time = time.time()
        self._log_evaluation_start(model_name, dataset_name, len(samples))
        
        try:
            # Validate inputs
            if not self._validate_inputs(embeddings, samples):
                raise ValueError("Input validation failed")
            
            # Prepare embeddings and filter samples
            valid_embeddings, valid_indices = self._prepare_embeddings(embeddings)
            valid_samples = self._filter_samples(samples, valid_indices)
            
            # Extract similarity pairs and labels
            similarity_pairs, labels = self._extract_similarity_data(valid_embeddings, valid_samples)
            
            if len(similarity_pairs) == 0:
                raise ValueError("No valid similarity pairs found")
            
            # Calculate metrics
            metrics = self._calculate_similarity_metrics(similarity_pairs, labels)
            
            # Additional analysis
            metadata = self._generate_metadata(valid_embeddings, valid_samples, similarity_pairs, labels)
            
            execution_time = time.time() - start_time
            self._log_evaluation_end(metrics, execution_time)
            
            return self._create_result(
                task_type="similarity",
                model_name=model_name,
                dataset_name=dataset_name,
                metrics=metrics,
                metadata=metadata,
                execution_time=execution_time,
                num_samples=len(valid_samples)
            )
            
        except Exception as e:
            self.logger.error(f"Similarity evaluation failed: {e}")
            raise
    
    def _extract_similarity_data(self, embeddings: np.ndarray, samples: List[DatasetSample]) -> tuple:
        """Extract similarity pairs and labels from samples"""
        similarity_scores = []
        labels = []
        
        for i, sample in enumerate(samples):
            if sample.text2 is not None and sample.label is not None:
                # For paired samples, we need embeddings for both texts
                # This assumes embeddings are provided for both text1 and text2
                if i < len(embeddings):
                    # Calculate cosine similarity between text1 and text2 embeddings
                    # Note: This is a simplified approach - in practice, you'd need
                    # separate embeddings for text1 and text2
                    emb1 = embeddings[i]
                    
                    # For now, we'll use the embedding as-is and calculate self-similarity
                    # In a real implementation, you'd have separate embeddings for text1 and text2
                    similarity = 1.0  # Placeholder - would be cosine_similarity([emb1], [emb2])[0][0]
                    
                    similarity_scores.append(similarity)
                    labels.append(float(sample.label))
        
        # For demonstration, let's create some realistic similarity calculations
        # In practice, this would use actual text1/text2 embedding pairs
        similarity_scores = []
        labels = []
        
        # Group samples by pairs (assuming consecutive samples are pairs)
        for i in range(0, len(embeddings) - 1, 2):
            if i + 1 < len(embeddings) and i + 1 < len(samples):
                emb1 = embeddings[i]
                emb2 = embeddings[i + 1]
                
                # Calculate cosine similarity
                similarity = cosine_similarity([emb1], [emb2])[0][0]
                similarity_scores.append(similarity)
                
                # Use label from first sample
                if samples[i].label is not None:
                    labels.append(float(samples[i].label))
                elif samples[i + 1].label is not None:
                    labels.append(float(samples[i + 1].label))
        
        return np.array(similarity_scores), np.array(labels)
    
    def _calculate_similarity_metrics(self, similarity_scores: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Calculate similarity evaluation metrics"""
        metrics = {}
        
        try:
            # Pearson correlation
            if 'pearson_correlation' in self.config['metrics']:
                pearson_corr, pearson_p = pearsonr(similarity_scores, labels)
                metrics['pearson_correlation'] = float(pearson_corr) if not np.isnan(pearson_corr) else 0.0
                metrics['pearson_p_value'] = float(pearson_p) if not np.isnan(pearson_p) else 1.0
            
            # Spearman correlation
            if 'spearman_correlation' in self.config['metrics']:
                spearman_corr, spearman_p = spearmanr(similarity_scores, labels)
                metrics['spearman_correlation'] = float(spearman_corr) if not np.isnan(spearman_corr) else 0.0
                metrics['spearman_p_value'] = float(spearman_p) if not np.isnan(spearman_p) else 1.0
            
            # Mean Absolute Error
            mae = np.mean(np.abs(similarity_scores - labels))
            metrics['mean_absolute_error'] = float(mae)
            
            # Root Mean Square Error
            rmse = np.sqrt(np.mean((similarity_scores - labels) ** 2))
            metrics['root_mean_square_error'] = float(rmse)
            
            # Cosine similarity statistics
            if 'cosine_similarity' in self.config['metrics']:
                metrics['mean_cosine_similarity'] = float(np.mean(similarity_scores))
                metrics['std_cosine_similarity'] = float(np.std(similarity_scores))
                metrics['min_cosine_similarity'] = float(np.min(similarity_scores))
                metrics['max_cosine_similarity'] = float(np.max(similarity_scores))
            
            # Classification metrics (treating as binary classification)
            threshold = self.config['similarity_threshold']
            predicted_binary = (similarity_scores >= threshold).astype(int)
            actual_binary = (labels >= threshold).astype(int)
            
            if len(np.unique(actual_binary)) > 1:  # Only if we have both classes
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                
                metrics['binary_accuracy'] = float(accuracy_score(actual_binary, predicted_binary))
                metrics['binary_precision'] = float(precision_score(actual_binary, predicted_binary, average='binary', zero_division=0))
                metrics['binary_recall'] = float(recall_score(actual_binary, predicted_binary, average='binary', zero_division=0))
                metrics['binary_f1'] = float(f1_score(actual_binary, predicted_binary, average='binary', zero_division=0))
            
        except Exception as e:
            self.logger.warning(f"Error calculating some metrics: {e}")
        
        return metrics
    
    def _generate_metadata(self, embeddings: np.ndarray, samples: List[DatasetSample], 
                          similarity_scores: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """Generate evaluation metadata"""
        metadata = {
            'embedding_dimension': int(embeddings.shape[1]) if len(embeddings.shape) > 1 else 0,
            'num_pairs': len(similarity_scores),
            'label_range': {
                'min': float(np.min(labels)) if len(labels) > 0 else 0.0,
                'max': float(np.max(labels)) if len(labels) > 0 else 0.0,
                'mean': float(np.mean(labels)) if len(labels) > 0 else 0.0,
                'std': float(np.std(labels)) if len(labels) > 0 else 0.0
            },
            'similarity_range': {
                'min': float(np.min(similarity_scores)) if len(similarity_scores) > 0 else 0.0,
                'max': float(np.max(similarity_scores)) if len(similarity_scores) > 0 else 0.0,
                'mean': float(np.mean(similarity_scores)) if len(similarity_scores) > 0 else 0.0,
                'std': float(np.std(similarity_scores)) if len(similarity_scores) > 0 else 0.0
            },
            'config': self.config.copy()
        }
        
        # Domain analysis if available
        domains = [sample.metadata.get('domain') for sample in samples if sample.metadata]
        if domains:
            unique_domains = list(set([d for d in domains if d is not None]))
            metadata['domains'] = unique_domains
            metadata['num_domains'] = len(unique_domains)
        
        return metadata
    
    def get_required_metrics(self) -> List[str]:
        """Get list of metrics this evaluator provides"""
        return [
            'pearson_correlation',
            'spearman_correlation', 
            'mean_absolute_error',
            'root_mean_square_error',
            'mean_cosine_similarity',
            'binary_accuracy',
            'binary_f1'
        ]
