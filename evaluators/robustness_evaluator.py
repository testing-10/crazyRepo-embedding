"""
Robustness Evaluator

Evaluates embedding models on robustness metrics including performance under
noise, adversarial inputs, domain shifts, and input variations.
"""

import time
import numpy as np
import random
import string
from typing import Dict, List, Any, Optional, Tuple, Callable
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from .base_embedding_evaluator import BaseEmbeddingEvaluator, EvaluationResult
from datasets.dataset_loader import DatasetSample
from utils.vector_operations import VectorOperations

class RobustnessEvaluator(BaseEmbeddingEvaluator):
    """Evaluator for robustness and stability metrics"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.vector_ops = VectorOperations()
        
        # Default configuration
        self.default_config = {
            'perturbation_types': ['typos', 'case_changes', 'punctuation', 'whitespace', 'synonyms'],
            'noise_levels': [0.1, 0.2, 0.3],
            'metrics': ['stability', 'consistency', 'degradation'],
            'num_perturbations_per_sample': 3,
            'max_test_samples': 500,
            'similarity_threshold': 0.8,
            'random_seed': 42
        }
        
        # Merge with provided config
        self.config = {**self.default_config, **self.config}
        
        # Set random seed for reproducibility
        random.seed(self.config['random_seed'])
        np.random.seed(self.config['random_seed'])
    
    def evaluate(self, 
                 embeddings: List[np.ndarray], 
                 samples: List[DatasetSample],
                 model_name: str,
                 dataset_name: str,
                 embedding_function: Optional[Callable] = None,
                 **kwargs) -> EvaluationResult:
        """Evaluate robustness and stability metrics"""
        
        start_time = time.time()
        self._log_evaluation_start(model_name, dataset_name, len(samples))
        
        try:
            # Validate inputs
            if not self._validate_inputs(embeddings, samples):
                raise ValueError("Input validation failed")
            
            # Prepare embeddings and filter samples
            valid_embeddings, valid_indices = self._prepare_embeddings(embeddings)
            valid_samples = self._filter_samples(samples, valid_indices)
            
            # Limit samples for performance
            if len(valid_samples) > self.config['max_test_samples']:
                indices = np.random.choice(len(valid_samples), self.config['max_test_samples'], replace=False)
                valid_embeddings = valid_embeddings[indices]
                valid_samples = [valid_samples[i] for i in indices]
                self.logger.info(f"Limited to {self.config['max_test_samples']} samples for robustness evaluation")
            
            # Perform robustness evaluation
            if embedding_function is not None:
                metrics = self._evaluate_robustness_with_function(
                    embedding_function, valid_samples, valid_embeddings
                )
            else:
                metrics = self._evaluate_robustness_precomputed(valid_embeddings, valid_samples)
            
            # Generate metadata
            metadata = self._generate_metadata(valid_embeddings, valid_samples)
            
            execution_time = time.time() - start_time
            self._log_evaluation_end(metrics, execution_time)
            
            return self._create_result(
                task_type="robustness",
                model_name=model_name,
                dataset_name=dataset_name,
                metrics=metrics,
                metadata=metadata,
                execution_time=execution_time,
                num_samples=len(valid_samples)
            )
            
        except Exception as e:
            self.logger.error(f"Robustness evaluation failed: {e}")
            raise
    
    def _evaluate_robustness_with_function(self, embedding_function: Callable, 
                                         samples: List[DatasetSample], 
                                         original_embeddings: np.ndarray) -> Dict[str, float]:
        """Evaluate robustness with access to embedding function"""
        metrics = {}
        
        # Test different perturbation types
        for perturbation_type in self.config['perturbation_types']:
            try:
                perturbation_metrics = self._test_perturbation_robustness(
                    embedding_function, samples, original_embeddings, perturbation_type
                )
                
                # Add perturbation type prefix
                for metric_name, value in perturbation_metrics.items():
                    metrics[f'{perturbation_type}_{metric_name}'] = value
                
            except Exception as e:
                self.logger.warning(f"Error testing {perturbation_type} perturbations: {e}")
        
        # Test noise robustness
        for noise_level in self.config['noise_levels']:
            try:
                noise_metrics = self._test_noise_robustness(
                    embedding_function, samples, original_embeddings, noise_level
                )
                
                # Add noise level prefix
                for metric_name, value in noise_metrics.items():
                    metrics[f'noise_{int(noise_level*100)}_{metric_name}'] = value
                
            except Exception as e:
                self.logger.warning(f"Error testing noise level {noise_level}: {e}")
        
        # Overall robustness metrics
        stability_scores = [v for k, v in metrics.items() if 'stability' in k]
        if stability_scores:
            metrics['overall_stability'] = float(np.mean(stability_scores))
            metrics['min_stability'] = float(np.min(stability_scores))
        
        return metrics
    
    def _evaluate_robustness_precomputed(self, embeddings: np.ndarray, 
                                       samples: List[DatasetSample]) -> Dict[str, float]:
        """Evaluate robustness for pre-computed embeddings (limited metrics)"""
        metrics = {}
        
        # Embedding consistency across similar samples
        consistency_scores = self._measure_embedding_consistency(embeddings, samples)
        if consistency_scores:
            metrics['embedding_consistency'] = float(np.mean(consistency_scores))
            metrics['consistency_std'] = float(np.std(consistency_scores))
        
        # Embedding stability (variance analysis)
        if len(embeddings) > 1:
            # Measure variance in embedding magnitudes
            embedding_norms = np.linalg.norm(embeddings, axis=1)
            metrics['embedding_norm_stability'] = float(1.0 / (np.std(embedding_norms) + 1e-8))
            
            # Measure cosine similarity distribution
            sample_indices = np.random.choice(len(embeddings), min(100, len(embeddings)), replace=False)
            sample_embeddings = embeddings[sample_indices]
            similarities = cosine_similarity(sample_embeddings)
            
            # Remove diagonal (self-similarities)
            similarities = similarities[np.triu_indices_from(similarities, k=1)]
            
            metrics['similarity_mean'] = float(np.mean(similarities))
            metrics['similarity_std'] = float(np.std(similarities))
            metrics['similarity_stability'] = float(1.0 / (np.std(similarities) + 1e-8))
        
        return metrics
    
    def _test_perturbation_robustness(self, embedding_function: Callable, 
                                    samples: List[DatasetSample], 
                                    original_embeddings: np.ndarray,
                                    perturbation_type: str) -> Dict[str, float]:
        """Test robustness against specific perturbation type"""
        stability_scores = []
        degradation_scores = []
        
        for i, sample in enumerate(samples):
            if not sample.text1:
                continue
            
            original_embedding = original_embeddings[i]
            
            # Generate perturbed versions
            perturbed_texts = self._generate_perturbations(
                sample.text1, perturbation_type, self.config['num_perturbations_per_sample']
            )
            
            if not perturbed_texts:
                continue
            
            try:
                # Get embeddings for perturbed texts
                perturbed_embeddings = embedding_function(perturbed_texts)
                
                if perturbed_embeddings is None or len(perturbed_embeddings) == 0:
                    continue
                
                # Calculate stability (similarity to original)
                similarities = cosine_similarity([original_embedding], perturbed_embeddings)[0]
                stability_score = np.mean(similarities)
                stability_scores.append(stability_score)
                
                # Calculate degradation (how much performance drops)
                degradation = 1.0 - stability_score
                degradation_scores.append(degradation)
                
            except Exception as e:
                self.logger.warning(f"Error processing perturbed sample {i}: {e}")
                continue
        
        metrics = {}
        if stability_scores:
            metrics['stability'] = float(np.mean(stability_scores))
            metrics['stability_std'] = float(np.std(stability_scores))
            metrics['min_stability'] = float(np.min(stability_scores))
        
        if degradation_scores:
            metrics['degradation'] = float(np.mean(degradation_scores))
            metrics['max_degradation'] = float(np.max(degradation_scores))
        
        return metrics
    
    def _test_noise_robustness(self, embedding_function: Callable, 
                             samples: List[DatasetSample], 
                             original_embeddings: np.ndarray,
                             noise_level: float) -> Dict[str, float]:
        """Test robustness against noise injection"""
        stability_scores = []
        
        for i, sample in enumerate(samples):
            if not sample.text1:
                continue
            
            original_embedding = original_embeddings[i]
            
            # Generate noisy versions
            noisy_texts = self._add_noise_to_text(sample.text1, noise_level, 3)
            
            if not noisy_texts:
                continue
            
            try:
                # Get embeddings for noisy texts
                noisy_embeddings = embedding_function(noisy_texts)
                
                if noisy_embeddings is None or len(noisy_embeddings) == 0:
                    continue
                
                # Calculate stability
                similarities = cosine_similarity([original_embedding], noisy_embeddings)[0]
                stability_score = np.mean(similarities)
                stability_scores.append(stability_score)
                
            except Exception as e:
                self.logger.warning(f"Error processing noisy sample {i}: {e}")
                continue
        
        metrics = {}
        if stability_scores:
            metrics['stability'] = float(np.mean(stability_scores))
            metrics['stability_std'] = float(np.std(stability_scores))
        
        return metrics
    
    def _generate_perturbations(self, text: str, perturbation_type: str, num_perturbations: int) -> List[str]:
        """Generate perturbed versions of text"""
        perturbations = []
        
        for _ in range(num_perturbations):
            try:
                if perturbation_type == 'typos':
                    perturbed = self._add_typos(text)
                elif perturbation_type == 'case_changes':
                    perturbed = self._change_case(text)
                elif perturbation_type == 'punctuation':
                    perturbed = self._modify_punctuation(text)
                elif perturbation_type == 'whitespace':
                    perturbed = self._modify_whitespace(text)
                elif perturbation_type == 'synonyms':
                    perturbed = self._replace_with_synonyms(text)
                else:
                    continue
                
                if perturbed and perturbed != text:
                    perturbations.append(perturbed)
                    
            except Exception as e:
                self.logger.warning(f"Error generating {perturbation_type} perturbation: {e}")
                continue
        
        return perturbations
    
    def _add_typos(self, text: str) -> str:
        """Add random typos to text"""
        words = text.split()
        if not words:
            return text
        
        # Randomly select words to modify
        num_typos = max(1, len(words) // 10)  # 10% of words
        word_indices = random.sample(range(len(words)), min(num_typos, len(words)))
        
        for idx in word_indices:
            word = words[idx]
            if len(word) > 2:
                # Random character substitution
                char_idx = random.randint(0, len(word) - 1)
                new_char = random.choice(string.ascii_lowercase)
                words[idx] = word[:char_idx] + new_char + word[char_idx + 1:]
        
        return ' '.join(words)
    
    def _change_case(self, text: str) -> str:
        """Randomly change case of characters"""
        result = []
        for char in text:
            if char.isalpha():
                if random.random() < 0.3:  # 30% chance to flip case
                    result.append(char.swapcase())
                else:
                    result.append(char)
            else:
                result.append(char)
        return ''.join(result)
    
    def _modify_punctuation(self, text: str) -> str:
        """Randomly modify punctuation"""
        punctuation_chars = '.,!?;:'
        result = []
        
        for char in text:
            if char in punctuation_chars:
                if random.random() < 0.5:  # 50% chance to modify
                    if random.random() < 0.3:  # Remove punctuation
                        continue
                    else:  # Replace with different punctuation
                        result.append(random.choice(punctuation_chars))
                else:
                    result.append(char)
            else:
                result.append(char)
        
        return ''.join(result)
    
    def _modify_whitespace(self, text: str) -> str:
        """Randomly modify whitespace"""
        # Add extra spaces
        words = text.split()
        result = []
        
        for i, word in enumerate(words):
            result.append(word)
            if i < len(words) - 1:
                # Random number of spaces (1-3)
                num_spaces = random.randint(1, 3)
                result.append(' ' * num_spaces)
        
        return ''.join(result)
    
    def _replace_with_synonyms(self, text: str) -> str:
        """Replace words with simple synonyms (basic implementation)"""
        # Simple synonym dictionary for common words
        synonyms = {
            'good': ['great', 'excellent', 'fine', 'nice'],
            'bad': ['poor', 'terrible', 'awful', 'horrible'],
            'big': ['large', 'huge', 'enormous', 'massive'],
            'small': ['tiny', 'little', 'mini', 'compact'],
            'fast': ['quick', 'rapid', 'swift', 'speedy'],
            'slow': ['sluggish', 'gradual', 'leisurely', 'delayed']
        }
        
        words = text.split()
        for i, word in enumerate(words):
            word_lower = word.lower().strip('.,!?;:')
            if word_lower in synonyms and random.random() < 0.2:  # 20% chance
                synonym = random.choice(synonyms[word_lower])
                # Preserve original case
                if word[0].isupper():
                    synonym = synonym.capitalize()
                words[i] = word.replace(word_lower, synonym)
        
        return ' '.join(words)
    
    def _add_noise_to_text(self, text: str, noise_level: float, num_versions: int) -> List[str]:
        """Add noise to text at specified level"""
        noisy_versions = []
        
        for _ in range(num_versions):
            words = text.split()
            if not words:
                continue
            
            # Determine number of words to modify based on noise level
            num_modifications = int(len(words) * noise_level)
            if num_modifications == 0:
                num_modifications = 1
            
            # Randomly select words to modify
            word_indices = random.sample(range(len(words)), min(num_modifications, len(words)))
            
            noisy_words = words.copy()
            for idx in word_indices:
                # Apply random modification
                modification_type = random.choice(['typo', 'case', 'duplicate', 'remove'])
                
                if modification_type == 'typo' and len(noisy_words[idx]) > 1:
                    # Add character substitution
                    word = noisy_words[idx]
                    char_idx = random.randint(0, len(word) - 1)
                    new_char = random.choice(string.ascii_lowercase)
                    noisy_words[idx] = word[:char_idx] + new_char + word[char_idx + 1:]
                    
                elif modification_type == 'case':
                    noisy_words[idx] = noisy_words[idx].swapcase()
                    
                elif modification_type == 'duplicate':
                    noisy_words[idx] = noisy_words[idx] + ' ' + noisy_words[idx]
                    
                elif modification_type == 'remove' and len(noisy_words) > 1:
                    noisy_words[idx] = ''
            
            noisy_text = ' '.join([w for w in noisy_words if w])
            if noisy_text and noisy_text != text:
                noisy_versions.append(noisy_text)
        
        return noisy_versions
    
    def _measure_embedding_consistency(self, embeddings: np.ndarray, 
                                     samples: List[DatasetSample]) -> List[float]:
        """Measure consistency of embeddings for similar samples"""
        consistency_scores = []
        
        # Group samples by domain or category if available
        groups = {}
        for i, sample in enumerate(samples):
            if sample.metadata:
                group_key = sample.metadata.get('domain') or sample.metadata.get('category')
                if group_key:
                    if group_key not in groups:
                        groups[group_key] = []
                    groups[group_key].append(i)
        
        # Calculate within-group consistency
        for group_indices in groups.values():
            if len(group_indices) < 2:
                continue
            
            group_embeddings = embeddings[group_indices]
            similarities = cosine_similarity(group_embeddings)
            
            # Average similarity within group (excluding diagonal)
            mask = np.ones_like(similarities, dtype=bool)
            np.fill_diagonal(mask, False)
            avg_similarity = np.mean(similarities[mask])
            consistency_scores.append(avg_similarity)
        
        return consistency_scores
    
    def _generate_metadata(self, embeddings: np.ndarray, samples: List[DatasetSample]) -> Dict[str, Any]:
        """Generate robustness evaluation metadata"""
        metadata = {
            'perturbation_types': self.config['perturbation_types'],
            'noise_levels': self.config['noise_levels'],
            'num_perturbations_per_sample': self.config['num_perturbations_per_sample'],
            'max_test_samples': self.config['max_test_samples'],
            'similarity_threshold': self.config['similarity_threshold'],
            'config': self.config.copy()
        }
        
        # Sample statistics
        if len(samples) > 0:
            text_lengths = [len(sample.text1.split()) if sample.text1 else 0 for sample in samples]
            metadata['text_stats'] = {
                'avg_length': float(np.mean(text_lengths)),
                'min_length': int(np.min(text_lengths)),
                'max_length': int(np.max(text_lengths))
            }
        
        # Embedding statistics
        if len(embeddings) > 0:
            embedding_norms = np.linalg.norm(embeddings, axis=1)
            metadata['embedding_stats'] = {
                'dimension': int(embeddings.shape[1]),
                'avg_norm': float(np.mean(embedding_norms)),
                'std_norm': float(np.std(embedding_norms))
            }
        
        return metadata
    
    def get_required_metrics(self) -> List[str]:
        """Get list of metrics this evaluator provides"""
        base_metrics = ['stability', 'consistency', 'degradation']
        
        # Add perturbation-specific metrics
        perturbation_metrics = []
        for perturbation in self.config['perturbation_types']:
            perturbation_metrics.extend([
                f'{perturbation}_stability',
                f'{perturbation}_degradation'
            ])
        
        # Add noise-specific metrics
        noise_metrics = []
        for noise_level in self.config['noise_levels']:
            noise_metrics.append(f'noise_{int(noise_level*100)}_stability')
        
        return base_metrics + perturbation_metrics + noise_metrics
