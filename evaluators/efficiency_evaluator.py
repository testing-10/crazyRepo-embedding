"""
Efficiency Evaluator

Evaluates embedding models on efficiency metrics including inference time,
memory usage, throughput, and computational cost analysis.
"""

import time
import psutil
import gc
import numpy as np
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
import threading
from concurrent.futures import ThreadPoolExecutor
import sys

from evaluators.base_embedding_evaluator import BaseEmbeddingEvaluator, EvaluationResult
from datasets.dataset_loader import DatasetSample
from utils.embedding_cost_tracker import CostTracker

@dataclass
class PerformanceMetrics:
    """Container for performance measurements"""
    inference_time: float
    memory_usage_mb: float
    cpu_usage_percent: float
    throughput_samples_per_second: float
    peak_memory_mb: float

class EfficiencyEvaluator(BaseEmbeddingEvaluator):
    """Evaluator for efficiency and performance metrics"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Default configuration
        self.default_config = {
            'metrics': ['inference_time', 'memory_usage', 'throughput', 'cost_efficiency'],
            'batch_sizes': [1, 8, 16, 32, 64],
            'warmup_iterations': 3,
            'measurement_iterations': 5,
            'monitor_memory': True,
            'monitor_cpu': True,
            'max_test_samples': 1000,
            'timeout_seconds': 300  # 5 minutes timeout
        }
        
        # Merge with provided config
        self.config = {**self.default_config, **self.config}
        
        # Performance monitoring
        self.memory_monitor = MemoryMonitor()
        self.cpu_monitor = CPUMonitor()
    
    def evaluate(self, 
                 embeddings: List[np.ndarray], 
                 samples: List[DatasetSample],
                 model_name: str,
                 dataset_name: str,
                 embedding_function: Optional[Callable] = None,
                 **kwargs) -> EvaluationResult:
        """Evaluate efficiency and performance metrics"""
        
        start_time = time.time()
        self._log_evaluation_start(model_name, dataset_name, len(samples))
        
        try:
            # We need the embedding function to measure inference time
            if embedding_function is None:
                self.logger.warning("No embedding function provided, using pre-computed embeddings for limited metrics")
                return self._evaluate_precomputed_efficiency(embeddings, samples, model_name, dataset_name)
            
            # Prepare test data
            test_texts = self._prepare_test_texts(samples)
            
            # Perform efficiency evaluation
            metrics = self._evaluate_efficiency(embedding_function, test_texts, model_name)
            
            # Generate metadata
            metadata = self._generate_metadata(test_texts, samples, embeddings)
            
            execution_time = time.time() - start_time
            self._log_evaluation_end(metrics, execution_time)
            
            return self._create_result(
                task_type="efficiency",
                model_name=model_name,
                dataset_name=dataset_name,
                metrics=metrics,
                metadata=metadata,
                execution_time=execution_time,
                num_samples=len(test_texts)
            )
            
        except Exception as e:
            self.logger.error(f"Efficiency evaluation failed: {e}")
            raise
    
    def _prepare_test_texts(self, samples: List[DatasetSample]) -> List[str]:
        """Prepare text samples for efficiency testing"""
        texts = []
        
        for sample in samples:
            if sample.text1:
                texts.append(sample.text1)
            if sample.text2:
                texts.append(sample.text2)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_texts = []
        for text in texts:
            if text not in seen:
                seen.add(text)
                unique_texts.append(text)
        
        # Limit for performance testing
        if len(unique_texts) > self.config['max_test_samples']:
            unique_texts = unique_texts[:self.config['max_test_samples']]
            self.logger.info(f"Limited test texts to {self.config['max_test_samples']} for efficiency evaluation")
        
        return unique_texts
    
    def _evaluate_efficiency(self, embedding_function: Callable, texts: List[str], model_name: str) -> Dict[str, float]:
        """Evaluate efficiency metrics with different batch sizes"""
        metrics = {}
        
        # Test different batch sizes
        for batch_size in self.config['batch_sizes']:
            if batch_size > len(texts):
                continue
            
            try:
                batch_metrics = self._measure_batch_performance(
                    embedding_function, texts, batch_size, model_name
                )
                
                # Add batch size prefix to metrics
                for metric_name, value in batch_metrics.items():
                    metrics[f'batch_{batch_size}_{metric_name}'] = value
                
            except Exception as e:
                self.logger.warning(f"Error measuring batch size {batch_size}: {e}")
        
        # Overall efficiency metrics
        if metrics:
            # Find optimal batch size based on throughput
            throughput_metrics = {k: v for k, v in metrics.items() if 'throughput' in k}
            if throughput_metrics:
                best_throughput_key = max(throughput_metrics.keys(), key=lambda k: throughput_metrics[k])
                optimal_batch_size = int(best_throughput_key.split('_')[1])
                metrics['optimal_batch_size'] = optimal_batch_size
                metrics['max_throughput'] = throughput_metrics[best_throughput_key]
            
            # Average metrics across batch sizes
            inference_times = [v for k, v in metrics.items() if 'inference_time_per_sample' in k]
            if inference_times:
                metrics['avg_inference_time_per_sample'] = float(np.mean(inference_times))
                metrics['min_inference_time_per_sample'] = float(np.min(inference_times))
        
        return metrics
    
    def _measure_batch_performance(self, embedding_function: Callable, texts: List[str], 
                                  batch_size: int, model_name: str) -> Dict[str, float]:
        """Measure performance metrics for a specific batch size"""
        
        # Prepare batches
        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
        
        # Warmup runs
        self.logger.info(f"Warming up with batch size {batch_size}")
        for _ in range(self.config['warmup_iterations']):
            if batches:
                try:
                    _ = embedding_function(batches[0])
                except Exception as e:
                    self.logger.warning(f"Warmup failed: {e}")
                    break
        
        # Clear memory
        gc.collect()
        
        # Start monitoring
        if self.config['monitor_memory']:
            self.memory_monitor.start()
        if self.config['monitor_cpu']:
            self.cpu_monitor.start()
        
        # Measurement runs
        inference_times = []
        total_samples = 0
        
        measurement_start = time.time()
        
        for iteration in range(self.config['measurement_iterations']):
            for batch in batches:
                batch_start = time.time()
                
                try:
                    # Measure inference time
                    embeddings = embedding_function(batch)
                    
                    batch_end = time.time()
                    batch_time = batch_end - batch_start
                    
                    inference_times.append(batch_time)
                    total_samples += len(batch)
                    
                    # Validate embeddings
                    if embeddings is None or len(embeddings) != len(batch):
                        self.logger.warning(f"Invalid embeddings returned for batch of size {len(batch)}")
                    
                except Exception as e:
                    self.logger.error(f"Embedding function failed on batch: {e}")
                    batch_end = time.time()
                    inference_times.append(batch_end - batch_start)  # Include failed time
        
        measurement_end = time.time()
        total_measurement_time = measurement_end - measurement_start
        
        # Stop monitoring
        memory_stats = self.memory_monitor.stop() if self.config['monitor_memory'] else {}
        cpu_stats = self.cpu_monitor.stop() if self.config['monitor_cpu'] else {}
        
        # Calculate metrics
        metrics = {}
        
        if inference_times:
            metrics['total_inference_time'] = float(np.sum(inference_times))
            metrics['avg_inference_time_per_batch'] = float(np.mean(inference_times))
            metrics['inference_time_per_sample'] = float(np.sum(inference_times) / total_samples)
            metrics['inference_time_std'] = float(np.std(inference_times))
            
            # Throughput
            metrics['throughput_samples_per_second'] = float(total_samples / total_measurement_time)
            metrics['throughput_batches_per_second'] = float(len(inference_times) / total_measurement_time)
        
        # Memory metrics
        if memory_stats:
            metrics['peak_memory_mb'] = memory_stats.get('peak_memory_mb', 0.0)
            metrics['avg_memory_mb'] = memory_stats.get('avg_memory_mb', 0.0)
            metrics['memory_efficiency'] = float(total_samples / memory_stats.get('peak_memory_mb', 1.0))
        
        # CPU metrics
        if cpu_stats:
            metrics['avg_cpu_usage_percent'] = cpu_stats.get('avg_cpu_percent', 0.0)
            metrics['peak_cpu_usage_percent'] = cpu_stats.get('peak_cpu_percent', 0.0)
        
        # Cost efficiency (if cost tracking available)
        if hasattr(self.cost_tracker, 'get_model_cost'):
            try:
                cost_per_sample = self.cost_tracker.get_model_cost(model_name, total_samples)
                if cost_per_sample > 0:
                    metrics['cost_per_sample'] = float(cost_per_sample)
                    metrics['cost_efficiency'] = float(1.0 / cost_per_sample)  # Higher is better
            except Exception as e:
                self.logger.warning(f"Could not calculate cost metrics: {e}")
        
        return metrics
    
    def _evaluate_precomputed_efficiency(self, embeddings: List[np.ndarray], samples: List[DatasetSample],
                                       model_name: str, dataset_name: str) -> EvaluationResult:
        """Evaluate efficiency metrics for pre-computed embeddings (limited metrics)"""
        start_time = time.time()
        
        try:
            valid_embeddings, valid_indices = self._prepare_embeddings(embeddings)
            valid_samples = self._filter_samples(samples, valid_indices)
            
            metrics = {}
            
            # Memory usage of embeddings
            embedding_memory_mb = valid_embeddings.nbytes / (1024 * 1024)
            metrics['embedding_memory_mb'] = float(embedding_memory_mb)
            
            # Embedding statistics
            if len(valid_embeddings) > 0:
                metrics['embedding_dimension'] = int(valid_embeddings.shape[1])
                metrics['num_embeddings'] = len(valid_embeddings)
                metrics['memory_per_embedding_kb'] = float(embedding_memory_mb * 1024 / len(valid_embeddings))
                
                # Vector operation efficiency (simple operations)
                vector_ops_start = time.time()
                
                # Cosine similarity computation (common operation)
                from sklearn.metrics.pairwise import cosine_similarity
                if len(valid_embeddings) > 1:
                    similarities = cosine_similarity(valid_embeddings[:100], valid_embeddings[:100])
                
                vector_ops_time = time.time() - vector_ops_start
                metrics['vector_ops_time_per_100_samples'] = float(vector_ops_time)
            
            # Generate metadata
            metadata = self._generate_metadata([], valid_samples, embeddings)
            metadata['evaluation_type'] = 'precomputed_embeddings'
            metadata['limited_metrics'] = True
            
            execution_time = time.time() - start_time
            
            return self._create_result(
                task_type="efficiency",
                model_name=model_name,
                dataset_name=dataset_name,
                metrics=metrics,
                metadata=metadata,
                execution_time=execution_time,
                num_samples=len(valid_samples)
            )
            
        except Exception as e:
            self.logger.error(f"Precomputed efficiency evaluation failed: {e}")
            raise
    
    def _generate_metadata(self, texts: List[str], samples: List[DatasetSample], 
                          embeddings: List[np.ndarray]) -> Dict[str, Any]:
        """Generate efficiency evaluation metadata"""
        metadata = {
            'batch_sizes_tested': self.config['batch_sizes'],
            'warmup_iterations': self.config['warmup_iterations'],
            'measurement_iterations': self.config['measurement_iterations'],
            'max_test_samples': self.config['max_test_samples'],
            'config': self.config.copy()
        }
        
        if texts:
            # Text statistics
            text_lengths = [len(text.split()) for text in texts]
            metadata['text_stats'] = {
                'num_texts': len(texts),
                'avg_text_length_words': float(np.mean(text_lengths)),
                'min_text_length_words': int(np.min(text_lengths)),
                'max_text_length_words': int(np.max(text_lengths)),
                'total_words': int(np.sum(text_lengths))
            }
        
        # System information
        metadata['system_info'] = {
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': round(psutil.virtual_memory().total / (1024**3), 2),
            'python_version': sys.version.split()[0]
        }
        
        return metadata
    
    def get_required_metrics(self) -> List[str]:
        """Get list of metrics this evaluator provides"""
        base_metrics = [
            'inference_time_per_sample',
            'throughput_samples_per_second',
            'peak_memory_mb',
            'avg_cpu_usage_percent',
            'cost_per_sample'
        ]
        
        # Add batch-specific metrics
        batch_metrics = []
        for batch_size in self.config['batch_sizes']:
            batch_metrics.extend([
                f'batch_{batch_size}_inference_time_per_sample',
                f'batch_{batch_size}_throughput_samples_per_second',
                f'batch_{batch_size}_peak_memory_mb'
            ])
        
        return base_metrics + batch_metrics

class MemoryMonitor:
    """Monitor memory usage during evaluation"""
    
    def __init__(self):
        self.monitoring = False
        self.memory_samples = []
        self.monitor_thread = None
    
    def start(self):
        """Start memory monitoring"""
        self.monitoring = True
        self.memory_samples = []
        self.monitor_thread = threading.Thread(target=self._monitor_memory)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop(self) -> Dict[str, float]:
        """Stop monitoring and return statistics"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        
        if not self.memory_samples:
            return {}
        
        return {
            'peak_memory_mb': float(np.max(self.memory_samples)),
            'avg_memory_mb': float(np.mean(self.memory_samples)),
            'min_memory_mb': float(np.min(self.memory_samples))
        }
    
    def _monitor_memory(self):
        """Monitor memory usage in background thread"""
        while self.monitoring:
            try:
                memory_mb = psutil.Process().memory_info().rss / (1024 * 1024)
                self.memory_samples.append(memory_mb)
                time.sleep(0.1)  # Sample every 100ms
            except Exception:
                break

class CPUMonitor:
    """Monitor CPU usage during evaluation"""
    
    def __init__(self):
        self.monitoring = False
        self.cpu_samples = []
        self.monitor_thread = None
    
    def start(self):
        """Start CPU monitoring"""
        self.monitoring = True
        self.cpu_samples = []
        self.monitor_thread = threading.Thread(target=self._monitor_cpu)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop(self) -> Dict[str, float]:
        """Stop monitoring and return statistics"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        
        if not self.cpu_samples:
            return {}
        
        return {
            'peak_cpu_percent': float(np.max(self.cpu_samples)),
            'avg_cpu_percent': float(np.mean(self.cpu_samples)),
            'min_cpu_percent': float(np.min(self.cpu_samples))
        }
    
    def _monitor_cpu(self):
        """Monitor CPU usage in background thread"""
        while self.monitoring:
            try:
                cpu_percent = psutil.cpu_percent(interval=0.1)
                self.cpu_samples.append(cpu_percent)
            except Exception:
                break
