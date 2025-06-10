"""
Efficiency metrics for embedding evaluation.

Provides lightweight, reusable efficiency evaluation metrics with proper edge case handling.
"""

import time
import psutil
import numpy as np
from typing import List, Dict, Union, Optional, Callable
import functools
import threading
from contextlib import contextmanager

def calculate_latency(func: Callable, *args, **kwargs) -> Dict[str, float]:
    """
    Calculate latency metrics for a function call.
    
    Args:
        func: Function to measure
        *args: Function arguments
        **kwargs: Function keyword arguments
        
    Returns:
        Dictionary with latency metrics in seconds
    """
    try:
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        
        latency = end_time - start_time
        
        return {
            'latency_seconds': latency,
            'latency_ms': latency * 1000,
            'result': result
        }
    except Exception as e:
        return {
            'latency_seconds': float('inf'),
            'latency_ms': float('inf'),
            'error': str(e),
            'result': None
        }

def calculate_throughput(func: Callable, inputs: List, batch_size: int = 1) -> Dict[str, float]:
    """
    Calculate throughput metrics for batch processing.
    
    Args:
        func: Function to measure
        inputs: List of inputs to process
        batch_size: Size of each batch
        
    Returns:
        Dictionary with throughput metrics
    """
    if len(inputs) == 0:
        return {
            'throughput_per_second': 0.0,
            'items_processed': 0,
            'total_time_seconds': 0.0,
            'avg_batch_time_seconds': 0.0
        }
    
    try:
        start_time = time.perf_counter()
        processed_items = 0
        batch_times = []
        
        # Process in batches
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i + batch_size]
            batch_start = time.perf_counter()
            
            # Process batch
            func(batch)
            
            batch_end = time.perf_counter()
            batch_times.append(batch_end - batch_start)
            processed_items += len(batch)
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        return {
            'throughput_per_second': processed_items / total_time if total_time > 0 else 0.0,
            'items_processed': processed_items,
            'total_time_seconds': total_time,
            'avg_batch_time_seconds': np.mean(batch_times) if batch_times else 0.0,
            'batch_times': batch_times
        }
    except Exception as e:
        return {
            'throughput_per_second': 0.0,
            'items_processed': 0,
            'total_time_seconds': float('inf'),
            'avg_batch_time_seconds': float('inf'),
            'error': str(e)
        }

@contextmanager
def memory_usage():
    """
    Context manager to measure memory usage during execution.
    
    Yields:
        Dictionary with memory usage metrics
    """
    process = psutil.Process()
    
    # Get initial memory usage
    initial_memory = process.memory_info()
    peak_memory = initial_memory.rss
    
    # Monitor memory in background thread
    monitoring = {'peak': peak_memory, 'stop': False}
    
    def monitor_memory():
        while not monitoring['stop']:
            try:
                current_memory = process.memory_info().rss
                if current_memory > monitoring['peak']:
                    monitoring['peak'] = current_memory
                time.sleep(0.01)  # Check every 10ms
            except:
                break
    
    monitor_thread = threading.Thread(target=monitor_memory, daemon=True)
    monitor_thread.start()
    
    try:
        yield monitoring
    finally:
        monitoring['stop'] = True
        monitor_thread.join(timeout=0.1)
        
        final_memory = process.memory_info()
        
        # Convert to MB
        monitoring.update({
            'initial_memory_mb': initial_memory.rss / (1024 * 1024),
            'final_memory_mb': final_memory.rss / (1024 * 1024),
            'peak_memory_mb': monitoring['peak'] / (1024 * 1024),
            'memory_increase_mb': (final_memory.rss - initial_memory.rss) / (1024 * 1024)
        })

def tokens_per_second(num_tokens: int, processing_time: float) -> float:
    """
    Calculate tokens processed per second.
    
    Args:
        num_tokens: Number of tokens processed
        processing_time: Time taken in seconds
        
    Returns:
        Tokens per second rate
    """
    if processing_time <= 0:
        return 0.0
    
    if num_tokens <= 0:
        return 0.0
    
    return num_tokens / processing_time

def cost_per_embedding(total_cost: float, num_embeddings: int) -> float:
    """
    Calculate cost per embedding.
    
    Args:
        total_cost: Total cost in dollars
        num_embeddings: Number of embeddings generated
        
    Returns:
        Cost per embedding in dollars
    """
    if num_embeddings <= 0:
        return float('inf')
    
    if total_cost < 0:
        return 0.0
    
    return total_cost / num_embeddings

def efficiency_score(latency: float, throughput: float, memory_mb: float, 
                    cost: float = 0.0, weights: Optional[Dict[str, float]] = None) -> float:
    """
    Calculate composite efficiency score.
    
    Args:
        latency: Average latency in seconds (lower is better)
        throughput: Throughput in items/second (higher is better)
        memory_mb: Memory usage in MB (lower is better)
        cost: Cost in dollars (lower is better)
        weights: Weights for each metric
        
    Returns:
        Efficiency score (0 to 1, higher is better)
    """
    if weights is None:
        weights = {
            'latency': 0.3,
            'throughput': 0.3,
            'memory': 0.2,
            'cost': 0.2
        }
    
    # Normalize metrics (handle edge cases)
    normalized_scores = {}
    
    # Latency score (inverse, lower is better)
    if latency <= 0:
        normalized_scores['latency'] = 1.0
    else:
        # Use exponential decay for latency penalty
        normalized_scores['latency'] = np.exp(-latency)
    
    # Throughput score (higher is better)
    if throughput <= 0:
        normalized_scores['throughput'] = 0.0
    else:
        # Normalize using sigmoid function
        normalized_scores['throughput'] = min(1.0, throughput / 100.0)
    
    # Memory score (inverse, lower is better)
    if memory_mb <= 0:
        normalized_scores['memory'] = 1.0
    else:
        # Penalize high memory usage
        normalized_scores['memory'] = max(0.0, 1.0 - (memory_mb / 1000.0))
    
    # Cost score (inverse, lower is better)
    if cost <= 0:
        normalized_scores['cost'] = 1.0
    else:
        # Penalize high cost
        normalized_scores['cost'] = max(0.0, 1.0 - (cost * 1000))  # Assuming cost in dollars
    
    # Calculate weighted score
    total_score = 0.0
    total_weight = 0.0
    
    for metric, weight in weights.items():
        if metric in normalized_scores and weight > 0:
            total_score += normalized_scores[metric] * weight
            total_weight += weight
    
    return total_score / total_weight if total_weight > 0 else 0.0

def benchmark_embedding_function(func: Callable, test_inputs: List[str], 
                               num_runs: int = 3) -> Dict[str, Union[float, List[float]]]:
    """
    Comprehensive benchmark of an embedding function.
    
    Args:
        func: Embedding function to benchmark
        test_inputs: List of test input strings
        num_runs: Number of benchmark runs
        
    Returns:
        Dictionary with comprehensive benchmark results
    """
    if len(test_inputs) == 0:
        return {
            'avg_latency_seconds': float('inf'),
            'min_latency_seconds': float('inf'),
            'max_latency_seconds': float('inf'),
            'throughput_per_second': 0.0,
            'memory_usage_mb': 0.0,
            'all_latencies': [],
            'error': 'No test inputs provided'
        }
    
    latencies = []
    memory_usages = []
    errors = []
    
    for run in range(num_runs):
        try:
            with memory_usage() as mem_monitor:
                start_time = time.perf_counter()
                
                # Process all inputs
                results = []
                for input_text in test_inputs:
                    result = func(input_text)
                    results.append(result)
                
                end_time = time.perf_counter()
                run_latency = end_time - start_time
                latencies.append(run_latency)
                memory_usages.append(mem_monitor.get('peak_memory_mb', 0.0))
                
        except Exception as e:
            errors.append(str(e))
            latencies.append(float('inf'))
            memory_usages.append(0.0)
    
    # Calculate statistics
    valid_latencies = [lat for lat in latencies if lat != float('inf')]
    
    if not valid_latencies:
        return {
            'avg_latency_seconds': float('inf'),
            'min_latency_seconds': float('inf'),
            'max_latency_seconds': float('inf'),
            'throughput_per_second': 0.0,
            'memory_usage_mb': 0.0,
            'all_latencies': latencies,
            'errors': errors
        }
    
    avg_latency = np.mean(valid_latencies)
    min_latency = np.min(valid_latencies)
    max_latency = np.max(valid_latencies)
    
    # Calculate throughput based on average latency
    throughput = len(test_inputs) / avg_latency if avg_latency > 0 else 0.0
    
    # Average memory usage
    valid_memory = [mem for mem in memory_usages if mem > 0]
    avg_memory = np.mean(valid_memory) if valid_memory else 0.0
    
    return {
        'avg_latency_seconds': avg_latency,
        'min_latency_seconds': min_latency,
        'max_latency_seconds': max_latency,
        'std_latency_seconds': np.std(valid_latencies),
        'throughput_per_second': throughput,
        'memory_usage_mb': avg_memory,
        'all_latencies': latencies,
        'memory_usages': memory_usages,
        'num_successful_runs': len(valid_latencies),
        'num_failed_runs': len(errors),
        'errors': errors
    }

def compare_efficiency(results1: Dict, results2: Dict, model1_name: str = "Model 1", 
                      model2_name: str = "Model 2") -> Dict[str, Union[str, float]]:
    """
    Compare efficiency metrics between two models.
    
    Args:
        results1: Benchmark results for first model
        results2: Benchmark results for second model
        model1_name: Name of first model
        model2_name: Name of second model
        
    Returns:
        Dictionary with comparison results
    """
    comparison = {
        'faster_model': None,
        'latency_improvement': 0.0,
        'throughput_improvement': 0.0,
        'memory_efficiency': None,
        'memory_difference_mb': 0.0,
        'overall_winner': None
    }
    
    # Compare latency
    lat1 = results1.get('avg_latency_seconds', float('inf'))
    lat2 = results2.get('avg_latency_seconds', float('inf'))
    
    if lat1 < lat2:
        comparison['faster_model'] = model1_name
        comparison['latency_improvement'] = (lat2 - lat1) / lat2 * 100 if lat2 > 0 else 0.0
    elif lat2 < lat1:
        comparison['faster_model'] = model2_name
        comparison['latency_improvement'] = (lat1 - lat2) / lat1 * 100 if lat1 > 0 else 0.0
    
    # Compare throughput
    thr1 = results1.get('throughput_per_second', 0.0)
    thr2 = results2.get('throughput_per_second', 0.0)
    
    if thr1 > thr2:
        comparison['throughput_improvement'] = (thr1 - thr2) / thr2 * 100 if thr2 > 0 else 0.0
    elif thr2 > thr1:
        comparison['throughput_improvement'] = (thr2 - thr1) / thr1 * 100 if thr1 > 0 else 0.0
    
    # Compare memory usage
    mem1 = results1.get('memory_usage_mb', 0.0)
    mem2 = results2.get('memory_usage_mb', 0.0)
    
    if mem1 < mem2:
        comparison['memory_efficiency'] = model1_name
        comparison['memory_difference_mb'] = mem2 - mem1
    elif mem2 < mem1:
        comparison['memory_efficiency'] = model2_name
        comparison['memory_difference_mb'] = mem1 - mem2
    
    # Determine overall winner (simple scoring)
    score1 = 0
    score2 = 0
    
    if comparison['faster_model'] == model1_name:
        score1 += 1
    elif comparison['faster_model'] == model2_name:
        score2 += 1
    
    if comparison['memory_efficiency'] == model1_name:
        score1 += 1
    elif comparison['memory_efficiency'] == model2_name:
        score2 += 1
    
    if thr1 > thr2:
        score1 += 1
    elif thr2 > thr1:
        score2 += 1
    
    if score1 > score2:
        comparison['overall_winner'] = model1_name
    elif score2 > score1:
        comparison['overall_winner'] = model2_name
    else:
        comparison['overall_winner'] = "Tie"
    
    return comparison
