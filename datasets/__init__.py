"""
Embedding Dataset Loaders

This module provides a unified interface for loading various datasets
used in embedding model evaluation, including standard benchmarks
and custom domain-specific datasets.
"""

from .dataset_loader import (
    BaseDatasetLoader,
    BenchmarkDatasetLoader, 
    CustomDatasetLoader,
    DatasetRegistry,
    load_dataset
)

from .benchmark_datasets.sts_benchmark import STSBenchmarkLoader
from .benchmark_datasets.msmarco import MSMARCOLoader
from .benchmark_datasets.nfcorpus import NFCorpusLoader
from .benchmark_datasets.quora_duplicates import QuoraDuplicatesLoader
from .benchmark_datasets.beir_datasets import BEIRDatasetLoader

__all__ = [
    'BaseDatasetLoader',
    'BenchmarkDatasetLoader',
    'CustomDatasetLoader', 
    'DatasetRegistry',
    'load_dataset',
    'STSBenchmarkLoader',
    'MSMARCOLoader',
    'NFCorpusLoader',
    'QuoraDuplicatesLoader',
    'BEIRDatasetLoader'
]

# Register all available dataset loaders
DATASET_REGISTRY = DatasetRegistry()

# Register benchmark datasets
DATASET_REGISTRY.register('sts_benchmark', STSBenchmarkLoader)
DATASET_REGISTRY.register('msmarco', MSMARCOLoader)
DATASET_REGISTRY.register('nfcorpus', NFCorpusLoader)
DATASET_REGISTRY.register('quora_duplicates', QuoraDuplicatesLoader)
DATASET_REGISTRY.register('beir', BEIRDatasetLoader)
