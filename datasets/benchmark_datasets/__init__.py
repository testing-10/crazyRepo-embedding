"""
Benchmark Dataset Loaders

Standard benchmark datasets for embedding evaluation including
STS, MS MARCO, NFCorpus, Quora Duplicates, and BEIR datasets.
"""

from .sts_benchmark import STSBenchmarkLoader
from .msmarco import MSMARCOLoader
from .nfcorpus import NFCorpusLoader
from .quora_duplicates import QuoraDuplicatesLoader
from .beir_datasets import BEIRDatasetLoader

__all__ = [
    'STSBenchmarkLoader',
    'MSMARCOLoader', 
    'NFCorpusLoader',
    'QuoraDuplicatesLoader',
    'BEIRDatasetLoader'
]
