"""
Benchmark Dataset Loaders

Standard benchmark datasets for embedding evaluation including
STS, MS MARCO, NFCorpus, Quora Duplicates, and BEIR datasets.
"""

from datasets.benchmark_datasets.sts_benchmark import STSBenchmarkLoader
from datasets.benchmark_datasets.msmarco import MSMARCOLoader
from datasets.benchmark_datasets.nfcorpus import NFCorpusLoader
from datasets.benchmark_datasets.quora_duplicates import QuoraDuplicatesLoader
from datasets.benchmark_datasets.beir_datasets import BEIRDatasetLoader

__all__ = [
    'STSBenchmarkLoader',
    'MSMARCOLoader', 
    'NFCorpusLoader',
    'QuoraDuplicatesLoader',
    'BEIRDatasetLoader'
]
