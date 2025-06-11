"""
Embedding Model Testing Framework - Reporting Module

This module provides comprehensive reporting capabilities for embedding model evaluations,
including visual summaries, performance comparisons, and multi-format outputs.
"""

from reporting.embedding_comparison_report import EmbeddingComparisonReport
from reporting.embedding_visualizations import EmbeddingVisualizations
from reporting.dimension_analysis import DimensionAnalysis

__all__ = [
    'EmbeddingComparisonReport',
    'EmbeddingVisualizations', 
    'DimensionAnalysis'
]

__version__ = "1.0.0"
