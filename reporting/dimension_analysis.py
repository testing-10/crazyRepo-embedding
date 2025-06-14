"""
Dimension Analysis Module

Analyzes embedding dimensions, vector distributions, and dimensional characteristics
across different embedding models for comprehensive evaluation.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging

from utils.embedding_logger import EmbeddingLogger

class DimensionAnalyzer:
    """
    Comprehensive analysis of embedding dimensions and vector characteristics
    """

    def __init__(self, output_dir: str = "reports/dimension_analysis"):
        """
        Initialize the dimension analysis module

        Args:
            output_dir: Directory to save generated analysis reports
        """
        self.output_dir = output_dir
        self.logger = EmbeddingLogger.get_logger(__name__)

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        # Set style preferences
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

    def analyze_embedding_dimensions(
        self,
        embeddings_data: Dict[str, np.ndarray],
        model_configs: Dict[str, Any],
        sample_texts: List[str]
    ) -> Dict[str, Any]:
        """
        Perform comprehensive dimension analysis on embedding data

        Args:
            embeddings_data: Dictionary mapping model names to embedding arrays
            model_configs: Configuration data for models
            sample_texts: Original texts used for embeddings

        Returns:
            Dictionary containing analysis results and visualization paths
        """
        try:
            self.logger.info("Starting comprehensive dimension analysis")

            analysis_results = {
                'dimensional_statistics': {},
                'visualization_paths': {},
                'clustering_analysis': {},
                'dimensionality_reduction': {},
                'vector_characteristics': {}
            }

            # Check if we have any data
            if not embeddings_data:
                self.logger.warning("No embedding data provided for dimension analysis")
                return analysis_results

            # 1. Basic dimensional statistics
            dim_stats = self._calculate_dimensional_statistics(embeddings_data, model_configs)
            analysis_results['dimensional_statistics'] = dim_stats

            # 2. Generate dimension comparison visualizations
            try:
                dim_viz_paths = self._generate_dimension_visualizations(embeddings_data, model_configs)
                analysis_results['visualization_paths']['dimensions'] = dim_viz_paths
            except Exception as e:
                self.logger.warning(f"Skipping dimension visualizations: {str(e)}")
                analysis_results['visualization_paths']['dimensions'] = []

            # 3. Vector distribution analysis
            try:
                dist_viz_paths = self._analyze_vector_distributions(embeddings_data)
                analysis_results['visualization_paths']['distributions'] = dist_viz_paths
            except Exception as e:
                self.logger.warning(f"Skipping distribution analysis: {str(e)}")
                analysis_results['visualization_paths']['distributions'] = []

            # 4. Vector characteristics analysis (simplified)
            try:
                char_results = self._analyze_vector_characteristics(embeddings_data)
                analysis_results['vector_characteristics'] = char_results
            except Exception as e:
                self.logger.warning(f"Skipping vector characteristics: {str(e)}")
                analysis_results['vector_characteristics'] = {}

            # 5. Generate comprehensive report
            try:
                report_path = self._generate_dimension_report(analysis_results)
                analysis_results['report_path'] = report_path
            except Exception as e:
                self.logger.warning(f"Skipping report generation: {str(e)}")
                analysis_results['report_path'] = ""

            self.logger.info("Dimension analysis completed successfully")
            return analysis_results

        except Exception as e:
            self.logger.error(f"Error in dimension analysis: {str(e)}")
            return {
                'dimensional_statistics': {},
                'visualization_paths': {},
                'clustering_analysis': {},
                'dimensionality_reduction': {},
                'vector_characteristics': {}
            }

    def _calculate_dimensional_statistics(
        self,
        embeddings_data: Dict[str, np.ndarray],
        model_configs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate basic dimensional statistics for each model"""
        stats = {}

        for model_name, embeddings in embeddings_data.items():
            try:
                config = model_configs.get(model_name, {})

                # Basic shape information
                n_samples, n_dimensions = embeddings.shape

                # Statistical measures
                mean_values = np.mean(embeddings, axis=0)
                std_values = np.std(embeddings, axis=0)

                # Vector norms
                vector_norms = np.linalg.norm(embeddings, axis=1)

                # Dimension utilization (how many dimensions have significant variance)
                significant_dims = np.sum(std_values > 0.01)  # Threshold for significance

                stats[model_name] = {
                    'dimensions': n_dimensions,
                    'samples': n_samples,
                    'configured_dimensions': config.get('dimensions', n_dimensions),
                    'mean_vector_norm': float(np.mean(vector_norms)),
                    'std_vector_norm': float(np.std(vector_norms)),
                    'min_vector_norm': float(np.min(vector_norms)),
                    'max_vector_norm': float(np.max(vector_norms)),
                    'significant_dimensions': int(significant_dims),
                    'dimension_utilization': float(significant_dims / n_dimensions) if n_dimensions > 0 else 0.0,
                    'mean_dimension_std': float(np.mean(std_values)),
                    'max_dimension_std': float(np.max(std_values)),
                    'min_dimension_std': float(np.min(std_values)),
                    'sparsity': float(np.mean(np.abs(embeddings) < 0.01))  # Proportion of near-zero values
                }

            except Exception as e:
                self.logger.warning(f"Error calculating stats for {model_name}: {str(e)}")
                continue

        return stats

    def _generate_dimension_visualizations(
        self,
        embeddings_data: Dict[str, np.ndarray],
        model_configs: Dict[str, Any]
    ) -> List[str]:
        """Generate visualizations comparing dimensional characteristics"""
        paths = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        try:
            # 1. Dimension count comparison
            plt.figure(figsize=(12, 6))

            models = list(embeddings_data.keys())
            dimensions = [embeddings_data[model].shape[1] for model in models]
            configured_dims = [model_configs.get(model, {}).get('dimensions', dim) 
                             for model, dim in zip(models, dimensions)]

            x = np.arange(len(models))
            width = 0.35

            plt.bar(x - width/2, dimensions, width, label='Actual Dimensions', alpha=0.8)
            plt.bar(x + width/2, configured_dims, width, label='Configured Dimensions', alpha=0.8)

            plt.xlabel('Models', fontweight='bold')
            plt.ylabel('Number of Dimensions', fontweight='bold')
            plt.title('Embedding Dimensions Comparison', fontweight='bold')
            plt.xticks(x, models, rotation=45)
            plt.legend()
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()

            path = os.path.join(self.output_dir, f'dimension_comparison_{timestamp}.png')
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plt.close()
            paths.append(path)

            # 2. Vector norm distributions (simplified)
            plt.figure(figsize=(12, 8))

            for i, (model_name, embeddings) in enumerate(embeddings_data.items()):
                vector_norms = np.linalg.norm(embeddings, axis=1)
                plt.hist(vector_norms, bins=30, alpha=0.6, label=model_name, edgecolor='black')

            plt.xlabel('Vector Norm', fontweight='bold')
            plt.ylabel('Frequency', fontweight='bold')
            plt.title('Vector Norm Distributions Comparison', fontweight='bold')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            path = os.path.join(self.output_dir, f'vector_norms_{timestamp}.png')
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plt.close()
            paths.append(path)

        except Exception as e:
            self.logger.error(f"Error generating dimension visualizations: {str(e)}")

        return paths

    def _analyze_vector_distributions(self, embeddings_data: Dict[str, np.ndarray]) -> List[str]:
        """Analyze and visualize vector value distributions"""
        paths = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        try:
            # Value distribution comparison
            plt.figure(figsize=(15, 10))

            n_models = len(embeddings_data)
            n_cols = 2
            n_rows = (n_models + 1) // 2

            for i, (model_name, embeddings) in enumerate(embeddings_data.items(), 1):
                plt.subplot(n_rows, n_cols, i)

                # Flatten embeddings for distribution analysis
                flat_values = embeddings.flatten()

                plt.hist(flat_values, bins=50, alpha=0.7, edgecolor='black', density=True)
                plt.title(f'{model_name} - Value Distribution', fontweight='bold')
                plt.xlabel('Embedding Values')
                plt.ylabel('Density')
                plt.grid(True, alpha=0.3)

                # Add statistics text
                mean_val = np.mean(flat_values)
                std_val = np.std(flat_values)
                plt.text(0.05, 0.95, f'μ={mean_val:.3f}\nσ={std_val:.3f}', 
                        transform=plt.gca().transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            plt.tight_layout()

            path = os.path.join(self.output_dir, f'value_distributions_{timestamp}.png')
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plt.close()
            paths.append(path)

        except Exception as e:
            self.logger.error(f"Error analyzing vector distributions: {str(e)}")

        return paths

    def _analyze_vector_characteristics(self, embeddings_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Analyze various vector characteristics"""
        characteristics = {}

        for model_name, embeddings in embeddings_data.items():
            try:
                # Calculate basic statistics without expensive similarity calculations
                characteristics[model_name] = {
                    'mean_vector_norm': float(np.mean(np.linalg.norm(embeddings, axis=1))),
                    'std_vector_norm': float(np.std(np.linalg.norm(embeddings, axis=1))),
                    'mean_value': float(np.mean(embeddings)),
                    'std_value': float(np.std(embeddings)),
                    'min_value': float(np.min(embeddings)),
                    'max_value': float(np.max(embeddings)),
                    'sparsity': float(np.mean(np.abs(embeddings) < 0.01))
                }

            except Exception as e:
                self.logger.warning(f"Error analyzing characteristics for {model_name}: {str(e)}")
                continue

        return characteristics

    def _generate_dimension_report(self, analysis_results: Dict[str, Any]) -> str:
        """Generate comprehensive dimension analysis report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(self.output_dir, f'dimension_analysis_report_{timestamp}.html')

        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Embedding Dimension Analysis Report</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 20px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; text-align: center; margin-bottom: 30px; }}
        h2 {{ color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        .stats-table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        .stats-table th, .stats-table td {{ padding: 12px; text-align: center; border: 1px solid #ddd; }}
        .stats-table th {{ background-color: #3498db; color: white; }}
        .stats-table tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .metric-card {{ background: #ecf0f1; padding: 15px; margin: 10px 0; border-radius: 5px; }}
        .footer {{ text-align: center; margin-top: 30px; color: #7f8c8d; font-size: 0.9em; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 Embedding Dimension Analysis Report</h1>

        <h2>📊 Dimensional Statistics</h2>
"""

        if analysis_results['dimensional_statistics']:
            html_content += """
        <table class="stats-table">
            <thead>
                <tr>
                    <th>Model</th>
                    <th>Dimensions</th>
                    <th>Utilization</th>
                    <th>Mean Norm</th>
                    <th>Sparsity</th>
                    <th>Variance</th>
                </tr>
            </thead>
            <tbody>
"""

            for model_name, stats in analysis_results['dimensional_statistics'].items():
                html_content += f"""
                <tr>
                    <td><strong>{model_name}</strong></td>
                    <td>{stats['dimensions']}</td>
                    <td>{stats['dimension_utilization']:.2%}</td>
                    <td>{stats['mean_vector_norm']:.3f}</td>
                    <td>{stats['sparsity']:.2%}</td>
                    <td>{stats['mean_dimension_std']:.3f}</td>
                </tr>
"""

            html_content += """
            </tbody>
        </table>

        <h2>🎯 Key Insights</h2>
"""

            # Add insights based on analysis
            best_utilization = max(analysis_results['dimensional_statistics'].items(), 
                                 key=lambda x: x[1]['dimension_utilization'])
            lowest_sparsity = min(analysis_results['dimensional_statistics'].items(), 
                                key=lambda x: x[1]['sparsity'])

            html_content += f"""
        <div class="metric-card">
            <strong>Best Dimension Utilization:</strong> {best_utilization[0]} 
            ({best_utilization[1]['dimension_utilization']:.2%})
        </div>
        <div class="metric-card">
            <strong>Lowest Sparsity:</strong> {lowest_sparsity[0]} 
            ({lowest_sparsity[1]['sparsity']:.2%})
        </div>
"""
        else:
            html_content += """
        <div class="metric-card">
            <strong>No dimensional statistics available</strong>
        </div>
"""

        html_content += f"""
        <div class="footer">
            <p>Generated by Embedding Model Testing Framework</p>
            <p>Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    </div>
</body>
</html>
"""

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        self.logger.info(f"Dimension analysis report generated: {report_path}")
        return report_path


# Keep backward compatibility
DimensionAnalysis = DimensionAnalyzer