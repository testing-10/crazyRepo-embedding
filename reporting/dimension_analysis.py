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

from ..utils.embedding_logger import EmbeddingLogger

class DimensionAnalysis:
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
            
            # 1. Basic dimensional statistics
            dim_stats = self._calculate_dimensional_statistics(embeddings_data, model_configs)
            analysis_results['dimensional_statistics'] = dim_stats
            
            # 2. Generate dimension comparison visualizations
            dim_viz_paths = self._generate_dimension_visualizations(embeddings_data, model_configs)
            analysis_results['visualization_paths']['dimensions'] = dim_viz_paths
            
            # 3. Vector distribution analysis
            dist_viz_paths = self._analyze_vector_distributions(embeddings_data)
            analysis_results['visualization_paths']['distributions'] = dist_viz_paths
            
            # 4. Dimensionality reduction analysis
            reduction_results = self._perform_dimensionality_reduction(embeddings_data, sample_texts)
            analysis_results['dimensionality_reduction'] = reduction_results
            
            # 5. Clustering analysis in reduced dimensions
            clustering_results = self._perform_clustering_analysis(embeddings_data)
            analysis_results['clustering_analysis'] = clustering_results
            
            # 6. Vector characteristics analysis
            char_results = self._analyze_vector_characteristics(embeddings_data)
            analysis_results['vector_characteristics'] = char_results
            
            # 7. Generate comprehensive report
            report_path = self._generate_dimension_report(analysis_results)
            analysis_results['report_path'] = report_path
            
            self.logger.info("Dimension analysis completed successfully")
            return analysis_results
            
        except Exception as e:
            self.logger.error(f"Error in dimension analysis: {str(e)}")
            raise
    
    def _calculate_dimensional_statistics(
        self,
        embeddings_data: Dict[str, np.ndarray],
        model_configs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate basic dimensional statistics for each model"""
        stats = {}
        
        for model_name, embeddings in embeddings_data.items():
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
                'dimension_utilization': float(significant_dims / n_dimensions),
                'mean_dimension_std': float(np.mean(std_values)),
                'max_dimension_std': float(np.max(std_values)),
                'min_dimension_std': float(np.min(std_values)),
                'sparsity': float(np.mean(np.abs(embeddings) < 0.01))  # Proportion of near-zero values
            }
        
        return stats
    
    def _generate_dimension_visualizations(
        self,
        embeddings_data: Dict[str, np.ndarray],
        model_configs: Dict[str, Any]
    ) -> List[str]:
        """Generate visualizations comparing dimensional characteristics"""
        paths = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
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
        
        # 2. Vector norm distributions
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        for i, (model_name, embeddings) in enumerate(embeddings_data.items()):
            if i >= 4:  # Limit to 4 models for visualization
                break
                
            vector_norms = np.linalg.norm(embeddings, axis=1)
            
            axes[i].hist(vector_norms, bins=30, alpha=0.7, edgecolor='black')
            axes[i].set_title(f'{model_name} - Vector Norm Distribution', fontweight='bold')
            axes[i].set_xlabel('Vector Norm')
            axes[i].set_ylabel('Frequency')
            axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(embeddings_data), 4):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        path = os.path.join(self.output_dir, f'vector_norms_{timestamp}.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        paths.append(path)
        
        return paths
    
    def _analyze_vector_distributions(self, embeddings_data: Dict[str, np.ndarray]) -> List[str]:
        """Analyze and visualize vector value distributions"""
        paths = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Value distribution comparison
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
        
        # 2. Dimension variance analysis
        plt.figure(figsize=(12, 8))
        
        for model_name, embeddings in embeddings_data.items():
            dim_variances = np.var(embeddings, axis=0)
            plt.plot(dim_variances, label=model_name, alpha=0.8, linewidth=2)
        
        plt.xlabel('Dimension Index', fontweight='bold')
        plt.ylabel('Variance', fontweight='bold')
        plt.title('Dimension-wise Variance Comparison', fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        path = os.path.join(self.output_dir, f'dimension_variance_{timestamp}.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        paths.append(path)
        
        return paths
    
    def _perform_dimensionality_reduction(
        self,
        embeddings_data: Dict[str, np.ndarray],
        sample_texts: List[str]
    ) -> Dict[str, Any]:
        """Perform PCA and t-SNE analysis on embeddings"""
        reduction_results = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for model_name, embeddings in embeddings_data.items():
            try:
                # PCA Analysis
                pca = PCA(n_components=min(50, embeddings.shape[1]))
                pca_embeddings = pca.fit_transform(embeddings)
                
                # Calculate explained variance
                explained_variance_ratio = pca.explained_variance_ratio_
                cumulative_variance = np.cumsum(explained_variance_ratio)
                
                # Find dimensions needed for 95% variance
                dims_95 = np.argmax(cumulative_variance >= 0.95) + 1
                dims_99 = np.argmax(cumulative_variance >= 0.99) + 1
                
                # t-SNE for 2D visualization (on PCA-reduced data for efficiency)
                tsne_input = pca_embeddings[:, :min(50, pca_embeddings.shape[1])]
                tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
                tsne_embeddings = tsne.fit_transform(tsne_input)
                
                reduction_results[model_name] = {
                    'pca_embeddings': pca_embeddings,
                    'tsne_embeddings': tsne_embeddings,
                    'explained_variance_ratio': explained_variance_ratio,
                    'cumulative_variance': cumulative_variance,
                    'dims_for_95_variance': int(dims_95),
                    'dims_for_99_variance': int(dims_99),
                    'total_variance_explained': float(np.sum(explained_variance_ratio))
                }
                
            except Exception as e:
                self.logger.warning(f"Dimensionality reduction failed for {model_name}: {str(e)}")
                continue
        
        # Generate PCA visualization
        self._visualize_pca_results(reduction_results, timestamp)
        
        # Generate t-SNE visualization
        self._visualize_tsne_results(reduction_results, sample_texts, timestamp)
        
        return reduction_results
    
    def _visualize_pca_results(self, reduction_results: Dict[str, Any], timestamp: str):
        """Visualize PCA analysis results"""
        # 1. Explained variance comparison
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        for model_name, results in reduction_results.items():
            explained_var = results['explained_variance_ratio'][:20]  # First 20 components
            plt.plot(range(1, len(explained_var) + 1), explained_var, 
                    marker='o', label=model_name, linewidth=2)
        
        plt.xlabel('Principal Component', fontweight='bold')
        plt.ylabel('Explained Variance Ratio', fontweight='bold')
        plt.title('PCA - Explained Variance by Component', fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 2)
        for model_name, results in reduction_results.items():
            cumulative_var = results['cumulative_variance'][:50]  # First 50 components
            plt.plot(range(1, len(cumulative_var) + 1), cumulative_var, 
                    marker='o', label=model_name, linewidth=2)
        
        plt.axhline(y=0.95, color='red', linestyle='--', alpha=0.7, label='95% Variance')
        plt.axhline(y=0.99, color='orange', linestyle='--', alpha=0.7, label='99% Variance')
        plt.xlabel('Number of Components', fontweight='bold')
        plt.ylabel('Cumulative Explained Variance', fontweight='bold')
        plt.title('PCA - Cumulative Explained Variance', fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 3)
        models = list(reduction_results.keys())
        dims_95 = [reduction_results[model]['dims_for_95_variance'] for model in models]
        dims_99 = [reduction_results[model]['dims_for_99_variance'] for model in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        plt.bar(x - width/2, dims_95, width, label='95% Variance', alpha=0.8)
        plt.bar(x + width/2, dims_99, width, label='99% Variance', alpha=0.8)
        
        plt.xlabel('Models', fontweight='bold')
        plt.ylabel('Dimensions Required', fontweight='bold')
        plt.title('Dimensions for Variance Thresholds', fontweight='bold')
        plt.xticks(x, models, rotation=45)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        path = os.path.join(self.output_dir, f'pca_analysis_{timestamp}.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _visualize_tsne_results(
        self,
        reduction_results: Dict[str, Any],
        sample_texts: List[str],
        timestamp: str
    ):
        """Visualize t-SNE results"""
        n_models = len(reduction_results)
        n_cols = 2
        n_rows = (n_models + 1) // 2
        
        plt.figure(figsize=(15, 5 * n_rows))
        
        for i, (model_name, results) in enumerate(reduction_results.items(), 1):
            plt.subplot(n_rows, n_cols, i)
            
            tsne_embeddings = results['tsne_embeddings']
            
            # Color points by text length as a proxy for complexity
            if sample_texts:
                colors = [len(text) for text in sample_texts[:len(tsne_embeddings)]]
                scatter = plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], 
                                    c=colors, cmap='viridis', alpha=0.7, s=50)
                plt.colorbar(scatter, label='Text Length')
            else:
                plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], 
                          alpha=0.7, s=50)
            
            plt.title(f'{model_name} - t-SNE Visualization', fontweight='bold')
            plt.xlabel('t-SNE Component 1')
            plt.ylabel('t-SNE Component 2')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        path = os.path.join(self.output_dir, f'tsne_visualization_{timestamp}.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _perform_clustering_analysis(self, embeddings_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Perform clustering analysis on embeddings"""
        clustering_results = {}
        
        for model_name, embeddings in embeddings_data.items():
            try:
                # Try different numbers of clusters
                k_range = range(2, min(11, len(embeddings)))
                silhouette_scores = []
                inertias = []
                
                for k in k_range:
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    cluster_labels = kmeans.fit_predict(embeddings)
                    
                    silhouette_avg = silhouette_score(embeddings, cluster_labels)
                    silhouette_scores.append(silhouette_avg)
                    inertias.append(kmeans.inertia_)
                
                # Find optimal k using silhouette score
                optimal_k = k_range[np.argmax(silhouette_scores)]
                
                clustering_results[model_name] = {
                    'k_range': list(k_range),
                    'silhouette_scores': silhouette_scores,
                    'inertias': inertias,
                    'optimal_k': optimal_k,
                    'best_silhouette_score': max(silhouette_scores)
                }
                
            except Exception as e:
                self.logger.warning(f"Clustering analysis failed for {model_name}: {str(e)}")
                continue
        
        return clustering_results
    
    def _analyze_vector_characteristics(self, embeddings_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Analyze various vector characteristics"""
        characteristics = {}
        
        for model_name, embeddings in embeddings_data.items():
            # Calculate pairwise similarities
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Sample subset for efficiency if too large
            if len(embeddings) > 1000:
                sample_indices = np.random.choice(len(embeddings), 1000, replace=False)
                sample_embeddings = embeddings[sample_indices]
            else:
                sample_embeddings = embeddings
            
            # Cosine similarity matrix
            similarity_matrix = cosine_similarity(sample_embeddings)
            
            # Remove diagonal (self-similarity)
            mask = ~np.eye(similarity_matrix.shape[0], dtype=bool)
            similarities = similarity_matrix[mask]
            
            characteristics[model_name] = {
                'mean_cosine_similarity': float(np.mean(similarities)),
                'std_cosine_similarity': float(np.std(similarities)),
                'min_cosine_similarity': float(np.min(similarities)),
                'max_cosine_similarity': float(np.max(similarities)),
                'median_cosine_similarity': float(np.median(similarities)),
                'similarity_distribution_skew': float(self._calculate_skewness(similarities)),
                'similarity_distribution_kurtosis': float(self._calculate_kurtosis(similarities))
            }
        
        return characteristics
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data distribution"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data distribution"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 4) - 3
    
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
        if analysis_results['dimensional_statistics']:
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
