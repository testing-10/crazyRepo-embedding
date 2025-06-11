"""
Embedding Visualizations Module

Generates comprehensive visualizations for embedding model evaluation results
using matplotlib, seaborn, and plotly for interactive charts.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import os
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging

from utils.embedding_logger import EmbeddingLogger

class EmbeddingVisualizations:
    """
    Comprehensive visualization generator for embedding model evaluations
    """
    
    def __init__(self, output_dir: str = "reports/visualizations"):
        """
        Initialize the visualization generator
        
        Args:
            output_dir: Directory to save generated visualizations
        """
        self.output_dir = output_dir
        self.logger = EmbeddingLogger.get_logger(__name__)
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set style preferences
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Color schemes
        self.colors = {
            'primary': '#3498db',
            'secondary': '#2ecc71', 
            'accent': '#e74c3c',
            'warning': '#f39c12',
            'info': '#9b59b6',
            'success': '#27ae60',
            'danger': '#c0392b'
        }
        
    def generate_all_visualizations(
    self,
    evaluation_results: Dict[str, Any],
    model_configs: Dict[str, Any]
) -> Dict[str, List[str]]:
        """Generate all visualization types for the evaluation results"""
        try:
            self.logger.info("Starting comprehensive visualization generation")
            
            # Convert results to DataFrame for easier manipulation
            df = self._prepare_dataframe(evaluation_results, model_configs)
            
            visualization_paths = {}
            
            # Performance comparison charts
            performance_paths = self._generate_performance_charts(df)
            visualization_paths['performance'] = performance_paths
            
            # Skip other visualizations if no data
            if df.empty:
                self.logger.warning("Skipping additional visualizations - no evaluation data available")
                return visualization_paths
            
            # Only generate other charts if we have sufficient data
            try:
                cost_paths = self._generate_cost_analysis(df)
                visualization_paths['cost'] = cost_paths
            except Exception as e:
                self.logger.warning(f"Skipping cost analysis: {str(e)}")
                visualization_paths['cost'] = []
            
            try:
                efficiency_paths = self._generate_efficiency_analysis(df)
                visualization_paths['efficiency'] = efficiency_paths
            except Exception as e:
                self.logger.warning(f"Skipping efficiency analysis: {str(e)}")
                visualization_paths['efficiency'] = []
            
            try:
                correlation_paths = self._generate_correlation_analysis(df)
                visualization_paths['correlation'] = correlation_paths
            except Exception as e:
                self.logger.warning(f"Skipping correlation analysis: {str(e)}")
                visualization_paths['correlation'] = []
            
            try:
                dashboard_path = self._generate_interactive_dashboard(df)
                visualization_paths['dashboard'] = [dashboard_path]
            except Exception as e:
                self.logger.warning(f"Skipping interactive dashboard: {str(e)}")
                visualization_paths['dashboard'] = []
            
            self.logger.info(f"Visualizations generated successfully")
            return visualization_paths
            
        except Exception as e:
            self.logger.error(f"Error generating visualizations: {str(e)}")
            return {}
    
    def _prepare_dataframe(
        self,
        evaluation_results: Dict[str, Any],
        model_configs: Dict[str, Any]
    ) -> pd.DataFrame:
        """Prepare pandas DataFrame from evaluation results"""
        data = []
        
        for model_name, results in evaluation_results.items():
            config = model_configs.get(model_name, {})
            
            row = {
                'model_name': model_name,
                'provider': config.get('provider', 'Unknown'),
                'dimensions': config.get('dimensions', 0),
                'max_tokens': config.get('max_tokens', 0),
                'cost_per_1k': config.get('cost_per_1k_tokens', 0),
                
                # Performance scores
                'similarity_score': self._safe_extract_score(results, 'similarity', 'overall_score'),
                'retrieval_score': self._safe_extract_score(results, 'retrieval', 'overall_score'),
                'clustering_score': self._safe_extract_score(results, 'clustering', 'overall_score'),
                'classification_score': self._safe_extract_score(results, 'classification', 'overall_score'),
                'efficiency_score': self._safe_extract_score(results, 'efficiency', 'overall_score'),
                
                # Cost and performance metrics
                'total_cost': results.get('cost_tracking', {}).get('total_cost', 0.0),
                'avg_latency': results.get('efficiency', {}).get('avg_latency', 0.0),
                'throughput': results.get('efficiency', {}).get('throughput', 0.0),
                'error_rate': results.get('error_tracking', {}).get('error_rate', 0.0),
                'tokens_processed': results.get('cost_tracking', {}).get('total_tokens', 0)
            }
            
            # Calculate overall score
            scores = [row['similarity_score'], row['retrieval_score'], 
                     row['clustering_score'], row['classification_score'], row['efficiency_score']]
            row['overall_score'] = sum(scores) / len(scores)
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def _safe_extract_score(self, results: Dict, category: str, metric: str) -> float:
        """Safely extract score with fallback"""
        try:
            return float(results.get(category, {}).get(metric, 0.0))
        except (ValueError, TypeError):
            return 0.0
    
    def _generate_performance_charts(self, df: pd.DataFrame) -> Dict[str, str]:
        """Generate performance comparison charts"""
        try:
            paths = {}
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Check if DataFrame is empty or has no useful data
            if df.empty:
                self.logger.warning("No data available for performance charts - DataFrame is empty")
                return paths
                
            # Check if we have the required columns
            required_cols = ['overall_score', 'model_name']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                self.logger.warning(f"Missing required columns for performance charts: {missing_cols}")
                # Try to create overall_score if we have any numeric columns
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols and 'overall_score' in missing_cols:
                    df['overall_score'] = df[numeric_cols].mean(axis=1)
                    self.logger.info("Created overall_score from available numeric columns")
                else:
                    return paths
            
            # Ensure we have model names
            if 'model_name' not in df.columns:
                df['model_name'] = df.index.astype(str)
            
            # 1. Overall Performance Bar Chart
            plt.figure(figsize=(12, 8))
            df_sorted = df.sort_values('overall_score', ascending=True)
            
            bars = plt.barh(df_sorted['model_name'], df_sorted['overall_score'], 
                        color=self.colors['primary'], alpha=0.8)
            
            # Add value labels on bars
            for i, bar in enumerate(bars):
                width = bar.get_width()
                plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                        f'{width:.3f}', ha='left', va='center', fontweight='bold')
            
            plt.xlabel('Overall Performance Score', fontsize=12, fontweight='bold')
            plt.ylabel('Model', fontsize=12, fontweight='bold')
            plt.title('Overall Performance Comparison Across Models', fontsize=14, fontweight='bold')
            plt.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            
            path = os.path.join(self.output_dir, f'overall_performance_{timestamp}.png')
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plt.close()
            paths['performance_chart'] = path
            
            # Only generate additional charts if we have sufficient data
            if len(df) > 1:
                # 2. Multi-metric Radar Chart (only if we have multiple metrics)
                metrics = ['similarity_score', 'retrieval_score', 'clustering_score', 
                        'classification_score', 'efficiency_score']
                available_metrics = [m for m in metrics if m in df.columns]
                
                if len(available_metrics) >= 3:  # Need at least 3 metrics for radar chart
                    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
                    
                    metric_labels = [m.replace('_score', '').title() for m in available_metrics]
                    angles = np.linspace(0, 2 * np.pi, len(available_metrics), endpoint=False).tolist()
                    angles += angles[:1]  # Complete the circle
                    
                    colors_list = plt.cm.Set3(np.linspace(0, 1, len(df)))
                    
                    for i, (_, row) in enumerate(df.iterrows()):
                        values = [row.get(metric, 0) for metric in available_metrics]
                        values += values[:1]  # Complete the circle
                        
                        ax.plot(angles, values, 'o-', linewidth=2, label=row['model_name'], 
                            color=colors_list[i], alpha=0.8)
                        ax.fill(angles, values, alpha=0.1, color=colors_list[i])
                    
                    ax.set_xticks(angles[:-1])
                    ax.set_xticklabels(metric_labels)
                    ax.set_ylim(0, 1)
                    ax.set_title('Multi-Metric Performance Radar Chart', size=14, fontweight='bold', pad=20)
                    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
                    ax.grid(True)
                    
                    path = os.path.join(self.output_dir, f'radar_chart_{timestamp}.png')
                    plt.savefig(path, dpi=300, bbox_inches='tight')
                    plt.close()
                    paths['radar_chart'] = path
            
            self.logger.info(f"Generated {len(paths)} performance charts")
            return paths
            
        except Exception as e:
            self.logger.error(f"Error generating performance charts: {str(e)}")
            return {}
    
    def _generate_cost_analysis(self, df: pd.DataFrame) -> List[str]:
        """Generate cost analysis visualizations"""
        try:
            paths = []
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Check if DataFrame is empty or missing required columns
            if df.empty:
                self.logger.warning("No data available for cost analysis - DataFrame is empty")
                return paths
                
            required_cols = ['total_cost', 'overall_score']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                self.logger.warning(f"Missing required columns for cost analysis: {missing_cols}")
                return paths
            
            # Check if we have any cost data (for local models, costs might be 0)
            if df['total_cost'].sum() == 0:
                self.logger.info("No cost data available (likely local models) - skipping cost analysis")
                return paths
            
            # 1. Cost vs Performance Scatter Plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Cost vs Overall Performance
            scatter1 = ax1.scatter(df['total_cost'], df['overall_score'], 
                                s=100, alpha=0.7, c=df['overall_score'], 
                                cmap='RdYlGn', edgecolors='black')
            
            for i, row in df.iterrows():
                ax1.annotate(row['model_name'], (row['total_cost'], row['overall_score']),
                            xytext=(5, 5), textcoords='offset points', fontsize=9)
            
            ax1.set_xlabel('Total Cost ($)', fontweight='bold')
            ax1.set_ylabel('Overall Performance Score', fontweight='bold')
            ax1.set_title('Cost vs Performance Trade-off', fontweight='bold')
            ax1.grid(True, alpha=0.3)
            
            # Cost per token vs Performance (only if cost_per_1k exists and > 0)
            if 'cost_per_1k' in df.columns and df['cost_per_1k'].sum() > 0:
                df['cost_efficiency'] = df['overall_score'] / (df['cost_per_1k'] + 0.0001)
                
                scatter2 = ax2.scatter(df['cost_per_1k'], df['overall_score'], 
                                    s=100, alpha=0.7, c=df['cost_efficiency'], 
                                    cmap='RdYlGn', edgecolors='black')
                
                for i, row in df.iterrows():
                    ax2.annotate(row['model_name'], (row['cost_per_1k'], row['overall_score']),
                                xytext=(5, 5), textcoords='offset points', fontsize=9)
                
                ax2.set_xlabel('Cost per 1K Tokens ($)', fontweight='bold')
                ax2.set_ylabel('Overall Performance Score', fontweight='bold')
                ax2.set_title('Cost Efficiency Analysis', fontweight='bold')
                ax2.grid(True, alpha=0.3)
            else:
                ax2.text(0.5, 0.5, 'No cost data available\n(Local models)', 
                        ha='center', va='center', transform=ax2.transAxes, 
                        fontsize=14, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
                ax2.set_title('Cost Efficiency Analysis - N/A', fontweight='bold')
            
            plt.tight_layout()
            
            path = os.path.join(self.output_dir, f'cost_analysis_{timestamp}.png')
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plt.close()
            paths.append(path)
            
            # 2. Cost breakdown by provider (only if we have cost data)
            if 'provider' in df.columns and df['total_cost'].sum() > 0:
                plt.figure(figsize=(10, 6))
                
                provider_costs = df.groupby('provider')['total_cost'].sum().sort_values(ascending=False)
                
                bars = plt.bar(provider_costs.index, provider_costs.values, 
                            color=plt.cm.Set3(np.linspace(0, 1, len(provider_costs))))
                
                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                            f'${height:.4f}', ha='center', va='bottom', fontweight='bold')
                
                plt.xlabel('Provider', fontweight='bold')
                plt.ylabel('Total Cost ($)', fontweight='bold')
                plt.title('Total Cost by Provider', fontweight='bold')
                plt.xticks(rotation=45)
                plt.grid(axis='y', alpha=0.3)
                plt.tight_layout()
                
                path = os.path.join(self.output_dir, f'cost_by_provider_{timestamp}.png')
                plt.savefig(path, dpi=300, bbox_inches='tight')
                plt.close()
                paths.append(path)
            
            return paths
            
        except Exception as e:
            self.logger.error(f"Error generating cost analysis: {str(e)}")
            return []
    
    def _generate_efficiency_analysis(self, df: pd.DataFrame) -> List[str]:
        """Generate efficiency analysis visualizations"""
        try:
            paths = []
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Check if DataFrame is empty
            if df.empty:
                self.logger.warning("No data available for efficiency analysis - DataFrame is empty")
                return paths
                
            required_cols = ['avg_latency', 'throughput', 'model_name']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                self.logger.warning(f"Missing required columns for efficiency analysis: {missing_cols}")
                return paths
            
            # Check if we have any efficiency data
            if df['avg_latency'].sum() == 0 and df['throughput'].sum() == 0:
                self.logger.info("No efficiency data available - skipping efficiency analysis")
                return paths
            
            # 1. Latency vs Throughput Analysis
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Latency comparison
            if df['avg_latency'].sum() > 0:
                df_sorted = df.sort_values('avg_latency', ascending=True)
                bars1 = ax1.barh(df_sorted['model_name'], df_sorted['avg_latency'], 
                                color=self.colors['warning'], alpha=0.8)
                
                for i, bar in enumerate(bars1):
                    width = bar.get_width()
                    if width > 0:
                        ax1.text(width + width*0.01, bar.get_y() + bar.get_height()/2, 
                                f'{width:.1f}ms', ha='left', va='center', fontweight='bold')
                
                ax1.set_xlabel('Average Latency (ms)', fontweight='bold')
                ax1.set_ylabel('Model', fontweight='bold')
                ax1.set_title('Average Response Latency', fontweight='bold')
                ax1.grid(axis='x', alpha=0.3)
            else:
                ax1.text(0.5, 0.5, 'No latency data available', 
                        ha='center', va='center', transform=ax1.transAxes, 
                        fontsize=14, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
                ax1.set_title('Average Response Latency - N/A', fontweight='bold')
            
            # Throughput comparison
            if df['throughput'].sum() > 0:
                df_sorted = df.sort_values('throughput', ascending=True)
                bars2 = ax2.barh(df_sorted['model_name'], df_sorted['throughput'], 
                                color=self.colors['success'], alpha=0.8)
                
                for i, bar in enumerate(bars2):
                    width = bar.get_width()
                    if width > 0:
                        ax2.text(width + width*0.01, bar.get_y() + bar.get_height()/2, 
                                f'{width:.1f}', ha='left', va='center', fontweight='bold')
                
                ax2.set_xlabel('Throughput (requests/second)', fontweight='bold')
                ax2.set_ylabel('Model', fontweight='bold')
                ax2.set_title('Request Throughput', fontweight='bold')
                ax2.grid(axis='x', alpha=0.3)
            else:
                ax2.text(0.5, 0.5, 'No throughput data available', 
                        ha='center', va='center', transform=ax2.transAxes, 
                        fontsize=14, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
                ax2.set_title('Request Throughput - N/A', fontweight='bold')
            
            plt.tight_layout()
            
            path = os.path.join(self.output_dir, f'efficiency_analysis_{timestamp}.png')
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plt.close()
            paths.append(path)
            
            # 2. Error Rate Analysis (only if there are errors to show)
            if 'error_rate' in df.columns and df['error_rate'].sum() > 0:
                plt.figure(figsize=(10, 6))
                
                df_sorted = df.sort_values('error_rate', ascending=False)
                bars = plt.bar(df_sorted['model_name'], df_sorted['error_rate'], 
                            color=self.colors['danger'], alpha=0.8)
                
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                                f'{height:.2f}%', ha='center', va='bottom', fontweight='bold')
                
                plt.xlabel('Model', fontweight='bold')
                plt.ylabel('Error Rate (%)', fontweight='bold')
                plt.title('Error Rate Comparison', fontweight='bold')
                plt.xticks(rotation=45)
                plt.grid(axis='y', alpha=0.3)
                plt.tight_layout()
                
                path = os.path.join(self.output_dir, f'error_rates_{timestamp}.png')
                plt.savefig(path, dpi=300, bbox_inches='tight')
                plt.close()
                paths.append(path)
            
            return paths
            
        except Exception as e:
            self.logger.error(f"Error generating efficiency analysis: {str(e)}")
            return []
    
    def _generate_correlation_analysis(self, df: pd.DataFrame) -> List[str]:
        """Generate correlation analysis visualizations"""
        try:
            paths = []
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Check if DataFrame is empty
            if df.empty:
                self.logger.warning("No data available for correlation analysis - DataFrame is empty")
                return paths
            
            # Select only available numeric columns for correlation
            numeric_cols = ['similarity_score', 'retrieval_score', 'clustering_score', 
                        'classification_score', 'efficiency_score', 'overall_score',
                        'total_cost', 'avg_latency', 'throughput', 'dimensions']
            
            available_cols = [col for col in numeric_cols if col in df.columns]
            
            if len(available_cols) < 2:
                self.logger.warning("Insufficient numeric columns for correlation analysis")
                return paths
            
            correlation_data = df[available_cols].corr()
            
            # Correlation heatmap
            plt.figure(figsize=(12, 10))
            
            mask = np.triu(np.ones_like(correlation_data, dtype=bool))
            sns.heatmap(correlation_data, mask=mask, annot=True, cmap='RdBu_r', center=0,
                    square=True, fmt='.2f', cbar_kws={'label': 'Correlation Coefficient'})
            
            plt.title('Correlation Matrix of Performance Metrics', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            path = os.path.join(self.output_dir, f'correlation_matrix_{timestamp}.png')
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plt.close()
            paths.append(path)
            
            return paths
            
        except Exception as e:
            self.logger.error(f"Error generating correlation analysis: {str(e)}")
            return []
    
    def _generate_interactive_dashboard(self, df: pd.DataFrame) -> str:
        """Generate interactive Plotly dashboard"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Check if DataFrame is empty
            if df.empty:
                self.logger.warning("No data available for interactive dashboard - DataFrame is empty")
                # Create empty dashboard
                fig = go.Figure()
                fig.add_annotation(
                    text="No evaluation data available",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, xanchor='center', yanchor='middle',
                    showarrow=False, font=dict(size=20)
                )
                fig.update_layout(title="Embedding Model Evaluation Dashboard - No Data")
                
                path = os.path.join(self.output_dir, f'interactive_dashboard_{timestamp}.html')
                fig.write_html(path)
                return path
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Overall Performance', 'Cost vs Performance', 
                            'Efficiency Metrics', 'Score Distribution'),
                specs=[[{"type": "bar"}, {"type": "scatter"}],
                    [{"type": "bar"}, {"type": "histogram"}]]
            )
            
            # 1. Overall Performance Bar Chart
            fig.add_trace(
                go.Bar(x=df['model_name'], y=df['overall_score'], 
                    name='Overall Score', marker_color='lightblue'),
                row=1, col=1
            )
            
            # 2. Cost vs Performance Scatter (only if cost data available)
            if 'total_cost' in df.columns and df['total_cost'].sum() > 0:
                fig.add_trace(
                    go.Scatter(x=df['total_cost'], y=df['overall_score'], 
                            mode='markers+text', text=df['model_name'],
                            textposition='top center', name='Models',
                            marker=dict(size=10, color=df['overall_score'], 
                                        colorscale='RdYlGn', showscale=True)),
                    row=1, col=2
                )
            else:
                # Add text annotation for no cost data
                fig.add_annotation(
                    text="No cost data available<br>(Local models)",
                    xref="x2", yref="y2",
                    x=0.5, y=0.5, xanchor='center', yanchor='middle',
                    showarrow=False, font=dict(size=14),
                    row=1, col=2
                )
            
            # 3. Efficiency Metrics (latency if available)
            if 'avg_latency' in df.columns and df['avg_latency'].sum() > 0:
                fig.add_trace(
                    go.Bar(x=df['model_name'], y=df['avg_latency'], 
                        name='Latency (ms)', marker_color='orange'),
                    row=2, col=1
                )
            else:
                fig.add_annotation(
                    text="No latency data available",
                    xref="x3", yref="y3",
                    x=0.5, y=0.5, xanchor='center', yanchor='middle',
                    showarrow=False, font=dict(size=14),
                    row=2, col=1
                )
            
            # 4. Score Distribution
            fig.add_trace(
                go.Histogram(x=df['overall_score'], nbinsx=10, 
                            name='Score Distribution', marker_color='green'),
                row=2, col=2
            )
            
            # Update layout
            fig.update_layout(
                title_text="Embedding Model Evaluation Dashboard",
                title_x=0.5,
                height=800,
                showlegend=False
            )
            
            # Update axes labels
            fig.update_xaxes(title_text="Model", row=1, col=1)
            fig.update_yaxes(title_text="Overall Score", row=1, col=1)
            
            fig.update_xaxes(title_text="Total Cost ($)", row=1, col=2)
            fig.update_yaxes(title_text="Overall Score", row=1, col=2)
            
            fig.update_xaxes(title_text="Model", row=2, col=1)
            fig.update_yaxes(title_text="Latency (ms)", row=2, col=1)
            
            fig.update_xaxes(title_text="Overall Score", row=2, col=2)
            fig.update_yaxes(title_text="Count", row=2, col=2)
            
            # Save interactive HTML
            path = os.path.join(self.output_dir, f'interactive_dashboard_{timestamp}.html')
            fig.write_html(path)
            
            self.logger.info(f"Interactive dashboard generated: {path}")
            return path
            
        except Exception as e:
            self.logger.error(f"Error generating interactive dashboard: {str(e)}")
            # Return empty dashboard path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            return os.path.join(self.output_dir, f'interactive_dashboard_{timestamp}.html')
    
    def generate_custom_visualization(
        self,
        df: pd.DataFrame,
        chart_type: str,
        x_column: str,
        y_column: str,
        title: str,
        filename: str
    ) -> str:
        """
        Generate custom visualization based on parameters
        
        Args:
            df: DataFrame with evaluation data
            chart_type: Type of chart ('bar', 'scatter', 'line', 'box')
            x_column: Column for x-axis
            y_column: Column for y-axis
            title: Chart title
            filename: Output filename
            
        Returns:
            Path to generated visualization
        """
        plt.figure(figsize=(10, 6))
        
        if chart_type == 'bar':
            plt.bar(df[x_column], df[y_column], alpha=0.8)
        elif chart_type == 'scatter':
            plt.scatter(df[x_column], df[y_column], alpha=0.7, s=100)
        elif chart_type == 'line':
            plt.plot(df[x_column], df[y_column], marker='o', linewidth=2)
        elif chart_type == 'box':
            plt.boxplot([df[y_column]], labels=[y_column])
        
        plt.xlabel(x_column.replace('_', ' ').title(), fontweight='bold')
        plt.ylabel(y_column.replace('_', ' ').title(), fontweight='bold')
        plt.title(title, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return path
