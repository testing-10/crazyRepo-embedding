"""
Embedding Comparison Report Generator

Generates comprehensive comparison reports across multiple embedding models
with support for HTML, JSON, and CSV outputs.
"""

import json
import csv
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import logging

from utils.embedding_logger import EmbeddingLogger
from utils.embedding_file_utils import FileUtils

@dataclass
class ModelPerformance:
    """Data class for model performance metrics"""
    model_name: str
    similarity_score: float
    retrieval_score: float
    clustering_score: float
    classification_score: float
    efficiency_score: float
    total_cost: float
    avg_latency: float
    throughput: float
    error_rate: float
    
class EmbeddingComparisonReport:
    """
    Generates comprehensive comparison reports for embedding model evaluations
    """
    
    def __init__(self, output_dir: str = "reports"):
        """
        Initialize the comparison report generator
        
        Args:
            output_dir: Directory to save generated reports
        """
        self.output_dir = output_dir
        self.logger = EmbeddingLogger.get_logger(__name__)
        self.file_utils = FileUtils()
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
    def generate_comparison_report(
        self,
        evaluation_results: Dict[str, Any],
        model_configs: Dict[str, Any],
        test_metadata: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Generate comprehensive comparison report in multiple formats
        
        Args:
            evaluation_results: Results from all model evaluations
            model_configs: Configuration data for all models
            test_metadata: Metadata about test execution
            
        Returns:
            Dictionary with paths to generated report files
        """
        try:
            self.logger.info("Starting comparison report generation")
            
            # Process and aggregate results
            processed_results = self._process_evaluation_results(evaluation_results)
            
            # Generate reports in different formats
            report_paths = {}
            
            # HTML Report
            html_path = self._generate_html_report(
                processed_results, model_configs, test_metadata
            )
            report_paths['html'] = html_path
            
            # JSON Report
            json_path = self._generate_json_report(
                processed_results, model_configs, test_metadata
            )
            report_paths['json'] = json_path
            
            # CSV Report
            csv_path = self._generate_csv_report(processed_results)
            report_paths['csv'] = csv_path
            
            self.logger.info(f"Comparison reports generated: {list(report_paths.keys())}")
            return report_paths
            
        except Exception as e:
            self.logger.error(f"Error generating comparison report: {str(e)}")
            raise
    
    def _process_evaluation_results(self, evaluation_results: Dict[str, Any]) -> List[ModelPerformance]:
        """Process raw evaluation results into structured performance data"""
        processed_results = []
        
        for model_name, results in evaluation_results.items():
            try:
                # Extract performance metrics with safe defaults
                similarity_score = self._safe_get_score(results, 'similarity', 'overall_score')
                retrieval_score = self._safe_get_score(results, 'retrieval', 'overall_score')
                clustering_score = self._safe_get_score(results, 'clustering', 'overall_score')
                classification_score = self._safe_get_score(results, 'classification', 'overall_score')
                efficiency_score = self._safe_get_score(results, 'efficiency', 'overall_score')
                
                # Extract cost and performance metrics
                total_cost = results.get('cost_tracking', {}).get('total_cost', 0.0)
                avg_latency = results.get('efficiency', {}).get('avg_latency', 0.0)
                throughput = results.get('efficiency', {}).get('throughput', 0.0)
                error_rate = results.get('error_tracking', {}).get('error_rate', 0.0)
                
                performance = ModelPerformance(
                    model_name=model_name,
                    similarity_score=similarity_score,
                    retrieval_score=retrieval_score,
                    clustering_score=clustering_score,
                    classification_score=classification_score,
                    efficiency_score=efficiency_score,
                    total_cost=total_cost,
                    avg_latency=avg_latency,
                    throughput=throughput,
                    error_rate=error_rate
                )
                
                processed_results.append(performance)
                
            except Exception as e:
                self.logger.warning(f"Error processing results for {model_name}: {str(e)}")
                continue
        
        return processed_results
    
    def _safe_get_score(self, results: Dict, category: str, metric: str) -> float:
        """Safely extract score with fallback to 0.0"""
        try:
            return float(results.get(category, {}).get(metric, 0.0))
        except (ValueError, TypeError):
            return 0.0
    
    def _generate_html_report(
        self,
        processed_results: List[ModelPerformance],
        model_configs: Dict[str, Any],
        test_metadata: Dict[str, Any]
    ) -> str:
        """Generate comprehensive HTML report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        html_path = os.path.join(self.output_dir, f"embedding_comparison_{timestamp}.html")
        
        # Sort results by overall performance (average of all scores)
        sorted_results = sorted(
            processed_results,
            key=lambda x: (x.similarity_score + x.retrieval_score + x.clustering_score + 
                          x.classification_score + x.efficiency_score) / 5,
            reverse=True
        )
        
        html_content = self._build_html_content(sorted_results, model_configs, test_metadata)
        
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self.logger.info(f"HTML report generated: {html_path}")
        return html_path
    
    def _build_html_content(
        self,
        results: List[ModelPerformance],
        model_configs: Dict[str, Any],
        test_metadata: Dict[str, Any]
    ) -> str:
        """Build HTML content for the report"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Embedding Model Comparison Report</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 20px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; text-align: center; margin-bottom: 30px; }}
        h2 {{ color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        .metadata {{ background: #ecf0f1; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
        .performance-table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        .performance-table th, .performance-table td {{ padding: 12px; text-align: center; border: 1px solid #ddd; }}
        .performance-table th {{ background-color: #3498db; color: white; }}
        .performance-table tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .score-high {{ background-color: #2ecc71; color: white; font-weight: bold; }}
        .score-medium {{ background-color: #f39c12; color: white; font-weight: bold; }}
        .score-low {{ background-color: #e74c3c; color: white; font-weight: bold; }}
        .summary-stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
        .stat-card {{ background: #3498db; color: white; padding: 20px; border-radius: 5px; text-align: center; }}
        .stat-value {{ font-size: 2em; font-weight: bold; }}
        .stat-label {{ font-size: 0.9em; opacity: 0.9; }}
        .footer {{ text-align: center; margin-top: 30px; color: #7f8c8d; font-size: 0.9em; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 Embedding Model Comparison Report</h1>
        
        <div class="metadata">
            <h2>📊 Test Execution Summary</h2>
            <p><strong>Generated:</strong> {timestamp}</p>
            <p><strong>Models Tested:</strong> {len(results)}</p>
            <p><strong>Test Categories:</strong> {', '.join(test_metadata.get('categories', ['Similarity', 'Retrieval', 'Clustering', 'Classification', 'Efficiency']))}</p>
            <p><strong>Total Test Cases:</strong> {test_metadata.get('total_test_cases', 'N/A')}</p>
        </div>
        
        <div class="summary-stats">
            <div class="stat-card">
                <div class="stat-value">{len(results)}</div>
                <div class="stat-label">Models Evaluated</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{max([r.similarity_score for r in results], default=0):.2f}</div>
                <div class="stat-label">Best Similarity Score</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{min([r.total_cost for r in results if r.total_cost > 0], default=0):.4f}</div>
                <div class="stat-label">Lowest Cost ($)</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{max([r.throughput for r in results], default=0):.0f}</div>
                <div class="stat-label">Max Throughput (req/s)</div>
            </div>
        </div>
        
        <h2>🏆 Performance Comparison</h2>
        <table class="performance-table">
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Model</th>
                    <th>Similarity</th>
                    <th>Retrieval</th>
                    <th>Clustering</th>
                    <th>Classification</th>
                    <th>Efficiency</th>
                    <th>Overall</th>
                    <th>Cost ($)</th>
                    <th>Latency (ms)</th>
                    <th>Error Rate (%)</th>
                </tr>
            </thead>
            <tbody>
"""
        
        for i, result in enumerate(results, 1):
            overall_score = (result.similarity_score + result.retrieval_score + 
                           result.clustering_score + result.classification_score + 
                           result.efficiency_score) / 5
            
            html += f"""
                <tr>
                    <td>{i}</td>
                    <td><strong>{result.model_name}</strong></td>
                    <td class="{self._get_score_class(result.similarity_score)}">{result.similarity_score:.3f}</td>
                    <td class="{self._get_score_class(result.retrieval_score)}">{result.retrieval_score:.3f}</td>
                    <td class="{self._get_score_class(result.clustering_score)}">{result.clustering_score:.3f}</td>
                    <td class="{self._get_score_class(result.classification_score)}">{result.classification_score:.3f}</td>
                    <td class="{self._get_score_class(result.efficiency_score)}">{result.efficiency_score:.3f}</td>
                    <td class="{self._get_score_class(overall_score)}">{overall_score:.3f}</td>
                    <td>${result.total_cost:.4f}</td>
                    <td>{result.avg_latency:.1f}</td>
                    <td>{result.error_rate:.2f}%</td>
                </tr>
"""
        
        html += """
            </tbody>
        </table>
        
        <div class="footer">
            <p>Generated by Embedding Model Testing Framework</p>
            <p>Scores range from 0.0 to 1.0 (higher is better)</p>
        </div>
    </div>
</body>
</html>
"""
        return html
    
    def _get_score_class(self, score: float) -> str:
        """Get CSS class based on score value"""
        if score >= 0.8:
            return "score-high"
        elif score >= 0.6:
            return "score-medium"
        else:
            return "score-low"
    
    def _generate_json_report(
        self,
        processed_results: List[ModelPerformance],
        model_configs: Dict[str, Any],
        test_metadata: Dict[str, Any]
    ) -> str:
        """Generate detailed JSON report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = os.path.join(self.output_dir, f"embedding_comparison_{timestamp}.json")
        
        report_data = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "framework_version": "1.0.0",
                "total_models": len(processed_results),
                "test_metadata": test_metadata
            },
            "model_configurations": model_configs,
            "performance_results": [asdict(result) for result in processed_results],
            "summary_statistics": self._calculate_summary_stats(processed_results)
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"JSON report generated: {json_path}")
        return json_path
    
    def _generate_csv_report(self, processed_results: List[ModelPerformance]) -> str:
        """Generate CSV report for easy data analysis"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(self.output_dir, f"embedding_comparison_{timestamp}.csv")
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow([
                'Model Name', 'Similarity Score', 'Retrieval Score', 'Clustering Score',
                'Classification Score', 'Efficiency Score', 'Overall Score',
                'Total Cost', 'Avg Latency (ms)', 'Throughput (req/s)', 'Error Rate (%)'
            ])
            
            # Write data rows
            for result in processed_results:
                overall_score = (result.similarity_score + result.retrieval_score + 
                               result.clustering_score + result.classification_score + 
                               result.efficiency_score) / 5
                
                writer.writerow([
                    result.model_name,
                    f"{result.similarity_score:.4f}",
                    f"{result.retrieval_score:.4f}",
                    f"{result.clustering_score:.4f}",
                    f"{result.classification_score:.4f}",
                    f"{result.efficiency_score:.4f}",
                    f"{overall_score:.4f}",
                    f"{result.total_cost:.6f}",
                    f"{result.avg_latency:.2f}",
                    f"{result.throughput:.2f}",
                    f"{result.error_rate:.3f}"
                ])
        
        self.logger.info(f"CSV report generated: {csv_path}")
        return csv_path
    
    def _calculate_summary_stats(self, results: List[ModelPerformance]) -> Dict[str, Any]:
        """Calculate summary statistics across all models"""
        if not results:
            return {}
        
        similarity_scores = [r.similarity_score for r in results]
        retrieval_scores = [r.retrieval_score for r in results]
        clustering_scores = [r.clustering_score for r in results]
        classification_scores = [r.classification_score for r in results]
        efficiency_scores = [r.efficiency_score for r in results]
        costs = [r.total_cost for r in results if r.total_cost > 0]
        latencies = [r.avg_latency for r in results if r.avg_latency > 0]
        
        return {
            "similarity": {
                "mean": sum(similarity_scores) / len(similarity_scores),
                "max": max(similarity_scores),
                "min": min(similarity_scores)
            },
            "retrieval": {
                "mean": sum(retrieval_scores) / len(retrieval_scores),
                "max": max(retrieval_scores),
                "min": min(retrieval_scores)
            },
            "clustering": {
                "mean": sum(clustering_scores) / len(clustering_scores),
                "max": max(clustering_scores),
                "min": min(clustering_scores)
            },
            "classification": {
                "mean": sum(classification_scores) / len(classification_scores),
                "max": max(classification_scores),
                "min": min(classification_scores)
            },
            "efficiency": {
                "mean": sum(efficiency_scores) / len(efficiency_scores),
                "max": max(efficiency_scores),
                "min": min(efficiency_scores)
            },
            "cost": {
                "mean": sum(costs) / len(costs) if costs else 0,
                "max": max(costs) if costs else 0,
                "min": min(costs) if costs else 0
            },
            "latency": {
                "mean": sum(latencies) / len(latencies) if latencies else 0,
                "max": max(latencies) if latencies else 0,
                "min": min(latencies) if latencies else 0
            }
        }
