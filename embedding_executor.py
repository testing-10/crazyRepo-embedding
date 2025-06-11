"""
Embedding Model Testing Framework - Main Executor

Main orchestrator that coordinates the entire embedding evaluation pipeline:
load config → generate embeddings → evaluate → report
"""

import os
import sys
import json
import yaml
import time
import traceback
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import argparse
import logging

# Framework imports
from embedding_client_factory import EmbeddingClientFactory
from utils.embedding_logger import EmbeddingLogger
from utils.embedding_file_utils import FileUtils
from utils.embedding_cost_tracker import CostTracker
from datasets.dataset_loader import DatasetLoader
from evaluators.evaluator_factory import EvaluatorFactory
from reporting.embedding_comparison_report import EmbeddingComparisonReport
from reporting.embedding_visualizations import EmbeddingVisualizations
from reporting.dimension_analysis import DimensionAnalysis

@dataclass
class ExecutionState:
    """Track execution state for resumption capability"""
    execution_id: str
    start_time: str
    current_step: str
    completed_models: List[str]
    failed_models: List[str]
    total_cost: float
    config_hash: str
    
class EmbeddingExecutor:
    """
    Main orchestrator for the embedding model testing framework
    """
    
    def __init__(self, config_path: str = "config/test_config.yaml"):
        """
        Initialize the executor
        
        Args:
            config_path: Path to the main configuration file
        """
        self.config_path = config_path
        self.config = None
        self.execution_id = self._generate_execution_id()
        self.execution_state = None
        
        # Initialize utilities
        self.logger = EmbeddingLogger.get_logger(__name__)
        self.file_utils = FileUtils()
        self.cost_tracker = CostTracker()
        
        # Initialize components
        self.client_factory = EmbeddingClientFactory()
        self.dataset_loader = DatasetLoader()
        self.evaluator_factory = EvaluatorFactory()
        
        # Results storage
        self.evaluation_results = {}
        self.model_clients = {}
        self.datasets = {}
        
        # Output directories
        self.output_dir = "reports"
        self.state_dir = "execution_state"
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.state_dir, exist_ok=True)
        
    def _generate_execution_id(self) -> str:
        """Generate unique execution ID"""
        return f"exec_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def load_configuration(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load and validate configuration
        
        Args:
            config_path: Optional override for config path
            
        Returns:
            Loaded configuration dictionary
        """
        try:
            config_file = config_path or self.config_path
            self.logger.info(f"Loading configuration from: {config_file}")
            
            if not os.path.exists(config_file):
                raise FileNotFoundError(f"Configuration file not found: {config_file}")
            
            with open(config_file, 'r') as f:
                self.config = yaml.safe_load(f)
            
            # Validate configuration
            self._validate_configuration()
            
            # Update output directory if specified in config
            if 'output_dir' in self.config:
                self.output_dir = self.config['output_dir']
                os.makedirs(self.output_dir, exist_ok=True)
            
            self.logger.info("Configuration loaded and validated successfully")
            return self.config
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {str(e)}")
            raise
    
    def _validate_configuration(self) -> None:
        """Validate the loaded configuration"""
        required_sections = ['models', 'datasets', 'evaluators']
        
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Validate models section
        if not self.config['models']:
            raise ValueError("No models specified in configuration")
        
        for model_name, model_config in self.config['models'].items():
            if 'provider' not in model_config:
                raise ValueError(f"Missing provider for model: {model_name}")
            if 'model_name' not in model_config:
                raise ValueError(f"Missing model_name for model: {model_name}")
        
        # Validate datasets section
        if not self.config['datasets']:
            raise ValueError("No datasets specified in configuration")
        
        # Validate evaluators section
        if not self.config['evaluators']:
            raise ValueError("No evaluators specified in configuration")
    
    def initialize_execution_state(self) -> None:
        """Initialize execution state for tracking and resumption"""
        config_hash = self.file_utils.calculate_file_hash(self.config_path)
        
        self.execution_state = ExecutionState(
            execution_id=self.execution_id,
            start_time=datetime.now().isoformat(),
            current_step="initialization",
            completed_models=[],
            failed_models=[],
            total_cost=0.0,
            config_hash=config_hash
        )
        
        self._save_execution_state()
        self.logger.info(f"Initialized execution state: {self.execution_id}")
    
    def _save_execution_state(self) -> None:
        """Save current execution state"""
        state_file = os.path.join(self.state_dir, f"{self.execution_id}_state.json")
        
        try:
            with open(state_file, 'w') as f:
                json.dump(asdict(self.execution_state), f, indent=2)
        except Exception as e:
            self.logger.warning(f"Failed to save execution state: {str(e)}")
    
    def load_execution_state(self, execution_id: str) -> bool:
        """
        Load previous execution state for resumption
        
        Args:
            execution_id: ID of execution to resume
            
        Returns:
            True if state loaded successfully, False otherwise
        """
        state_file = os.path.join(self.state_dir, f"{execution_id}_state.json")
        
        try:
            if not os.path.exists(state_file):
                self.logger.warning(f"State file not found: {state_file}")
                return False
            
            with open(state_file, 'r') as f:
                state_data = json.load(f)
            
            self.execution_state = ExecutionState(**state_data)
            self.execution_id = execution_id
            
            self.logger.info(f"Loaded execution state: {execution_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load execution state: {str(e)}")
            return False
    
    def setup_model_clients(self) -> Dict[str, Any]:
        """
        Initialize all model clients
        
        Returns:
            Dictionary of successfully created clients
        """
        try:
            self.execution_state.current_step = "client_setup"
            self._save_execution_state()
            
            self.logger.info("Setting up model clients...")
            
            # Load model configurations
            model_configs = {}
            for model_name, config in self.config['models'].items():
                # Load model-specific config file if specified
                if 'config_file' in config:
                    config_file = config['config_file']
                    if os.path.exists(config_file):
                        with open(config_file, 'r') as f:
                            model_specific_config = yaml.safe_load(f)
                        # Merge configurations
                        merged_config = {**model_specific_config, **config}
                        model_configs[model_name] = merged_config
                    else:
                        self.logger.warning(f"Model config file not found: {config_file}")
                        model_configs[model_name] = config
                else:
                    model_configs[model_name] = config
            
            # Create clients
            self.model_clients = self.client_factory.create_multiple_clients(model_configs)
            
            if not self.model_clients:
                raise RuntimeError("No model clients were successfully created")
            
            self.logger.info(f"Successfully created {len(self.model_clients)} model clients")
            return self.model_clients
            
        except Exception as e:
            self.logger.error(f"Failed to setup model clients: {str(e)}")
            raise
    
    def load_datasets(self) -> Dict[str, Any]:
        """
        Load all configured datasets
        
        Returns:
            Dictionary of loaded datasets
        """
        try:
            self.execution_state.current_step = "dataset_loading"
            self._save_execution_state()
            
            self.logger.info("Loading datasets...")
            
            for dataset_config in self.config['datasets']:
                dataset_type = dataset_config.get('type', 'custom')
                dataset_name = dataset_config.get('name', 'unknown')
                
                if dataset_type == 'benchmark':
                    # Load benchmark dataset
                    benchmark_name = dataset_config.get('benchmark_name')
                    dataset = self.dataset_loader.load_benchmark_dataset(benchmark_name)
                elif dataset_type == 'custom':
                    # Load custom dataset
                    file_path = dataset_config.get('file_path')
                    dataset = self.dataset_loader.load_custom_dataset(file_path)
                else:
                    self.logger.warning(f"Unknown dataset type: {dataset_type}")
                    continue
                
                if dataset:
                    self.datasets[dataset_name] = dataset
                    self.logger.info(f"Loaded dataset: {dataset_name}")
            
            if not self.datasets:
                raise RuntimeError("No datasets were successfully loaded")
            
            self.logger.info(f"Successfully loaded {len(self.datasets)} datasets")
            return self.datasets
            
        except Exception as e:
            self.logger.error(f"Failed to load datasets: {str(e)}")
            raise
    
    def run_evaluations(self) -> Dict[str, Any]:
        """
        Run evaluations for all models and datasets
        
        Returns:
            Dictionary of evaluation results
        """
        try:
            self.execution_state.current_step = "evaluation"
            self._save_execution_state()
            
            self.logger.info("Starting evaluations...")
            
            total_models = len(self.model_clients)
            total_evaluators = len(self.config['evaluators'])
            
            for i, (model_name, client) in enumerate(self.model_clients.items(), 1):
                # Skip if already completed (for resumption)
                if model_name in self.execution_state.completed_models:
                    self.logger.info(f"Skipping already completed model: {model_name}")
                    continue
                
                try:
                    self.logger.info(f"Evaluating model {i}/{total_models}: {model_name}")
                    
                    model_results = {}
                    model_cost = 0.0
                    
                    # Run each evaluator
                    for j, evaluator_config in enumerate(self.config['evaluators'], 1):
                        evaluator_type = evaluator_config.get('type')
                        evaluator_name = evaluator_config.get('name', evaluator_type)
                        
                        self.logger.info(f"Running evaluator {j}/{total_evaluators}: {evaluator_name}")
                        
                        try:
                            # Create evaluator
                            evaluator = self.evaluator_factory.create_evaluator(
                                evaluator_type, evaluator_config
                            )
                            
                            # Get relevant datasets for this evaluator
                            evaluator_datasets = self._get_evaluator_datasets(evaluator_config)
                            
                            # Run evaluation
                            start_time = time.time()
                            results = evaluator.evaluate(client, evaluator_datasets)
                            evaluation_time = time.time() - start_time
                            
                            # Track costs
                            evaluation_cost = self.cost_tracker.get_session_cost()
                            model_cost += evaluation_cost
                            
                            # Store results
                            model_results[evaluator_name] = {
                                **results,
                                'evaluation_time': evaluation_time,
                                'evaluation_cost': evaluation_cost
                            }
                            
                            self.logger.info(f"Completed {evaluator_name} evaluation in {evaluation_time:.2f}s")
                            
                        except Exception as e:
                            self.logger.error(f"Evaluator {evaluator_name} failed: {str(e)}")
                            model_results[evaluator_name] = {
                                'error': str(e),
                                'evaluation_time': 0,
                                'evaluation_cost': 0
                            }
                    
                    # Store model results
                    self.evaluation_results[model_name] = {
                        **model_results,
                        'total_cost': model_cost,
                        'model_info': client.get_model_info()
                    }
                    
                    # Update execution state
                    self.execution_state.completed_models.append(model_name)
                    self.execution_state.total_cost += model_cost
                    self._save_execution_state()
                    
                    self.logger.info(f"Completed evaluation for {model_name} (Cost: ${model_cost:.4f})")
                    
                except Exception as e:
                    self.logger.error(f"Model {model_name} evaluation failed: {str(e)}")
                    self.execution_state.failed_models.append(model_name)
                    self._save_execution_state()
                    
                    # Store error information
                    self.evaluation_results[model_name] = {
                        'error': str(e),
                        'total_cost': 0,
                        'model_info': {}
                    }
            
            self.logger.info(f"Evaluations completed. Total cost: ${self.execution_state.total_cost:.4f}")
            return self.evaluation_results
            
        except Exception as e:
            self.logger.error(f"Evaluation process failed: {str(e)}")
            raise
    
    def _get_evaluator_datasets(self, evaluator_config: Dict[str, Any]) -> Dict[str, Any]:
        """Get datasets relevant to a specific evaluator"""
        evaluator_datasets = {}
        
        # Get dataset filters from evaluator config
        dataset_filters = evaluator_config.get('datasets', [])
        
        if not dataset_filters:
            # If no filters specified, use all datasets
            return self.datasets
        
        for dataset_name, dataset in self.datasets.items():
            if dataset_name in dataset_filters:
                evaluator_datasets[dataset_name] = dataset
        
        return evaluator_datasets
    
    def generate_reports(self) -> Dict[str, str]:
        """
        Generate comprehensive reports
        
        Returns:
            Dictionary of generated report paths
        """
        try:
            self.execution_state.current_step = "reporting"
            self._save_execution_state()
            
            self.logger.info("Generating reports...")
            
            report_paths = {}
            
            # 1. Generate comparison report
            comparison_report = EmbeddingComparisonReport(self.output_dir)
            comparison_paths = comparison_report.generate_comparison_report(
                self.evaluation_results,
                self.config['models'],
                {
                    'execution_id': self.execution_id,
                    'total_models': len(self.model_clients),
                    'total_cost': self.execution_state.total_cost,
                    'categories': [eval_config.get('type') for eval_config in self.config['evaluators']]
                }
            )
            report_paths.update(comparison_paths)
            
            # 2. Generate visualizations
            if self.config.get('generate_visualizations', True):
                visualizations = EmbeddingVisualizations(
                    os.path.join(self.output_dir, "visualizations")
                )
                viz_paths = visualizations.generate_all_visualizations(
                    self.evaluation_results,
                    self.config['models']
                )
                report_paths['visualizations'] = viz_paths
            
            # 3. Generate dimension analysis (if embeddings are available)
            if self.config.get('generate_dimension_analysis', True):
                # This would require storing embeddings during evaluation
                # For now, we'll skip if embeddings aren't available
                self.logger.info("Dimension analysis skipped (embeddings not stored)")
            
            # 4. Generate execution summary
            summary_path = self._generate_execution_summary()
            report_paths['execution_summary'] = summary_path
            
            self.logger.info(f"Reports generated successfully: {list(report_paths.keys())}")
            return report_paths
            
        except Exception as e:
            self.logger.error(f"Report generation failed: {str(e)}")
            raise
    
    def _generate_execution_summary(self) -> str:
        """Generate execution summary report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = os.path.join(self.output_dir, f"execution_summary_{timestamp}.json")
        
        summary = {
            'execution_info': asdict(self.execution_state),
            'configuration': self.config,
            'results_summary': {
                'total_models_evaluated': len(self.evaluation_results),
                'successful_models': len(self.execution_state.completed_models),
                'failed_models': len(self.execution_state.failed_models),
                'total_cost': self.execution_state.total_cost
            },
            'model_performance_summary': self._summarize_model_performance()
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary_path
    
    def _summarize_model_performance(self) -> Dict[str, Any]:
        """Summarize model performance across all evaluations"""
        performance_summary = {}
        
        for model_name, results in self.evaluation_results.items():
            if 'error' in results:
                performance_summary[model_name] = {'status': 'failed', 'error': results['error']}
                continue
            
            # Calculate average performance across evaluators
            evaluator_scores = []
            for evaluator_name, evaluator_results in results.items():
                if evaluator_name in ['total_cost', 'model_info']:
                    continue
                
                if isinstance(evaluator_results, dict) and 'overall_score' in evaluator_results:
                    evaluator_scores.append(evaluator_results['overall_score'])
            
            avg_score = sum(evaluator_scores) / len(evaluator_scores) if evaluator_scores else 0
            
            performance_summary[model_name] = {
                'status': 'completed',
                'average_score': avg_score,
                'total_cost': results.get('total_cost', 0),
                'evaluator_count': len(evaluator_scores)
            }
        
        return performance_summary
    
    def run_full_evaluation(self, resume_execution_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Run the complete evaluation pipeline
        
        Args:
            resume_execution_id: Optional execution ID to resume from
            
        Returns:
            Dictionary containing all results and report paths
        """
        try:
            start_time = time.time()
            
            # Load previous state if resuming
            if resume_execution_id:
                if self.load_execution_state(resume_execution_id):
                    self.logger.info(f"Resuming execution: {resume_execution_id}")
                else:
                    self.logger.warning("Failed to load execution state, starting fresh")
                    resume_execution_id = None
            
            # Initialize if not resuming
            if not resume_execution_id:
                self.initialize_execution_state()
            
            # Load configuration
            self.load_configuration()
            
            # Setup model clients
            if not self.model_clients:
                self.setup_model_clients()
            
            # Load datasets
            if not self.datasets:
                self.load_datasets()
            
            # Run evaluations
            self.run_evaluations()
            
            # Generate reports
            report_paths = self.generate_reports()
            
            # Calculate total execution time
            total_time = time.time() - start_time
            
            # Final execution state update
            self.execution_state.current_step = "completed"
            self._save_execution_state()
            
            # Prepare final results
            final_results = {
                'execution_id': self.execution_id,
                'execution_time': total_time,
                'total_cost': self.execution_state.total_cost,
                'evaluation_results': self.evaluation_results,
                'report_paths': report_paths,
                'execution_state': asdict(self.execution_state)
            }
            
            self.logger.info(f"Full evaluation completed in {total_time:.2f}s")
            self.logger.info(f"Total cost: ${self.execution_state.total_cost:.4f}")
            self.logger.info(f"Reports available at: {self.output_dir}")
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"Full evaluation failed: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise

def main():
    """Main entry point for command-line execution"""
    parser = argparse.ArgumentParser(description="Embedding Model Testing Framework")
    parser.add_argument(
        '--config', '-c',
        default='config/test_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--resume', '-r',
        help='Resume execution from given execution ID'
    )
    parser.add_argument(
        '--log-level', '-l',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    parser.add_argument(
        '--output-dir', '-o',
        help='Override output directory'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Create executor
        executor = EmbeddingExecutor(args.config)
        
        # Override output directory if specified
        if args.output_dir:
            executor.output_dir = args.output_dir
            os.makedirs(executor.output_dir, exist_ok=True)
        
        # Run evaluation
        results = executor.run_full_evaluation(args.resume)
        
        print(f"\n{'='*60}")
        print("EVALUATION COMPLETED SUCCESSFULLY")
        print(f"{'='*60}")
        print(f"Execution ID: {results['execution_id']}")
        print(f"Total Time: {results['execution_time']:.2f} seconds")
        print(f"Total Cost: ${results['total_cost']:.4f}")
        print(f"Models Evaluated: {len(results['evaluation_results'])}")
        print(f"Reports Directory: {executor.output_dir}")
        print(f"{'='*60}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\nExecution interrupted by user")
        return 1
    except Exception as e:
        print(f"\nExecution failed: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
