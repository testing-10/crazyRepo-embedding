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
from utils.embedding_logger import get_logger
from utils.embedding_file_utils import FileUtils
from utils.embedding_cost_tracker import CostTracker
from datasets.dataset_loader import DatasetRegistry, load_dataset
from evaluators.evaluator_factory import EvaluatorFactory
# Fixed imports - using correct class names
from reporting.embedding_comparison_report import EmbeddingComparisonReport
from reporting.embedding_visualizations import EmbeddingVisualizations
from reporting.dimension_analysis import DimensionAnalyzer

def load_environment():
    """Load environment variables from .env file."""
    import os
    from pathlib import Path
    
    env_file = Path(".env")
    if env_file.exists():
        with open(env_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    os.environ[key] = value
        print("✅ Environment variables loaded from .env file")
    else:
        print("⚠️  .env file not found")

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

    def __init__(self, config_path: str = "configs/embedding_test_config.yaml"):
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
        self.logger = get_logger(__name__)
        self.file_utils = FileUtils()
        self.cost_tracker = CostTracker()

        # Initialize components - Fixed: removed cost_tracker parameter
        self.client_factory = EmbeddingClientFactory()
        self.dataset_registry = DatasetRegistry()
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

        # Handle different model configuration structures
        if 'enabled_models' in self.config['models']:
            # Your structure - enabled_models is a list
            enabled_models = self.config['models']['enabled_models']
            if not enabled_models:
                raise ValueError("No models enabled in configuration")

            # Check if model configs exist for enabled models
            model_configs_path = self.config['models'].get('model_configs_path', './configs/models/')

            for model_name in enabled_models:
                model_config_file = Path(model_configs_path) / f"{model_name}.yaml"
                if model_config_file.exists():
                    self.logger.info(f"Model '{model_name}' config expected in external file at {model_configs_path}")
                else:
                    self.logger.warning(f"Model config file not found: {model_config_file}")

        # Validate evaluators section - handle your config structure
        evaluators_config = self.config.get('evaluators', {})
        if 'metrics' in evaluators_config:
            # Your structure - evaluators.metrics with enabled flags
            enabled_evaluators = []
            for eval_type, eval_config in evaluators_config['metrics'].items():
                if eval_config.get('enabled', False):
                    enabled_evaluators.append(eval_type)

            if not enabled_evaluators:
                raise ValueError("No evaluators enabled in configuration")

            self.logger.info(f"Enabled evaluators: {enabled_evaluators}")
        else:
            # Standard structure - list of evaluator configs
            if isinstance(self.config['evaluators'], list):
                if len(self.config['evaluators']) == 0:
                    raise ValueError("No evaluators specified in configuration")

    def initialize_execution_state(self) -> None:
        """Initialize execution state for tracking and resumption"""
        # Use a simple hash instead of file hash for now
        config_hash = str(hash(str(self.config)))

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
        """Setup embedding model clients based on configuration"""
        self.logger.info("Setting up model clients...")

        try:
            models_config = self.config['models']

            # Handle your config structure with enabled_models
            if 'enabled_models' in models_config:
                enabled_models = models_config['enabled_models']
                model_configs_path = models_config.get('model_configs_path', './configs/models/')

                for model_name in enabled_models:
                    try:
                        # First check if model config is inline
                        if model_name in models_config and isinstance(models_config[model_name], dict):
                            model_config = models_config[model_name]
                        else:
                            # Load from external config file
                            config_file_path = os.path.join(model_configs_path, f"{model_name}.yaml")
                            if os.path.exists(config_file_path):
                                with open(config_file_path, 'r') as f:
                                    model_config = yaml.safe_load(f)
                            else:
                                self.logger.warning(f"Config file not found for model: {model_name} at {config_file_path}")
                                continue

                        # Create client using factory
                        provider = model_config.get('provider')
                        if not provider:
                            self.logger.error(f"No provider specified for model: {model_name}")
                            continue

                        # Create client - pass model_config as second positional argument
                        client = self.client_factory.create_client(
                            provider=provider,
                            model_config=model_config,  # Pass as positional argument
                            cost_tracker=self.cost_tracker
                        )

                        if client:
                            # Test client initialization
                            try:
                                client.ensure_initialized()
                                self.logger.info(f"Model {model_name} initialized successfully")

                                # Test with a simple embedding
                                test_result = client.embed_single("test")
                                self.logger.info(f"Test embedding successful: {len(test_result.embeddings[0])} dimensions")

                            except Exception as e:
                                self.logger.error(f"Failed to initialize model {model_name}: {e}")
                                continue

                            self.model_clients[model_name] = client
                            self.logger.info(f"Successfully created client for model: {model_name}")
                        else:
                            self.logger.error(f"Failed to create client for model: {model_name}")

                    except Exception as e:
                        self.logger.error(f"Error setting up client for model {model_name}: {e}")
                        if self.config.get('execution', {}).get('error_handling', {}).get('fail_fast', False):
                            raise
                        continue
            else:
                # Handle standard config structure
                for model_name, model_config in models_config.items():
                    # Skip non-dict entries (like load_all_at_startup, model_configs_path, etc.)
                    if not isinstance(model_config, dict):
                        continue

                    try:
                        # Check if config_file is specified
                        if 'config_file' in model_config:
                            config_file_path = model_config['config_file']
                            with open(config_file_path, 'r') as f:
                                external_config = yaml.safe_load(f)
                            model_config.update(external_config)

                        provider = model_config.get('provider')
                        if not provider:
                            self.logger.error(f"No provider specified for model: {model_name}")
                            continue

                        client = self.client_factory.create_client(provider, model_config)
                        if client:
                            self.model_clients[model_name] = client
                            self.logger.info(f"Successfully created client for model: {model_name}")
                        else:
                            self.logger.error(f"Failed to create client for model: {model_name}")

                    except Exception as e:
                        self.logger.error(f"Error setting up client for model {model_name}: {e}")
                        if self.config.get('execution', {}).get('error_handling', {}).get('fail_fast', False):
                            raise
                        continue

            if not self.model_clients:
                raise ValueError("No model clients were successfully created")

            self.logger.info(f"Successfully set up {len(self.model_clients)} model clients")
            return self.model_clients

        except Exception as e:
            self.logger.error(f"Failed to setup model clients: {e}")
            raise

    def load_datasets(self):
        """Load all configured datasets."""
        try:
            self.logger.info("Loading datasets...")

            # Get the enabled datasets list from config
            datasets_config = self.config.get('datasets', {})
            enabled_datasets = datasets_config.get('enabled_datasets', [])

            if not enabled_datasets:
                self.logger.warning("No enabled datasets found in configuration")
                return

            self.logger.info(f"Found {len(enabled_datasets)} enabled datasets: {enabled_datasets}")

            # Load each enabled dataset
            for dataset_name in enabled_datasets:
                try:
                    self.logger.info(f"Loading dataset: {dataset_name}")

                    # Use the standalone load_dataset function
                    dataset = load_dataset(dataset_name)

                    if dataset and len(dataset) > 0:
                        self.datasets[dataset_name] = dataset
                        self.logger.info(f"Successfully loaded dataset: {dataset_name} ({len(dataset)} samples)")
                    else:
                        self.logger.warning(f"No samples loaded for dataset: {dataset_name}")

                except Exception as e:
                    self.logger.error(f"Error loading dataset {dataset_name}: {e}")
                    continue

            if not self.datasets:
                raise ValueError("No datasets were successfully loaded")

            self.logger.info(f"Successfully loaded {len(self.datasets)} datasets")

        except Exception as e:
            self.logger.error(f"Failed to load datasets: {e}")
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

            if not self.model_clients:
                self.logger.warning("No model clients available for evaluation")
                return {}

            if not self.datasets:
                self.logger.warning("No datasets available for evaluation")
                return {}

            # Get enabled evaluators from your config structure
            evaluators_config = self.config.get('evaluators', {}).get('metrics', {})
            enabled_evaluators = []

            for eval_type, eval_config in evaluators_config.items():
                if eval_config.get('enabled', False):
                    enabled_evaluators.append(eval_type)

            if not enabled_evaluators:
                self.logger.warning("No evaluators enabled")
                return {}

            total_evaluations = len(self.model_clients) * len(self.datasets) * len(enabled_evaluators)
            self.logger.info(f"Running {total_evaluations} evaluations...")

            evaluation_count = 0

            # Run evaluations for each model-dataset-evaluator combination
            for model_name, client in self.model_clients.items():
                # Skip if already completed (for resumption)
                if model_name in self.execution_state.completed_models:
                    self.logger.info(f"Skipping already completed model: {model_name}")
                    continue

                try:
                    self.logger.info(f"Evaluating model: {model_name}")

                    self.evaluation_results[model_name] = {}
                    model_cost = 0.0

                    for dataset_name, dataset in self.datasets.items():
                        self.evaluation_results[model_name][dataset_name] = {}

                        for evaluator_type in enabled_evaluators:
                            try:
                                evaluation_count += 1
                                self.logger.info(f"[{evaluation_count}/{total_evaluations}] Evaluating {model_name} on {dataset_name} with {evaluator_type}")

                                # Get evaluator
                                evaluator = self.evaluator_factory.create_evaluator(evaluator_type)

                                # Generate embeddings for the dataset
                                self.logger.info(f"Generating embeddings for {len(dataset)} samples...")
                                
                                # Extract texts from dataset samples
                                texts = []
                                for sample in dataset:
                                    if hasattr(sample, 'text1') and sample.text1:
                                        texts.append(sample.text1)
                                    elif hasattr(sample, 'text') and sample.text:
                                        texts.append(sample.text)
                                    else:
                                        texts.append(str(sample))  # Fallback
                                
                                # Generate embeddings using the client
                                start_time = time.time()
                                embedding_result = client.embed_texts(texts)
                                embeddings = embedding_result.embeddings
                                
                                # Convert to list of NumPy arrays for evaluators
                                import numpy as np
                                embeddings = [np.array(emb) for emb in embeddings]
                                
                                # Run evaluation - NEW (CORRECT)
                                result = evaluator.evaluate(
                                    embeddings=embeddings,  # List[np.ndarray]
                                    samples=dataset,        # List[DatasetSample]
                                    model_name=model_name,
                                    dataset_name=dataset_name
                                )
                                
                                evaluation_time = time.time() - start_time

                                # Track costs - FIXED
                                cost_before = self.cost_tracker.get_session_cost()
                                # Cost is already tracked by the embed_texts call above
                                cost_after = self.cost_tracker.get_session_cost()
                                evaluation_cost = cost_after - cost_before
                                model_cost += evaluation_cost

                                # Store result
                                self.evaluation_results[model_name][dataset_name][evaluator_type] = {
                                    'result': result,
                                    'evaluation_time': evaluation_time,
                                    'evaluation_cost': evaluation_cost,
                                    'timestamp': datetime.now().isoformat()
                                }

                                self.logger.info(f"Completed {evaluator_type} evaluation in {evaluation_time:.2f}s")

                            except Exception as e:
                                self.logger.error(f"Error in evaluation {model_name}/{dataset_name}/{evaluator_type}: {e}")
                                self.evaluation_results[model_name][dataset_name][evaluator_type] = {
                                    'error': str(e),
                                    'timestamp': datetime.now().isoformat()
                                }
                                continue

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

            reports_generated = {}

            # Ensure reports directory exists
            reports_dir = Path(self.output_dir)
            reports_dir.mkdir(parents=True, exist_ok=True)

            # Generate comparison report - Fixed: using correct class name
            report_generator = EmbeddingComparisonReport(
                output_dir=str(reports_dir)
            )

            # Prepare model configs for report
            model_configs = {}
            for model_name, client in self.model_clients.items():
                try:
                    model_configs[model_name] = client.get_model_info()
                except:
                    model_configs[model_name] = {'model_name': model_name}

            # Prepare test metadata
            test_metadata = {
                'execution_id': self.execution_id,
                'categories': list(self.config.get('evaluators', {}).get('metrics', {}).keys()),
                'total_test_cases': sum(len(dataset) for dataset in self.datasets.values()),
                'models_count': len(self.model_clients),
                'datasets_count': len(self.datasets)
            }

            comparison_reports = report_generator.generate_comparison_report(
                evaluation_results=self.evaluation_results,
                model_configs=model_configs,
                test_metadata=test_metadata
            )
            reports_generated.update(comparison_reports)

            # Generate visualizations - Fixed: using correct class name
            viz_generator = EmbeddingVisualizations(
                output_dir=str(reports_dir / "visualizations")
            )

            viz_reports = viz_generator.generate_all_visualizations(
                evaluation_results=self.evaluation_results,
                model_configs=model_configs
            )
            reports_generated['visualizations'] = viz_reports

            # Generate dimension analysis if embeddings are available
            try:
                dimension_analyzer = DimensionAnalyzer(
                    output_dir=str(reports_dir / "dimension_analysis")
                )
                self.logger.info("Dimension analysis available but requires stored embeddings")
                reports_generated['dimension_analysis'] = "available"
            except Exception as e:
                self.logger.info(f"Dimension analysis skipped: {str(e)}")

            # Generate execution summary
            self._generate_execution_summary(reports_dir)
            reports_generated['execution_summary'] = str(reports_dir / f"execution_summary_{self.execution_id}.json")

            return reports_generated

        except Exception as e:
            self.logger.error(f"Error generating reports: {e}")
            return {}

    def _generate_execution_summary(self, reports_dir: Path):
        """Generate a summary of the execution."""
        summary = {
            'execution_id': self.execution_id,
            'timestamp': datetime.now().isoformat(),
            'models_evaluated': list(self.model_clients.keys()),
            'datasets_used': list(self.datasets.keys()),
            'total_cost': self.execution_state.total_cost if self.execution_state else 0.0,
            'evaluation_results_summary': {
                model: {
                    dataset: list(evals.keys()) 
                    for dataset, evals in datasets.items()
                }
                for model, datasets in self.evaluation_results.items()
                if isinstance(datasets, dict)
            }
        }

        summary_file = reports_dir / f"execution_summary_{self.execution_id}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        self.logger.info(f"Execution summary saved: {summary_file}")

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
            if self.execution_state:
                self.execution_state.current_step = "completed"
                self._save_execution_state()

            # Prepare final results
            final_results = {
                'execution_id': self.execution_id,
                'execution_time': total_time,
                'total_cost': self.execution_state.total_cost if self.execution_state else 0.0,
                'evaluation_results': self.evaluation_results,
                'report_paths': report_paths,
                'execution_state': asdict(self.execution_state) if self.execution_state else {}
            }

            self.logger.info(f"Reports generated successfully: {report_paths}")
            self.logger.info(f"Full evaluation completed in {total_time:.2f}s")
            self.logger.info(f"Total cost: ${final_results['total_cost']:.4f}")
            self.logger.info(f"Reports available at: {self.output_dir}")

            return final_results

        except Exception as e:
            self.logger.error(f"Full evaluation failed: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise


def main():
    """Main entry point for command-line execution"""
    load_environment()

    parser = argparse.ArgumentParser(description="Embedding Model Testing Framework")
    parser.add_argument(
        '--config', '-c',
        default='configs/embedding_test_config.yaml',
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
