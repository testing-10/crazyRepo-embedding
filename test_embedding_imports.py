"""
Import Validation Tests

Tests to validate that all framework components can be imported correctly
and that dependencies are properly installed.
"""

import pytest
import sys
import importlib
from typing import List, Tuple

class TestEmbeddingImports:
    """Test suite for validating framework imports"""
    
    def test_core_dependencies(self):
        """Test that all core dependencies are available"""
        core_deps = [
            'numpy',
            'pandas', 
            'sklearn',
            'scipy',
            'yaml',
            'json',
            'os',
            'logging',
            'datetime',
            'typing'
        ]
        
        missing_deps = []
        for dep in core_deps:
            try:
                importlib.import_module(dep)
            except ImportError:
                missing_deps.append(dep)
        
        assert not missing_deps, f"Missing core dependencies: {missing_deps}"
    
    def test_ml_dependencies(self):
        """Test machine learning and NLP dependencies"""
        ml_deps = [
            'sentence_transformers',
            'transformers',
            'torch',
            'tokenizers'
        ]
        
        missing_deps = []
        for dep in ml_deps:
            try:
                importlib.import_module(dep)
            except ImportError:
                missing_deps.append(dep)
        
        # Allow some ML dependencies to be missing in test environments
        if missing_deps:
            pytest.skip(f"ML dependencies not available: {missing_deps}")
    
    def test_api_client_dependencies(self):
        """Test API client dependencies"""
        api_deps = [
            'openai',
            'cohere', 
            'requests',
            'httpx'
        ]
        
        missing_deps = []
        for dep in api_deps:
            try:
                importlib.import_module(dep)
            except ImportError:
                missing_deps.append(dep)
        
        # Allow API dependencies to be missing in test environments
        if missing_deps:
            pytest.skip(f"API client dependencies not available: {missing_deps}")
    
    def test_visualization_dependencies(self):
        """Test visualization dependencies"""
        viz_deps = [
            'matplotlib',
            'seaborn',
            'plotly'
        ]
        
        missing_deps = []
        for dep in viz_deps:
            try:
                importlib.import_module(dep)
            except ImportError:
                missing_deps.append(dep)
        
        assert not missing_deps, f"Missing visualization dependencies: {missing_deps}"
    
    def test_utility_imports(self):
        """Test utility module imports"""
        try:
            from utils.embedding_logger import EmbeddingLogger
            from utils.embedding_file_utils import FileUtils
            from utils.embedding_cost_tracker import CostTracker
            from utils.vector_operations import VectorOperations
        except ImportError as e:
            pytest.fail(f"Failed to import utility modules: {e}")
    
    def test_client_imports(self):
        """Test client module imports"""
        try:
            from clients.base_embedding_client import BaseEmbeddingClient
            from clients.openai_embedding_client import OpenAIEmbeddingClient
            from clients.cohere_embedding_client import CohereEmbeddingClient
            from clients.sentence_transformer_client import SentenceTransformersClient
            from clients.huggingface_embedding_client import HuggingFaceEmbeddingClient
            from clients.jina_embedding_client import JinaEmbeddingClient
            from clients.azure_embedding_client import AzureEmbeddingClient
        except ImportError as e:
            pytest.fail(f"Failed to import client modules: {e}")
    
    def test_evaluator_imports(self):
        """Test evaluator module imports"""
        try:
            from evaluators.base_embedding_evaluator import BaseEmbeddingEvaluator
            from evaluators.semantic_similarity_evaluator import SemanticSimilarityEvaluator
            from evaluators.retrieval_evaluator import RetrievalEvaluator
            from evaluators.clustering_evaluator import ClusteringEvaluator
            from evaluators.classification_evaluator import ClassificationEvaluator
            from evaluators.efficiency_evaluator import EfficiencyEvaluator
            from evaluators.robustness_evaluator import RobustnessEvaluator
            from evaluators.evaluator_factory import EvaluatorFactory
        except ImportError as e:
            pytest.fail(f"Failed to import evaluator modules: {e}")
    
    def test_metrics_imports(self):
        """Test metrics module imports"""
        try:
            from metrics.similarity_metrics import SimilarityMetrics
            from metrics.retrieval_metrics import RetrievalMetrics
            from metrics.clustering_metrics import ClusteringMetrics
            from metrics.classification_metrics import ClassificationMetrics
            from metrics.efficiency_metrics import EfficiencyMetrics
        except ImportError as e:
            pytest.fail(f"Failed to import metrics modules: {e}")
    
    def test_dataset_imports(self):
        """Test dataset module imports"""
        try:
            from datasets.dataset_loader import BaseDatasetLoader
            from datasets.dataset_loader import BenchmarkDatasetLoader
            from datasets.dataset_loader import CustomDatasetLoader
            from datasets.dataset_loader import DatasetLoader
        except ImportError as e:
            pytest.fail(f"Failed to import dataset modules: {e}")
    
    def test_reporting_imports(self):
        """Test reporting module imports"""
        try:
            from reporting.embedding_comparison_report import EmbeddingComparisonReport
            from reporting.embedding_visualizations import EmbeddingVisualizations
            from reporting.dimension_analysis import DimensionAnalysis
        except ImportError as e:
            pytest.fail(f"Failed to import reporting modules: {e}")
    
    def test_main_components_imports(self):
        """Test main framework component imports"""
        try:
            from embedding_client_factory import EmbeddingClientFactory
            from embedding_executor import EmbeddingExecutor
        except ImportError as e:
            pytest.fail(f"Failed to import main components: {e}")
    
    def test_framework_instantiation(self):
        """Test that main framework components can be instantiated"""
        try:
            # Test factory instantiation
            from embedding_client_factory import EmbeddingClientFactory
            factory = EmbeddingClientFactory()
            assert factory is not None
            
            # Test executor instantiation (with dummy config)
            from embedding_executor import EmbeddingExecutor
            # This might fail if config file doesn't exist, which is expected
            try:
                executor = EmbeddingExecutor("dummy_config.yaml")
            except FileNotFoundError:
                # Expected if config file doesn't exist
                pass
            
        except Exception as e:
            pytest.fail(f"Failed to instantiate framework components: {e}")
    
    def test_environment_validation(self):
        """Test environment validation functionality"""
        try:
            from embedding_client_factory import EmbeddingClientFactory
            
            factory = EmbeddingClientFactory()
            validation_results = factory.validate_environment()
            
            # Should return a dictionary with validation results
            assert isinstance(validation_results, dict)
            assert 'valid' in validation_results
            assert 'missing_dependencies' in validation_results
            assert 'missing_api_keys' in validation_results
            assert 'available_providers' in validation_results
            
        except Exception as e:
            pytest.fail(f"Environment validation failed: {e}")
    
    def test_logger_functionality(self):
        """Test that logging functionality works"""
        try:
            from utils.embedding_logger import EmbeddingLogger
            
            logger = EmbeddingLogger.get_logger("test_logger")
            assert logger is not None
            
            # Test logging methods exist
            assert hasattr(logger, 'info')
            assert hasattr(logger, 'error')
            assert hasattr(logger, 'warning')
            assert hasattr(logger, 'debug')
            
            # Test logging doesn't crash
            logger.info("Test log message")
            
        except Exception as e:
            pytest.fail(f"Logger functionality test failed: {e}")
    
    def test_file_utils_functionality(self):
        """Test file utilities functionality"""
        try:
            from utils.embedding_file_utils import EmbeddingFileUtils
            
            file_utils = EmbeddingFileUtils()
            assert file_utils is not None
            
            # Test that methods exist
            assert hasattr(file_utils, 'ensure_directory')
            assert hasattr(file_utils, 'save_json')
            assert hasattr(file_utils, 'load_json')
            assert hasattr(file_utils, 'calculate_file_hash')
            
        except Exception as e:
            pytest.fail(f"File utils functionality test failed: {e}")

class TestOptionalDependencies:
    """Test optional dependencies that might not be available in all environments"""
    
    def test_gpu_dependencies(self):
        """Test GPU-related dependencies (optional)"""
        gpu_deps = ['torch', 'transformers']
        
        available_gpu_deps = []
        for dep in gpu_deps:
            try:
                module = importlib.import_module(dep)
                available_gpu_deps.append(dep)
                
                # Test GPU availability if torch is available
                if dep == 'torch':
                    import torch
                    gpu_available = torch.cuda.is_available()
                    print(f"GPU available: {gpu_available}")
                    
            except ImportError:
                continue
        
        print(f"Available GPU dependencies: {available_gpu_deps}")
    
    def test_advanced_nlp_dependencies(self):
        """Test advanced NLP dependencies (optional)"""
        nlp_deps = ['spacy', 'nltk']
        
        available_nlp_deps = []
        for dep in nlp_deps:
            try:
                importlib.import_module(dep)
                available_nlp_deps.append(dep)
            except ImportError:
                continue
        
        print(f"Available advanced NLP dependencies: {available_nlp_deps}")

def run_import_validation():
    """
    Standalone function to run import validation
    Can be called directly without pytest
    """
    print("Running import validation...")
    
    # Test core imports
    try:
        test_imports = TestEmbeddingImports()
        
        print("✓ Testing core dependencies...")
        test_imports.test_core_dependencies()
        
        print("✓ Testing utility imports...")
        test_imports.test_utility_imports()
        
        print("✓ Testing client imports...")
        test_imports.test_client_imports()
        
        print("✓ Testing evaluator imports...")
        test_imports.test_evaluator_imports()
        
        print("✓ Testing metrics imports...")
        test_imports.test_metrics_imports()
        
        print("✓ Testing dataset imports...")
        test_imports.test_dataset_imports()
        
        print("✓ Testing reporting imports...")
        test_imports.test_reporting_imports()
        
        print("✓ Testing main component imports...")
        test_imports.test_main_components_imports()
        
        print("✓ Testing framework instantiation...")
        test_imports.test_framework_instantiation()
        
        print("✓ Testing environment validation...")
        test_imports.test_environment_validation()
        
        print("✓ Testing logger functionality...")
        test_imports.test_logger_functionality()
        
        print("✓ Testing file utils functionality...")
        test_imports.test_file_utils_functionality()
        
        print("\n✅ All import validations passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Import validation failed: {e}")
        return False

if __name__ == "__main__":
    # Run validation when script is executed directly
    success = run_import_validation()
    sys.exit(0 if success else 1)
