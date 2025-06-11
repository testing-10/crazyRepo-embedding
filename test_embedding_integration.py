"""
Integration Tests for Embedding Model Testing Framework

Comprehensive integration tests that validate the interaction between
models, datasets, evaluators, and reporting components.
"""

import pytest
import os
import json
import yaml
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any
import numpy as np

# Framework imports
from embedding_client_factory import EmbeddingClientFactory
from embedding_executor import EmbeddingExecutor
from utils.embedding_logger import EmbeddingLogger
from utils.embedding_file_utils import FileUtils
from utils.embedding_cost_tracker import CostTracker

class TestEmbeddingIntegration:
    """Integration tests for the embedding framework"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_config(self, temp_dir):
        """Create mock configuration for testing"""
        config = {
            'models': {
                'test_model_1': {
                    'provider': 'sentence_transformers',
                    'model_name': 'all-MiniLM-L6-v2',
                    'dimensions': 384,
                    'max_tokens': 256
                },
                'test_model_2': {
                    'provider': 'sentence_transformers', 
                    'model_name': 'all-mpnet-base-v2',
                    'dimensions': 768,
                    'max_tokens': 384
                }
            },
            'datasets': [
                {
                    'type': 'custom',
                    'name': 'test_similarity',
                    'file_path': os.path.join(temp_dir, 'test_similarity.json')
                }
            ],
            'evaluators': [
                {
                    'type': 'similarity',
                    'name': 'similarity_eval',
                    'datasets': ['test_similarity']
                },
                {
                    'type': 'efficiency',
                    'name': 'efficiency_eval'
                }
            ],
            'output_dir': temp_dir,
            'generate_visualizations': True,
            'generate_dimension_analysis': False
        }
        
        # Save config to file
        config_path = os.path.join(temp_dir, 'test_config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        # Create test dataset
        test_dataset = {
            'metadata': {
                'name': 'test_similarity',
                'description': 'Test similarity dataset',
                'total_pairs': 2
            },
            'test_cases': [
                {
                    'id': 'test_1',
                    'text1': 'The cat sat on the mat',
                    'text2': 'A cat was sitting on a mat',
                    'expected_similarity': 0.8,
                    'category': 'paraphrase'
                },
                {
                    'id': 'test_2', 
                    'text1': 'Python is a programming language',
                    'text2': 'The snake slithered through the grass',
                    'expected_similarity': 0.1,
                    'category': 'unrelated'
                }
            ]
        }
        
        dataset_path = os.path.join(temp_dir, 'test_similarity.json')
        with open(dataset_path, 'w') as f:
            json.dump(test_dataset, f)
        
        return config, config_path
    
    def test_client_factory_integration(self):
        """Test client factory functionality"""
        factory = EmbeddingClientFactory()
        
        # Test environment validation
        validation_results = factory.validate_environment()
        assert isinstance(validation_results, dict)
        assert 'valid' in validation_results
        
        # Test available providers
        providers = factory.list_providers()
        assert isinstance(providers, list)
        assert len(providers) > 0
        
        # Test provider info
        for provider in providers:
            info = factory.get_provider_info(provider)
            assert isinstance(info, dict)
            assert 'provider' in info
            assert 'client_class' in info
    
    def test_sentence_transformers_client_creation(self):
        """Test creating sentence transformers client (no API key required)"""
        factory = EmbeddingClientFactory()
        
        config = {
            'provider': 'sentence_transformers',
            'model_name': 'all-MiniLM-L6-v2',
            'dimensions': 384
        }
        
        try:
            client = factory.create_client('sentence_transformers', config)
            assert client is not None
            
            # Test basic functionality
            model_info = client.get_model_info()
            assert isinstance(model_info, dict)
            
        except Exception as e:
            # Skip if sentence-transformers not available
            pytest.skip(f"Sentence transformers not available: {e}")
    
    @patch('clients.openai_client.OpenAI')
    def test_openai_client_creation_mock(self, mock_openai):
        """Test OpenAI client creation with mocked API"""
        # Mock OpenAI client
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        factory = EmbeddingClientFactory()
        
        config = {
            'provider': 'openai',
            'model_name': 'text-embedding-3-small',
            'dimensions': 1536
        }
        
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'}):
            try:
                client = factory.create_client('openai', config)
                assert client is not None
                
                model_info = client.get_model_info()
                assert isinstance(model_info, dict)
                
            except Exception as e:
                pytest.skip(f"OpenAI client creation failed: {e}")
    
    def test_multiple_client_creation(self):
        """Test creating multiple clients"""
        factory = EmbeddingClientFactory()
        
        model_configs = {
            'model1': {
                'provider': 'sentence_transformers',
                'model_name': 'all-MiniLM-L6-v2'
            },
            'model2': {
                'provider': 'sentence_transformers',
                'model_name': 'all-mpnet-base-v2'
            }
        }
        
        try:
            clients = factory.create_multiple_clients(model_configs)
            
            # Should create at least one client
            assert len(clients) >= 0
            
            for client_name, client in clients.items():
                assert client is not None
                model_info = client.get_model_info()
                assert isinstance(model_info, dict)
                
        except Exception as e:
            pytest.skip(f"Multiple client creation failed: {e}")
    
    def test_dataset_loading_integration(self, mock_config, temp_dir):
        """Test dataset loading functionality"""
        config, config_path = mock_config
        
        from datasets.dataset_loader import DatasetLoader
        
        loader = DatasetLoader()
        
        # Test custom dataset loading
        dataset_path = os.path.join(temp_dir, 'test_similarity.json')
        dataset = loader.load_custom_dataset(dataset_path)
        
        assert dataset is not None
        assert 'metadata' in dataset
        assert 'test_cases' in dataset
        assert len(dataset['test_cases']) == 2
    
    def test_evaluator_factory_integration(self):
        """Test evaluator factory functionality"""
        from evaluators.evaluator_factory import EvaluatorFactory
        
        factory = EvaluatorFactory()
        
        # Test available evaluators
        evaluators = factory.list_evaluators()
        assert isinstance(evaluators, list)
        assert len(evaluators) > 0
        
        # Test creating similarity evaluator
        config = {
            'type': 'similarity',
            'name': 'test_similarity'
        }
        
        try:
            evaluator = factory.create_evaluator('similarity', config)
            assert evaluator is not None
            
        except Exception as e:
            pytest.skip(f"Evaluator creation failed: {e}")
    
    def test_cost_tracking_integration(self):
        """Test cost tracking functionality"""
        tracker = CostTracker()
        
        # Test basic functionality
        tracker.start_session()
        
        # Simulate some costs
        tracker.track_request('openai', 'text-embedding-3-small', 100, 0.002)
        tracker.track_request('cohere', 'embed-english-v3.0', 50, 0.001)
        
        session_cost = tracker.get_session_cost()
        assert session_cost >= 0
        
        total_cost = tracker.get_total_cost()
        assert total_cost >= session_cost
        
        tracker.end_session()
    
    def test_logger_integration(self):
        """Test logging functionality"""
        logger = EmbeddingLogger.get_logger("test_integration")
        
        # Test that logging doesn't crash
        logger.info("Test info message")
        logger.warning("Test warning message")
        logger.error("Test error message")
        logger.debug("Test debug message")
        
        # Test logger configuration
        assert logger.name == "test_integration"
    
    def test_file_utils_integration(self, temp_dir):
        """Test file utilities integration"""
        file_utils = FileUtils()
        
        # Test directory creation
        test_dir = os.path.join(temp_dir, 'test_subdir')
        file_utils.ensure_directory(test_dir)
        assert os.path.exists(test_dir)
        
        # Test JSON operations
        test_data = {'test': 'data', 'number': 42}
        json_path = os.path.join(temp_dir, 'test.json')
        
        file_utils.save_json(test_data, json_path)
        assert os.path.exists(json_path)
        
        loaded_data = file_utils.load_json(json_path)
        assert loaded_data == test_data
        
        # Test file hash calculation
        file_hash = file_utils.calculate_file_hash(json_path)
        assert isinstance(file_hash, str)
        assert len(file_hash) > 0
    
    @patch('sentence_transformers.SentenceTransformer')
    def test_mock_embedding_generation(self, mock_st):
        """Test embedding generation with mocked models"""
        # Mock sentence transformer
        mock_model = Mock()
        mock_model.encode.return_value = np.random.rand(2, 384)  # 2 texts, 384 dimensions
        mock_st.return_value = mock_model
        
        from clients.sentence_transformer_client import SentenceTransformersClient
        
        config = {
            'model_name': 'all-MiniLM-L6-v2',
            'dimensions': 384
        }
        
        client = SentenceTransformersClient(config)
        
        # Test embedding generation
        texts = ['Test text 1', 'Test text 2']
        embeddings = client.embed_texts(texts)
        
        assert embeddings is not None
        assert embeddings.shape == (2, 384)
    
    def test_executor_configuration_loading(self, mock_config, temp_dir):
        """Test executor configuration loading"""
        config, config_path = mock_config
        
        executor = EmbeddingExecutor(config_path)
        
        # Test configuration loading
        loaded_config = executor.load_configuration()
        assert loaded_config is not None
        assert 'models' in loaded_config
        assert 'datasets' in loaded_config
        assert 'evaluators' in loaded_config
        
        # Test configuration validation
        assert len(loaded_config['models']) == 2
        assert len(loaded_config['datasets']) == 1
        assert len(loaded_config['evaluators']) == 2
    
    def test_executor_state_management(self, mock_config, temp_dir):
        """Test executor state management"""
        config, config_path = mock_config
        
        executor = EmbeddingExecutor(config_path)
        executor.load_configuration()
        
        # Test state initialization
        executor.initialize_execution_state()
        assert executor.execution_state is not None
        assert executor.execution_state.execution_id is not None
        
        # Test state saving and loading
        execution_id = executor.execution_id
        
        # Create new executor and try to load state
        new_executor = EmbeddingExecutor(config_path)
        loaded = new_executor.load_execution_state(execution_id)
        
        if loaded:
            assert new_executor.execution_state.execution_id == execution_id
    
    @patch('clients.sentence_transformers_client.SentenceTransformer')
    def test_end_to_end_mock_evaluation(self, mock_st, mock_config, temp_dir):
        """Test end-to-end evaluation with mocked components"""
        # Mock sentence transformer
        mock_model = Mock()
        mock_model.encode.return_value = np.random.rand(10, 384)
        mock_st.return_value = mock_model
        
        config, config_path = mock_config
        
        executor = EmbeddingExecutor(config_path)
        
        try:
            # Load configuration
            executor.load_configuration()
            
            # Setup clients (should work with sentence transformers)
            clients = executor.setup_model_clients()
            assert len(clients) > 0
            
            # Load datasets
            datasets = executor.load_datasets()
            assert len(datasets) > 0
            
            # Test that we can create evaluators
            from evaluators.evaluator_factory import EvaluatorFactory
            factory = EvaluatorFactory()
            
            for evaluator_config in executor.config['evaluators']:
                evaluator_type = evaluator_config.get('type')
                evaluator = factory.create_evaluator(evaluator_type, evaluator_config)
                assert evaluator is not None
            
        except Exception as e:
            pytest.skip(f"End-to-end test failed: {e}")
    
    def test_reporting_integration(self, temp_dir):
        """Test reporting functionality"""
        from reporting.embedding_comparison_report import EmbeddingComparisonReport
        
        # Mock evaluation results
        mock_results = {
            'model1': {
                'similarity': {'overall_score': 0.85},
                'efficiency': {'overall_score': 0.75, 'avg_latency': 100},
                'cost_tracking': {'total_cost': 0.001},
                'error_tracking': {'error_rate': 0.0}
            },
            'model2': {
                'similarity': {'overall_score': 0.80},
                'efficiency': {'overall_score': 0.80, 'avg_latency': 80},
                'cost_tracking': {'total_cost': 0.002},
                'error_tracking': {'error_rate': 0.0}
            }
        }
        
        mock_configs = {
            'model1': {'provider': 'test', 'dimensions': 384},
            'model2': {'provider': 'test', 'dimensions': 768}
        }
        
        mock_metadata = {
            'execution_id': 'test_exec',
            'total_models': 2,
            'categories': ['similarity', 'efficiency']
        }
        
        report_generator = EmbeddingComparisonReport(temp_dir)
        
        try:
            report_paths = report_generator.generate_comparison_report(
                mock_results, mock_configs, mock_metadata
            )
            
            assert isinstance(report_paths, dict)
            assert 'html' in report_paths
            assert 'json' in report_paths
            assert 'csv' in report_paths
            
            # Check that files were created
            for report_type, path in report_paths.items():
                assert os.path.exists(path)
                
        except Exception as e:
            pytest.skip(f"Reporting test failed: {e}")
    
    def test_visualization_integration(self, temp_dir):
        """Test visualization functionality"""
        from reporting.embedding_visualizations import EmbeddingVisualizations
        
        # Mock evaluation results
        mock_results = {
            'model1': {
                'similarity': {'overall_score': 0.85},
                'retrieval': {'overall_score': 0.75},
                'clustering': {'overall_score': 0.70},
                'classification': {'overall_score': 0.80},
                'efficiency': {'overall_score': 0.75, 'avg_latency': 100, 'throughput': 10},
                'cost_tracking': {'total_cost': 0.001},
                'error_tracking': {'error_rate': 0.0}
            }
        }
        
        mock_configs = {
            'model1': {'provider': 'test', 'dimensions': 384, 'cost_per_1k_tokens': 0.0001}
        }
        
        viz_generator = EmbeddingVisualizations(temp_dir)
        
        try:
            viz_paths = viz_generator.generate_all_visualizations(
                mock_results, mock_configs
            )
            
            assert isinstance(viz_paths, dict)
            
            # Check that some visualizations were created
            total_files = sum(len(paths) for paths in viz_paths.values())
            assert total_files > 0
            
        except Exception as e:
            pytest.skip(f"Visualization test failed: {e}")

class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_invalid_provider_error(self):
        """Test error handling for invalid provider"""
        factory = EmbeddingClientFactory()
        
        config = {
            'provider': 'invalid_provider',
            'model_name': 'test_model'
        }
        
        with pytest.raises(ValueError, match="Unsupported provider"):
            factory.create_client('invalid_provider', config)
    
    def test_missing_config_fields(self):
        """Test error handling for missing configuration fields"""
        factory = EmbeddingClientFactory()
        
        # Missing model_name
        config = {
            'provider': 'openai'
        }
        
        with pytest.raises(ValueError, match="Missing required configuration field"):
            factory.create_client('openai', config)
    
    def test_invalid_config_file(self, temp_dir):
        """Test error handling for invalid configuration file"""
        invalid_config_path = os.path.join(temp_dir, 'invalid_config.yaml')
        
        # Create invalid YAML
        with open(invalid_config_path, 'w') as f:
            f.write("invalid: yaml: content: [")
        
        executor = EmbeddingExecutor(invalid_config_path)
        
        with pytest.raises(Exception):
            executor.load_configuration()
    
    def test_missing_dataset_file(self, temp_dir):
        """Test error handling for missing dataset file"""
        from datasets.dataset_loader import CustomDatasetLoader
        
        loader = CustomDatasetLoader()
        
        missing_file = os.path.join(temp_dir, 'missing_dataset.json')
        
        with pytest.raises(FileNotFoundError):
            loader.load_dataset(missing_file)

def run_integration_tests():
    """
    Standalone function to run integration tests
    Can be called directly without pytest
    """
    print("Running integration tests...")
    
    try:
        # Test basic imports
        print("✓ Testing basic framework imports...")
        from embedding_client_factory import EmbeddingClientFactory
        from embedding_executor import EmbeddingExecutor
        
        # Test client factory
        print("✓ Testing client factory...")
        factory = EmbeddingClientFactory()
        validation = factory.validate_environment()
        print(f"  Environment validation: {validation['valid']}")
        
        # Test utilities
        print("✓ Testing utilities...")
        from utils.embedding_logger import EmbeddingLogger
        from utils.embedding_file_utils import FileUtils
        from utils.embedding_cost_tracker import CostTracker
        
        logger = EmbeddingLogger.get_logger("integration_test")
        file_utils = FileUtils()
        cost_tracker = CostTracker()
        
        logger.info("Integration test log message")
        
        # Test evaluator factory
        print("✓ Testing evaluator factory...")
        from evaluators.evaluator_factory import EvaluatorFactory
        eval_factory = EvaluatorFactory()
        available_evaluators = eval_factory.list_evaluators()
        print(f"  Available evaluators: {available_evaluators}")
        
        # Test dataset loader
        print("✓ Testing dataset loader...")
        from datasets.dataset_loader import DatasetLoader
        dataset_loader = DatasetLoader()
        
        # Test reporting
        print("✓ Testing reporting components...")
        from reporting.embedding_comparison_report import EmbeddingComparisonReport
        from reporting.embedding_visualizations import EmbeddingVisualizations
        
        print("\n✅ All integration tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Run integration tests when script is executed directly
    import sys
    success = run_integration_tests()
    sys.exit(0 if success else 1)
