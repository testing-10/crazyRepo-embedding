"""
Embedding Client Factory

Centralized factory for creating and managing embedding model clients
with proper configuration, validation, and error handling.
"""

import os
import json
from typing import Dict, Any, Optional, Type, List
from abc import ABC, abstractmethod
import logging

from clients.base_embedding_client import BaseEmbeddingClient
from clients.openai_embedding_client import OpenAIEmbeddingClient
from clients.cohere_embedding_client import CohereEmbeddingClient
from clients.sentence_transformer_client import SentenceTransformerClient
from clients.huggingface_embedding_client import HuggingFaceEmbeddingClient
from clients.jina_embedding_client import JinaEmbeddingClient
from clients.azure_embedding_client import AzureEmbeddingClient
from utils.embedding_logger import EmbeddingLogger
from utils.embedding_file_utils import FileUtils

class EmbeddingClientFactory:
    """
    Factory class for creating embedding model clients with proper configuration
    """
    
    # Registry of available client classes
    _client_registry: Dict[str, Type[BaseEmbeddingClient]] = {
        'openai': OpenAIEmbeddingClient,
        'cohere': CohereEmbeddingClient,
        'sentence_transformers': SentenceTransformerClient,
        'huggingface': HuggingFaceEmbeddingClient,
        'jina': JinaEmbeddingClient,
        'azure_openai': AzureEmbeddingClient
    }
    
    def __init__(self, cost_config_path: str = "costs/embedding_api_costs.json"):
        """
        Initialize the client factory
        
        Args:
            cost_config_path: Path to cost configuration file
        """
        self.logger = EmbeddingLogger.get_logger(__name__)
        self.file_utils = FileUtils()
        self.cost_config_path = cost_config_path
        self.cost_config = self._load_cost_config()
        
    def _load_cost_config(self) -> Dict[str, Any]:
        """Load cost configuration from JSON file"""
        try:
            if os.path.exists(self.cost_config_path):
                with open(self.cost_config_path, 'r') as f:
                    return json.load(f)
            else:
                self.logger.warning(f"Cost config file not found: {self.cost_config_path}")
                return {}
        except Exception as e:
            self.logger.error(f"Error loading cost config: {str(e)}")
            return {}
    
    def create_client(
    self,
    provider: str,
    model_config: Dict[str, Any],
    **kwargs
) -> BaseEmbeddingClient:
        """
        Create an embedding client instance
        
        Args:
            provider: Provider name (e.g., 'openai', 'cohere')
            model_config: Model configuration dictionary
            **kwargs: Additional arguments for client initialization
        
        Returns:
            Configured embedding client instance
        
        Raises:
            ValueError: If provider is not supported
            RuntimeError: If client creation fails
        """
        try:
            self.logger.info(f"Creating client for provider: {provider}")
            
            # Validate provider
            if provider not in self._client_registry:
                available_providers = list(self._client_registry.keys())
                raise ValueError(
                    f"Unsupported provider '{provider}'. "
                    f"Available providers: {available_providers}"
                )
            
            # Get client class
            client_class = self._client_registry[provider]
            
            # Enhance config with cost information
            enhanced_config = self._enhance_config_with_costs(provider, model_config)
            
            # Validate configuration
            self._validate_config(provider, enhanced_config)
            
            # Prepare client initialization parameters
            # Extract model_name as the primary parameter
            model_name = enhanced_config.get('model_name')
            if not model_name:
                raise ValueError(f"Missing model_name in configuration for provider: {provider}")
            
            # Filter config to only include parameters that BaseEmbeddingClient expects
            base_client_params = {
                'config_path', 'api_key', 'cost_tracker', 'logger_name'
            }
            
            # Prepare kwargs for client initialization
            client_kwargs = {}
            for key, value in enhanced_config.items():
                if key in base_client_params:
                    client_kwargs[key] = value
            
            # Add any additional kwargs passed to this method
            client_kwargs.update(kwargs)
            
            # Create client instance
            # Pass model_name as first argument and filtered kwargs
            client = client_class(model_name=model_name, **client_kwargs)
            
            # Validate client after creation
            self._validate_client(client, provider)
            
            self.logger.info(f"Successfully created {provider} client for model: {model_name}")
            return client
            
        except Exception as e:
            self.logger.error(f"Failed to create client for provider '{provider}': {str(e)}")
            raise RuntimeError(f"Client creation failed: {str(e)}")
    
    def _enhance_config_with_costs(self, provider: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance configuration with cost information from cost config"""
        enhanced_config = config.copy()
        
        try:
            cost_models = self.cost_config.get('embedding_api_costs', {}).get('cost_models', {})
            provider_costs = cost_models.get(provider, {})
            
            model_name = config.get('model_name', '')
            if model_name in provider_costs:
                cost_info = provider_costs[model_name]
                enhanced_config.update({
                    'cost_per_1k_tokens': cost_info.get('cost_per_1k_tokens', 0.0),
                    'max_tokens': cost_info.get('max_tokens', 0),
                    'dimensions': cost_info.get('dimensions', enhanced_config.get('dimensions', 0)),
                    'billing_unit': cost_info.get('billing_unit', 'tokens')
                })
                self.logger.debug(f"Enhanced config with cost info for {provider}/{model_name}")
            else:
                self.logger.warning(f"No cost information found for {provider}/{model_name}")
                
        except Exception as e:
            self.logger.warning(f"Failed to enhance config with cost info: {str(e)}")
        
        return enhanced_config
    
    def _validate_config(self, provider: str, config: Dict[str, Any]) -> None:
        """Validate configuration for the specified provider"""
        required_fields = {
            'openai': ['model_name'],
            'cohere': ['model_name'],
            'sentence_transformers': ['model_name'],
            'huggingface': ['model_name'],
            'jina': ['model_name'],
            'azure_openai': ['model_name', 'azure_endpoint']
        }
        
        provider_required = required_fields.get(provider, ['model_name'])
        
        for field in provider_required:
            if field not in config or not config[field]:
                raise ValueError(f"Missing required configuration field '{field}' for provider '{provider}'")
        
        # Validate API keys for paid providers
        api_key_mapping = {
            'openai': 'OPENAI_API_KEY',
            'cohere': 'COHERE_API_KEY',
            'jina': 'JINA_API_KEY',
            'azure_openai': 'AZURE_OPENAI_API_KEY'
        }
        
        if provider in api_key_mapping:
            env_var = api_key_mapping[provider]
            if not os.getenv(env_var):
                self.logger.warning(f"API key not found in environment: {env_var}")
    
    def _validate_client(self, client: BaseEmbeddingClient, provider: str):
        """
        Validate that the client has required methods and attributes.
        
        Args:
            client: Client instance to validate
            provider: Provider name for error messages
        
        Raises:
            ValueError: If client is invalid
        """
        required_methods = ['embed_texts', 'embed_single', '_load_model']
        
        for method_name in required_methods:
            if not hasattr(client, method_name):
                raise ValueError(f"Client missing required method: {method_name}")
            
            method = getattr(client, method_name)
            if not callable(method):
                raise ValueError(f"Client attribute '{method_name}' is not callable")
        
        # Validate that client inherits from BaseEmbeddingClient
        if not isinstance(client, BaseEmbeddingClient):
            raise ValueError(f"Client must inherit from BaseEmbeddingClient")
        
        self.logger.debug(f"Client validation passed for provider: {provider}")
    
    def create_multiple_clients(
        self,
        model_configs: Dict[str, Dict[str, Any]]
    ) -> Dict[str, BaseEmbeddingClient]:
        """
        Create multiple embedding clients from configurations
        
        Args:
            model_configs: Dictionary mapping model names to their configurations
            
        Returns:
            Dictionary mapping model names to client instances
        """
        clients = {}
        failed_models = []
        
        for model_name, config in model_configs.items():
            try:
                provider = config.get('provider')
                if not provider:
                    self.logger.error(f"No provider specified for model: {model_name}")
                    failed_models.append(model_name)
                    continue
                
                client = self.create_client(provider, config)
                clients[model_name] = client
                
            except Exception as e:
                self.logger.error(f"Failed to create client for {model_name}: {str(e)}")
                failed_models.append(model_name)
                continue
        
        if failed_models:
            self.logger.warning(f"Failed to create clients for models: {failed_models}")
        
        self.logger.info(f"Successfully created {len(clients)} clients out of {len(model_configs)} requested")
        return clients
    
    def get_available_providers(self) -> List[str]:
        """Get list of available providers"""
        return list(self._client_registry.keys())
    
    def get_provider_info(self, provider: str) -> Dict[str, Any]:
        """Get information about a specific provider"""
        if provider not in self._client_registry:
            raise ValueError(f"Unknown provider: {provider}")
        
        client_class = self._client_registry[provider]
        
        # Get cost information
        cost_models = self.cost_config.get('embedding_api_costs', {}).get('cost_models', {})
        provider_costs = cost_models.get(provider, {})
        
        return {
            'provider': provider,
            'client_class': client_class.__name__,
            'available_models': list(provider_costs.keys()) if provider_costs else [],
            'requires_api_key': provider in ['openai', 'cohere', 'jina', 'azure_openai'],
            'supports_batching': True,  # All our clients support batching
            'cost_info': provider_costs
        }
    
    def validate_environment(self) -> Dict[str, Any]:
        """
        Validate that the environment is properly configured
        
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'valid': True,
            'missing_dependencies': [],
            'missing_api_keys': [],
            'available_providers': [],
            'warnings': []
        }
        
        # Check dependencies
        required_packages = [
            'numpy', 'pandas', 'scikit-learn', 'sentence_transformers',
            'openai', 'cohere', 'requests', 'yaml', 'matplotlib'
        ]
        
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                validation_results['missing_dependencies'].append(package)
                validation_results['valid'] = False
        
        # Check API keys
        api_keys = {
            'openai': 'OPENAI_API_KEY',
            'cohere': 'COHERE_API_KEY',
            'jina': 'JINA_API_KEY',
            'azure_openai': 'AZURE_OPENAI_API_KEY'
        }
        
        for provider, env_var in api_keys.items():
            if not os.getenv(env_var):
                validation_results['missing_api_keys'].append(env_var)
                validation_results['warnings'].append(f"Missing API key for {provider}: {env_var}")
        
        # Check available providers
        for provider in self._client_registry.keys():
            try:
                # Try to get provider info (basic validation)
                info = self.get_provider_info(provider)
                validation_results['available_providers'].append(provider)
            except Exception as e:
                validation_results['warnings'].append(f"Provider {provider} validation failed: {str(e)}")
        
        return validation_results
    
    @classmethod
    def register_client(cls, provider: str, client_class: Type[BaseEmbeddingClient]) -> None:
        """
        Register a new client class for a provider
        
        Args:
            provider: Provider name
            client_class: Client class that extends BaseEmbeddingClient
        """
        if not issubclass(client_class, BaseEmbeddingClient):
            raise ValueError("Client class must extend BaseEmbeddingClient")
        
        cls._client_registry[provider] = client_class
        logging.getLogger(__name__).info(f"Registered new client for provider: {provider}")
    
    @classmethod
    def unregister_client(cls, provider: str) -> None:
        """Unregister a client for a provider"""
        if provider in cls._client_registry:
            del cls._client_registry[provider]
            logging.getLogger(__name__).info(f"Unregistered client for provider: {provider}")
