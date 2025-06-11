"""
Azure OpenAI Embedding Client - Client for Azure OpenAI embedding models
"""

import os
import time
from typing import List, Dict, Any, Optional
from openai import AzureOpenAI

from clients.base_embedding_client import BaseEmbeddingClient, EmbeddingResult
from utils.embedding_cost_tracker import APICallTracker


class AzureEmbeddingClient(BaseEmbeddingClient):
    """
    Client for Azure OpenAI embedding models.
    
    Supports the same models as OpenAI but through Azure endpoints:
    - text-embedding-3-small
    - text-embedding-3-large  
    - text-embedding-ada-002
    """
    
    def __init__(self, 
                 model_name: str = "text-embedding-3-small",
                 deployment_name: Optional[str] = None,
                 **kwargs):
        """
        Initialize Azure OpenAI embedding client.
        
        Args:
            model_name: OpenAI model name
            deployment_name: Azure deployment name (defaults to model_name)
            **kwargs: Additional arguments passed to base class
        """
        super().__init__(model_name=model_name, **kwargs)
        self.deployment_name = deployment_name or model_name
        self.azure_endpoint = None
        self.api_version = "2024-02-01"
        self.client = None
    
    def _get_api_key_from_env(self) -> Optional[str]:
        """Get Azure OpenAI API key from environment variables."""
        return os.getenv('AZURE_OPENAI_API_KEY')
    
    def _get_azure_endpoint_from_env(self) -> Optional[str]:
        """Get Azure OpenAI endpoint from environment variables."""
        return os.getenv('AZURE_OPENAI_ENDPOINT')
    
    def _get_api_version_from_env(self) -> str:
        """Get Azure OpenAI API version from environment variables."""
        return os.getenv('AZURE_OPENAI_API_VERSION', self.api_version)
    
    def _load_model(self):
        """Initialize Azure OpenAI client."""
        if not self.api_key:
            raise ValueError("Azure OpenAI API key not found. Set AZURE_OPENAI_API_KEY environment variable.")
        
        self.azure_endpoint = self._get_azure_endpoint_from_env()
        if not self.azure_endpoint:
            raise ValueError("Azure OpenAI endpoint not found. Set AZURE_OPENAI_ENDPOINT environment variable.")
        
        self.api_version = self._get_api_version_from_env()
        
        try:
            self.client = AzureOpenAI(
                api_key=self.api_key,
                azure_endpoint=self.azure_endpoint,
                api_version=self.api_version
            )
            
            # Test the connection with a simple call
            test_response = self.client.embeddings.create(
                model=self.deployment_name,
                input="test",
                encoding_format="float"
            )
            
            # Update dimensions from actual response
            if test_response.data:
                actual_dimensions = len(test_response.data[0].embedding)
                if actual_dimensions != self.config.dimensions:
                    self.logger.info(f"Updated dimensions from {self.config.dimensions} to {actual_dimensions}")
                    self.config.dimensions = actual_dimensions
            
            self.logger.info(f"Azure OpenAI client initialized for deployment: {self.deployment_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Azure OpenAI client: {e}")
            raise
    
    def _estimate_tokens(self, texts: List[str]) -> int:
        """
        Estimate tokens for Azure OpenAI models (same as OpenAI).
        """
        # More accurate estimation for OpenAI: ~0.75 tokens per word
        total_words = sum(len(text.split()) for text in texts)
        return max(1, int(total_words * 0.75))
    
    def embed_texts(self, texts: List[str], **kwargs) -> EmbeddingResult:
        """
        Generate embeddings for multiple texts using Azure OpenAI API.
        
        Args:
            texts: List of texts to embed
            **kwargs: Additional parameters (dimensions, user, etc.)
            
        Returns:
            EmbeddingResult containing embeddings and metadata
        """
        self.ensure_initialized()
        texts = self._validate_texts(texts)
        
        start_time = time.time()
        all_embeddings = []
        total_tokens = 0
        
        # Create batches
        batches = self._create_batches(texts)
        self.logger.info(f"Processing {len(texts)} texts in {len(batches)} batches")
        
        for batch_idx, batch in enumerate(batches):
            estimated_tokens = self._estimate_tokens(batch)
            self._check_rate_limits(estimated_tokens)
            
            with APICallTracker(
                cost_tracker=self.cost_tracker,
                provider="azure",
                model=self.model_name,
                endpoint="embeddings"
            ) as tracker:
                try:
                    # Prepare request parameters
                    request_params = {
                        'model': self.deployment_name,  # Use deployment name for Azure
                        'input': batch,
                        'encoding_format': 'float'
                    }
                    
                    # Add optional parameters
                    if 'dimensions' in kwargs and self.model_name in ['text-embedding-3-small', 'text-embedding-3-large']:
                        request_params['dimensions'] = kwargs['dimensions']
                    
                    if 'user' in kwargs:
                        request_params['user'] = kwargs['user']
                    
                    # Make API call
                    response = self.client.embeddings.create(**request_params)
                    
                    # Extract embeddings
                    batch_embeddings = [item.embedding for item in response.data]
                    all_embeddings.extend(batch_embeddings)
                    
                    # Track usage
                    batch_tokens = response.usage.total_tokens
                    total_tokens += batch_tokens
                    tracker.set_tokens_used(batch_tokens)
                    tracker.add_metadata('batch_index', batch_idx)
                    tracker.add_metadata('batch_size', len(batch))
                    tracker.add_metadata('deployment_name', self.deployment_name)
                    tracker.add_metadata('azure_endpoint', self.azure_endpoint)
                    
                    self.logger.debug(f"Batch {batch_idx + 1}/{len(batches)} completed: {len(batch)} texts, {batch_tokens} tokens")
                    
                except Exception as e:
                    self.logger.error(f"Azure OpenAI API error in batch {batch_idx}: {e}")
                    raise
        
        processing_time = time.time() - start_time
        
        # Create result
        result = EmbeddingResult(
            embeddings=all_embeddings,
            model_name=self.model_name,
            provider="azure",
            tokens_used=total_tokens,
            processing_time=processing_time,
            metadata={
                'batch_count': len(batches),
                'total_texts': len(texts),
                'average_embedding_dimension': len(all_embeddings[0]) if all_embeddings else 0,
                'deployment_name': self.deployment_name,
                'azure_endpoint': self.azure_endpoint,
                'api_version': self.api_version,
                'request_params': {k: v for k, v in kwargs.items() if k != 'input'}
            }
        )
        
        self.logger.info(f"Generated {len(all_embeddings)} embeddings in {processing_time:.2f}s using {total_tokens} tokens")
        return result
    
    def embed_single(self, text: str, **kwargs) -> EmbeddingResult:
        """
        Generate embedding for a single text using Azure OpenAI API.
        
        Args:
            text: Text to embed
            **kwargs: Additional parameters
            
        Returns:
            EmbeddingResult containing embedding and metadata
        """
        return self.embed_texts([text], **kwargs)
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available Azure OpenAI embedding models.
        
        Returns:
            List of model names (same as OpenAI)
        """
        return [
            "text-embedding-3-small",
            "text-embedding-3-large", 
            "text-embedding-ada-002"
        ]
    
    def get_model_dimensions(self, model_name: Optional[str] = None) -> int:
        """
        Get embedding dimensions for a model.
        
        Args:
            model_name: Model name (uses current model if None)
            
        Returns:
            Number of dimensions
        """
        model = model_name or self.model_name
        
        # Same dimensions as OpenAI models
        dimensions_map = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536
        }
        
        return dimensions_map.get(model, 1536)
    
    def supports_custom_dimensions(self) -> bool:
        """
        Check if current model supports custom dimensions.
        
        Returns:
            True if custom dimensions are supported
        """
        return self.model_name in ['text-embedding-3-small', 'text-embedding-3-large']
    
    def get_max_input_length(self) -> int:
        """
        Get maximum input length for current model.
        
        Returns:
            Maximum input length in tokens
        """
        # Same as OpenAI: 8192 tokens
        return 8192
    
    def get_deployment_info(self) -> Dict[str, Any]:
        """
        Get Azure deployment information.
        
        Returns:
            Dictionary containing deployment details
        """
        return {
            'deployment_name': self.deployment_name,
            'model_name': self.model_name,
            'azure_endpoint': self.azure_endpoint,
            'api_version': self.api_version,
            'provider': 'azure'
        }
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for Azure OpenAI models."""
        return {
            'provider': 'azure',
            'max_tokens': 8192,
            'max_batch_size': 2048,  # Same as OpenAI
            'dimensions': self.get_model_dimensions(),
            'cost_per_1k_tokens': self._get_model_cost(),
            'rate_limit_rpm': 3000,  # May vary by Azure subscription
            'rate_limit_tpm': 1000000,
            'supports_batching': True,
            'metadata': {
                'supports_custom_dimensions': self.supports_custom_dimensions(),
                'max_input_length': self.get_max_input_length(),
                'deployment_name': self.deployment_name,
                'azure_endpoint': self.azure_endpoint,
                'api_version': self.api_version
            }
        }
    
    def _get_model_cost(self) -> float:
        """Get cost per 1K tokens for current model (same as OpenAI)."""
        cost_map = {
            "text-embedding-3-small": 0.00002,
            "text-embedding-3-large": 0.00013,
            "text-embedding-ada-002": 0.0001
        }
        return cost_map.get(self.model_name, 0.0001)
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the Azure client.
        
        Returns:
            Dictionary containing health status
        """
        try:
            self.ensure_initialized()
            
            # Try a simple embedding
            test_result = self.embed_single("test")
            
            return {
                'status': 'healthy',
                'model_name': self.model_name,
                'deployment_name': self.deployment_name,
                'provider': 'azure',
                'azure_endpoint': self.azure_endpoint,
                'api_version': self.api_version,
                'api_key_configured': bool(self.api_key),
                'model_initialized': self._is_initialized,
                'test_embedding_dimensions': len(test_result.embeddings[0]) if test_result.embeddings else 0,
                'timestamp': time.time()
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'model_name': self.model_name,
                'deployment_name': self.deployment_name,
                'provider': 'azure',
                'azure_endpoint': self.azure_endpoint,
                'error': str(e),
                'api_key_configured': bool(self.api_key),
                'model_initialized': self._is_initialized,
                'timestamp': time.time()
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model and deployment.
        
        Returns:
            Dictionary containing model and deployment information
        """
        base_info = super().get_model_info()
        
        # Add Azure-specific information
        base_info.update({
            'deployment_name': self.deployment_name,
            'azure_endpoint': self.azure_endpoint,
            'api_version': self.api_version,
            'supports_custom_dimensions': self.supports_custom_dimensions()
        })
        
        return base_info
    
    def __str__(self) -> str:
        return f"AzureEmbeddingClient(model={self.model_name}, deployment={self.deployment_name})"
