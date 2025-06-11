"""
Jina Embedding Client - Client for Jina AI embedding models
"""

import os
import time
from typing import List, Dict, Any, Optional
import requests
import json

from clients.base_embedding_client import BaseEmbeddingClient, EmbeddingResult
from utils.embedding_cost_tracker import APICallTracker


class JinaEmbeddingClient(BaseEmbeddingClient):
    """
    Client for Jina AI embedding models.
    
    Supports models:
    - jina-embeddings-v2-base-en
    - jina-embeddings-v2-small-en
    - jina-embeddings-v2-base-zh
    - jina-embeddings-v2-base-de
    - jina-embeddings-v2-base-es
    """
    
    def __init__(self, model_name: str = "jina-embeddings-v2-base-en", **kwargs):
        """
        Initialize Jina embedding client.
        
        Args:
            model_name: Jina model name
            **kwargs: Additional arguments passed to base class
        """
        super().__init__(model_name=model_name, **kwargs)
        self.base_url = "https://api.jina.ai/v1/embeddings"
        self.session = None
    
    def _get_api_key_from_env(self) -> Optional[str]:
        """Get Jina API key from environment variables."""
        return os.getenv('JINA_API_KEY')
    
    def _load_model(self):
        """Initialize Jina client session."""
        if not self.api_key:
            raise ValueError("Jina API key not found. Set JINA_API_KEY environment variable.")
        
        try:
            # Create session with headers
            self.session = requests.Session()
            self.session.headers.update({
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            })
            
            # Test the connection with a simple call
            test_response = self._make_request(["test"])
            
            # Update dimensions from actual response
            if test_response and 'data' in test_response and test_response['data']:
                actual_dimensions = len(test_response['data'][0]['embedding'])
                if actual_dimensions != self.config.dimensions:
                    self.logger.info(f"Updated dimensions from {self.config.dimensions} to {actual_dimensions}")
                    self.config.dimensions = actual_dimensions
            
            self.logger.info(f"Jina client initialized for model: {self.model_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Jina client: {e}")
            raise
    
    def _make_request(self, texts: List[str], **kwargs) -> Dict[str, Any]:
        """
        Make request to Jina API.
        
        Args:
            texts: List of texts to embed
            **kwargs: Additional parameters
            
        Returns:
            API response dictionary
            
        Raises:
            Exception: If API request fails
        """
        payload = {
            'model': self.model_name,
            'input': texts
        }
        
        # Add optional parameters
        if 'encoding_format' in kwargs:
            payload['encoding_format'] = kwargs['encoding_format']
        
        try:
            response = self.session.post(self.base_url, json=payload, timeout=60)
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Jina API request failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_detail = e.response.json()
                    self.logger.error(f"API error details: {error_detail}")
                except:
                    self.logger.error(f"API response: {e.response.text}")
            raise
    
    def _estimate_tokens(self, texts: List[str]) -> int:
        """
        Estimate tokens for Jina models.
        """
        # Estimation: ~0.75 tokens per word
        total_words = sum(len(text.split()) for text in texts)
        return max(1, int(total_words * 0.75))
    
    def embed_texts(self, 
                   texts: List[str],
                   encoding_format: str = "float",
                   **kwargs) -> EmbeddingResult:
        """
        Generate embeddings for multiple texts using Jina API.
        
        Args:
            texts: List of texts to embed
            encoding_format: Encoding format ("float", "base64")
            **kwargs: Additional parameters
            
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
                provider="jina",
                model=self.model_name,
                endpoint="embeddings"
            ) as tracker:
                try:
                    # Make API call
                    response = self._make_request(
                        batch, 
                        encoding_format=encoding_format,
                        **kwargs
                    )
                    
                    # Extract embeddings
                    batch_embeddings = [item['embedding'] for item in response['data']]
                    all_embeddings.extend(batch_embeddings)
                    
                    # Track usage
                    batch_tokens = response.get('usage', {}).get('total_tokens', estimated_tokens)
                    total_tokens += batch_tokens
                    tracker.set_tokens_used(batch_tokens)
                    tracker.add_metadata('batch_index', batch_idx)
                    tracker.add_metadata('batch_size', len(batch))
                    tracker.add_metadata('encoding_format', encoding_format)
                    
                    self.logger.debug(f"Batch {batch_idx + 1}/{len(batches)} completed: {len(batch)} texts, {batch_tokens} tokens")
                    
                except Exception as e:
                    self.logger.error(f"Jina API error in batch {batch_idx}: {e}")
                    raise
        
        processing_time = time.time() - start_time
        
        # Create result
        result = EmbeddingResult(
            embeddings=all_embeddings,
            model_name=self.model_name,
            provider="jina",
            tokens_used=total_tokens,
            processing_time=processing_time,
            metadata={
                'batch_count': len(batches),
                'total_texts': len(texts),
                'average_embedding_dimension': len(all_embeddings[0]) if all_embeddings else 0,
                'encoding_format': encoding_format,
                'request_params': {k: v for k, v in kwargs.items() if k != 'input'}
            }
        )
        
        self.logger.info(f"Generated {len(all_embeddings)} embeddings in {processing_time:.2f}s using {total_tokens} tokens")
        return result
    
    def embed_single(self, 
                    text: str,
                    encoding_format: str = "float",
                    **kwargs) -> EmbeddingResult:
        """
        Generate embedding for a single text using Jina API.
        
        Args:
            text: Text to embed
            encoding_format: Encoding format
            **kwargs: Additional parameters
            
        Returns:
            EmbeddingResult containing embedding and metadata
        """
        return self.embed_texts([text], encoding_format=encoding_format, **kwargs)
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available Jina embedding models.
        
        Returns:
            List of model names
        """
        return [
            "jina-embeddings-v2-base-en",
            "jina-embeddings-v2-small-en",
            "jina-embeddings-v2-base-zh",
            "jina-embeddings-v2-base-de",
            "jina-embeddings-v2-base-es",
            "jina-embeddings-v2-base-code"
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
        
        # Dimensions for Jina models
        dimensions_map = {
            "jina-embeddings-v2-base-en": 768,
            "jina-embeddings-v2-small-en": 512,
            "jina-embeddings-v2-base-zh": 768,
            "jina-embeddings-v2-base-de": 768,
            "jina-embeddings-v2-base-es": 768,
            "jina-embeddings-v2-base-code": 768
        }
        
        return dimensions_map.get(model, 768)
    
    def get_max_input_length(self) -> int:
        """
        Get maximum input length for current model.
        
        Returns:
            Maximum input length in tokens
        """
        # Jina models typically support up to 8192 tokens
        return 8192
    
    def get_supported_languages(self) -> List[str]:
        """
        Get supported languages for current model.
        
        Returns:
            List of supported language codes
        """
        language_map = {
            "jina-embeddings-v2-base-en": ["en"],
            "jina-embeddings-v2-small-en": ["en"],
            "jina-embeddings-v2-base-zh": ["zh"],
            "jina-embeddings-v2-base-de": ["de"],
            "jina-embeddings-v2-base-es": ["es"],
            "jina-embeddings-v2-base-code": ["code"]
        }
        
        return language_map.get(self.model_name, ["en"])
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for Jina models."""
        return {
            'provider': 'jina',
            'max_tokens': self.get_max_input_length(),
            'max_batch_size': 2048,  # Jina supports large batches
            'dimensions': self.get_model_dimensions(),
            'cost_per_1k_tokens': self._get_model_cost(),
            'rate_limit_rpm': 200,  # Conservative estimate
            'rate_limit_tpm': 1000000,
            'supports_batching': True,
            'metadata': {
                'max_input_length': self.get_max_input_length(),
                'supported_languages': self.get_supported_languages(),
                'supported_formats': ["float", "base64"]
            }
        }
    
    def _get_model_cost(self) -> float:
        """Get cost per 1K tokens for current model."""
        cost_map = {
            "jina-embeddings-v2-base-en": 0.00002,
            "jina-embeddings-v2-small-en": 0.00001,
            "jina-embeddings-v2-base-zh": 0.00002,
            "jina-embeddings-v2-base-de": 0.00002,
            "jina-embeddings-v2-base-es": 0.00002,
            "jina-embeddings-v2-base-code": 0.00002
        }
        return cost_map.get(self.model_name, 0.00002)
    
    def _create_batches(self, texts: List[str]) -> List[List[str]]:
        """
        Create batches optimized for Jina API limits.
        
        Args:
            texts: List of texts to batch
            
        Returns:
            List of text batches
        """
        # Jina supports large batches, but we'll be conservative
        max_batch_size = min(self.config.max_batch_size, 1000)
        
        batches = []
        for i in range(0, len(texts), max_batch_size):
            batch = texts[i:i + max_batch_size]
            batches.append(batch)
        
        return batches
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the Jina client.
        
        Returns:
            Dictionary containing health status
        """
        try:
            self.ensure_initialized()
            
            # Try a simple embedding
            test_result = self.embed_single("health check test")
            
            return {
                'status': 'healthy',
                'model_name': self.model_name,
                'provider': 'jina',
                'api_key_configured': bool(self.api_key),
                'model_initialized': self._is_initialized,
                'test_embedding_dimensions': len(test_result.embeddings[0]) if test_result.embeddings else 0,
                'supported_languages': self.get_supported_languages(),
                'timestamp': time.time()
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'model_name': self.model_name,
                'provider': 'jina',
                'error': str(e),
                'api_key_configured': bool(self.api_key),
                'model_initialized': self._is_initialized,
                'timestamp': time.time()
            }
