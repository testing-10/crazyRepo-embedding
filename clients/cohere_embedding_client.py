"""
Cohere Embedding Client - Client for Cohere embedding models
"""

import os
import time
from typing import List, Dict, Any, Optional
import cohere

from .base_embedding_client import BaseEmbeddingClient, EmbeddingResult
from ..utils.embedding_cost_tracker import APICallTracker


class CohereEmbeddingClient(BaseEmbeddingClient):
    """
    Client for Cohere embedding models.
    
    Supports models:
    - embed-english-v3.0
    - embed-multilingual-v3.0
    - embed-english-light-v3.0
    - embed-multilingual-light-v3.0
    """
    
    def __init__(self, model_name: str = "embed-english-v3.0", **kwargs):
        """
        Initialize Cohere embedding client.
        
        Args:
            model_name: Cohere model name
            **kwargs: Additional arguments passed to base class
        """
        super().__init__(model_name=model_name, **kwargs)
        self.client = None
    
    def _get_api_key_from_env(self) -> Optional[str]:
        """Get Cohere API key from environment variables."""
        return os.getenv('COHERE_API_KEY') or os.getenv('CO_API_KEY')
    
    def _load_model(self):
        """Initialize Cohere client."""
        if not self.api_key:
            raise ValueError("Cohere API key not found. Set COHERE_API_KEY or CO_API_KEY environment variable.")
        
        try:
            self.client = cohere.Client(api_key=self.api_key)
            
            # Test the connection with a simple call
            test_response = self.client.embed(
                texts=["test"],
                model=self.model_name,
                input_type="search_document"
            )
            
            # Update dimensions from actual response
            if test_response.embeddings:
                actual_dimensions = len(test_response.embeddings[0])
                if actual_dimensions != self.config.dimensions:
                    self.logger.info(f"Updated dimensions from {self.config.dimensions} to {actual_dimensions}")
                    self.config.dimensions = actual_dimensions
            
            self.logger.info(f"Cohere client initialized for model: {self.model_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Cohere client: {e}")
            raise
    
    def _estimate_tokens(self, texts: List[str]) -> int:
        """
        Estimate tokens for Cohere models.
        Cohere uses similar tokenization to other models.
        """
        # Estimation: ~0.75 tokens per word
        total_words = sum(len(text.split()) for text in texts)
        return max(1, int(total_words * 0.75))
    
    def embed_texts(self, 
                   texts: List[str], 
                   input_type: str = "search_document",
                   truncate: str = "END",
                   **kwargs) -> EmbeddingResult:
        """
        Generate embeddings for multiple texts using Cohere API.
        
        Args:
            texts: List of texts to embed
            input_type: Type of input ("search_document", "search_query", "classification", "clustering")
            truncate: How to truncate long texts ("NONE", "START", "END")
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
                provider="cohere",
                model=self.model_name,
                endpoint="embed"
            ) as tracker:
                try:
                    # Prepare request parameters
                    request_params = {
                        'texts': batch,
                        'model': self.model_name,
                        'input_type': input_type,
                        'truncate': truncate
                    }
                    
                    # Add optional parameters
                    if 'embedding_types' in kwargs:
                        request_params['embedding_types'] = kwargs['embedding_types']
                    
                    # Make API call
                    response = self.client.embed(**request_params)
                    
                    # Extract embeddings
                    batch_embeddings = response.embeddings
                    all_embeddings.extend(batch_embeddings)
                    
                    # Estimate tokens used (Cohere doesn't always return token count)
                    batch_tokens = estimated_tokens
                    if hasattr(response, 'meta') and response.meta and hasattr(response.meta, 'billed_units'):
                        if hasattr(response.meta.billed_units, 'input_tokens'):
                            batch_tokens = response.meta.billed_units.input_tokens
                    
                    total_tokens += batch_tokens
                    tracker.set_tokens_used(batch_tokens)
                    tracker.add_metadata('batch_index', batch_idx)
                    tracker.add_metadata('batch_size', len(batch))
                    tracker.add_metadata('input_type', input_type)
                    
                    self.logger.debug(f"Batch {batch_idx + 1}/{len(batches)} completed: {len(batch)} texts, ~{batch_tokens} tokens")
                    
                except Exception as e:
                    self.logger.error(f"Cohere API error in batch {batch_idx}: {e}")
                    raise
        
        processing_time = time.time() - start_time
        
        # Create result
        result = EmbeddingResult(
            embeddings=all_embeddings,
            model_name=self.model_name,
            provider="cohere",
            tokens_used=total_tokens,
            processing_time=processing_time,
            metadata={
                'batch_count': len(batches),
                'total_texts': len(texts),
                'average_embedding_dimension': len(all_embeddings[0]) if all_embeddings else 0,
                'input_type': input_type,
                'truncate': truncate,
                'request_params': {k: v for k, v in kwargs.items() if k not in ['texts']}
            }
        )
        
        self.logger.info(f"Generated {len(all_embeddings)} embeddings in {processing_time:.2f}s using ~{total_tokens} tokens")
        return result
    
    def embed_single(self, 
                    text: str, 
                    input_type: str = "search_document",
                    **kwargs) -> EmbeddingResult:
        """
        Generate embedding for a single text using Cohere API.
        
        Args:
            text: Text to embed
            input_type: Type of input
            **kwargs: Additional parameters
            
        Returns:
            EmbeddingResult containing embedding and metadata
        """
        return self.embed_texts([text], input_type=input_type, **kwargs)
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available Cohere embedding models.
        
        Returns:
            List of model names
        """
        return [
            "embed-english-v3.0",
            "embed-multilingual-v3.0",
            "embed-english-light-v3.0",
            "embed-multilingual-light-v3.0",
            "embed-english-v2.0",
            "embed-multilingual-v2.0"
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
        
        # Dimensions for Cohere models
        dimensions_map = {
            "embed-english-v3.0": 1024,
            "embed-multilingual-v3.0": 1024,
            "embed-english-light-v3.0": 384,
            "embed-multilingual-light-v3.0": 384,
            "embed-english-v2.0": 4096,
            "embed-multilingual-v2.0": 768
        }
        
        return dimensions_map.get(model, 1024)
    
    def get_supported_input_types(self) -> List[str]:
        """
        Get supported input types for embeddings.
        
        Returns:
            List of supported input types
        """
        return [
            "search_document",
            "search_query", 
            "classification",
            "clustering"
        ]
    
    def get_max_input_length(self) -> int:
        """
        Get maximum input length for current model.
        
        Returns:
            Maximum input length in tokens
        """
        # Cohere models support different max lengths
        max_length_map = {
            "embed-english-v3.0": 512,
            "embed-multilingual-v3.0": 512,
            "embed-english-light-v3.0": 512,
            "embed-multilingual-light-v3.0": 512,
            "embed-english-v2.0": 2048,
            "embed-multilingual-v2.0": 2048
        }
        
        return max_length_map.get(self.model_name, 512)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for Cohere models."""
        return {
            'provider': 'cohere',
            'max_tokens': self.get_max_input_length(),
            'max_batch_size': 96,  # Cohere allows up to 96 texts per request
            'dimensions': self.get_model_dimensions(),
            'cost_per_1k_tokens': self._get_model_cost(),
            'rate_limit_rpm': 1000,
            'rate_limit_tpm': 100000,
            'supports_batching': True,
            'metadata': {
                'supported_input_types': self.get_supported_input_types(),
                'max_input_length': self.get_max_input_length(),
                'supports_truncation': True
            }
        }
    
    def _get_model_cost(self) -> float:
        """Get cost per 1K tokens for current model."""
        cost_map = {
            "embed-english-v3.0": 0.0001,
            "embed-multilingual-v3.0": 0.0001,
            "embed-english-light-v3.0": 0.0001,
            "embed-multilingual-light-v3.0": 0.0001,
            "embed-english-v2.0": 0.0001,
            "embed-multilingual-v2.0": 0.0001
        }
        return cost_map.get(self.model_name, 0.0001)
    
    def _create_batches(self, texts: List[str]) -> List[List[str]]:
        """
        Create batches optimized for Cohere API limits.
        
        Args:
            texts: List of texts to batch
            
        Returns:
            List of text batches
        """
        # Cohere has a limit of 96 texts per request
        max_batch_size = min(self.config.max_batch_size, 96)
        
        batches = []
        for i in range(0, len(texts), max_batch_size):
            batch = texts[i:i + max_batch_size]
            batches.append(batch)
        
        return batches
