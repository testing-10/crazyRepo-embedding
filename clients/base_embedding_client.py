"""
Base Embedding Client - Abstract base class for all embedding model clients
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Tuple
import os
import time
from dataclasses import dataclass
import numpy as np
from pathlib import Path

# Import utilities
import sys
sys.path.append('..')
from utils.embedding_logger import get_logger
from utils.embedding_file_utils import FileUtils
from utils.embedding_cost_tracker import CostTracker, APICallTracker


@dataclass
class EmbeddingResult:
    """Data class for embedding results."""
    embeddings: List[List[float]]
    model_name: str
    provider: str
    tokens_used: int
    processing_time: float
    metadata: Dict[str, Any]


@dataclass
class ModelConfig:
    """Data class for model configuration."""
    name: str
    provider: str
    max_tokens: int
    max_batch_size: int
    dimensions: int
    cost_per_1k_tokens: float
    rate_limit_rpm: int
    rate_limit_tpm: int
    supports_batching: bool
    metadata: Dict[str, Any]


class BaseEmbeddingClient(ABC):
    """
    Abstract base class for all embedding model clients.
    
    All embedding clients must inherit from this class and implement:
    - embed_texts: Batch embedding of multiple texts
    - embed_single: Single text embedding
    - _load_model: Model initialization
    """
    
    def __init__(self, 
                 model_name: str,
                 config_path: Optional[str] = None,
                 api_key: Optional[str] = None,
                 cost_tracker: Optional[CostTracker] = None,
                 logger_name: Optional[str] = None):
        """
        Initialize the embedding client.
        
        Args:
            model_name: Name of the model to use
            config_path: Path to model configuration YAML file
            api_key: API key (if None, will try to load from environment)
            cost_tracker: Cost tracker instance
            logger_name: Logger name for this client
        """
        self.model_name = model_name
        self.config_path = config_path
        self.api_key = api_key
        self.cost_tracker = cost_tracker or CostTracker()
        
        # Initialize logger
        logger_name = logger_name or f"{self.__class__.__name__.lower()}"
        self.logger = get_logger(logger_name)
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize API key from environment if not provided
        if not self.api_key:
            self.api_key = self._get_api_key_from_env()
        
        # Rate limiting
        self._last_request_time = 0
        self._request_count = 0
        self._token_count = 0
        self._minute_start = time.time()
        
        # Model initialization
        self.model = None
        self._is_initialized = False
        
        self.logger.info(f"Initialized {self.__class__.__name__} for model: {model_name}")
    
    def _load_config(self) -> ModelConfig:
        """
        Load model configuration from YAML file.
        
        Returns:
            ModelConfig object with model settings
        """
        if not self.config_path:
            # Use default config path
            self.config_path = Path("configs") / "models.yaml"
        
        try:
            config_data = FileUtils.load_yaml(self.config_path, default={})
            
            # Find model configuration
            model_config = None
            for provider, models in config_data.get('models', {}).items():
                if self.model_name in models:
                    model_config = models[self.model_name].copy()
                    model_config['provider'] = provider
                    break
            
            if not model_config:
                self.logger.warning(f"Model {self.model_name} not found in config, using defaults")
                model_config = self._get_default_config()
            
            return ModelConfig(
                name=self.model_name,
                provider=model_config.get('provider', 'unknown'),
                max_tokens=model_config.get('max_tokens', 8192),
                max_batch_size=model_config.get('max_batch_size', 100),
                dimensions=model_config.get('dimensions', 1536),
                cost_per_1k_tokens=model_config.get('cost_per_1k_tokens', 0.0),
                rate_limit_rpm=model_config.get('rate_limit_rpm', 3000),
                rate_limit_tpm=model_config.get('rate_limit_tpm', 1000000),
                supports_batching=model_config.get('supports_batching', True),
                metadata=model_config.get('metadata', {})
            )
            
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            return ModelConfig(
                name=self.model_name,
                provider='unknown',
                max_tokens=8192,
                max_batch_size=100,
                dimensions=1536,
                cost_per_1k_tokens=0.0,
                rate_limit_rpm=3000,
                rate_limit_tpm=1000000,
                supports_batching=True,
                metadata={}
            )
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for unknown models."""
        return {
            'provider': 'unknown',
            'max_tokens': 8192,
            'max_batch_size': 100,
            'dimensions': 1536,
            'cost_per_1k_tokens': 0.0,
            'rate_limit_rpm': 3000,
            'rate_limit_tpm': 1000000,
            'supports_batching': True,
            'metadata': {}
        }
    
    @abstractmethod
    def _get_api_key_from_env(self) -> Optional[str]:
        """
        Get API key from environment variables.
        Each client should implement this based on their provider's conventions.
        
        Returns:
            API key string or None if not found
        """
        pass
    
    @abstractmethod
    def _load_model(self):
        """
        Load and initialize the embedding model.
        Each client should implement this based on their provider's requirements.
        """
        pass
    
    @abstractmethod
    def embed_texts(self, texts: List[str], **kwargs) -> EmbeddingResult:
        """
        Generate embeddings for multiple texts (batch operation).
        
        Args:
            texts: List of texts to embed
            **kwargs: Additional parameters specific to the provider
            
        Returns:
            EmbeddingResult containing embeddings and metadata
        """
        pass
    
    @abstractmethod
    def embed_single(self, text: str, **kwargs) -> EmbeddingResult:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            **kwargs: Additional parameters specific to the provider
            
        Returns:
            EmbeddingResult containing embedding and metadata
        """
        pass
    
    def ensure_initialized(self):
        """Ensure the model is loaded and initialized."""
        if not self._is_initialized:
            try:
                self._load_model()
                self._is_initialized = True
                self.logger.info(f"Model {self.model_name} initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize model {self.model_name}: {e}")
                raise
    
    def _validate_texts(self, texts: Union[str, List[str]]) -> List[str]:
        """
        Validate and normalize input texts.
        
        Args:
            texts: Single text or list of texts
            
        Returns:
            List of validated texts
            
        Raises:
            ValueError: If texts are invalid
        """
        if isinstance(texts, str):
            texts = [texts]
        
        if not texts:
            raise ValueError("No texts provided for embedding")
        
        if not isinstance(texts, list):
            raise ValueError("Texts must be a string or list of strings")
        
        # Filter out empty texts
        valid_texts = [text for text in texts if text and text.strip()]
        
        if not valid_texts:
            raise ValueError("No valid texts found (all texts are empty)")
        
        if len(valid_texts) != len(texts):
            self.logger.warning(f"Filtered out {len(texts) - len(valid_texts)} empty texts")
        
        return valid_texts
    
    def _estimate_tokens(self, texts: List[str]) -> int:
        """
        Estimate token count for texts.
        Basic estimation - can be overridden by specific clients.
        
        Args:
            texts: List of texts
            
        Returns:
            Estimated token count
        """
        # Simple estimation: ~4 characters per token
        total_chars = sum(len(text) for text in texts)
        return max(1, total_chars // 4)
    
    def _check_rate_limits(self, estimated_tokens: int):
        """
        Check and enforce rate limits.
        
        Args:
            estimated_tokens: Estimated tokens for the request
            
        Raises:
            Exception: If rate limits would be exceeded
        """
        current_time = time.time()
        
        # Reset counters if a minute has passed
        if current_time - self._minute_start >= 60:
            self._request_count = 0
            self._token_count = 0
            self._minute_start = current_time
        
        # Check RPM limit
        if self._request_count >= self.config.rate_limit_rpm:
            wait_time = 60 - (current_time - self._minute_start)
            if wait_time > 0:
                self.logger.warning(f"Rate limit reached, waiting {wait_time:.1f}s")
                time.sleep(wait_time)
                self._request_count = 0
                self._token_count = 0
                self._minute_start = time.time()
        
        # Check TPM limit
        if self._token_count + estimated_tokens > self.config.rate_limit_tpm:
            wait_time = 60 - (current_time - self._minute_start)
            if wait_time > 0:
                self.logger.warning(f"Token rate limit reached, waiting {wait_time:.1f}s")
                time.sleep(wait_time)
                self._request_count = 0
                self._token_count = 0
                self._minute_start = time.time()
        
        # Update counters
        self._request_count += 1
        self._token_count += estimated_tokens
    
    def _create_batches(self, texts: List[str]) -> List[List[str]]:
        """
        Create batches of texts based on model limits.
        
        Args:
            texts: List of texts to batch
            
        Returns:
            List of text batches
        """
        if not self.config.supports_batching:
            return [[text] for text in texts]
        
        batches = []
        current_batch = []
        current_tokens = 0
        
        for text in texts:
            text_tokens = self._estimate_tokens([text])
            
            # Check if adding this text would exceed limits
            if (len(current_batch) >= self.config.max_batch_size or
                current_tokens + text_tokens > self.config.max_tokens):
                
                if current_batch:
                    batches.append(current_batch)
                    current_batch = []
                    current_tokens = 0
            
            current_batch.append(text)
            current_tokens += text_tokens
        
        # Add remaining batch
        if current_batch:
            batches.append(current_batch)
        
        return batches
    
    def _merge_embedding_results(self, results: List[EmbeddingResult]) -> EmbeddingResult:
        """
        Merge multiple embedding results into a single result.
        
        Args:
            results: List of EmbeddingResult objects
            
        Returns:
            Merged EmbeddingResult
        """
        if not results:
            raise ValueError("No results to merge")
        
        if len(results) == 1:
            return results[0]
        
        # Merge embeddings
        all_embeddings = []
        for result in results:
            all_embeddings.extend(result.embeddings)
        
        # Sum up tokens and time
        total_tokens = sum(result.tokens_used for result in results)
        total_time = sum(result.processing_time for result in results)
        
        # Merge metadata
        merged_metadata = {}
        for result in results:
            merged_metadata.update(result.metadata)
        
        merged_metadata['batch_count'] = len(results)
        merged_metadata['individual_times'] = [r.processing_time for r in results]
        
        return EmbeddingResult(
            embeddings=all_embeddings,
            model_name=results[0].model_name,
            provider=results[0].provider,
            tokens_used=total_tokens,
            processing_time=total_time,
            metadata=merged_metadata
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.
        
        Returns:
            Dictionary containing model information
        """
        return {
            'model_name': self.model_name,
            'provider': self.config.provider,
            'dimensions': self.config.dimensions,
            'max_tokens': self.config.max_tokens,
            'max_batch_size': self.config.max_batch_size,
            'supports_batching': self.config.supports_batching,
            'cost_per_1k_tokens': self.config.cost_per_1k_tokens,
            'rate_limits': {
                'rpm': self.config.rate_limit_rpm,
                'tpm': self.config.rate_limit_tpm
            },
            'is_initialized': self._is_initialized,
            'metadata': self.config.metadata
        }
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the client.
        
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
                'provider': self.config.provider,
                'api_key_configured': bool(self.api_key),
                'model_initialized': self._is_initialized,
                'test_embedding_dimensions': len(test_result.embeddings[0]) if test_result.embeddings else 0,
                'timestamp': time.time()
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'model_name': self.model_name,
                'provider': self.config.provider,
                'error': str(e),
                'api_key_configured': bool(self.api_key),
                'model_initialized': self._is_initialized,
                'timestamp': time.time()
            }
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name}, provider={self.config.provider})"
    
    def __repr__(self) -> str:
        return self.__str__()
