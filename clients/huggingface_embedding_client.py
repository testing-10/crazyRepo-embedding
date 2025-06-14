"""
HuggingFace Embedding Client - Client for HuggingFace embedding models
"""

import os
import time
from typing import List, Dict, Any, Optional, Union
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

from clients.base_embedding_client import BaseEmbeddingClient, EmbeddingResult
from utils.embedding_cost_tracker import APICallTracker


class HuggingFaceEmbeddingClient(BaseEmbeddingClient):
    """
    Client for HuggingFace embedding models.

    Supports models like:
    - BAAI/bge-m3
    - intfloat/e5-mistral-7b-instruct
    - thenlper/gte-large
    - BAAI/bge-large-en-v1.5
    - intfloat/e5-large-v2
    """

    def __init__(self, model_name: str = "BAAI/bge-m3", **kwargs):
        """
        Initialize HuggingFace embedding client.

        Args:
            model_name: HuggingFace model name
            **kwargs: Additional arguments passed to base class
        """
        super().__init__(model_name=model_name, **kwargs)
        self.device = self._get_device()
        self.tokenizer = None
        self.model = None
        self.max_length = 512  # Default, will be updated from model config

    def _get_api_key_from_env(self) -> Optional[str]:
        """Get HuggingFace token from environment variables."""
        return os.getenv('HUGGINGFACE_TOKEN') or os.getenv('HF_TOKEN')

    def _get_device(self) -> str:
        """Determine the best device for model inference."""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"  # Apple Silicon
        else:
            return "cpu"

    def _load_model(self):
        """Load HuggingFace model and tokenizer."""
        try:
            self.logger.info(f"Loading HuggingFace model: {self.model_name} on {self.device}")

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                token=self.api_key,
                trust_remote_code=True
            )

            # Load model
            self.model = AutoModel.from_pretrained(
                self.model_name,
                token=self.api_key,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )

            # Move model to device
            self.model = self.model.to(self.device)
            self.model.eval()

            # Get max length from tokenizer
            if hasattr(self.tokenizer, 'model_max_length'):
                self.max_length = min(self.tokenizer.model_max_length, 8192)

            # Test embedding to get dimensions
            test_embedding = self._encode_texts(["test"])[0]
            actual_dimensions = len(test_embedding)

            if actual_dimensions != self.config.dimensions:
                self.logger.info(f"Updated dimensions from {self.config.dimensions} to {actual_dimensions}")
                self.config.dimensions = actual_dimensions

            self.logger.info(f"HuggingFace model loaded: {self.model_name} ({actual_dimensions}D, max_len={self.max_length})")

        except Exception as e:
            self.logger.error(f"Failed to load HuggingFace model: {e}")
            raise

    def _mean_pooling(self, model_output, attention_mask):
        """Apply mean pooling to get sentence embeddings."""
        token_embeddings = model_output[0]  # First element contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def _encode_texts(self, texts: List[str], normalize: bool = True) -> List[List[float]]:
        """
        Encode texts to embeddings using the HuggingFace model.

        Args:
            texts: List of texts to encode
            normalize: Whether to normalize embeddings

        Returns:
            List of embeddings
        """
        # Tokenize texts
        encoded_input = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        # Move to device
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}

        # Generate embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # Apply pooling strategy based on model type
        if self.model_name.startswith('BAAI/bge'):
            # BGE models use CLS token
            embeddings = model_output.last_hidden_state[:, 0]
        elif 'e5' in self.model_name.lower():
            # E5 models use mean pooling
            embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
        elif 'gte' in self.model_name.lower():
            # GTE models use mean pooling
            embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
        else:
            # Default to mean pooling
            embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])

        # Normalize embeddings if requested
        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().numpy().tolist()

    def _estimate_tokens(self, texts: List[str]) -> int:
        """
        Estimate tokens for HuggingFace models using the tokenizer.

        Args:
            texts: List of texts

        Returns:
            Estimated token count
        """
        if self.tokenizer:
            try:
                # Use actual tokenizer for accurate count
                total_tokens = 0
                for text in texts:
                    tokens = self.tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=self.max_length)
                    total_tokens += len(tokens)
                return total_tokens
            except Exception:
                pass

        # Fallback to word-based estimation
        total_words = sum(len(text.split()) for text in texts)
        return max(1, int(total_words * 0.75))

    def embed_texts(self, 
                   texts: List[str],
                   batch_size: Optional[int] = None,
                   normalize: bool = True,
                   **kwargs) -> EmbeddingResult:
        """
        Generate embeddings for multiple texts using HuggingFace models.

        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            normalize: Whether to normalize embeddings
            **kwargs: Additional parameters

        Returns:
            EmbeddingResult containing embeddings and metadata
        """
        self.ensure_initialized()
        texts = self._validate_texts(texts)

        start_time = time.time()
        all_embeddings = []

        # Use provided batch size or config default
        if batch_size is None:
            batch_size = min(self.config.max_batch_size, 8)  # Conservative for large models

        # Create batches
        batches = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batches.append(batch)

        self.logger.info(f"Processing {len(texts)} texts in {len(batches)} batches")

        estimated_tokens = self._estimate_tokens(texts)

        try:
            for batch_idx, batch in enumerate(batches):
                # Generate embeddings for batch
                batch_embeddings = self._encode_texts(batch, normalize=normalize)
                all_embeddings.extend(batch_embeddings)

                self.logger.debug(f"Batch {batch_idx + 1}/{len(batches)} completed: {len(batch)} texts")

        except Exception as e:
            self.logger.error(f"HuggingFace encoding error: {e}")
            raise

        processing_time = time.time() - start_time

        # Track usage with correct method
        self._track_usage(texts, processing_time)

        # Create result
        result = EmbeddingResult(
            embeddings=all_embeddings,
            model_name=self.model_name,
            provider="huggingface",
            tokens_used=estimated_tokens,
            processing_time=processing_time,
            metadata={
                'batch_count': len(batches),
                'total_texts': len(texts),
                'batch_size': batch_size,
                'device': self.device,
                'embedding_dimension': len(all_embeddings[0]) if all_embeddings else 0,
                'normalize': normalize,
                'max_length': self.max_length,
                'model_dtype': str(self.model.dtype) if self.model else None,
                'request_params': {k: v for k, v in kwargs.items()}
            }
        )

        self.logger.info(f"Generated {len(all_embeddings)} embeddings in {processing_time:.2f}s on {self.device}")
        return result

    def embed_single(self, 
                    text: str,
                    normalize: bool = True,
                    **kwargs) -> EmbeddingResult:
        """
        Generate embedding for a single text using HuggingFace models.

        Args:
            text: Text to embed
            normalize: Whether to normalize embedding
            **kwargs: Additional parameters

        Returns:
            EmbeddingResult containing embedding and metadata
        """
        return self.embed_texts([text], batch_size=1, normalize=normalize, **kwargs)

    def _track_usage(self, texts: List[str], processing_time: float) -> Dict[str, Any]:
        """
        Track usage for local HuggingFace models (no API costs).

        Args:
            texts: List of processed texts
            processing_time: Time taken for processing

        Returns:
            Usage tracking information
        """
        estimated_tokens = self._estimate_tokens(texts)

        # For local models, we track usage but no costs
        usage_info = {
            'total_tokens': estimated_tokens,
            'total_cost': 0.0,  # Local models have no API cost
            'cost_per_1k_tokens': 0.0,
            'processing_time': processing_time,
            'texts_processed': len(texts),
            'model_type': 'local',
            'device': self.device
        }

        # Update cost tracker with correct method
        if self.cost_tracker:
            self.cost_tracker.track_api_call(
                provider="huggingface",
                model=self.model_name,
                endpoint="embed",
                tokens_used=estimated_tokens,
                duration_seconds=processing_time,
                success=True,
                metadata={
                    'texts_processed': len(texts),
                    'device': self.device,
                    'model_type': 'local'
                }
            )

        return usage_info

    def get_available_models(self) -> List[str]:
        """
        Get list of popular HuggingFace embedding models.

        Returns:
            List of model names
        """
        return [
            # BGE models
            "BAAI/bge-m3",
            "BAAI/bge-large-en-v1.5",
            "BAAI/bge-base-en-v1.5",
            "BAAI/bge-small-en-v1.5",

            # E5 models
            "intfloat/e5-mistral-7b-instruct",
            "intfloat/e5-large-v2",
            "intfloat/e5-base-v2",
            "intfloat/e5-small-v2",

            # GTE models
            "thenlper/gte-large",
            "thenlper/gte-base",
            "thenlper/gte-small",

            # Other popular models
            "sentence-transformers/all-mpnet-base-v2",
            "microsoft/DialoGPT-medium",
            "facebook/contriever"
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

        # Dimensions for popular HuggingFace models
        dimensions_map = {
            "BAAI/bge-m3": 1024,
            "BAAI/bge-large-en-v1.5": 1024,
            "BAAI/bge-base-en-v1.5": 768,
            "BAAI/bge-small-en-v1.5": 384,
            "intfloat/e5-mistral-7b-instruct": 4096,
            "intfloat/e5-large-v2": 1024,
            "intfloat/e5-base-v2": 768,
            "intfloat/e5-small-v2": 384,
            "thenlper/gte-large": 1024,
            "thenlper/gte-base": 768,
            "thenlper/gte-small": 384
        }

        return dimensions_map.get(model, 768)  # Default to 768

    def get_max_sequence_length(self) -> int:
        """
        Get maximum sequence length for current model.

        Returns:
            Maximum sequence length in tokens
        """
        return self.max_length

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for HuggingFace models."""
        return {
            'provider': 'huggingface',
            'max_tokens': 8192,  # Use a default value instead of self.get_max_sequence_length()
            'max_batch_size': 8,  # Conservative for large models
            'dimensions': self.get_model_dimensions(),
            'cost_per_1k_tokens': 0.0,  # Local models have no API cost
            'rate_limit_rpm': 1000,  # Reasonable limit for local processing
            'rate_limit_tpm': 10000,
            'supports_batching': True,
            'metadata': {
                'device': self.device,
                'max_sequence_length': 8192,  # Use default value
                'is_local_model': True,
                'supports_normalization': True,
                'model_type': self._get_model_type()
            }
        }

    def _get_model_type(self) -> str:
        """Determine the model type based on model name."""
        model_name_lower = self.model_name.lower()

        if 'bge' in model_name_lower:
            return 'bge'
        elif 'e5' in model_name_lower:
            return 'e5'
        elif 'gte' in model_name_lower:
            return 'gte'
        else:
            return 'generic'

    def _create_batches(self, texts: List[str]) -> List[List[str]]:
        """
        Create batches optimized for HuggingFace models.

        Args:
            texts: List of texts to batch

        Returns:
            List of text batches
        """
        # Use smaller batches for large models to avoid memory issues
        batch_size = min(self.config.max_batch_size, 16)

        batches = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batches.append(batch)

        return batches

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the current model.

        Returns:
            Dictionary containing model information
        """
        base_info = super().get_model_info()

        # Add HuggingFace specific info
        if self.model and self.tokenizer:
            base_info.update({
                'device': self.device,
                'max_sequence_length': self.max_length,
                'model_dtype': str(self.model.dtype),
                'tokenizer_type': type(self.tokenizer).__name__,
                'vocab_size': self.tokenizer.vocab_size if hasattr(self.tokenizer, 'vocab_size') else None,
                'model_type': self._get_model_type(),
                'num_parameters': sum(p.numel() for p in self.model.parameters()) if self.model else None
            })

        return base_info
