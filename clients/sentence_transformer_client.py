"""
Sentence Transformer Client - Client for Sentence-BERT models
"""

import os
import time
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

from .base_embedding_client import BaseEmbeddingClient, EmbeddingResult
from ..utils.embedding_cost_tracker import APICallTracker


class SentenceTransformerClient(BaseEmbeddingClient):
    """
    Client for Sentence Transformer models.
    
    Supports popular models like:
    - all-MiniLM-L6-v2
    - all-mpnet-base-v2
    - all-distilroberta-v1
    - paraphrase-multilingual-MiniLM-L12-v2
    - multi-qa-MiniLM-L6-cos-v1
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", **kwargs):
        """
        Initialize Sentence Transformer client.
        
        Args:
            model_name: Sentence Transformer model name
            **kwargs: Additional arguments passed to base class
        """
        super().__init__(model_name=model_name, **kwargs)
        self.device = self._get_device()
        self.model = None
    
    def _get_api_key_from_env(self) -> Optional[str]:
        """Sentence Transformers don't require API keys (local models)."""
        return None
    
    def _get_device(self) -> str:
        """Determine the best device for model inference."""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"  # Apple Silicon
        else:
            return "cpu"
    
    def _load_model(self):
        """Load Sentence Transformer model."""
        try:
            self.logger.info(f"Loading Sentence Transformer model: {self.model_name} on {self.device}")
            
            # Load model with device specification
            self.model = SentenceTransformer(self.model_name, device=self.device)
            
            # Get actual dimensions
            test_embedding = self.model.encode("test", convert_to_numpy=True)
            actual_dimensions = len(test_embedding)
            
            if actual_dimensions != self.config.dimensions:
                self.logger.info(f"Updated dimensions from {self.config.dimensions} to {actual_dimensions}")
                self.config.dimensions = actual_dimensions
            
            self.logger.info(f"Sentence Transformer model loaded: {self.model_name} ({actual_dimensions}D)")
            
        except Exception as e:
            self.logger.error(f"Failed to load Sentence Transformer model: {e}")
            raise
    
    def _estimate_tokens(self, texts: List[str]) -> int:
        """
        Estimate tokens for Sentence Transformer models.
        Since these are local models, token estimation is less critical.
        """
        # Simple estimation based on words
        total_words = sum(len(text.split()) for text in texts)
        return max(1, int(total_words * 0.75))
    
    def embed_texts(self, 
                   texts: List[str],
                   batch_size: Optional[int] = None,
                   show_progress_bar: bool = False,
                   convert_to_numpy: bool = True,
                   normalize_embeddings: bool = False,
                   **kwargs) -> EmbeddingResult:
        """
        Generate embeddings for multiple texts using Sentence Transformers.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing (None for auto)
            show_progress_bar: Whether to show progress bar
            convert_to_numpy: Convert to numpy arrays
            normalize_embeddings: Whether to normalize embeddings
            **kwargs: Additional parameters
            
        Returns:
            EmbeddingResult containing embeddings and metadata
        """
        self.ensure_initialized()
        texts = self._validate_texts(texts)
        
        start_time = time.time()
        
        # Use provided batch size or config default
        if batch_size is None:
            batch_size = min(self.config.max_batch_size, 32)  # Reasonable default for local models
        
        estimated_tokens = self._estimate_tokens(texts)
        
        # Track the operation (no actual API cost for local models)
        with APICallTracker(
            cost_tracker=self.cost_tracker,
            provider="sentence-transformers",
            model=self.model_name,
            endpoint="encode"
        ) as tracker:
            try:
                # Generate embeddings
                embeddings = self.model.encode(
                    texts,
                    batch_size=batch_size,
                    show_progress_bar=show_progress_bar,
                    convert_to_numpy=convert_to_numpy,
                    normalize_embeddings=normalize_embeddings,
                    **kwargs
                )
                
                # Convert to list of lists for consistency
                if convert_to_numpy:
                    embeddings_list = embeddings.tolist()
                else:
                    embeddings_list = [emb.tolist() if hasattr(emb, 'tolist') else emb for emb in embeddings]
                
                # Track usage (no actual tokens/cost for local models)
                tracker.set_tokens_used(estimated_tokens)
                tracker.add_metadata('batch_size', batch_size)
                tracker.add_metadata('device', self.device)
                tracker.add_metadata('normalize_embeddings', normalize_embeddings)
                
            except Exception as e:
                self.logger.error(f"Sentence Transformer encoding error: {e}")
                raise
        
        processing_time = time.time() - start_time
        
        # Create result
        result = EmbeddingResult(
            embeddings=embeddings_list,
            model_name=self.model_name,
            provider="sentence-transformers",
            tokens_used=estimated_tokens,
            processing_time=processing_time,
            metadata={
                'total_texts': len(texts),
                'batch_size': batch_size,
                'device': self.device,
                'embedding_dimension': len(embeddings_list[0]) if embeddings_list else 0,
                'normalize_embeddings': normalize_embeddings,
                'model_max_seq_length': getattr(self.model, 'max_seq_length', None),
                'request_params': {k: v for k, v in kwargs.items()}
            }
        )
        
        self.logger.info(f"Generated {len(embeddings_list)} embeddings in {processing_time:.2f}s on {self.device}")
        return result
    
    def embed_single(self, 
                    text: str,
                    convert_to_numpy: bool = True,
                    normalize_embeddings: bool = False,
                    **kwargs) -> EmbeddingResult:
        """
        Generate embedding for a single text using Sentence Transformers.
        
        Args:
            text: Text to embed
            convert_to_numpy: Convert to numpy array
            normalize_embeddings: Whether to normalize embedding
            **kwargs: Additional parameters
            
        Returns:
            EmbeddingResult containing embedding and metadata
        """
        return self.embed_texts(
            [text], 
            batch_size=1,
            convert_to_numpy=convert_to_numpy,
            normalize_embeddings=normalize_embeddings,
            **kwargs
        )
    
    def get_available_models(self) -> List[str]:
        """
        Get list of popular Sentence Transformer models.
        
        Returns:
            List of model names
        """
        return [
            # General purpose models
            "all-MiniLM-L6-v2",
            "all-mpnet-base-v2",
            "all-distilroberta-v1",
            
            # Multilingual models
            "paraphrase-multilingual-MiniLM-L12-v2",
            "paraphrase-multilingual-mpnet-base-v2",
            
            # Specialized models
            "multi-qa-MiniLM-L6-cos-v1",
            "multi-qa-mpnet-base-cos-v1",
            "msmarco-distilbert-base-tas-b",
            
            # Code models
            "flax-sentence-embeddings/all_datasets_v4_MiniLM-L6",
            
            # Large models
            "sentence-transformers/gtr-t5-large",
            "sentence-transformers/gtr-t5-xl"
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
        
        # Common dimensions for popular models
        dimensions_map = {
            "all-MiniLM-L6-v2": 384,
            "all-mpnet-base-v2": 768,
            "all-distilroberta-v1": 768,
            "paraphrase-multilingual-MiniLM-L12-v2": 384,
            "paraphrase-multilingual-mpnet-base-v2": 768,
            "multi-qa-MiniLM-L6-cos-v1": 384,
            "multi-qa-mpnet-base-cos-v1": 768,
            "msmarco-distilbert-base-tas-b": 768,
            "sentence-transformers/gtr-t5-large": 768,
            "sentence-transformers/gtr-t5-xl": 768
        }
        
        return dimensions_map.get(model, 768)  # Default to 768
    
    def get_max_sequence_length(self) -> int:
        """
        Get maximum sequence length for current model.
        
        Returns:
            Maximum sequence length in tokens
        """
        if self.model and hasattr(self.model, 'max_seq_length'):
            return self.model.max_seq_length
        
        # Default max lengths for common models
        max_length_map = {
            "all-MiniLM-L6-v2": 256,
            "all-mpnet-base-v2": 384,
            "all-distilroberta-v1": 512,
            "paraphrase-multilingual-MiniLM-L12-v2": 128,
            "multi-qa-MiniLM-L6-cos-v1": 512,
            "msmarco-distilbert-base-tas-b": 512
        }
        
        return max_length_map.get(self.model_name, 512)
    
    def similarity(self, embeddings1: List[List[float]], embeddings2: List[List[float]]) -> np.ndarray:
        """
        Compute cosine similarity between two sets of embeddings.
        
        Args:
            embeddings1: First set of embeddings
            embeddings2: Second set of embeddings
            
        Returns:
            Similarity matrix
        """
        from sentence_transformers.util import cos_sim
        
        emb1 = torch.tensor(embeddings1)
        emb2 = torch.tensor(embeddings2)
        
        return cos_sim(emb1, emb2).numpy()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for Sentence Transformer models."""
        return {
            'provider': 'sentence-transformers',
            'max_tokens': self.get_max_sequence_length(),
            'max_batch_size': 32,  # Reasonable batch size for local processing
            'dimensions': self.get_model_dimensions(),
            'cost_per_1k_tokens': 0.0,  # Local models have no API cost
            'rate_limit_rpm': 10000,  # High limit for local models
            'rate_limit_tpm': 1000000,  # High limit for local models
            'supports_batching': True,
            'metadata': {
                'device': self.device,
                'max_sequence_length': self.get_max_sequence_length(),
                'is_local_model': True,
                'supports_similarity': True
            }
        }
    
    def _create_batches(self, texts: List[str]) -> List[List[str]]:
        """
        Create batches optimized for local processing.
        
        Args:
            texts: List of texts to batch
            
        Returns:
            List of text batches
        """
        # For local models, we can use larger batches limited by memory
        batch_size = min(self.config.max_batch_size, 64)
        
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
        
        # Add Sentence Transformer specific info
        if self.model:
            base_info.update({
                'device': self.device,
                'max_seq_length': getattr(self.model, 'max_seq_length', None),
                'tokenizer': str(type(self.model.tokenizer).__name__) if hasattr(self.model, 'tokenizer') else None,
                'pooling_mode': str(self.model._modules.get('1', 'Unknown')) if hasattr(self.model, '_modules') else None
            })
        
        return base_info
