"""
Embedding Model Clients Package

This package provides client implementations for various embedding model providers:
- OpenAI Embeddings
- Cohere Embeddings  
- Sentence Transformers
- HuggingFace Models
- Jina Embeddings
- Azure OpenAI Embeddings

All clients extend from BaseEmbeddingClient and provide consistent interfaces.
"""

from clients.base_embedding_client import BaseEmbeddingClient
from clients.openai_embedding_client import OpenAIEmbeddingClient
from clients.cohere_embedding_client import CohereEmbeddingClient
from clients.sentence_transformer_client import SentenceTransformerClient
from clients.huggingface_embedding_client import HuggingFaceEmbeddingClient
from clients.jina_embedding_client import JinaEmbeddingClient
from clients.azure_embedding_client import AzureEmbeddingClient

__all__ = [
    'BaseEmbeddingClient',
    'OpenAIEmbeddingClient',
    'CohereEmbeddingClient', 
    'SentenceTransformerClient',
    'HuggingFaceEmbeddingClient',
    'JinaEmbeddingClient',
    'AzureEmbeddingClient'
]

# Client registry for factory pattern
CLIENT_REGISTRY = {
    'openai': OpenAIEmbeddingClient,
    'cohere': CohereEmbeddingClient,
    'sentence-transformers': SentenceTransformerClient,
    'huggingface': HuggingFaceEmbeddingClient,
    'jina': JinaEmbeddingClient,
    'azure': AzureEmbeddingClient
}

def get_client_class(provider: str):
    """Get client class by provider name."""
    return CLIENT_REGISTRY.get(provider.lower())

def list_available_providers():
    """List all available embedding providers."""
    return list(CLIENT_REGISTRY.keys())
