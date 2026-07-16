"""
Embeddings package for SmartRag.

Provides `get_embedding_model` for cached initialization of sentence-transformers
vector embeddings (`all-MiniLM-L6-v2`).
"""

from embeddings.embedding_model import get_embedding_model, clear_embedding_cache

__all__ = ["get_embedding_model", "clear_embedding_cache"]
