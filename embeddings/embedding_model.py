"""
Embedding Model Module for SmartRag.

Provides a cached singleton wrapper around HuggingFace's sentence-transformers
(default: `all-MiniLM-L6-v2`) to generate dense vector embeddings without
repeated loading overhead.
"""

import logging
from typing import Optional, Dict, Any
from langchain_core.embeddings import Embeddings

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    # Fallback for older langchain_community installations if needed
    from langchain_community.embeddings import HuggingFaceEmbeddings

from config import rag_config

logger = logging.getLogger(__name__)

# Global cache for instantiated embedding models to prevent duplicate memory loading
_EMBEDDING_MODEL_CACHE: Dict[str, Embeddings] = {}


def get_embedding_model(
    model_name: Optional[str] = None, 
    device: str = "cpu",
    normalize_embeddings: bool = True
) -> Embeddings:
    """
    Retrieve or initialize a cached HuggingFace embedding model instance.
    
    Args:
        model_name (Optional[str]): HuggingFace model identifier. Defaults to `all-MiniLM-L6-v2`.
        device (str): Compute device ('cpu' or 'cuda'). Defaults to 'cpu'.
        normalize_embeddings (bool): Whether to L2-normalize vectors for cosine similarity. Defaults to True.
        
    Returns:
        Embeddings: LangChain Embeddings instance.
        
    Raises:
        RuntimeError: If the embedding model fails to download or initialize.
    """
    target_model = model_name if model_name else rag_config.EMBEDDING_MODEL_NAME
    cache_key = f"{target_model}_{device}_{normalize_embeddings}"
    
    if cache_key in _EMBEDDING_MODEL_CACHE:
        logger.debug(f"Returning cached embedding model: '{target_model}' on device '{device}'")
        return _EMBEDDING_MODEL_CACHE[cache_key]
        
    try:
        logger.info(f"Loading embedding model '{target_model}' on device '{device}' (normalize={normalize_embeddings})...")
        model_kwargs = {"device": device}
        encode_kwargs = {"normalize_embeddings": normalize_embeddings}
        
        embeddings = HuggingFaceEmbeddings(
            model_name=target_model,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        
        _EMBEDDING_MODEL_CACHE[cache_key] = embeddings
        logger.info(f"Successfully loaded and cached embedding model: '{target_model}'")
        return embeddings
        
    except Exception as e:
        logger.error(f"Failed to initialize embedding model '{target_model}': {str(e)}")
        raise RuntimeError(
            f"Embedding initialization error for '{target_model}'. Check internet connection/PyTorch installation: {str(e)}"
        ) from e


def clear_embedding_cache() -> None:
    """Clear cached embedding models from RAM to free system memory."""
    global _EMBEDDING_MODEL_CACHE
    _EMBEDDING_MODEL_CACHE.clear()
    logger.info("Embedding model cache cleared.")
