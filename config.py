"""
Configuration module for SmartRag.

This module centralizes all system settings, API keys, path constants,
and RAG hyper-parameters (chunk size, overlap, top_k, models, temperatures).
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()


# =====================================================================
# PATH CONSTANTS
# =====================================================================
BASE_DIR: Path = Path(__file__).resolve().parent
ASSETS_DIR: Path = BASE_DIR / "assets"
UPLOADS_DIR: Path = BASE_DIR / "uploads"
VECTOR_STORE_DIR: Path = BASE_DIR / "vector_store"

# Local vector store paths
FAISS_INDEX_PATH: Path = VECTOR_STORE_DIR / "faiss_index"
CHROMA_DB_PATH: Path = VECTOR_STORE_DIR / "chroma_db"

# Ensure runtime directories exist
for folder in [ASSETS_DIR, UPLOADS_DIR, VECTOR_STORE_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


# =====================================================================
# RAG PIPELINE CONFIGURATIONS
# =====================================================================
@dataclass(frozen=True)
class RAGConfig:
    """Hyper-parameters and default settings for the RAG pipeline."""
    
    # Document Chunking
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 800))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 150))
    
    # Vector Retrieval
    TOP_K: int = int(os.getenv("TOP_K", 4))
    
    # Embeddings
    EMBEDDING_MODEL_NAME: str = os.getenv(
        "DEFAULT_EMBEDDING_MODEL", 
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # LLM Settings
    DEFAULT_LLM_PROVIDER: str = os.getenv("DEFAULT_LLM_PROVIDER", "openai").lower()
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", 0.0))  # Strict 0.0 to prevent hallucination


@dataclass(frozen=True)
class LLMProviderConfig:
    """Supported LLM providers and their default models."""
    
    OPENAI_DEFAULT_MODEL: str = "gpt-4o-mini"
    GEMINI_DEFAULT_MODEL: str = "gemini-2.5-flash"
    OPENROUTER_DEFAULT_MODEL: str = "openai/gpt-4o-mini"
    
    @classmethod
    def get_supported_providers(cls) -> list[str]:
        """Returns a list of supported LLM provider names."""
        return ["openai", "gemini", "openrouter"]
    
    @classmethod
    def get_default_model(cls, provider: str) -> str:
        """Get the default model ID for a given provider."""
        provider_clean = provider.strip().lower()
        if provider_clean == "openai":
            return cls.OPENAI_DEFAULT_MODEL
        elif provider_clean == "gemini":
            return cls.GEMINI_DEFAULT_MODEL
        elif provider_clean == "openrouter":
            return cls.OPENROUTER_DEFAULT_MODEL
        else:
            raise ValueError(f"Unsupported LLM provider: '{provider}'. Supported: {cls.get_supported_providers()}")


# =====================================================================
# API KEY VALIDATION HELPERS
# =====================================================================
def get_api_key(provider: str) -> Optional[str]:
    """
    Retrieve the API key for the requested provider from environment variables.
    
    Args:
        provider (str): The name of the LLM provider ('openai', 'gemini', 'openrouter').
        
    Returns:
        Optional[str]: The API key if found, or None.
    """
    provider_clean = provider.strip().lower()
    if provider_clean == "openai":
        return os.getenv("OPENAI_API_KEY")
    elif provider_clean == "gemini":
        return os.getenv("GOOGLE_API_KEY")
    elif provider_clean == "openrouter":
        return os.getenv("OPENROUTER_API_KEY")
    return None


def validate_api_key(provider: str, override_key: Optional[str] = None) -> bool:
    """
    Check if a valid API key is available for the specified provider.
    
    Args:
        provider (str): Provider name ('openai', 'gemini', 'openrouter').
        override_key (Optional[str]): A key passed directly from the UI sidebar.
        
    Returns:
        bool: True if an API key is present, False otherwise.
    """
    if override_key and override_key.strip():
        return True
    key = get_api_key(provider)
    return bool(key and key.strip())


# Global instantiated config instance for convenient imports
rag_config = RAGConfig()
llm_config = LLMProviderConfig()
