"""
FAISS Vector Database Manager for SmartRag.

Manages creation, local persistence (`vector_store/faiss_index`), loading,
and similarity searching over FAISS vector indices.
"""

import shutil
import logging
from pathlib import Path
from typing import List, Tuple, Optional
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS

from config import FAISS_INDEX_PATH, rag_config
from embeddings import get_embedding_model

logger = logging.getLogger(__name__)


class FAISSDatabaseManager:
    """
    Manager class around LangChain's FAISS vector store, handling local disk saving,
    loading with deserialization safety, and top-k similarity searches.
    """
    
    def __init__(self, index_path: Optional[Path] = None) -> None:
        """
        Initialize the FAISS database manager.
        
        Args:
            index_path (Optional[Path]): Directory where FAISS index (.faiss/.pkl) files are stored.
        """
        self.index_path = index_path or FAISS_INDEX_PATH
        self.vector_store: Optional[FAISS] = None

    def _get_embeddings(self, embeddings: Optional[Embeddings] = None) -> Embeddings:
        """Helper to get provided embeddings or fall back to global cached model."""
        return embeddings or get_embedding_model()

    def add_documents(
        self, 
        documents: List[Document], 
        embeddings: Optional[Embeddings] = None
    ) -> FAISS:
        """
        Create a new index or append documents to an existing FAISS index, then persist to disk.
        
        Args:
            documents (List[Document]): Chunked Document objects with metadata.
            embeddings (Optional[Embeddings]): Embedding model instance.
            
        Returns:
            FAISS: The updated FAISS vector store instance.
            
        Raises:
            ValueError: If no documents are provided.
        """
        if not documents:
            raise ValueError("No documents provided to add to FAISS database.")
            
        embed_model = self._get_embeddings(embeddings)
        
        # Check if an index already exists on disk
        existing_store = self.load_index(embeddings=embed_model)
        
        try:
            if existing_store is not None:
                logger.info(f"Adding {len(documents)} chunks to existing FAISS index at {self.index_path}...")
                existing_store.add_documents(documents)
                self.vector_store = existing_store
            else:
                logger.info(f"Creating new FAISS index from {len(documents)} chunks...")
                self.vector_store = FAISS.from_documents(documents, embed_model)
                
            # Persist to local disk
            self.index_path.mkdir(parents=True, exist_ok=True)
            self.vector_store.save_local(str(self.index_path))
            logger.info(f"Successfully saved FAISS index to {self.index_path}")
            return self.vector_store
            
        except Exception as e:
            logger.error(f"Error while adding documents to FAISS: {str(e)}")
            raise RuntimeError(f"Failed to index documents into FAISS: {str(e)}") from e

    def load_index(self, embeddings: Optional[Embeddings] = None) -> Optional[FAISS]:
        """
        Load an existing FAISS index from disk if present.
        
        Args:
            embeddings (Optional[Embeddings]): Embedding model used to create the index.
            
        Returns:
            Optional[FAISS]: Loaded vector store instance, or None if no index exists.
        """
        index_file = self.index_path / "index.faiss"
        pkl_file = self.index_path / "index.pkl"
        
        if not (index_file.exists() and pkl_file.exists()):
            logger.debug(f"No existing FAISS index found at {self.index_path}")
            return None
            
        try:
            embed_model = self._get_embeddings(embeddings)
            logger.info(f"Loading FAISS index from {self.index_path}...")
            # allow_dangerous_deserialization is required for loading local pickle files securely in LangChain >= 0.2
            self.vector_store = FAISS.load_local(
                str(self.index_path), 
                embed_model, 
                allow_dangerous_deserialization=True
            )
            return self.vector_store
        except Exception as e:
            logger.error(f"Failed to load FAISS index from {self.index_path}: {str(e)}")
            return None

    def similarity_search_with_score(
        self, 
        query: str, 
        top_k: Optional[int] = None, 
        embeddings: Optional[Embeddings] = None
    ) -> List[Tuple[Document, float]]:
        """
        Perform vector similarity search for a user query.
        
        Args:
            query (str): Natural language question.
            top_k (Optional[int]): Number of top relevant chunks to retrieve. Defaults to config (4).
            embeddings (Optional[Embeddings]): Embedding model.
            
        Returns:
            List[Tuple[Document, float]]: List of (Document, similarity_score) tuples.
            
        Raises:
            RuntimeError: If no index is loaded or available on disk.
        """
        k = top_k if top_k is not None else rag_config.TOP_K
        
        if self.vector_store is None:
            self.load_index(embeddings)
            
        if self.vector_store is None:
            logger.warning("Attempted similarity search, but FAISS database is empty/unindexed.")
            return []
            
        try:
            logger.info(f"Running FAISS similarity search for query='{query}' (top_k={k})...")
            # FAISS similarity_search_with_score returns L2 distance (lower = more similar)
            # Or cosine distance depending on normalization
            results = self.vector_store.similarity_search_with_score(query, k=k)
            
            # Ensure similarity score is standardized across chunks
            formatted_results: List[Tuple[Document, float]] = []
            for doc, raw_score in results:
                # Convert distance into a normalized similarity percentage or raw score representation
                formatted_results.append((doc, float(raw_score)))
                
            return formatted_results
        except Exception as e:
            logger.error(f"Error during FAISS similarity search: {str(e)}")
            raise RuntimeError(f"Similarity search failed: {str(e)}") from e

    def clear_database(self) -> bool:
        """
        Delete the persistent local FAISS index from disk and clear in-memory store.
        
        Returns:
            bool: True if cleared or already empty, False on deletion failure.
        """
        self.vector_store = None
        if self.index_path.exists():
            try:
                shutil.rmtree(self.index_path)
                self.index_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Cleared FAISS database files at {self.index_path}")
                return True
            except Exception as e:
                logger.error(f"Failed to delete FAISS index directory: {str(e)}")
                return False
        return True


# Global instantiated manager
faiss_manager = FAISSDatabaseManager()
