"""
ChromaDB Vector Database Manager for SmartRag.

Provides optional/alternative persistent storage (`vector_store/chroma_db`)
using LangChain's Chroma integration.
"""

import shutil
import logging
from pathlib import Path
from typing import List, Tuple, Optional
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import Chroma

from config import CHROMA_DB_PATH, rag_config
from embeddings import get_embedding_model

logger = logging.getLogger(__name__)


class ChromaDatabaseManager:
    """
    Manager class around LangChain's Chroma vector store for persistent indexing
    and similarity search with metadata filtering.
    """
    
    def __init__(self, persist_directory: Optional[Path] = None, collection_name: str = "smartrag_collection") -> None:
        """
        Initialize the ChromaDB manager.
        
        Args:
            persist_directory (Optional[Path]): Folder path to persist Chroma SQLite database files.
            collection_name (str): Chroma collection identifier.
        """
        self.persist_directory = persist_directory or CHROMA_DB_PATH
        self.collection_name = collection_name
        self.vector_store: Optional[Chroma] = None

    def _get_embeddings(self, embeddings: Optional[Embeddings] = None) -> Embeddings:
        """Helper to get provided embeddings or fall back to global cached model."""
        return embeddings or get_embedding_model()

    def add_documents(
        self, 
        documents: List[Document], 
        embeddings: Optional[Embeddings] = None
    ) -> Chroma:
        """
        Add document chunks into ChromaDB and persist to local storage.
        
        Args:
            documents (List[Document]): Chunked Document objects.
            embeddings (Optional[Embeddings]): Embedding model.
            
        Returns:
            Chroma: Updated Chroma vector store instance.
        """
        if not documents:
            raise ValueError("No documents provided to add to Chroma database.")
            
        embed_model = self._get_embeddings(embeddings)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        try:
            logger.info(f"Adding {len(documents)} chunks to ChromaDB at {self.persist_directory}...")
            # Chroma.from_documents automatically persists in modern LangChain when persist_directory is provided
            if self.vector_store is None:
                self.vector_store = Chroma.from_documents(
                    documents=documents,
                    embedding=embed_model,
                    collection_name=self.collection_name,
                    persist_directory=str(self.persist_directory)
                )
            else:
                self.vector_store.add_documents(documents)
                
            logger.info("Successfully indexed documents into ChromaDB.")
            return self.vector_store
            
        except Exception as e:
            logger.error(f"Error while adding documents to ChromaDB: {str(e)}")
            raise RuntimeError(f"Failed to index documents into ChromaDB: {str(e)}") from e

    def load_index(self, embeddings: Optional[Embeddings] = None) -> Optional[Chroma]:
        """
        Load an existing Chroma database from disk if present.
        
        Args:
            embeddings (Optional[Embeddings]): Embedding model.
            
        Returns:
            Optional[Chroma]: Loaded vector store or None.
        """
        if not self.persist_directory.exists() or not any(self.persist_directory.iterdir()):
            logger.debug(f"No existing ChromaDB files at {self.persist_directory}")
            return None
            
        try:
            embed_model = self._get_embeddings(embeddings)
            logger.info(f"Loading ChromaDB collection '{self.collection_name}' from {self.persist_directory}...")
            self.vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=embed_model,
                persist_directory=str(self.persist_directory)
            )
            return self.vector_store
        except Exception as e:
            logger.error(f"Failed to load ChromaDB collection: {str(e)}")
            return None

    def similarity_search_with_score(
        self, 
        query: str, 
        top_k: Optional[int] = None, 
        embeddings: Optional[Embeddings] = None
    ) -> List[Tuple[Document, float]]:
        """
        Perform top-k similarity search over Chroma database.
        
        Args:
            query (str): Natural language question.
            top_k (Optional[int]): Number of chunks to retrieve.
            embeddings (Optional[Embeddings]): Embedding model.
            
        Returns:
            List[Tuple[Document, float]]: Top matching chunks and scores.
        """
        k = top_k if top_k is not None else rag_config.TOP_K
        
        if self.vector_store is None:
            self.load_index(embeddings)
            
        if self.vector_store is None:
            logger.warning("Attempted similarity search, but ChromaDB is empty.")
            return []
            
        try:
            results = self.vector_store.similarity_search_with_score(query, k=k)
            return [(doc, float(score)) for doc, score in results]
        except Exception as e:
            logger.error(f"Error during ChromaDB similarity search: {str(e)}")
            raise RuntimeError(f"Chroma similarity search failed: {str(e)}") from e

    def clear_database(self) -> bool:
        """
        Delete persistent local Chroma database files.
        
        Returns:
            bool: True on success.
        """
        self.vector_store = None
        if self.persist_directory.exists():
            try:
                shutil.rmtree(self.persist_directory)
                self.persist_directory.mkdir(parents=True, exist_ok=True)
                logger.info(f"Cleared ChromaDB database at {self.persist_directory}")
                return True
            except Exception as e:
                logger.error(f"Failed to delete ChromaDB directory: {str(e)}")
                return False
        return True


# Global instantiated manager
chroma_manager = ChromaDatabaseManager()
