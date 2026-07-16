"""
Semantic Retriever Module for SmartRag.

Encapsulates similarity search over the active vector database (FAISS or Chroma),
retrieving Top K relevant chunks and attaching normalized similarity scores for citations.
"""

import logging
from typing import List, Tuple, Optional
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from config import rag_config
from database import get_vector_database

logger = logging.getLogger(__name__)


class SmartRetriever:
    """
    Retriever engine that queries the selected vector store for top-k context chunks,
    ensuring standardized metadata and similarity scores are attached to each result.
    """
    
    def __init__(self, db_type: str = "faiss", top_k: Optional[int] = None) -> None:
        """
        Initialize the retriever.
        
        Args:
            db_type (str): Active vector database engine ('faiss' or 'chroma').
            top_k (Optional[int]): Number of chunks to retrieve. Defaults to config (4).
        """
        self.db_type = db_type
        self.top_k = top_k if top_k is not None else rag_config.TOP_K
        self.db_manager = get_vector_database(self.db_type)

    def set_db_type(self, db_type: str) -> None:
        """Switch active database backend dynamically."""
        self.db_type = db_type
        self.db_manager = get_vector_database(self.db_type)
        logger.info(f"Retriever switched to database backend: '{db_type}'")

    def retrieve(
        self, 
        query: str, 
        top_k: Optional[int] = None, 
        embeddings: Optional[Embeddings] = None
    ) -> List[Document]:
        """
        Retrieve the top-k most relevant Document chunks for a query.
        
        Args:
            query (str): Natural language search query.
            top_k (Optional[int]): Override default top_k.
            embeddings (Optional[Embeddings]): Override default embedding model.
            
        Returns:
            List[Document]: List of retrieved Document objects with embedded `similarity_score` in metadata.
        """
        if not query or not query.strip():
            logger.warning("Empty query passed to retriever.")
            return []
            
        k = top_k if top_k is not None else self.top_k
        logger.info(f"Retrieving top {k} chunks from '{self.db_type}' for query: '{query[:50]}...'")
        
        try:
            raw_results: List[Tuple[Document, float]] = self.db_manager.similarity_search_with_score(
                query=query, 
                top_k=k, 
                embeddings=embeddings
            )
            
            if not raw_results:
                logger.warning("No documents returned by similarity search.")
                return []
                
            enriched_docs: List[Document] = []
            for rank, (doc, raw_score) in enumerate(raw_results, start=1):
                # Standardize similarity score formatting
                # For FAISS L2 distance, distance >= 0. We convert to a clean bounded metric or keep raw
                # Using 1 / (1 + distance) maps [0, inf) to [1, 0)
                if self.db_type == "faiss":
                    normalized_score = round(1.0 / (1.0 + max(0.0, float(raw_score))), 4)
                else:
                    # Chroma usually returns cosine or L2 distance depending on collection setup
                    normalized_score = round(float(raw_score), 4)
                    
                doc.metadata["similarity_score"] = normalized_score
                doc.metadata["raw_distance"] = round(float(raw_score), 4)
                doc.metadata["retrieval_rank"] = rank
                
                # Verify required citation fields are present
                if "document_name" not in doc.metadata:
                    doc.metadata["document_name"] = doc.metadata.get("source", "Unknown Document")
                if "page_number" not in doc.metadata:
                    doc.metadata["page_number"] = 1
                    
                enriched_docs.append(doc)
                
            logger.info(f"Retriever returned {len(enriched_docs)} grounded chunks.")
            return enriched_docs
            
        except Exception as e:
            logger.error(f"Error during retrieval: {str(e)}")
            raise RuntimeError(f"Failed to retrieve context chunks: {str(e)}") from e


def retrieve_chunks(
    query: str, 
    db_type: str = "faiss", 
    top_k: Optional[int] = None
) -> List[Document]:
    """
    Functional convenience wrapper to retrieve top-k chunks without manually instantiating SmartRetriever.
    
    Args:
        query (str): Question to search.
        db_type (str): Database backend ('faiss' or 'chroma').
        top_k (Optional[int]): Number of chunks.
        
    Returns:
        List[Document]: Enriched Document chunks.
    """
    retriever_instance = SmartRetriever(db_type=db_type, top_k=top_k)
    return retriever_instance.retrieve(query)
