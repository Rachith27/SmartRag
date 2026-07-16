"""
Document Chunking and Splitting Module for SmartRag.

Uses LangChain's RecursiveCharacterTextSplitter to split loaded documents into
contextually dense chunks (`CHUNK_SIZE=800`, `CHUNK_OVERLAP=150`), preserving all
source and page metadata while assigning unique chunk identifiers.
"""

import logging
from typing import List, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import rag_config

logger = logging.getLogger(__name__)


class DocumentSplitter:
    """
    Wrapper class around RecursiveCharacterTextSplitter for consistent chunking
    and metadata enhancement across all ingestion workflows.
    """
    
    def __init__(
        self, 
        chunk_size: Optional[int] = None, 
        chunk_overlap: Optional[int] = None
    ) -> None:
        """
        Initialize the splitter with specified or configured parameters.
        
        Args:
            chunk_size (Optional[int]): Maximum character size per chunk. Defaults to config (800).
            chunk_overlap (Optional[int]): Overlap between adjacent chunks. Defaults to config (150).
        """
        self.chunk_size = chunk_size if chunk_size is not None else rag_config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap if chunk_overlap is not None else rag_config.CHUNK_OVERLAP
        
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(f"Chunk overlap ({self.chunk_overlap}) must be smaller than chunk size ({self.chunk_size}).")
            
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
            keep_separator=False
        )
        logger.info(f"Initialized DocumentSplitter(chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap})")

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split a list of Document objects into smaller chunks while preserving
        and enhancing metadata (`chunk_index`).
        
        Args:
            documents (List[Document]): List of raw loaded Document objects.
            
        Returns:
            List[Document]: List of split Document chunks.
        """
        if not documents:
            logger.warning("No documents provided for splitting.")
            return []
            
        try:
            chunks = self.splitter.split_documents(documents)
            
            # Enrich every chunk with an explicit chunk index for tracking and UI display
            for idx, chunk in enumerate(chunks):
                chunk.metadata["chunk_index"] = idx
                # Ensure core citation keys exist even if a custom loader was used
                if "document_name" not in chunk.metadata:
                    chunk.metadata["document_name"] = chunk.metadata.get("source", "Unknown Document")
                if "page_number" not in chunk.metadata:
                    chunk.metadata["page_number"] = 1
                    
            logger.info(f"Split {len(documents)} document(s) into {len(chunks)} chunks.")
            return chunks
            
        except Exception as e:
            logger.error(f"Error during document splitting: {str(e)}")
            raise RuntimeError(f"Document splitting failed: {str(e)}") from e


def split_documents(
    documents: List[Document], 
    chunk_size: Optional[int] = None, 
    chunk_overlap: Optional[int] = None
) -> List[Document]:
    """
    Functional convenience helper to split documents without instantiating DocumentSplitter manually.
    
    Args:
        documents (List[Document]): Documents to split.
        chunk_size (Optional[int]): Custom chunk size.
        chunk_overlap (Optional[int]): Custom chunk overlap.
        
    Returns:
        List[Document]: Split Document chunks with standardized metadata.
    """
    splitter_instance = DocumentSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter_instance.split_documents(documents)
