"""
PDF Document Loader for SmartRag.

This module encapsulates loading and parsing PDF files using LangChain's PyPDFLoader,
injecting standardized metadata required for accurate citation grounding.
"""

import logging
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

logger = logging.getLogger(__name__)


def load_pdf(file_path: str | Path) -> List[Document]:
    """
    Load a PDF document from disk, extract text page by page, and inject standardized metadata.
    
    Args:
        file_path (str | Path): Absolute or relative path to the PDF file.
        
    Returns:
        List[Document]: A list of LangChain Document objects, each representing a page with metadata.
        
    Raises:
        FileNotFoundError: If the specified PDF file does not exist.
        ValueError: If the PDF file is empty or corrupted.
    """
    path_obj = Path(file_path).resolve()
    if not path_obj.exists():
        raise FileNotFoundError(f"PDF file not found at path: {path_obj}")
        
    try:
        logger.info(f"Loading PDF file: {path_obj.name}")
        loader = PyPDFLoader(str(path_obj))
        raw_docs = loader.load()
        
        if not raw_docs:
            logger.warning(f"No text content extracted from PDF: {path_obj.name}")
            return []
            
        standardized_docs: List[Document] = []
        for doc in raw_docs:
            # PyPDFLoader returns 0-indexed page numbers in doc.metadata['page']
            raw_page = doc.metadata.get("page", 0)
            page_num = raw_page + 1 if isinstance(raw_page, int) else 1
            
            doc_name = path_obj.name
            
            # Enrich and standardize metadata for citation display
            doc.metadata.update({
                "source": doc_name,
                "document_name": doc_name,
                "page_number": page_num,
                "file_path": str(path_obj)
            })
            standardized_docs.append(doc)
            
        logger.info(f"Successfully loaded {len(standardized_docs)} pages from {path_obj.name}")
        return standardized_docs
        
    except Exception as e:
        logger.error(f"Error loading corrupted or unreadable PDF '{path_obj.name}': {str(e)}")
        raise ValueError(f"Failed to parse PDF document '{path_obj.name}'. Reason: {str(e)}") from e
