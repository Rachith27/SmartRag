"""
Text and Markdown Document Loader for SmartRag.

This module loads plain text (.txt) and markdown (.md) documents using LangChain's TextLoader,
handling multiple file encodings and standardizing metadata.
"""

import logging
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader

logger = logging.getLogger(__name__)


def load_text(file_path: str | Path) -> List[Document]:
    """
    Load a plain text or markdown document from disk, handling encoding fallback,
    and inject standardized citation metadata.
    
    Args:
        file_path (str | Path): Path to the .txt or .md file.
        
    Returns:
        List[Document]: A list containing the loaded document with standardized metadata.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file cannot be decoded or read.
    """
    path_obj = Path(file_path).resolve()
    if not path_obj.exists():
        raise FileNotFoundError(f"Text/Markdown file not found at path: {path_obj}")
        
    raw_docs: List[Document] = []
    
    # Try UTF-8 first, fallback to autodetect or latin-1 if character decoding fails
    encodings_to_try = ["utf-8", "utf-8-sig", "latin-1"]
    last_error = None
    
    for encoding in encodings_to_try:
        try:
            logger.info(f"Trying to load {path_obj.name} with encoding='{encoding}'")
            loader = TextLoader(str(path_obj), encoding=encoding, autodetect_encoding=False)
            raw_docs = loader.load()
            if raw_docs:
                break
        except Exception as e:
            last_error = e
            continue
            
    if not raw_docs:
        logger.error(f"Failed to decode text document '{path_obj.name}' with all tested encodings.")
        raise ValueError(f"Could not decode text document '{path_obj.name}'. Last error: {str(last_error)}")
        
    # Standardize metadata
    standardized_docs: List[Document] = []
    for doc in raw_docs:
        doc_name = path_obj.name
        doc.metadata.update({
            "source": doc_name,
            "document_name": doc_name,
            "page_number": 1,  # Text/MD documents treat the whole file or chunk section as Page 1
            "file_path": str(path_obj)
        })
        standardized_docs.append(doc)
        
    logger.info(f"Successfully loaded {len(standardized_docs)} document from {path_obj.name}")
    return standardized_docs
