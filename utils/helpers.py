"""
Utilities and Helper Functions for SmartRag.

Provides helper routines for file management, Streamlit session state initialization,
and document/chunk statistics.
"""

import os
import shutil
import logging
from pathlib import Path
from typing import List, Dict, Any
from langchain_core.documents import Document

from config import UPLOADS_DIR, VECTOR_STORE_DIR

logger = logging.getLogger(__name__)


def save_uploaded_file(uploaded_file: Any, target_dir: Optional[Path] = None) -> Path:
    """
    Save a Streamlit UploadedFile object to local disk (`uploads/`).
    
    Args:
        uploaded_file (Any): Streamlit UploadedFile object.
        target_dir (Optional[Path]): Target directory. Defaults to `config.UPLOADS_DIR`.
        
    Returns:
        Path: Absolute path where the file was saved.
    """
    save_dir = target_dir or UPLOADS_DIR
    save_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = save_dir / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    logger.info(f"Saved uploaded file to: {file_path}")
    return file_path


def clear_uploaded_files(target_dir: Optional[Path] = None) -> bool:
    """
    Clear all saved document files from `uploads/` directory while keeping `.gitkeep`.
    
    Args:
        target_dir (Optional[Path]): Directory to clear. Defaults to `config.UPLOADS_DIR`.
        
    Returns:
        bool: True on success.
    """
    clear_dir = target_dir or UPLOADS_DIR
    if not clear_dir.exists():
        return True
        
    try:
        for item in clear_dir.iterdir():
            if item.name == ".gitkeep":
                continue
            if item.is_file() or item.is_symlink():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
        logger.info(f"Cleared uploaded documents in: {clear_dir}")
        return True
    except Exception as e:
        logger.error(f"Failed to clear uploads directory: {str(e)}")
        return False


def compute_document_statistics(documents: List[Document]) -> Dict[str, Any]:
    """
    Calculate summary statistics over a list of Document chunks.
    
    Args:
        documents (List[Document]): List of Document chunks.
        
    Returns:
        Dict[str, Any]: Dictionary containing total_chunks, total_characters,
                        unique_documents, and average_chunk_size.
    """
    if not documents:
        return {
            "total_chunks": 0,
            "total_characters": 0,
            "unique_documents": 0,
            "average_chunk_size": 0
        }
        
    total_chars = sum(len(doc.page_content) for doc in documents)
    unique_sources = set(
        doc.metadata.get("document_name") or doc.metadata.get("source", "Unknown")
        for doc in documents
    )
    
    return {
        "total_chunks": len(documents),
        "total_characters": total_chars,
        "unique_documents": len(unique_sources),
        "average_chunk_size": round(total_chars / len(documents), 1)
    }


def format_chat_history_for_display(chat_history: List[Tuple[str, str]]) -> List[Dict[str, str]]:
    """
    Format internal `(user_query, ai_response)` tuples into Streamlit `st.chat_message` compatible dicts.
    """
    display_messages = []
    for user_msg, ai_msg in chat_history:
        display_messages.append({"role": "user", "content": user_msg})
        display_messages.append({"role": "assistant", "content": ai_msg})
    return display_messages
