"""
Vector Database package for SmartRag.

Provides unified interface over `faiss_manager` (Primary) and `chroma_manager` (Optional),
allowing runtime switching via `get_vector_database(db_type)`.
"""

from typing import Union
from database.faiss_db import FAISSDatabaseManager, faiss_manager
from database.chroma_db import ChromaDatabaseManager, chroma_manager


def get_vector_database(db_type: str = "faiss") -> Union[FAISSDatabaseManager, ChromaDatabaseManager]:
    """
    Retrieve the requested vector database manager instance.
    
    Args:
        db_type (str): Either 'faiss' (default) or 'chroma'.
        
    Returns:
        Union[FAISSDatabaseManager, ChromaDatabaseManager]: The database manager.
        
    Raises:
        ValueError: If db_type is not supported.
    """
    clean_type = db_type.strip().lower()
    if clean_type == "faiss":
        return faiss_manager
    elif clean_type in ["chroma", "chromadb"]:
        return chroma_manager
    else:
        raise ValueError(f"Unsupported vector database type: '{db_type}'. Supported: 'faiss' and 'chroma'.")


__all__ = [
    "FAISSDatabaseManager", 
    "faiss_manager", 
    "ChromaDatabaseManager", 
    "chroma_manager", 
    "get_vector_database"
]
