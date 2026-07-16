"""
Document Loaders package for SmartRag.

Provides specialized loaders for PDFs, plain text, markdown, and web URLs,
as well as a unified dispatch function (`load_document`) that routes inputs automatically.
"""

from pathlib import Path
from typing import List
from langchain_core.documents import Document

from loaders.pdf_loader import load_pdf
from loaders.text_loader import load_text
from loaders.web_loader import load_web_url


def load_document(source: str | Path) -> List[Document]:
    """
    Automatically dispatch a file path or URL to the appropriate specialized loader.
    
    Args:
        source (str | Path): File path (.pdf, .txt, .md) or web URL (http:// / https://).
        
    Returns:
        List[Document]: Standardized LangChain Document objects ready for chunking.
        
    Raises:
        ValueError: If the source format or extension is unsupported.
    """
    source_str = str(source).strip()
    
    # Check if input is a Web URL
    if source_str.startswith("http://") or source_str.startswith("https://"):
        return load_web_url(source_str)
        
    path_obj = Path(source_str)
    suffix = path_obj.suffix.lower()
    
    if suffix == ".pdf":
        return load_pdf(path_obj)
    elif suffix in [".txt", ".md", ".markdown"]:
        return load_text(path_obj)
    else:
        raise ValueError(f"Unsupported document format: '{suffix}'. Supported formats: PDF, TXT, MD, and Web URLs.")


__all__ = ["load_pdf", "load_text", "load_web_url", "load_document"]
