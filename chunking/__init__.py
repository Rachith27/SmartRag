"""
Chunking package for SmartRag.

Provides `DocumentSplitter` and `split_documents` to segment loaded text into
contextually dense chunks with `RecursiveCharacterTextSplitter`.
"""

from chunking.splitter import DocumentSplitter, split_documents

__all__ = ["DocumentSplitter", "split_documents"]
