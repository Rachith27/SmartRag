"""
Utilities package for SmartRag.

Exports `save_uploaded_file`, `clear_uploaded_files`, `compute_document_statistics`,
and `format_chat_history_for_display`.
"""

from utils.helpers import (
    save_uploaded_file,
    clear_uploaded_files,
    compute_document_statistics,
    format_chat_history_for_display,
)

__all__ = [
    "save_uploaded_file",
    "clear_uploaded_files",
    "compute_document_statistics",
    "format_chat_history_for_display",
]
