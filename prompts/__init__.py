"""
Prompts package for SmartRag.

Exports anti-hallucination system prompt (`STRICT_SYSTEM_PROMPT`) and ChatPromptTemplates
(`RAG_PROMPT`, `CONVERSATIONAL_RAG_PROMPT`).
"""

from prompts.prompt import (
    STRICT_SYSTEM_PROMPT,
    get_rag_prompt_template,
    get_conversational_rag_prompt_template,
    RAG_PROMPT,
    CONVERSATIONAL_RAG_PROMPT,
)

__all__ = [
    "STRICT_SYSTEM_PROMPT",
    "get_rag_prompt_template",
    "get_conversational_rag_prompt_template",
    "RAG_PROMPT",
    "CONVERSATIONAL_RAG_PROMPT",
]
