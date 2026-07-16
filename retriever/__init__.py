"""
Retriever package for SmartRag.

Provides `SmartRetriever` and `retrieve_chunks` for querying vector indices
and standardizing similarity scores for UI display and LLM prompt grounding.
"""

from retriever.retriever import SmartRetriever, retrieve_chunks

__all__ = ["SmartRetriever", "retrieve_chunks"]
