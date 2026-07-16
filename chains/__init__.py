"""
Chains package for SmartRag.

Exports `SmartRAGChain` and `run_rag_query` for executing LCEL retrieval-augmented
generation pipelines with strict source grounding and conversation memory.
"""

from chains.rag_chain import SmartRAGChain, run_rag_query

__all__ = ["SmartRAGChain", "run_rag_query"]
