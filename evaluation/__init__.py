"""
Evaluation package for SmartRag.

Exports `evaluate_rag_turn` for automated RAGAS metric calculation
(Faithfulness, Relevancy, Precision, Recall) with graceful fallback scoring.
"""

from evaluation.ragas_eval import evaluate_rag_turn

__all__ = ["evaluate_rag_turn"]
