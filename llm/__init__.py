"""
LLM package for SmartRag.

Exports `LLMFactory` and `create_llm` to instantiate multi-provider LLMs
(OpenAI, Google Gemini, OpenRouter) with strict temperature control.
"""

from llm.llm_factory import LLMFactory, create_llm

__all__ = ["LLMFactory", "create_llm"]
