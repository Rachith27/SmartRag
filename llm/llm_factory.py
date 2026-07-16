"""
LLM Factory Module for SmartRag.

Implements the Factory Design Pattern (`LLMFactory`) to dynamically initialize
and switch between OpenAI, Google Gemini, and OpenRouter chat models.
"""

import os
import logging
from typing import Optional
from langchain_core.language_models import BaseChatModel

from config import rag_config, llm_config, get_api_key

logger = logging.getLogger(__name__)


class LLMFactory:
    """
    Factory class responsible for instantiating LangChain chat models
    from multiple providers (OpenAI, Gemini, OpenRouter) with strict temperature settings.
    """

    @staticmethod
    def create_llm(
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        api_key: Optional[str] = None
    ) -> BaseChatModel:
        """
        Create and return a configured BaseChatModel instance.
        
        Args:
            provider (Optional[str]): Provider ID ('openai', 'gemini', 'openrouter'). Defaults to config.
            model_name (Optional[str]): Specific model ID. Defaults to provider's default model.
            temperature (Optional[float]): Sampling temperature. Defaults to config (0.0).
            api_key (Optional[str]): Direct API key override from UI sidebar.
            
        Returns:
            BaseChatModel: Initialized LangChain chat model.
            
        Raises:
            ValueError: If the required API key is missing or provider is unsupported.
        """
        target_provider = (provider or rag_config.DEFAULT_LLM_PROVIDER).strip().lower()
        target_model = model_name or llm_config.get_default_model(target_provider)
        target_temp = temperature if temperature is not None else rag_config.TEMPERATURE
        
        # Resolve API key (override parameter takes precedence over environment variables)
        resolved_api_key = (api_key or get_api_key(target_provider) or "").strip()
        
        if not resolved_api_key:
            raise ValueError(
                f"Missing API key for provider '{target_provider}'. "
                f"Please enter your API key in the Streamlit sidebar or set it in your .env file."
            )
            
        logger.info(
            f"Initializing LLM provider='{target_provider}', model='{target_model}', temp={target_temp}"
        )
        
        try:
            if target_provider == "openai":
                from langchain_openai import ChatOpenAI
                return ChatOpenAI(
                    model=target_model,
                    temperature=target_temp,
                    api_key=resolved_api_key
                )
                
            elif target_provider == "gemini":
                from langchain_google_genai import ChatGoogleGenerativeAI
                # Remap deprecated 'gemini-1.5-*' strings to active 'gemini-2.5-flash'
                if "1.5" in target_model:
                    logger.warning(f"Deprecated model '{target_model}' requested. Auto-upgrading to 'gemini-2.5-flash'.")
                    target_model = "gemini-2.5-flash"
                # Set environment variable temporarily if needed by Google client internals
                os.environ["GOOGLE_API_KEY"] = resolved_api_key
                return ChatGoogleGenerativeAI(
                    model=target_model,
                    temperature=target_temp,
                    google_api_key=resolved_api_key,
                    convert_system_message_to_human=True  # For smooth compatibility with strict system messages
                )
                
            elif target_provider == "openrouter":
                from langchain_openai import ChatOpenAI
                return ChatOpenAI(
                    model=target_model,
                    temperature=target_temp,
                    api_key=resolved_api_key,
                    base_url="https://openrouter.ai/api/v1"
                )
                
            else:
                raise ValueError(
                    f"Unsupported LLM provider '{target_provider}'. "
                    f"Supported providers: {llm_config.get_supported_providers()}"
                )
                
        except Exception as e:
            logger.error(f"Error initializing model '{target_model}' for '{target_provider}': {str(e)}")
            raise RuntimeError(
                f"Failed to initialize LLM '{target_model}' ({target_provider}): {str(e)}"
            ) from e


def create_llm(
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    api_key: Optional[str] = None
) -> BaseChatModel:
    """Convenience helper wrapping LLMFactory.create_llm."""
    return LLMFactory.create_llm(
        provider=provider,
        model_name=model_name,
        temperature=temperature,
        api_key=api_key
    )
