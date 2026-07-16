"""
Prompt Templates Module for SmartRag.

Defines strict system prompts and LangChain ChatPromptTemplates enforced to prevent
hallucination and require citation of document sources.
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Exact strict system prompt required to eliminate hallucination
STRICT_SYSTEM_PROMPT = """You are SmartRag.
Answer ONLY using the provided context.
If the answer is unavailable in the context, reply:
'I could not find this information in the uploaded documents.'
Never fabricate facts.
Always cite document sources.

Context Chunks:
{context}"""


def get_rag_prompt_template() -> ChatPromptTemplate:
    """
    Generate the standard single-turn RAG ChatPromptTemplate.
    
    Returns:
        ChatPromptTemplate: Prompt template expecting `context` and `question`.
    """
    return ChatPromptTemplate.from_messages([
        ("system", STRICT_SYSTEM_PROMPT),
        ("human", "{question}")
    ])


def get_conversational_rag_prompt_template() -> ChatPromptTemplate:
    """
    Generate the multi-turn conversational RAG ChatPromptTemplate supporting
    previous conversation memory (`chat_history`).
    
    Returns:
        ChatPromptTemplate: Prompt template expecting `context`, `chat_history`, and `question`.
    """
    return ChatPromptTemplate.from_messages([
        ("system", STRICT_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{question}")
    ])


# Pre-instantiated default templates for fast import
RAG_PROMPT = get_rag_prompt_template()
CONVERSATIONAL_RAG_PROMPT = get_conversational_rag_prompt_template()
