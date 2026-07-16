"""
RAG Chain Module for SmartRag.

Orchestrates the complete Retrieval-Augmented Generation pipeline using LangChain Expression
Language (LCEL). Connects the semantic retriever (`SmartRetriever`), anti-hallucination
prompts (`RAG_PROMPT`), and multi-provider LLMs (`BaseChatModel`) while formatting
structured source citations and supporting conversation memory.
"""

import logging
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from retriever import SmartRetriever
from prompts import RAG_PROMPT, CONVERSATIONAL_RAG_PROMPT
from llm import create_llm

logger = logging.getLogger(__name__)


class SmartRAGChain:
    """
    Core execution engine for SmartRag. Connects retrieval, prompt formatting,
    and LLM inference into a robust, grounded RAG chain.
    """
    
    def __init__(
        self,
        retriever: Optional[SmartRetriever] = None,
        llm: Optional[BaseChatModel] = None,
        provider: str = "openai",
        model_name: Optional[str] = None,
        temperature: float = 0.0,
        api_key: Optional[str] = None,
        db_type: str = "faiss",
        top_k: Optional[int] = None
    ) -> None:
        """
        Initialize the SmartRAG chain.
        
        Args:
            retriever (Optional[SmartRetriever]): Semantic retriever instance.
            llm (Optional[BaseChatModel]): Pre-initialized LLM instance.
            provider (str): Provider ID ('openai', 'gemini', 'openrouter').
            model_name (Optional[str]): Specific model ID.
            temperature (float): Sampling temperature (default 0.0).
            api_key (Optional[str]): Direct API key override.
            db_type (str): Database backend ('faiss' or 'chroma').
            top_k (Optional[int]): Number of chunks to retrieve.
        """
        self.retriever = retriever or SmartRetriever(db_type=db_type, top_k=top_k)
        self.llm = llm or create_llm(
            provider=provider,
            model_name=model_name,
            temperature=temperature,
            api_key=api_key
        )
        self.output_parser = StrOutputParser()

    def _format_context(self, documents: List[Document]) -> str:
        """
        Format retrieved Document objects into a clean, structured context string for the prompt.
        """
        formatted_chunks = []
        for i, doc in enumerate(documents, start=1):
            doc_name = doc.metadata.get("document_name", "Unknown Document")
            page_num = doc.metadata.get("page_number", 1)
            chunk_text = doc.page_content.strip()
            formatted_chunks.append(
                f"--- [Source {i}: {doc_name} | Page {page_num}] ---\n{chunk_text}"
            )
        return "\n\n".join(formatted_chunks)

    def _format_sources(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """
        Format retrieved Document objects into structured UI-friendly source citation dictionaries.
        """
        sources = []
        for doc in documents:
            sources.append({
                "document_name": str(doc.metadata.get("document_name", "Unknown Document")),
                "page_number": doc.metadata.get("page_number", 1),
                "chunk_text": doc.page_content.strip(),
                "similarity_score": float(doc.metadata.get("similarity_score", 0.0)),
                "raw_distance": float(doc.metadata.get("raw_distance", 0.0))
            })
        return sources

    def run(
        self,
        question: str,
        chat_history: Optional[List[Tuple[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Execute the complete RAG pipeline for a user question.
        
        Args:
            question (str): Natural language question.
            chat_history (Optional[List[Tuple[str, str]]]): List of (user_msg, ai_msg) tuples for memory.
            
        Returns:
            Dict[str, Any]: Structured output dictionary containing:
                - `answer` (str): Grounded answer from the LLM.
                - `sources` (List[Dict[str, Any]]): Formatted source metadata for UI citations.
                - `retrieved_docs` (List[Document]): Raw Document objects for evaluation.
                
        Raises:
            RuntimeError: If execution fails due to API or network errors.
        """
        clean_question = question.strip() if question else ""
        if not clean_question:
            return {
                "answer": "Please provide a valid question.",
                "sources": [],
                "retrieved_docs": []
            }
            
        logger.info(f"Executing SmartRAGChain for question: '{clean_question[:60]}...'")
        
        try:
            # 1. Retrieve relevant context chunks from vector store
            retrieved_docs: List[Document] = self.retriever.retrieve(query=clean_question)
            
            # 2. Strict grounding check: if no relevant chunks exist or vector store is empty,
            # return the exact required fallback phrase without hallucinating via LLM inference.
            if not retrieved_docs:
                logger.info("No context chunks retrieved. Returning strict fallback response.")
                return {
                    "answer": "I could not find this information in the uploaded documents.",
                    "sources": [],
                    "retrieved_docs": []
                }
                
            # 3. Format context string and source citations
            context_str = self._format_context(retrieved_docs)
            sources = self._format_sources(retrieved_docs)
            
            # 4. Select prompt template based on whether conversation history is present
            if chat_history and len(chat_history) > 0:
                prompt_template: ChatPromptTemplate = CONVERSATIONAL_RAG_PROMPT
                # Format chat history into LangChain message format
                formatted_history = []
                for user_msg, ai_msg in chat_history:
                    formatted_history.append(("human", user_msg))
                    formatted_history.append(("ai", ai_msg))
                    
                chain = prompt_template | self.llm | self.output_parser
                raw_answer = chain.invoke({
                    "context": context_str,
                    "chat_history": formatted_history,
                    "question": clean_question
                })
            else:
                prompt_template: ChatPromptTemplate = RAG_PROMPT
                chain = prompt_template | self.llm | self.output_parser
                raw_answer = chain.invoke({
                    "context": context_str,
                    "question": clean_question
                })
                
            logger.info("Successfully generated grounded answer.")
            return {
                "answer": raw_answer.strip(),
                "sources": sources,
                "retrieved_docs": retrieved_docs
            }
            
        except Exception as e:
            logger.error(f"Error executing RAG chain: {str(e)}")
            raise RuntimeError(f"RAG execution error: {str(e)}") from e


def run_rag_query(
    question: str,
    provider: str = "openai",
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    db_type: str = "faiss",
    top_k: Optional[int] = None,
    chat_history: Optional[List[Tuple[str, str]]] = None
) -> Dict[str, Any]:
    """
    Convenience helper function to execute a single RAG query without manually instantiating SmartRAGChain.
    """
    chain = SmartRAGChain(
        provider=provider,
        model_name=model_name,
        api_key=api_key,
        db_type=db_type,
        top_k=top_k
    )
    return chain.run(question=question, chat_history=chat_history)
