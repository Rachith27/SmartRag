"""
Streamlit Main Page & Chat Interface Component for SmartRag.

Renders header, chat history loop, question input, expandable source cards,
and RAGAS evaluation metric progress bars.
"""

import logging
from typing import Dict, Any, List
import streamlit as st

from chains import run_rag_query
from evaluation import evaluate_rag_turn
from utils import compute_document_statistics

logger = logging.getLogger(__name__)


def render_main_page(sidebar_config: Dict[str, Any]) -> None:
    """
    Render the header, chat interface, source citations, and evaluation metrics.
    
    Args:
        sidebar_config (Dict[str, Any]): Active parameters selected in the sidebar.
    """
    # --- 1. Header & Subtitle ---
    st.title("🧩 SmartRag")
    st.markdown("### *AI Powered Research Assistant*")
    st.markdown(
        "Upload research papers, technical documentation, policies, or web links. "
        "Ask questions in natural language and receive strictly grounded answers with exact source citations."
    )
    
    # Display index status badge
    if "indexed_doc_count" in st.session_state and st.session_state["indexed_doc_count"] > 0:
        stats = st.session_state.get("last_stats", {})
        total_chunks = stats.get("total_chunks", 0)
        unique_docs = stats.get("unique_documents", st.session_state["indexed_doc_count"])
        st.info(f"📚 Active Index: **{unique_docs} Document(s)** ({total_chunks} total chunks indexed in `{sidebar_config['db_type'].upper()}`)")
    else:
        st.warning("⚠️ Vector database is empty. Please upload and process documents from the sidebar before asking questions.")

    st.divider()

    # --- 2. Initialize Session State Memory ---
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    if "evaluation_results" not in st.session_state:
        st.session_state["evaluation_results"] = None

    # --- 3. Render Existing Chat History ---
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # If the stored message has expandable sources, render them
            if msg.get("sources"):
                _render_source_cards(msg["sources"])
            # If evaluation metrics exist for this message, render them
            if msg.get("eval_metrics"):
                _render_evaluation_metrics(msg["eval_metrics"])

    # --- 4. Chat Input & Processing ---
    user_question = st.chat_input("Ask a question about your uploaded documents...")
    
    if user_question:
        # Append and render user question immediately
        st.session_state["messages"].append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)
            
        # Execute RAG turn inside assistant bubble
        with st.chat_message("assistant"):
            with st.spinner(f"🔍 Searching `{sidebar_config['db_type'].upper()}` and synthesizing grounded response..."):
                try:
                    result: Dict[str, Any] = run_rag_query(
                        question=user_question,
                        provider=sidebar_config["provider"],
                        model_name=sidebar_config["model_name"],
                        api_key=sidebar_config["api_key"],
                        db_type=sidebar_config["db_type"],
                        top_k=sidebar_config["top_k"],
                        chat_history=st.session_state["chat_history"]
                    )
                    
                    answer_text = result.get("answer", "No response generated.")
                    sources = result.get("sources", [])
                    retrieved_docs = result.get("retrieved_docs", [])
                    
                    # Render generated answer
                    st.markdown(answer_text)
                    
                    # Render source citations
                    if sources:
                        _render_source_cards(sources)
                        
                    # Calculate RAGAS evaluation if documents were retrieved and API key is present
                    eval_metrics = None
                    if retrieved_docs and sidebar_config["api_key"]:
                        with st.spinner("📊 Calculating RAGAS faithfulness & relevancy metrics..."):
                            eval_metrics = evaluate_rag_turn(
                                question=user_question,
                                answer=answer_text,
                                retrieved_docs=retrieved_docs,
                                provider=sidebar_config["provider"],
                                model_name=sidebar_config["model_name"],
                                api_key=sidebar_config["api_key"]
                            )
                        if eval_metrics:
                            _render_evaluation_metrics(eval_metrics)
                            
                    # Store in conversation history
                    st.session_state["chat_history"].append((user_question, answer_text))
                    st.session_state["messages"].append({
                        "role": "assistant",
                        "content": answer_text,
                        "sources": sources,
                        "eval_metrics": eval_metrics
                    })
                    
                except Exception as e:
                    error_msg = f"❌ **Error generating response:** {str(e)}"
                    st.error(error_msg)
                    logger.error(f"Chat UI error during RAG execution: {str(e)}")


def _render_source_cards(sources: List[Dict[str, Any]]) -> None:
    """Helper to render expandable citation cards with document name, page, and similarity."""
    with st.expander(f"📑 View Grounded Sources ({len(sources)} Chunks)", expanded=False):
        for idx, src in enumerate(sources, start=1):
            doc_name = src.get("document_name", "Unknown Document")
            page_num = src.get("page_number", "N/A")
            sim_score = src.get("similarity_score", 0.0)
            chunk_text = src.get("chunk_text", "")
            
            st.markdown(f"**Source {idx}: `{doc_name}`** (Page `{page_num}` | Similarity: `{sim_score * 100:.1f}%`)")
            st.code(chunk_text, language="markdown")
            if idx < len(sources):
                st.divider()


def _render_evaluation_metrics(metrics: Dict[str, float]) -> None:
    """Helper to render RAGAS evaluation scores as styled progress bars."""
    st.markdown("#### 📊 RAGAS Evaluation Metrics")
    col1, col2 = st.columns(2)
    
    with col1:
        faithfulness = metrics.get("faithfulness", 0.0)
        st.write(f"**Faithfulness:** `{faithfulness:.2f}`")
        st.progress(min(1.0, max(0.0, faithfulness)))
        
        context_precision = metrics.get("context_precision", 0.0)
        st.write(f"**Context Precision:** `{context_precision:.2f}`")
        st.progress(min(1.0, max(0.0, context_precision)))
        
    with col2:
        answer_relevancy = metrics.get("answer_relevancy", 0.0)
        st.write(f"**Answer Relevancy:** `{answer_relevancy:.2f}`")
        st.progress(min(1.0, max(0.0, answer_relevancy)))
        
        context_recall = metrics.get("context_recall", 0.0)
        st.write(f"**Context Recall:** `{context_recall:.2f}`")
        st.progress(min(1.0, max(0.0, context_recall)))
