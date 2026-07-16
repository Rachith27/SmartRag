"""
Streamlit Sidebar UI Component for SmartRag.

Renders document upload controls, hyper-parameter sliders (`Top K`, `Chunk Size`, `Overlap`),
LLM provider configuration, and database management actions with live progress indicators.
"""

import logging
from typing import Dict, Any, List
import streamlit as st

from config import rag_config, llm_config
from loaders import load_document
from chunking import split_documents
from database import get_vector_database
from utils import save_uploaded_file, clear_uploaded_files, compute_document_statistics

logger = logging.getLogger(__name__)


def render_sidebar() -> Dict[str, Any]:
    """
    Render the SmartRag sidebar and return the current user-configured settings.
    
    Returns:
        Dict[str, Any]: Dictionary containing active UI configuration values:
                        (provider, model_name, api_key, top_k, chunk_size, chunk_overlap, db_type).
    """
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        
        # --- 1. LLM Provider & Model Selection ---
        st.markdown("### 🤖 LLM Provider")
        provider_options = llm_config.get_supported_providers()
        selected_provider = st.selectbox(
            "Select Provider",
            options=provider_options,
            index=0,
            help="Choose your target LLM inference backend."
        )
        
        # Model selector based on chosen provider
        if selected_provider == "openai":
            model_options = ["gpt-4o-mini", "gpt-4o"]
        elif selected_provider == "gemini":
            model_options = [
                "gemini-2.5-flash",
                "gemini-2.5-pro",
                "gemini-2.0-flash",
                "gemini-3.5-flash",
                "gemini-3.1-pro-preview",
                "gemini-flash-latest",
                "gemini-pro-latest"
            ]
        elif selected_provider == "openrouter":
            model_options = ["openai/gpt-4o-mini", "anthropic/claude-3.5-sonnet"]
        else:
            model_options = [llm_config.get_default_model(selected_provider)]
            
        selected_model = st.selectbox(
            "Select Model",
            options=model_options,
            index=0
        )
        
        api_key_input = st.text_input(
            f"{selected_provider.upper()} API Key",
            type="password",
            placeholder="Paste your API key here...",
            help=f"Required for querying {selected_provider.upper()} models."
        )
        
        st.divider()
        
        # --- 2. Vector DB & Hyper-parameters ---
        st.markdown("### 🔍 Retrieval Parameters")
        selected_db = st.selectbox(
            "Vector Database",
            options=["faiss", "chroma"],
            index=0,
            format_func=lambda x: x.upper(),
            help="FAISS is fast & local; Chroma provides optional SQLite storage."
        )
        
        top_k = st.slider(
            "Top K Chunks",
            min_value=1,
            max_value=10,
            value=rag_config.TOP_K,
            help="Number of relevant document chunks retrieved per query."
        )
        
        chunk_size = st.slider(
            "Chunk Size (Characters)",
            min_value=200,
            max_value=2000,
            value=rag_config.CHUNK_SIZE,
            step=50,
            help="Size of text chunks during document splitting."
        )
        
        chunk_overlap = st.slider(
            "Chunk Overlap (Characters)",
            min_value=0,
            max_value=500,
            value=rag_config.CHUNK_OVERLAP,
            step=25,
            help="Character overlap between consecutive chunks."
        )
        
        st.divider()
        
        # --- 3. Document Ingestion ---
        st.markdown("### 📥 Document Upload")
        uploaded_files = st.file_uploader(
            "Upload Documents (PDF, TXT, MD)",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
            help="Select one or more files to index into the vector database."
        )
        
        web_url_input = st.text_input(
            "Or Index Web URL (Optional)",
            placeholder="https://example.com/research-paper",
            help="Enter a public web page URL to ingest and index its content."
        )
        
        col_proc1, col_proc2 = st.columns(2)
        with col_proc1:
            process_btn = st.button("🚀 Process Docs", type="primary", use_container_width=True)
        with col_proc2:
            reindex_btn = st.button("🔄 Re-index All", use_container_width=True)
            
        if process_btn or reindex_btn:
            _handle_document_ingestion(
                uploaded_files=uploaded_files,
                web_url=web_url_input,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                db_type=selected_db,
                force_reindex=reindex_btn
            )
            
        st.divider()
        
        # --- 4. Session & Database Management ---
        st.markdown("### 🧹 Management Actions")
        col_clear1, col_clear2 = st.columns(2)
        with col_clear1:
            if st.button("💬 Clear Chat", use_container_width=True):
                st.session_state["messages"] = []
                st.session_state["chat_history"] = []
                st.session_state["evaluation_results"] = None
                st.success("Chat memory cleared!")
                st.rerun()
                
        with col_clear2:
            if st.button("🗑️ Clear DB", use_container_width=True):
                db_manager = get_vector_database(selected_db)
                if db_manager.clear_database():
                    clear_uploaded_files()
                    st.session_state["indexed_doc_count"] = 0
                    st.success(f"{selected_db.upper()} database cleared!")
                    st.rerun()
                else:
                    st.error("Failed to clear vector database.")

    return {
        "provider": selected_provider,
        "model_name": selected_model,
        "api_key": api_key_input,
        "top_k": top_k,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "db_type": selected_db
    }


def _handle_document_ingestion(
    uploaded_files: List[Any],
    web_url: str,
    chunk_size: int,
    chunk_overlap: int,
    db_type: str,
    force_reindex: bool = False
) -> None:
    """Internal workflow orchestrator for processing uploaded files and URLs into chunks."""
    if not uploaded_files and not web_url.strip():
        st.sidebar.error("❌ Please upload at least one file or enter a valid Web URL.")
        return
        
    db_manager = get_vector_database(db_type)
    
    # If re-indexing is requested, clear existing index first
    if force_reindex:
        db_manager.clear_database()
        
    progress_bar = st.sidebar.progress(0, text="Initializing ingestion...")
    all_chunks = []
    
    total_sources = len(uploaded_files) + (1 if web_url.strip() else 0)
    current_source = 0
    
    try:
        # Process uploaded files
        for ufile in uploaded_files:
            current_source += 1
            progress_bar.progress(
                int((current_source / total_sources) * 50),
                text=f"Loading file ({current_source}/{total_sources}): {ufile.name}..."
            )
            file_path = save_uploaded_file(ufile)
            raw_docs = load_document(file_path)
            chunks = split_documents(raw_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            all_chunks.extend(chunks)
            
        # Process Web URL if provided
        if web_url.strip():
            current_source += 1
            progress_bar.progress(
                int((current_source / total_sources) * 50),
                text="Fetching and splitting Web URL..."
            )
            raw_docs = load_document(web_url.strip())
            chunks = split_documents(raw_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            all_chunks.extend(chunks)
            
        if not all_chunks:
            progress_bar.empty()
            st.sidebar.error("⚠️ No text could be extracted from the provided documents.")
            return
            
        progress_bar.progress(75, text=f"Embedding {len(all_chunks)} chunks into {db_type.upper()}...")
        db_manager.add_documents(all_chunks)
        
        progress_bar.progress(100, text="Index complete!")
        stats = compute_document_statistics(all_chunks)
        st.session_state["indexed_doc_count"] = stats["unique_documents"]
        st.session_state["last_stats"] = stats
        
        st.sidebar.success(
            f"✅ Indexed {stats['total_chunks']} chunks from {stats['unique_documents']} document(s)!"
        )
        logger.info(f"Successfully ingested and indexed {stats['total_chunks']} chunks.")
        
    except Exception as e:
        progress_bar.empty()
        st.sidebar.error(f"❌ Ingestion Error: {str(e)}")
        logger.error(f"Error during document ingestion: {str(e)}")
