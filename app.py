"""
SmartRag Main Application Entry Point (`app.py`).

Run this file using:
    streamlit run app.py

Initializes system logging, sets dark-theme page configuration,
and orchestrates the interactive sidebar and chat components.
"""

import logging
import streamlit as st

# Configure application-level logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("SmartRag")


def main() -> None:
    """Main execution entry point for SmartRag Streamlit application."""
    st.set_page_config(
        page_title="SmartRag | AI Research Assistant",
        page_icon="🧩",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Optional dark theme enhancement & custom styling
    st.markdown("""
        <style>
        .stApp {
            background-color: #0E1117;
            color: #FAFAFA;
        }
        .stChatMessage {
            border-radius: 12px;
            padding: 1rem;
            margin-bottom: 1rem;
        }
        div[data-testid="stExpander"] {
            border: 1px solid #30363D;
            border-radius: 8px;
            background-color: #161B22;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Import UI components inside main() to ensure clean Streamlit execution lifecycle
    from ui import render_sidebar, render_main_page
    
    # Render sidebar and capture user configuration changes
    sidebar_config = render_sidebar()
    
    # Render main chat interface and evaluation metrics
    render_main_page(sidebar_config)


if __name__ == "__main__":
    main()
