"""
Streamlit UI package for SmartRag.

Exports `render_sidebar` and `render_main_page` to construct the full
interactive web interface.
"""

from ui.sidebar import render_sidebar
from ui.chat import render_main_page

__all__ = ["render_sidebar", "render_main_page"]
