"""
Financial RAG Benchmark — Streamlit App
Entry point: sets page config, renders sidebar, routes to active page.
"""

import streamlit as st

st.set_page_config(
    page_title="Financial RAG Benchmark",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

from components.sidebar import render_sidebar
from pages.router import route

render_sidebar()
route()
