#app.py

from pathlib import Path
import streamlit as st
from streamlit_app.auth.session_manager import (
    logout_user
)
# Pages Configuration

st.set_page_config(
  page_title = "Financial RAG System",
  page_icon = "$$$",
  layout = "wide",
  initial_sidebar_state = "expanded"
)

# Project Path

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

EVALUATION_DIR = PROJECT_ROOT / "evaluation"
RESULTS_DIR = EVALUATION_DIR / "results"

# Initialise Sessions

def initialise_session():

  # Creates application-wide session variables.

  defaults = {
    # Current retrieval mode
    "selected_method": "Vectorless",

    # Chat history
    "chat_history": [],

    # Uploaded files during session
    "uploaded_files": [],

    # Evaluation status
    "evaluation_running": False,

    # Cached results
    "latest_results": None,

    # Current page info
    "app_loaded": True,
    
  }
  
  for key, value in defaults.items():
    if key not in st.session_state:
      st.session_state[key] = value

initialise_session()
if st.session_state.get(
    "logged_in",
    False
):

    with st.sidebar:

        st.divider()

        st.markdown(
            f"""
            ### 👤 User

            **Username:** {st.session_state.username}

            **Role:** {st.session_state.role}
            """
        )

        if st.button(
            "🚪 Logout",
            use_container_width=True,
            key="logout_button"
        ):

            logout_user()

            st.switch_page(
                "pages/00_login.py"
            )

# Styling

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .sub-title {
        color: #888;
        margin-bottom: 2rem;
    }

    .metric-card {
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid #333;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# Start Up health checks

def run_startup_checks():
  # Verifies required project folders exist.

  issues = []

  required_paths = [
    RAW_DIR, PROCESSED_DIR, EVALUATION_DIR, RESULTS_DIR
  ]

  for path in required_paths:
    if not path.exists():
      issues.append(f"Missing: {path}")

  return issues

issues = run_startup_checks()

# Header

st.markdown(
    '<div class="main-title">📊 Financial RAG Benchmark</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    Compare and evaluate:

    - Vector RAG (ChromaDB + BGE)
    - Vectorless RAG (BM25)
    - Hybrid RAG (RRF Fusion)

    using Financial 10-K reports.
    """
)

# System Status

if issues:
  st.error("Project Startup issues detected")

  for issue in issues:
    st.warning(issue)

else:
  st.success("System Initialised Successfully")

# Quick Stats

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Raw Documents",
        value=len(list(RAW_DIR.glob("*.pdf")))
    )

with col2:
    st.metric(
        label="Evaluation Files",
        value=len(list(RESULTS_DIR.glob("*")))
    )

with col3:
    st.metric(
        label="Chat Messages",
        value=len(st.session_state.chat_history)
    )

with col4:
    st.metric(
        label="Current Method",
        value=st.session_state.selected_method
    )

# Project Overview

st.divider()

st.subheader("Project Architecture")

st.code(
"""
Vector RAG
    ├── ChromaDB
    ├── BGE Embeddings
    └── Cross Encoder Reranking

Vectorless RAG
    ├── BM25
    └── Cross Encoder Reranking

Hybrid RAG
    ├── Vector Retrieval
    ├── BM25 Retrieval
    ├── RRF Fusion
    └── Cross Encoder Reranking
""",
    language="text"
)

# Navigation Guide

st.divider()

st.subheader("Navigation")

st.markdown(
"""
Use the left sidebar to access:

1. Dashboard
2. Upload Documents
3. Index Manager
4. Chat
5. Conversations
6. Evaluation
"""
)

# Footer

st.divider()

st.caption("Financial RAG Benchmark")