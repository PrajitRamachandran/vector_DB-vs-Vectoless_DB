"""
RAG Service Layer

Provides a single interface for:

- Vector RAG
- Vectorless RAG
- Hybrid RAG

Uses Streamlit caching so pipelines
are only loaded once.
"""

import traceback

import streamlit as st

from vector_rag.pipeline import (
    VectorRAGPipeline
)

from vectorless_rag.pipeline import (
    VectorlessRAGPipeline
)

from hybrid_rag.pipeline import (
    HybridRAGPipeline
)

# ============================================================
# PIPELINE CACHE
# ============================================================

@st.cache_resource
def get_vector_pipeline():
    """
    Load Vector RAG once.
    """

    return VectorRAGPipeline()


@st.cache_resource
def get_vectorless_pipeline():
    """
    Load Vectorless RAG once.
    """

    return VectorlessRAGPipeline()


@st.cache_resource
def get_hybrid_pipeline():
    """
    Load Hybrid RAG once.
    """

    return HybridRAGPipeline()

# ============================================================
# PIPELINE ACCESS
# ============================================================

def get_pipeline(method: str):

    method = method.lower()

    if method == "vector":
        return get_vector_pipeline()

    if method == "vectorless":
        return get_vectorless_pipeline()

    if method == "hybrid":
        return get_hybrid_pipeline()

    raise ValueError(
        f"Unknown method: {method}"
    )

# ============================================================
# ASK
# ============================================================

def ask_question(
    question: str,
    method: str
):
    """
    Execute RAG query.
    """

    try:

        pipeline = get_pipeline(
            method
        )

        result = pipeline.ask(
            question
        )

        return {
            "success": True,
            "method": method,
            "result": result
        }

    except Exception as e:

        return {
            "success": False,
            "error": str(e),
            "traceback":
                traceback.format_exc()
        }

# ============================================================
# HEALTH CHECK
# ============================================================

def pipeline_status():
    """
    Basic status reporting.
    """

    status = {
        "vector": False,
        "vectorless": False,
        "hybrid": False
    }

    try:
        get_vector_pipeline()
        status["vector"] = True
    except:
        pass

    try:
        get_vectorless_pipeline()
        status["vectorless"] = True
    except:
        pass

    try:
        get_hybrid_pipeline()
        status["hybrid"] = True
    except:
        pass

    return status

# ============================================================
# CLEAR CACHE
# ============================================================

def clear_pipeline_cache():
    """
    Reload all pipelines.
    Useful after rebuilding indexes.
    """

    st.cache_resource.clear()