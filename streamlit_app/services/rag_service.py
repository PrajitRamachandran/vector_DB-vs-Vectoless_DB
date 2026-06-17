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

from random_rag.pipeline import (
    RandomRAGPipeline
)

from auto_rag.pipeline import (
    AutoRAGPipeline
)

from query_router.router import (
    route
)

from llm import (
    load_llm,
    generate_chat_response
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

@st.cache_resource
def get_random_pipeline():

    return RandomRAGPipeline()

@st.cache_resource
def get_auto_pipeline():

    return AutoRAGPipeline()

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
    
    if method == "random":
        return get_random_pipeline()
    
    if method == "auto":
        return get_auto_pipeline()

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

    try:

        # ==========================================
        # ROUTE QUESTION
        # ==========================================

        routing = route(
            question
        )

        # ==========================================
        # NON-RAG QUESTIONS
        # ==========================================

        if not routing["use_rag"]:

            llm = load_llm()

            answer = generate_chat_response(
                llm,
                question
            )

            return {

                "success": True,

                "method": "chat",

                "intent":
                    routing["intent"],

                "result": {

                    "answer": answer,

                    "method": "chat",

                    "intent":
                        routing["intent"],

                    "retrieved": [],

                    "retrieval_time": 0,

                    "rerank_time": 0,

                    "generation_time": 0,

                    "total_time": 0
                }
            }

        # ==========================================
        # RAG QUESTIONS
        # ==========================================

        pipeline = get_pipeline(
            method
        )

        result = pipeline.ask(
            question
        )

        result["intent"] = (
            routing["intent"]
        )

        return {

            "success": True,

            "method": method,

            "intent":
                routing["intent"],

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
        "hybrid": False,
        "random": False,
        "auto": False
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

    try:
        get_random_pipeline()
        status["random"] = True
    except:
        pass

    try:
        get_auto_pipeline()
        status["auto"] = True
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