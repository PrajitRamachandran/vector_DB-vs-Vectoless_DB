"""
RAG Service Layer

Provides a single interface for:

- Vector RAG
- Vectorless RAG
- Hybrid RAG

Uses Streamlit caching so pipelines
are only loaded once.
"""

import inspect
import time

start = time.time()
print("Loading rag_service...")

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

from streamlit_app.services.document_service import (
    get_document_metadata
)

from streamlit_app.services.evaluation_explorer import (
    get_evaluation_summary
)

from utils.query_processor import (
    detect_company
)

from streamlit_app.services.company_summary_service import (
    get_company_summary_context
)

from streamlit_app.services.analytics_service import (
    classify_query,
    estimate_confidence,
    estimate_hallucination_risk,
    estimate_cost,
)

from llm import (
    load_llm,
    generate_answer
)

import config

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
# ADMIN OVERRIDES
# ============================================================
# Retrieval parameter controls (Top-K, rerank threshold, etc.)
# are only forwarded to `pipeline.ask` when that pipeline's
# signature actually accepts the given keyword — this keeps the
# UI's admin controls safe even if a given pipeline doesn't
# support live tuning of that parameter.

def _call_pipeline_ask(pipeline, question: str, overrides: dict):

    ask_fn = pipeline.ask
    accepted = set()

    try:
        accepted = set(inspect.signature(ask_fn).parameters.keys())
    except (TypeError, ValueError):
        pass

    kwargs = {
        k: v for k, v in (overrides or {}).items()
        if k in accepted
    }

    return ask_fn(question, **kwargs)


# ============================================================
# ENRICHMENT
# ============================================================

def _enrich_result(question: str, result: dict, model_name: str) -> dict:
    """
    Adds query classification, confidence, hallucination risk,
    and token/cost estimates to a raw pipeline result, without
    overwriting values a pipeline already provides.
    """

    retrieved = result.get("retrieved", [])

    if "query_type" not in result or not result.get("query_type"):
        result["query_type"] = classify_query(question)

    if "confidence" not in result or result.get("confidence") is None:
        result["confidence"] = estimate_confidence(retrieved)

    if "hallucination_risk" not in result or not result.get("hallucination_risk"):
        result["hallucination_risk"] = estimate_hallucination_risk(
            result.get("answer", ""), retrieved, result["confidence"]
        )

    usage = estimate_cost(
        prompt_text=question,
        completion_text=result.get("answer", ""),
        model_name=model_name,
    )
    result.setdefault("tokens_prompt", usage["tokens_prompt"])
    result.setdefault("tokens_completion", usage["tokens_completion"])
    result.setdefault("estimated_cost_usd", usage["estimated_cost_usd"])

    return result


# ============================================================
# ASK
# ============================================================

def ask_question(
    question: str,
    method: str,
    overrides: dict = None,
    model_name: str = None
):

    model_name = model_name or config.LLM_MODEL_ID

    try:

        # ==========================================
        # ROUTE QUESTION
        # ==========================================

        routing = route(
            question
        )

        print("\n========== ROUTER DEBUG ==========")
        print(routing)
        print("==================================\n")

        intent = routing["intent"]

        # ==========================================
        # DOCUMENT METADATA
        # ==========================================

        if intent == "document_metadata":

            metadata = get_document_metadata()

            companies = "\n".join(
                [
                    f"• {c}"
                    for c in metadata["companies"]
                ]
            )

            answer = f"""
Available Companies

{companies}

Total Documents:
{metadata['total_documents']}
"""

            return {

                "success": True,

                "method": "metadata",

                "intent": intent,

                "result": _enrich_result(question, {

                    "answer": answer,

                    "retrieved": [],

                    "retrieval_time": 0,

                    "rerank_time": 0,

                    "generation_time": 0,

                    "total_time": 0

                }, model_name)
            }

        # ==========================================
        # EVALUATION EXPLORATION
        # ==========================================

        if intent == "evaluation_exploration":

            answer = get_evaluation_summary()

            return {

                "success": True,

                "method": "evaluation",

                "intent": intent,

                "result": _enrich_result(question, {

                    "answer": answer,

                    "retrieved": [],

                    "retrieval_time": 0,

                    "rerank_time": 0,

                    "generation_time": 0,

                    "total_time": 0

                }, model_name)
            }

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

                "intent": intent,

                "result": _enrich_result(question, {

                    "answer": answer,

                    "method": "chat",

                    "intent": intent,

                    "retrieved": [],

                    "retrieval_time": 0,

                    "rerank_time": 0,

                    "generation_time": 0,

                    "total_time": 0

                }, model_name)
            }

        # ==========================================
        # LOAD PIPELINE
        # ==========================================

        pipeline = get_pipeline(
            method
        )

        # ==========================================
        # DOCUMENT EXPLORATION
        # ==========================================

        if intent == "document_exploration":
            print("\nDOCUMENT EXPLORATION HIT\n")

            company = detect_company(
                question
            )

            print("\n===== COMPANY DETECTION =====")
            print(f"Question: {question}")
            print(f"Detected Company: {company}")
            print("=============================\n")

            if not company:

                return {

                    "success": True,

                    "method": "summary",

                    "intent": intent,

                    "result": _enrich_result(question, {

                        "answer":
                            "Company could not be identified.",

                        "retrieved": []

                    }, model_name)
                }

            context = (
                get_company_summary_context(
                    company
                )
            )

            if not context:

                return {

                    "success": True,

                    "method": "summary",

                    "intent": intent,

                    "result": _enrich_result(question, {

                        "answer":
                            f"No data found for {company}.",

                        "retrieved": []

                    }, model_name)
                }

            llm = load_llm()

            summary_prompt = f"""
        Create a professional company overview.

        Company:
        {company}

        Include:

        1. Company Overview

        2. Business Segments

        3. Products and Services

        4. Revenue Drivers

        5. Strategic Priorities

        6. Key Risks

        Only use information from the context.
        """

            answer = generate_answer(
                llm,
                context,
                summary_prompt
            )

            return {

                "success": True,

                "method": "company_summary",

                "intent": intent,

                "result": _enrich_result(question, {

                    "answer": answer,

                    "retrieved": []

                }, model_name)
            }

        # ==========================================
        # NORMAL DOCUMENT QA
        # ==========================================

        result = _call_pipeline_ask(
            pipeline, question, overrides
        )

        result["intent"] = intent

        result = _enrich_result(question, result, model_name)

        return {

            "success": True,

            "method": method,

            "intent": intent,

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
    except Exception:
        pass

    try:
        get_vectorless_pipeline()
        status["vectorless"] = True
    except Exception:
        pass

    try:
        get_hybrid_pipeline()
        status["hybrid"] = True
    except Exception:
        pass

    try:
        get_random_pipeline()
        status["random"] = True
    except Exception:
        pass

    try:
        get_auto_pipeline()
        status["auto"] = True
    except Exception:
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


print(
    f"rag_service loaded in "
    f"{time.time()-start:.2f}s"
)