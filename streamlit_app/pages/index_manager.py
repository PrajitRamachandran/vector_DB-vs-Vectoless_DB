"""
Index Manager Page
==================
Build, inspect, and rebuild the ChromaDB vector index and BM25 index
from the preprocessed chunk data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from services.index_service import (
    build_vector_index,
    build_bm25_index,
    get_vector_stats,
    get_bm25_stats,
    load_chunks,
)
from services.pipeline_manager import get_pipeline_manager


def render() -> None:
    st.title("🗂️ Index Manager")
    st.caption("Build and inspect the retrieval indices used by each RAG pipeline.")

    pm = get_pipeline_manager()
    chunks = load_chunks()
    n_parents  = len(chunks.get("parents",  []))
    n_children = len(chunks.get("children", []))

    # ── Corpus snapshot ───────────────────────────────────────────────────────
    st.subheader("Corpus Snapshot")
    c1, c2 = st.columns(2)
    c1.metric("Parent chunks (sent to LLM)", f"{n_parents:,}")
    c2.metric("Child chunks (indexed)",       f"{n_children:,}")

    if n_children == 0:
        st.warning("No chunks found. Go to **Upload Documents** and run preprocessing first.")
        return

    st.divider()

    # ── Vector index ──────────────────────────────────────────────────────────
    st.subheader("🧠 Vector Index (ChromaDB + BGE)")
    vec_stat = get_vector_stats()

    v1, v2, v3 = st.columns(3)
    v1.metric("Vectors in index", f"{vec_stat['count']:,}")
    v2.metric("Status", vec_stat["status"])
    v3.metric("Pipeline loaded", "Yes" if pm.vector_loaded else "No")

    with st.expander("Index configuration"):
        import config
        st.json({
            "embedding_model":   config.EMBEDDING_MODEL,
            "collection":        config.CHROMA_COLLECTION,
            "hnsw_m":            config.HNSW_M,
            "hnsw_construction_ef": config.HNSW_CONSTRUCTION_EF,
            "hnsw_search_ef":    config.HNSW_SEARCH_EF,
            "child_chunk_size":  config.CHILD_CHUNK_SIZE,
            "parent_chunk_size": config.PARENT_CHUNK_SIZE,
            "top_k":             config.TOP_K,
            "fetch_k":           config.FETCH_K,
        })

    col_vec, _ = st.columns([1, 3])
    with col_vec:
        if st.button("🔨 Build / Rebuild Vector Index", type="primary", use_container_width=True):
            status_box = st.empty()
            with st.spinner("Building vector index…"):
                try:
                    # Unload pipeline so a fresh collection is used on next load
                    pm._vector = None
                    build_vector_index(status_container=status_box)
                    st.rerun()
                except Exception as exc:
                    status_box.error(f"❌ Vector index build failed: {exc}")
                    st.exception(exc)

    st.divider()

    # ── BM25 index ────────────────────────────────────────────────────────────
    st.subheader("📝 BM25 Index (Vectorless RAG)")
    bm_stat = get_bm25_stats()

    b1, b2, b3 = st.columns(3)
    b1.metric("Documents in BM25", f"{bm_stat['count']:,}")
    b2.metric("Status", bm_stat["status"])
    b3.metric("Pipeline loaded", "Yes" if pm.vectorless_loaded else "No")

    col_bm, _ = st.columns([1, 3])
    with col_bm:
        if st.button("🔨 Build / Rebuild BM25 Index", type="primary", use_container_width=True):
            status_box = st.empty()
            with st.spinner("Building BM25 index…"):
                try:
                    pm._vectorless = None
                    pm._hybrid = None
                    build_bm25_index(status_container=status_box)
                    st.rerun()
                except Exception as exc:
                    status_box.error(f"❌ BM25 index build failed: {exc}")
                    st.exception(exc)

    st.divider()

    # ── Chunk browser ─────────────────────────────────────────────────────────
    st.subheader("Chunk Browser")
    companies = sorted({c.get("company", "Unknown") for c in chunks.get("children", [])})
    selected_company = st.selectbox("Filter by company", ["All"] + companies)
    chunk_type = st.radio("Chunk type", ["children", "parents"], horizontal=True)
    n_preview = st.slider("Chunks to preview", 5, 50, 10)

    sample = [
        c for c in chunks.get(chunk_type, [])
        if selected_company == "All" or c.get("company") == selected_company
    ][:n_preview]

    if sample:
        import pandas as pd
        rows = [
            {
                "id":      c.get("chunk_id", "—"),
                "company": c.get("company",  "—"),
                "page":    c.get("page",     "—"),
                "text":    (c.get("text", "") or "")[:180] + "…",
            }
            for c in sample
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.info("No chunks match the current filter.")
