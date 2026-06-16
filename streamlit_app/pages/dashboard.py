"""
Dashboard Page
==============
High-level overview: corpus stats, index health, and latest evaluation results.
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from services.pipeline_manager import get_pipeline_manager
from services.index_service import load_manifest, load_chunks, get_vector_stats, get_bm25_stats
from services.conversation_store import list_conversations

_ROOT = Path(__file__).resolve().parent.parent.parent
_RESULTS_DIR = _ROOT / "evaluation" / "results"


def _latest_eval_result() -> dict | None:
    """Load the most recently written evaluation JSON, if any."""
    if not _RESULTS_DIR.exists():
        return None
    jsons = sorted(_RESULTS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for j in jsons:
        try:
            return json.loads(j.read_text(encoding="utf-8"))
        except Exception:
            continue
    return None


def render() -> None:
    st.title("📊 Dashboard")
    st.caption("Financial RAG Benchmark — corpus health, pipeline status, and latest results")

    pm = get_pipeline_manager()

    # ── Row 1: corpus + index stats ──────────────────────────────────────────
    manifest = load_manifest()
    chunks   = load_chunks()
    vec_stat = get_vector_stats()
    bm25_stat = get_bm25_stats()
    convs    = list_conversations()

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("📄 PDFs Indexed",     len(manifest))
    col2.metric("🧩 Parent Chunks",    f"{len(chunks.get('parents', [])):,}")
    col3.metric("🔬 Child Chunks",     f"{len(chunks.get('children', [])):,}")
    col4.metric("🧠 Chroma Vectors",   f"{vec_stat['count']:,}")
    col5.metric("💬 Conversations",    len(convs))

    st.divider()

    # ── Row 2: pipeline status cards ─────────────────────────────────────────
    st.subheader("Pipeline Status")
    p1, p2, p3 = st.columns(3)

    def _pipeline_card(col, name: str, loaded: bool, description: str, load_fn):
        with col:
            status = "🟢 Loaded" if loaded else "🔴 Not loaded"
            st.markdown(f"### {name}")
            st.caption(description)
            st.markdown(f"**Status:** {status}")
            if not loaded:
                if st.button(f"Load {name}", key=f"load_{name}", use_container_width=True):
                    with st.spinner(f"Loading {name}…"):
                        try:
                            load_fn()
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Failed: {exc}")

    _pipeline_card(
        p1, "Vector RAG", pm.vector_loaded,
        "ChromaDB · BGE embeddings · HNSW · Parent-Child chunks",
        pm.get_vector,
    )
    _pipeline_card(
        p2, "Vectorless RAG", pm.vectorless_loaded,
        "BM25 keyword search · Financial tokenizer · Cross-encoder rerank",
        pm.get_vectorless,
    )
    _pipeline_card(
        p3, "Hybrid RAG", pm.hybrid_loaded,
        "Reciprocal Rank Fusion · Vector + BM25 · Cross-encoder rerank",
        pm.get_hybrid,
    )

    st.divider()

    # ── Row 3: corpus breakdown ───────────────────────────────────────────────
    if manifest:
        st.subheader("Corpus Breakdown")
        rows = []
        for fname, meta in manifest.items():
            rows.append({
                "File":          fname,
                "Parents":       meta.get("parents_count", "—"),
                "Children":      meta.get("children_count", "—"),
                "Processed At":  meta.get("processed_at", "—")[:19],
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)

    # ── Row 4: latest eval snapshot ──────────────────────────────────────────
    st.divider()
    st.subheader("Latest Evaluation Snapshot")

    result = _latest_eval_result()
    if result is None:
        st.info("No evaluation results found yet. Run an evaluation from the **Evaluation** page.")
    else:
        import pandas as pd
        df = pd.DataFrame(result) if isinstance(result, list) else None
        if df is not None and not df.empty:
            score_cols = [c for c in df.columns if "score" in c.lower() or "f1" in c.lower()]
            if score_cols:
                st.dataframe(
                    df[["method", "question_id", *score_cols]].head(20),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.dataframe(df.head(20), use_container_width=True, hide_index=True)
        else:
            st.json(result)
