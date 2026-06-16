"""
IndexService
============
Thin wrapper around the existing data_loader, vector_rag.indexer, and
vectorless_rag.indexer modules.  All heavy imports are deferred so the
Streamlit UI remains responsive.
"""

from __future__ import annotations

import sys
import json
from pathlib import Path
from typing import Any

import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ── Preprocessing ─────────────────────────────────────────────────────────────

def run_preprocessing() -> dict:
    """Run the PDF → chunks pipeline and return the chunk dict."""
    from data_loader import run_preprocessing_pipeline
    return run_preprocessing_pipeline()


def load_chunks() -> dict:
    from data_loader import load_existing_chunks
    return load_existing_chunks()


def load_manifest() -> dict:
    from data_loader import load_manifest
    return load_manifest()


# ── Vector index ─────────────────────────────────────────────────────────────

def build_vector_index(status_container=None) -> None:
    """Build / rebuild the ChromaDB vector index."""
    from vector_rag.indexer import build_index
    data = load_chunks()
    children = data.get("children", [])
    if status_container:
        status_container.info(f"Indexing {len(children):,} child chunks into ChromaDB…")
    build_index(children)
    if status_container:
        status_container.success("✅ Vector index built.")


def get_vector_stats() -> dict[str, Any]:
    try:
        from vector_rag.indexer import load_index
        col = load_index()
        return {"count": col.count(), "status": "ready"}
    except Exception as exc:
        return {"count": 0, "status": f"error: {exc}"}


# ── BM25 index ────────────────────────────────────────────────────────────────

def build_bm25_index(status_container=None) -> None:
    """Build / rebuild the BM25 index."""
    from vectorless_rag.indexer import build_bm25_index as _build
    data = load_chunks()
    children = data.get("children", [])
    if status_container:
        status_container.info(f"Building BM25 index over {len(children):,} child chunks…")
    _build(children)
    if status_container:
        status_container.success("✅ BM25 index built.")


def get_bm25_stats() -> dict[str, Any]:
    try:
        from vectorless_rag.indexer import load_bm25_index
        _, children = load_bm25_index()
        return {"count": len(children), "status": "ready"}
    except Exception as exc:
        return {"count": 0, "status": f"error: {exc}"}
