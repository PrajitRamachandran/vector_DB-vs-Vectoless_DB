"""
PipelineManager
===============
Centralised singleton that lazy-loads each RAG pipeline.

Design decisions:
- Stored in st.session_state so pipelines survive page navigations.
- Each pipeline is loaded on first use, not at app start, so the UI is
  responsive even before heavy models finish loading.
- `loaded` flags let the sidebar show live status without triggering loads.
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path
from typing import Optional

import streamlit as st

# ── Make the existing RAG package importable ─────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

log = logging.getLogger(__name__)


class PipelineManager:
    """Holds lazy references to all three RAG pipelines."""

    def __init__(self) -> None:
        self._vector: Optional[object] = None
        self._vectorless: Optional[object] = None
        self._hybrid: Optional[object] = None

    # ── Status flags ─────────────────────────────────────────────────────────

    @property
    def vector_loaded(self) -> bool:
        return self._vector is not None

    @property
    def vectorless_loaded(self) -> bool:
        return self._vectorless is not None

    @property
    def hybrid_loaded(self) -> bool:
        return self._hybrid is not None

    # ── Loaders ──────────────────────────────────────────────────────────────

    def get_vector(self):
        if self._vector is None:
            from vector_rag.pipeline import VectorRAGPipeline
            self._vector = VectorRAGPipeline()
        return self._vector

    def get_vectorless(self):
        if self._vectorless is None:
            from vectorless_rag.pipeline import VectorlessRAGPipeline
            self._vectorless = VectorlessRAGPipeline()
        return self._vectorless

    def get_hybrid(self):
        if self._hybrid is None:
            from hybrid_rag.pipeline import HybridRAGPipeline
            self._hybrid = HybridRAGPipeline()
        return self._hybrid

    def get_pipeline(self, method: str):
        """Return the pipeline for `method` ∈ {'vector', 'vectorless', 'hybrid'}."""
        dispatch = {
            "vector":     self.get_vector,
            "vectorless": self.get_vectorless,
            "hybrid":     self.get_hybrid,
        }
        if method not in dispatch:
            raise ValueError(f"Unknown method: {method!r}")
        return dispatch[method]()

    def unload_all(self) -> None:
        """Release all pipeline references (free GPU/RAM)."""
        self._vector = None
        self._vectorless = None
        self._hybrid = None


# ── Session-state helper ─────────────────────────────────────────────────────

def get_pipeline_manager() -> PipelineManager:
    """Return the singleton PipelineManager from session state."""
    if "pipeline_manager" not in st.session_state:
        st.session_state.pipeline_manager = PipelineManager()
    return st.session_state.pipeline_manager
