"""
Sidebar — navigation and global state indicators.
"""

import streamlit as st
from services.pipeline_manager import PipelineManager


_NAV_ITEMS = [
    ("📊", "Dashboard"),
    ("📂", "Upload Documents"),
    ("🗂️", "Index Manager"),
    ("💬", "Chat"),
    ("🗃️", "Conversations"),
    ("🔬", "Evaluation"),
]


def render_sidebar() -> None:
    with st.sidebar:
        st.image(
            "https://img.icons8.com/fluency/96/combo-chart.png",
            width=56,
        )
        st.title("RAG Benchmark")
        st.caption("Financial 10-K Analysis")
        st.divider()

        # ── Page selector ────────────────────────────────────────────────────
        labels = [f"{icon}  {name}" for icon, name in _NAV_ITEMS]
        page_names = [name for _, name in _NAV_ITEMS]

        if "current_page" not in st.session_state:
            st.session_state.current_page = "Dashboard"

        for icon, name in _NAV_ITEMS:
            active = st.session_state.current_page == name
            if st.button(
                f"{icon}  {name}",
                key=f"nav_{name}",
                use_container_width=True,
                type="primary" if active else "secondary",
            ):
                st.session_state.current_page = name
                st.rerun()

        st.divider()

        # ── Pipeline status indicators ────────────────────────────────────────
        st.caption("Pipeline Status")
        pm: PipelineManager = st.session_state.get("pipeline_manager")

        def _status(loaded: bool, label: str):
            icon = "🟢" if loaded else "🔴"
            st.markdown(f"{icon} **{label}**")

        if pm:
            _status(pm.vector_loaded,     "Vector RAG")
            _status(pm.vectorless_loaded, "Vectorless RAG")
            _status(pm.hybrid_loaded,     "Hybrid RAG")
        else:
            for _, name in [("", "Vector RAG"), ("", "Vectorless RAG"), ("", "Hybrid RAG")]:
                _status(False, name)

        st.divider()
        st.caption("v1.0 · Financial 10-K Benchmark")
