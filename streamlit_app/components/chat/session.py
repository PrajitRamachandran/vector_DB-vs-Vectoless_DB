"""
Session summary card + system/pipeline health indicators.
"""

import streamlit as st

from streamlit_app.database.repository import get_system_health
from streamlit_app.services.rag_service import pipeline_status


def render_session_summary_card(stats: dict):

    minutes, seconds = divmod(int(stats.get("duration_seconds", 0)), 60)
    duration_str = f"{minutes}m {seconds}s" if minutes else f"{seconds}s"

    cols = st.columns(4)

    with cols[0]:
        st.metric("Messages", stats.get("total_messages", 0))
    with cols[1]:
        st.metric("Session Duration", duration_str)
    with cols[2]:
        st.metric("Avg. Latency", f"{stats.get('avg_latency', 0):.2f}s")
    with cols[3]:
        st.metric("Est. Cost", f"${stats.get('total_cost', 0):.4f}")


def render_health_panel():
    """
    Sidebar system health indicators: DB connectivity (cheap,
    checked every render) and retrieval pipeline health (only
    checked on demand, since instantiating a pipeline that
    isn't cached yet is expensive and shouldn't happen on
    every rerun).
    """

    st.caption("System Health")

    db_health = get_system_health()

    if db_health.get("database"):
        st.success("Database: connected", icon="🟢")
    else:
        st.error("Database: unreachable", icon="🔴")

    if st.button("Check pipeline health", key="check_pipeline_health"):
        st.session_state["_pipeline_health"] = pipeline_status()

    pipelines = st.session_state.get("_pipeline_health")

    if pipelines is None:
        st.caption("Pipeline status not checked yet this session.")
    else:
        for name, ok in pipelines.items():
            icon = "🟢" if ok else "🔴"
            label = "healthy" if ok else "unavailable"
            st.caption(f"{icon} {name.title()} pipeline — {label}")