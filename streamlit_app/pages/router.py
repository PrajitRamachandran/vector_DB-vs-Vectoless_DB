"""
Page router
===========
Maps page names (set in session state by the sidebar) to their render functions.
"""

import streamlit as st


def route() -> None:
    page = st.session_state.get("current_page", "Dashboard")

    if page == "Dashboard":
        from pages.dashboard import render
    elif page == "Upload Documents":
        from pages.upload_documents import render
    elif page == "Index Manager":
        from pages.index_manager import render
    elif page == "Chat":
        from pages.chat import render
    elif page == "Conversations":
        from pages.conversations import render
    elif page == "Evaluation":
        from pages.evaluation import render
    else:
        def render():
            st.error(f"Unknown page: {page!r}")

    render()
