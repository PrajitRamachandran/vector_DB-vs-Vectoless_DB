"""
Export panel: lets the user download their current session's
conversations in CSV, Markdown, JSON, or PDF format.
"""

import streamlit as st

from streamlit_app.services import export_service


def render_export_panel(conversations: list, key_prefix: str = "export"):

    if not conversations:
        st.caption("Nothing to export yet.")
        return

    st.caption(f"Export this session ({len(conversations)} messages)")

    cols = st.columns(4)
    formats = [
        ("CSV", "csv", cols[0]),
        ("Markdown", "markdown", cols[1]),
        ("JSON", "json", cols[2]),
        ("PDF", "pdf", cols[3]),
    ]

    for label, fmt, col in formats:
        with col:
            try:
                data, mime, filename = export_service.export(conversations, fmt)
                st.download_button(
                    label,
                    data=data,
                    file_name=filename,
                    mime=mime,
                    key=f"{key_prefix}_{fmt}",
                    use_container_width=True
                )
            except ImportError as e:
                st.caption(f"{label}: unavailable ({e})")