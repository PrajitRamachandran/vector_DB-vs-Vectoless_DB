"""
Recent-conversations sidebar: search, filter, and reopen past
conversations (loads a past conversation's Q&A back into the
active chat_history so the user can keep going from there).
"""

import streamlit as st

from streamlit_app.database.repository import search_conversations


def render_history_browser(user_id, is_admin: bool):
    """
    Returns the chat_id the user asked to reopen, or None.
    """

    st.subheader("🗂 Conversation History")

    keyword = st.text_input("Search", key="history_search")

    filter_cols = st.columns(2)

    with filter_cols[0]:
        method_filter = st.selectbox(
            "Method",
            ["All", "Hybrid", "Vector", "Vectorless", "Random", "Auto", "CHAT"],
            key="history_method_filter"
        )

    with filter_cols[1]:
        show_only = st.selectbox(
            "Show",
            ["All", "Favorites", "Bookmarked"],
            key="history_show_filter"
        )

    date_cols = st.columns(2)
    with date_cols[0]:
        date_from = st.date_input("From", value=None, key="history_date_from")
    with date_cols[1]:
        date_to = st.date_input("To", value=None, key="history_date_to")

    results = search_conversations(
        user_id=None if is_admin else user_id,
        keyword=keyword or None,
        method_filter=method_filter,
        date_from=str(date_from) if date_from else None,
        date_to=str(date_to) if date_to else None,
        favorites_only=(show_only == "Favorites"),
        bookmarked_only=(show_only == "Bookmarked"),
        limit=25
    )

    if not results:
        st.caption("No matching conversations yet.")
        return None

    reopen_id = None

    for row in results:
        title = row.get("title") or row.get("prompt", "")[:60]
        flags = ""
        if row.get("is_favorite"):
            flags += "⭐"
        if row.get("is_bookmarked"):
            flags += "🔖"

        with st.container(border=True):
            st.markdown(f"**{title}** {flags}")
            st.caption(f"{row.get('timestamp', '-')} · {row.get('method', '-')}")
            if st.button("Reopen", key=f"reopen_{row['chat_id']}", use_container_width=True):
                reopen_id = row["chat_id"]

    return reopen_id