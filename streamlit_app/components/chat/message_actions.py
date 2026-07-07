"""
Per-message action row: copy, regenerate, edit & resend,
thumbs up/down feedback, favorite/bookmark toggles, and a
simple numbered source-citation list under the answer.
"""

import streamlit as st

from streamlit_app.database.repository import (
    save_feedback,
    toggle_favorite,
    toggle_bookmark,
)


def render_message_actions(chat_id: str, answer: str, question: str, key_prefix: str):
    """
    Renders the action row under an assistant message.
    Returns a dict of user intents the caller should act on:
      {"regenerate": bool, "edit_resend": str|None}
    """

    intents = {"regenerate": False, "edit_resend": None}

    cols = st.columns([1, 1, 1, 1, 1, 1])

    with cols[0]:
        if st.button("📋 Copy", key=f"{key_prefix}_copy"):
            st.session_state[f"{key_prefix}_copy_payload"] = answer
            st.toast("Answer copied to the copy buffer below.", icon="📋")

    with cols[1]:
        if st.button("🔁 Regenerate", key=f"{key_prefix}_regen"):
            intents["regenerate"] = True

    with cols[2]:
        if st.button("✏️ Edit & resend", key=f"{key_prefix}_edit_btn"):
            st.session_state[f"{key_prefix}_editing"] = True

    with cols[3]:
        if st.button("👍", key=f"{key_prefix}_up"):
            save_feedback(chat_id, st.session_state.get("user_id"), "up")
            st.toast("Thanks for the feedback!", icon="👍")

    with cols[4]:
        if st.button("👎", key=f"{key_prefix}_down"):
            save_feedback(chat_id, st.session_state.get("user_id"), "down")
            st.toast("Thanks — we'll use this to improve retrieval.", icon="👎")

    with cols[5]:
        fav_key = f"{key_prefix}_fav_state"
        if st.button("⭐ Favorite", key=f"{key_prefix}_fav"):
            st.session_state[fav_key] = toggle_favorite(chat_id)

    if st.session_state.get(f"{key_prefix}_copy_payload"):
        st.code(st.session_state[f"{key_prefix}_copy_payload"], language=None)

    if st.session_state.get(f"{key_prefix}_editing"):
        edited = st.text_area(
            "Edit your question", value=question, key=f"{key_prefix}_edit_text"
        )
        if st.button("Resend", key=f"{key_prefix}_resend_confirm"):
            st.session_state[f"{key_prefix}_editing"] = False
            intents["edit_resend"] = edited

    return intents


def render_citations(retrieved_chunks: list, key_prefix: str):
    """
    Numbered source list under an answer; clicking a citation
    number highlights/expands the matching chunk in the
    Retrieved Chunks panel (via a shared session-state key the
    chunk panel can read).
    """

    if not retrieved_chunks:
        return

    st.caption("Sources")

    marker_html = "".join(
        f'<span class="citation-marker">{i+1}</span>'
        for i in range(len(retrieved_chunks))
    )
    st.markdown(marker_html, unsafe_allow_html=True)

    with st.expander("Jump to a source", expanded=False):
        for i, chunk in enumerate(retrieved_chunks, start=1):
            metadata = chunk.get("metadata", {}) or {}
            label = f"[{i}] {metadata.get('company', 'Unknown')} — {metadata.get('source', '-')} (p.{metadata.get('page', '-')})"
            if st.button(label, key=f"{key_prefix}_cite_{i}"):
                st.session_state[f"{key_prefix}_highlighted_chunk"] = i - 1
                st.info(chunk.get("text", "")[:600])