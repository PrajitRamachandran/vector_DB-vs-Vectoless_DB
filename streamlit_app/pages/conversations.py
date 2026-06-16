"""
Conversations Page
==================
Browse, search, and delete saved conversations.
Click any conversation to open it in a read-only view with full source attribution.
"""

from __future__ import annotations

import streamlit as st

from services.conversation_store import (
    list_conversations,
    get_conversation,
    delete_conversation,
)

_METHOD_BADGE = {
    "vector":     "🧠 Vector",
    "vectorless": "📝 BM25",
    "hybrid":     "⚡ Hybrid",
}


def _render_message(msg: dict) -> None:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            meta = msg.get("metadata", {})
            retrieved = meta.get("retrieved", [])
            if retrieved:
                with st.expander(f"📚 {len(retrieved)} source(s)", expanded=False):
                    for i, chunk in enumerate(retrieved, 1):
                        chunk_meta = chunk.get("metadata", {})
                        company = chunk_meta.get("company", chunk.get("company", "Unknown"))
                        page    = chunk_meta.get("page",    chunk.get("page",    "?"))
                        score   = chunk.get("rerank_score", chunk.get("score", 0))
                        text    = (chunk.get("text", "") or "")[:250]
                        st.caption(
                            f"**{i}.** {company} · Page {page} · Score `{score:.4f}`\n\n"
                            f"{text}…"
                        )
            r = meta.get("retrieval_time", 0)
            g = meta.get("generation_time", 0)
            t = meta.get("total_time", 0)
            if t:
                st.caption(f"⏱ Retrieval {r:.3f}s · Generation {g:.3f}s · Total {t:.3f}s")


def render() -> None:
    st.title("🗃️ Conversations")
    st.caption("Browse and replay past RAG conversations.")

    convs = list_conversations()

    if not convs:
        st.info("No conversations yet. Start one from the **Chat** page.")
        return

    # ── Search ────────────────────────────────────────────────────────────────
    search = st.text_input("🔍 Search conversations", placeholder="Type to filter by title…")
    method_filter = st.multiselect(
        "Filter by method",
        ["vector", "vectorless", "hybrid"],
        default=["vector", "vectorless", "hybrid"],
        format_func=lambda m: _METHOD_BADGE.get(m, m),
    )

    filtered = [
        c for c in convs
        if c["method"] in method_filter
        and (not search or search.lower() in c["title"].lower())
    ]

    st.markdown(f"**{len(filtered)}** conversation(s)")
    st.divider()

    # ── Conversation list ─────────────────────────────────────────────────────
    if "selected_conv_id" not in st.session_state:
        st.session_state.selected_conv_id = None

    list_col, detail_col = st.columns([1, 2])

    with list_col:
        for conv in filtered:
            badge = _METHOD_BADGE.get(conv["method"], conv["method"])
            n_msgs = len(conv["messages"])
            label  = f"{badge} · {conv['title'][:35]}"
            sub    = f"{conv['created_at']} · {n_msgs} messages"

            is_selected = st.session_state.selected_conv_id == conv["id"]

            with st.container():
                btn_col, del_col = st.columns([5, 1])
                with btn_col:
                    if st.button(
                        label,
                        key=f"sel_{conv['id']}",
                        use_container_width=True,
                        type="primary" if is_selected else "secondary",
                    ):
                        st.session_state.selected_conv_id = conv["id"]
                        st.rerun()
                with del_col:
                    if st.button("🗑", key=f"del_{conv['id']}", help="Delete conversation"):
                        delete_conversation(conv["id"])
                        if st.session_state.selected_conv_id == conv["id"]:
                            st.session_state.selected_conv_id = None
                        st.rerun()
                st.caption(sub)

    # ── Detail view ───────────────────────────────────────────────────────────
    with detail_col:
        cid = st.session_state.selected_conv_id
        if cid:
            conv = get_conversation(cid)
            if conv:
                badge = _METHOD_BADGE.get(conv["method"], conv["method"])
                st.subheader(conv["title"])
                st.caption(f"{badge} · {conv['created_at']}")
                st.divider()

                if conv["messages"]:
                    for msg in conv["messages"]:
                        _render_message(msg)
                else:
                    st.info("This conversation has no messages.")
            else:
                st.info("Select a conversation from the list.")
        else:
            st.info("Select a conversation from the list.")
