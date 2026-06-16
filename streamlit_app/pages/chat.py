"""
Chat Page
=========
Interactive RAG chat — pick a method, ask questions, inspect retrieved sources.
Each conversation is stored in ConversationStore and is viewable on the
Conversations page.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from services.pipeline_manager import get_pipeline_manager
from services.conversation_store import (
    new_conversation,
    add_message,
    active_conversation,
    get_conversation,
    list_conversations,
)

_METHOD_LABELS = {
    "vector":     "🧠 Vector RAG (ChromaDB + BGE)",
    "vectorless": "📝 Vectorless RAG (BM25)",
    "hybrid":     "⚡ Hybrid RAG (RRF Fusion)",
}

_EXAMPLE_QUESTIONS = [
    "What was NVIDIA's total revenue in the latest fiscal year?",
    "How much did Microsoft spend on research and development?",
    "What are Netflix's primary sources of revenue?",
    "Describe Amazon's cloud computing segment performance.",
    "What risk factors did NVIDIA highlight related to AI chip demand?",
]


def _render_sources(chunks: list[dict]) -> None:
    """Collapsible source attribution for retrieved chunks."""
    if not chunks:
        return
    with st.expander(f"📚 {len(chunks)} source(s) retrieved", expanded=False):
        for i, chunk in enumerate(chunks, 1):
            meta = chunk.get("metadata", {})
            if not meta:
                # vectorless/hybrid sometimes put metadata at top level
                meta = {k: chunk.get(k) for k in ("company", "page", "source") if k in chunk}
            company = meta.get("company", chunk.get("company", "Unknown"))
            page    = meta.get("page",    chunk.get("page",    "?"))
            score   = chunk.get("rerank_score", chunk.get("score", chunk.get("rrf_score", 0)))
            text    = chunk.get("text", chunk.get("child_text", ""))[:300]

            st.markdown(
                f"**Source {i}** · {company} · Page {page} · Score `{score:.4f}`"
            )
            st.caption(text + ("…" if len(text) == 300 else ""))
            if i < len(chunks):
                st.markdown("---")


def _render_timing(result: dict) -> None:
    r = result.get("retrieval_time", 0) or result.get("retrieval_latency", 0)
    rr = result.get("rerank_time",    0) or result.get("rerank_latency",    0)
    g  = result.get("generation_time", 0)
    t  = result.get("total_time",      0)
    st.caption(
        f"⏱ Retrieval: {r:.3f}s · Rerank: {rr:.3f}s · "
        f"Generation: {g:.3f}s · **Total: {t:.3f}s**"
    )


def render() -> None:
    st.title("💬 Chat")
    st.caption("Ask questions about the 10-K corpus using any RAG pipeline.")

    pm = get_pipeline_manager()

    # ── Sidebar-style controls in top row ────────────────────────────────────
    ctrl1, ctrl2, ctrl3 = st.columns([2, 2, 1])

    with ctrl1:
        method = st.selectbox(
            "RAG method",
            list(_METHOD_LABELS.keys()),
            format_func=lambda m: _METHOD_LABELS[m],
            key="chat_method",
        )

    with ctrl2:
        # Conversation selector
        convs = list_conversations()
        conv_options = {"➕ New conversation": None}
        for c in convs:
            conv_options[f"{c['title'][:40]}  [{c['method']}]"] = c["id"]

        selected_label = st.selectbox(
            "Conversation",
            list(conv_options.keys()),
            key="chat_conv_selector",
        )
        selected_conv_id: Optional[str] = conv_options[selected_label]

    with ctrl3:
        st.write("")  # vertical spacing
        new_conv_btn = st.button("➕ New", use_container_width=True)

    if new_conv_btn:
        conv = new_conversation(method=method)
        st.rerun()

    # ── Resolve active conversation ───────────────────────────────────────────
    if selected_conv_id:
        st.session_state.active_conversation_id = selected_conv_id
        conv = get_conversation(selected_conv_id)
    else:
        conv = active_conversation()

    if conv is None:
        # No conversation exists yet — create one silently
        conv = new_conversation(method=method)

    # ── Render message history ────────────────────────────────────────────────
    st.divider()
    for msg in conv["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            meta = msg.get("metadata", {})
            if msg["role"] == "assistant":
                _render_sources(meta.get("retrieved", []))
                _render_timing(meta)

    # ── Example question buttons (shown only on empty conversations) ──────────
    if not conv["messages"]:
        st.markdown("**Try an example question:**")
        cols = st.columns(len(_EXAMPLE_QUESTIONS))
        for col, q in zip(cols, _EXAMPLE_QUESTIONS):
            if col.button(q[:40] + "…", key=f"ex_{q[:20]}", use_container_width=True):
                st.session_state["_prefill_question"] = q
                st.rerun()

    # ── Chat input ────────────────────────────────────────────────────────────
    prefill = st.session_state.pop("_prefill_question", "")
    question = st.chat_input("Ask a question about the 10-K reports…", key="chat_input")

    if not question and prefill:
        question = prefill

    if question:
        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(question)
        add_message(conv["id"], "user", question)

        # Load pipeline and generate
        with st.chat_message("assistant"):
            status = st.empty()
            status.markdown("_Retrieving and generating…_")

            try:
                pipeline = pm.get_pipeline(conv["method"])
            except Exception as exc:
                status.error(f"Failed to load **{conv['method']}** pipeline: {exc}")
                return

            try:
                result = pipeline.ask(question)
                answer = result.get("answer", "No answer generated.")
                status.markdown(answer)
                _render_sources(result.get("retrieved", []))
                _render_timing(result)

                add_message(
                    conv["id"], "assistant", answer,
                    metadata={
                        "retrieved":       result.get("retrieved", []),
                        "retrieval_time":  result.get("retrieval_time",  0),
                        "rerank_time":     result.get("rerank_time",     0),
                        "generation_time": result.get("generation_time", 0),
                        "total_time":      result.get("total_time",      0),
                    }
                )
            except Exception as exc:
                status.error(f"Pipeline error: {exc}")
                st.exception(exc)
