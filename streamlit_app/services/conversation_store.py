"""
ConversationStore
=================
Lightweight in-memory store for chat conversations, backed by session state.

Each conversation is a dict:
{
    "id":       str,          # uuid4
    "title":    str,          # auto-derived from first question
    "method":   str,          # 'vector' | 'vectorless' | 'hybrid'
    "messages": list[dict],   # [{role, content, metadata}, ...]
    "created_at": str,        # ISO timestamp
}
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

import streamlit as st


def _store() -> list[dict]:
    if "conversations" not in st.session_state:
        st.session_state.conversations = []
    return st.session_state.conversations


def new_conversation(method: str, title: Optional[str] = None) -> dict:
    conv = {
        "id":         str(uuid.uuid4()),
        "title":      title or "New conversation",
        "method":     method,
        "messages":   [],
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    _store().append(conv)
    st.session_state.active_conversation_id = conv["id"]
    return conv


def get_conversation(conv_id: str) -> Optional[dict]:
    return next((c for c in _store() if c["id"] == conv_id), None)


def list_conversations() -> list[dict]:
    return list(reversed(_store()))   # newest first


def add_message(conv_id: str, role: str, content: str, metadata: Optional[dict] = None) -> None:
    conv = get_conversation(conv_id)
    if conv is None:
        return
    conv["messages"].append({
        "role":       role,
        "content":    content,
        "metadata":   metadata or {},
        "timestamp":  datetime.now().isoformat(timespec="seconds"),
    })
    # Auto-title from first user message
    if role == "user" and conv["title"] == "New conversation":
        conv["title"] = content[:60] + ("…" if len(content) > 60 else "")


def delete_conversation(conv_id: str) -> None:
    st.session_state.conversations = [
        c for c in _store() if c["id"] != conv_id
    ]
    if st.session_state.get("active_conversation_id") == conv_id:
        del st.session_state["active_conversation_id"]


def active_conversation() -> Optional[dict]:
    cid = st.session_state.get("active_conversation_id")
    return get_conversation(cid) if cid else None
