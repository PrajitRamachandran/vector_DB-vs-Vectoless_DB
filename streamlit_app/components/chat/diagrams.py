"""
Lightweight, dependency-free visual components:
- a simple CSS "boxes and arrows" retrieval workflow diagram
- a multi-stage progress indicator shown while a response
  is being generated (replaces a generic progress bar).
"""

import streamlit as st

_WORKFLOW_STEPS = {
    "vector": ["Question", "Embed Query", "Vector Search (ChromaDB)", "Rerank", "LLM Answer"],
    "vectorless": ["Question", "Keyword Extraction", "BM25 / Structural Search", "Rerank", "LLM Answer"],
    "hybrid": ["Question", "Vector Search", "BM25 Search", "RRF Fusion", "Rerank", "LLM Answer"],
    "random": ["Question", "Random Method Pick", "Retrieve", "Rerank", "LLM Answer"],
    "auto": ["Question", "Query Classification", "Method Routing", "Retrieve", "Rerank", "LLM Answer"],
    "chat": ["Question", "Chat Router", "LLM Answer"],
    "general_knowledge": ["Question", "Router", "LLM Answer"],
}


def render_workflow_diagram(method: str, highlight_index: int = None):

    steps = _WORKFLOW_STEPS.get(method.lower(), _WORKFLOW_STEPS["hybrid"])

    nodes_html = []
    for i, step in enumerate(steps):
        cls = "workflow-node highlight" if i == highlight_index else "workflow-node"
        nodes_html.append(f'<div class="{cls}">{step}</div>')
        if i < len(steps) - 1:
            nodes_html.append('<div class="workflow-arrow">→</div>')

    st.markdown(
        f'<div class="workflow-diagram">{"".join(nodes_html)}</div>',
        unsafe_allow_html=True
    )


_STAGE_LABELS = [
    ("routing", "🧭 Classifying question"),
    ("retrieving", "🔎 Retrieving candidate chunks"),
    ("reranking", "🏅 Reranking results"),
    ("generating", "✍️ Generating answer"),
    ("done", "✅ Done"),
]


def render_stage_tracker(container, current_stage: str):
    """
    Draws a vertical list of pipeline stages, marking the
    current one as active and earlier ones as done.
    """

    stage_keys = [s[0] for s in _STAGE_LABELS]
    current_idx = stage_keys.index(current_stage) if current_stage in stage_keys else 0

    rows = []
    for i, (key, label) in enumerate(_STAGE_LABELS):
        if i < current_idx:
            css = "stage-step done"
            icon = "✔"
        elif i == current_idx:
            css = "stage-step active"
            icon = "▶"
        else:
            css = "stage-step"
            icon = "•"

        rows.append(f'<div class="{css}">{icon} {label}</div>')

    container.markdown("".join(rows), unsafe_allow_html=True)