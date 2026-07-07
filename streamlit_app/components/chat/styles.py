"""
Chat page styling.

Centralizes all custom CSS so the page module stays focused
on layout/logic instead of markup.
"""

import streamlit as st

CHAT_CSS = """
<style>
.method-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    margin-right: 6px;
}
.badge-hybrid      { background: rgba(99,102,241,0.18);  color: #a5b4fc; border: 1px solid rgba(99,102,241,0.4); }
.badge-vector      { background: rgba(59,130,246,0.18);  color: #93c5fd; border: 1px solid rgba(59,130,246,0.4); }
.badge-vectorless  { background: rgba(16,185,129,0.18);  color: #6ee7b7; border: 1px solid rgba(16,185,129,0.4); }
.badge-random      { background: rgba(234,179,8,0.18);   color: #fde047; border: 1px solid rgba(234,179,8,0.4); }
.badge-auto        { background: rgba(236,72,153,0.18);  color: #f9a8d4; border: 1px solid rgba(236,72,153,0.4); }
.badge-chat        { background: rgba(148,163,184,0.18); color: #cbd5e1; border: 1px solid rgba(148,163,184,0.4); }
.badge-general     { background: rgba(56,189,248,0.18);  color: #7dd3fc; border: 1px solid rgba(56,189,248,0.4); }

.confidence-high   { color: #4ade80; font-weight: 600; }
.confidence-medium { color: #facc15; font-weight: 600; }
.confidence-low    { color: #f87171; font-weight: 600; }

.risk-low    { color: #4ade80; }
.risk-medium { color: #facc15; }
.risk-high   { color: #f87171; }
.risk-unknown{ color: #94a3b8; }

.session-card {
    background: rgba(148,163,184,0.06);
    border: 1px solid rgba(148,163,184,0.22);
    border-radius: 14px;
    padding: 14px 18px;
    margin-bottom: 10px;
}

.welcome-hero {
    background: linear-gradient(135deg, rgba(99,102,241,0.12), rgba(59,130,246,0.05));
    border: 1px solid rgba(148,163,184,0.25);
    border-radius: 16px;
    padding: 28px 26px;
    margin-bottom: 18px;
}
.welcome-method-card {
    border: 1px solid rgba(148,163,184,0.25);
    border-radius: 12px;
    padding: 14px;
    background: rgba(148,163,184,0.05);
    height: 100%;
}

.msg-container {
    border: 1px solid rgba(148,163,184,0.18);
    border-radius: 12px;
    padding: 12px 16px;
    margin-bottom: 6px;
    background: rgba(148,163,184,0.04);
}
.msg-timestamp {
    font-size: 0.72rem;
    opacity: 0.55;
    margin-top: 4px;
}

.stage-step {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 4px 0;
    font-size: 0.85rem;
    opacity: 0.55;
}
.stage-step.active { opacity: 1; font-weight: 600; }
.stage-step.done { opacity: 0.85; }

.workflow-diagram {
    display: flex;
    align-items: center;
    gap: 6px;
    flex-wrap: wrap;
    padding: 14px 4px;
}
.workflow-node {
    border: 1px solid rgba(148,163,184,0.3);
    border-radius: 10px;
    padding: 8px 14px;
    font-size: 0.8rem;
    background: rgba(148,163,184,0.06);
    white-space: nowrap;
}
.workflow-node.highlight {
    border-color: rgba(99,102,241,0.6);
    background: rgba(99,102,241,0.15);
    font-weight: 600;
}
.workflow-arrow { opacity: 0.5; font-size: 1rem; }

.chunk-score-bar-bg {
    background: rgba(148,163,184,0.18);
    border-radius: 6px;
    height: 8px;
    width: 100%;
    overflow: hidden;
}
.chunk-score-bar-fill {
    height: 100%;
    border-radius: 6px;
    background: linear-gradient(90deg, #6366f1, #22d3ee);
}
.company-badge {
    display: inline-block;
    background: rgba(148,163,184,0.15);
    border: 1px solid rgba(148,163,184,0.3);
    border-radius: 8px;
    padding: 2px 8px;
    font-size: 0.72rem;
    margin-right: 6px;
}
.citation-marker {
    display: inline-block;
    background: rgba(99,102,241,0.2);
    color: #a5b4fc;
    border-radius: 999px;
    padding: 0 7px;
    font-size: 0.72rem;
    font-weight: 700;
    margin-left: 2px;
}
</style>
"""


def inject_chat_styles():
    st.markdown(CHAT_CSS, unsafe_allow_html=True)


def method_badge_html(label: str) -> str:

    key = label.lower()

    css_class = {
        "hybrid": "badge-hybrid",
        "vector": "badge-vector",
        "vectorless": "badge-vectorless",
        "random": "badge-random",
        "auto": "badge-auto",
        "chat": "badge-chat",
        "general_knowledge": "badge-general",
    }.get(key.split("(")[0].strip(), "badge-chat")

    return f'<span class="method-badge {css_class}">{label}</span>'


def confidence_html(confidence: float, level: str) -> str:

    return (
        f'<span class="confidence-{level}">'
        f'{confidence * 100:.0f}% confidence</span>'
    )


def risk_html(risk: str) -> str:

    return f'<span class="risk-{risk}">⚠ {risk.title()} hallucination risk</span>'