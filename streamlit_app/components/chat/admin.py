"""
Admin controls: live retrieval parameter tuning and model
selection, gated behind is_admin(). Also houses the financial
query template library used by both admins and regular users.
"""

import streamlit as st

import config


def render_admin_controls() -> dict:
    """
    Renders Top-K / rerank-threshold / chunk-size / model
    controls. Returns a dict of overrides to forward into
    ask_question(..., overrides=..., model_name=...).
    """

    st.caption("Admin — Retrieval Parameters")

    top_k_min, top_k_max = config.ADMIN_TOP_K_RANGE
    fetch_k_min, fetch_k_max = config.ADMIN_FETCH_K_RANGE
    rr_min, rr_max = config.ADMIN_RERANK_THRESHOLD_RANGE

    top_k = st.slider(
        "Top-K (final chunks)",
        top_k_min, top_k_max, config.TOP_K, key="admin_top_k"
    )

    fetch_k = st.slider(
        "Fetch-K (pre-rerank candidates)",
        fetch_k_min, fetch_k_max, config.FETCH_K, key="admin_fetch_k"
    )

    rerank_threshold = st.slider(
        "Rerank score threshold",
        rr_min, rr_max, config.DEFAULT_RERANK_THRESHOLD, 0.05,
        key="admin_rerank_threshold"
    )

    chunk_size_hint = st.select_slider(
        "Chunk size experimentation (child chunk tokens)",
        options=[150, 200, 250, 300, 400, 500],
        value=config.CHILD_CHUNK_SIZE,
        key="admin_chunk_size"
    )

    model_name = st.selectbox(
        "Model",
        config.AVAILABLE_LLM_MODELS,
        index=config.AVAILABLE_LLM_MODELS.index(config.LLM_MODEL_ID)
        if config.LLM_MODEL_ID in config.AVAILABLE_LLM_MODELS else 0,
        key="admin_model_select"
    )

    st.caption(
        "Parameters are only applied when the active pipeline "
        "supports that keyword — unsupported ones are ignored safely."
    )

    return {
        "overrides": {
            "top_k": top_k,
            "fetch_k": fetch_k,
            "rerank_threshold": rerank_threshold,
            "chunk_size_hint": chunk_size_hint,
        },
        "model_name": model_name,
    }


_TEMPLATES = {
    "Financial ratio analysis": (
        "Calculate and explain the key financial ratios "
        "(liquidity, profitability, leverage) for {company} "
        "based on their latest 10-K."
    ),
    "Earnings report analysis": (
        "Summarize {company}'s most recent earnings report, "
        "including revenue, net income, and guidance."
    ),
    "Company comparison": (
        "Compare {company_a} and {company_b} on revenue growth, "
        "margins, and R&D spending."
    ),
    "Annual report analysis": (
        "Provide a structured summary of {company}'s annual "
        "report: business overview, risks, and outlook."
    ),
}


def render_query_templates():
    """
    Returns a filled-in template string if the user picked one
    and clicked 'Use template', otherwise None.
    """

    st.caption("📋 Query Templates")

    choice = st.selectbox(
        "Template", ["None"] + list(_TEMPLATES.keys()), key="query_template_choice"
    )

    if choice == "None":
        return None

    template = _TEMPLATES[choice]

    if "{company_a}" in template:
        company_a = st.text_input("Company A", key="tmpl_company_a")
        company_b = st.text_input("Company B", key="tmpl_company_b")
        filled = template.format(company_a=company_a or "[Company A]", company_b=company_b or "[Company B]")
    else:
        company = st.text_input("Company", key="tmpl_company")
        filled = template.format(company=company or "[Company]")

    st.text_area("Preview", filled, height=80, disabled=True, key="tmpl_preview")

    if st.button("Use template", key="tmpl_use_button"):
        return filled

    return None