"""
Retrieved chunk rendering.

Handles the "Retrieved Chunks" panel: ranking numbers, score
bars, company/page badges, per-panel filtering/search/sorting,
and collapsed-by-default display so a busy answer doesn't get
buried under a wall of raw context.
"""

import html
import re

import streamlit as st


def _highlight(text: str, terms: list) -> str:
    """
    Wraps any of `terms` found in `text` with <mark> for a
    lightweight "answer-relevant text" highlight. Falls back to
    the escaped, un-highlighted text if no terms are given.
    """

    escaped = html.escape(text)

    if not terms:
        return escaped

    pattern = "|".join(re.escape(t) for t in terms if len(t) > 3)

    if not pattern:
        return escaped

    return re.sub(
        f"({pattern})",
        r"<mark>\1</mark>",
        escaped,
        flags=re.IGNORECASE
    )


def render_chunk_panel(chunks: list, key_prefix: str, answer_terms: list = None):

    if not chunks:
        st.info("No relevant chunks were retrieved for this answer.")
        return

    with st.expander(f"📄 Retrieved Chunks ({len(chunks)})", expanded=False):

        controls = st.columns([2, 2, 2])

        with controls[0]:
            companies = sorted({
                (c.get("metadata", {}) or {}).get("company", "Unknown")
                for c in chunks
            })
            company_filter = st.selectbox(
                "Filter by company",
                ["All"] + companies,
                key=f"{key_prefix}_company_filter"
            )

        with controls[1]:
            min_score = st.slider(
                "Min. relevance score",
                0.0, 1.0, 0.0, 0.05,
                key=f"{key_prefix}_score_filter"
            )

        with controls[2]:
            sort_desc = st.checkbox(
                "Sort by score (desc)",
                value=True,
                key=f"{key_prefix}_sort"
            )

        search_term = st.text_input(
            "🔍 Search within chunks",
            key=f"{key_prefix}_chunk_search"
        )

        working = list(chunks)

        if company_filter != "All":
            working = [
                c for c in working
                if (c.get("metadata", {}) or {}).get("company") == company_filter
            ]

        working = [
            c for c in working
            if (c.get("score") or 0) >= min_score
        ]

        if search_term:
            working = [
                c for c in working
                if search_term.lower() in (c.get("text") or "").lower()
            ]

        if sort_desc:
            working = sorted(
                working, key=lambda c: c.get("score") or 0, reverse=True
            )

        if not working:
            st.warning("No chunks match the current filters.")
            return

        for i, chunk in enumerate(working, start=1):

            metadata = chunk.get("metadata", {}) or {}
            score = chunk.get("score") or 0
            rerank_score = chunk.get("rerank_score")

            st.markdown(f"**#{i}**")

            badge_cols = st.columns([3, 2, 2, 2])

            with badge_cols[0]:
                st.markdown(
                    f'<span class="company-badge">🏢 {metadata.get("company", "Unknown")}</span>'
                    f'<span class="company-badge">📄 {metadata.get("source", "-")}</span>'
                    f'<span class="company-badge">📃 p.{metadata.get("page", "-")}</span>',
                    unsafe_allow_html=True
                )

            with badge_cols[1]:
                st.caption(f"Score: {score:.3f}")
                st.markdown(
                    f'<div class="chunk-score-bar-bg">'
                    f'<div class="chunk-score-bar-fill" style="width:{min(100, max(0, score*100)):.0f}%;">'
                    f'</div></div>',
                    unsafe_allow_html=True
                )

            with badge_cols[2]:
                if rerank_score is not None:
                    st.caption(f"Rerank: {rerank_score:.3f}")

            with badge_cols[3]:
                st.caption(f"Rank #{i}")

            highlighted = _highlight(chunk.get("text", ""), answer_terms)
            st.markdown(
                f'<div style="max-height:180px; overflow-y:auto; '
                f'font-size:0.85rem; padding:8px; border-radius:8px; '
                f'background:rgba(148,163,184,0.06); margin-top:4px;">{highlighted}</div>',
                unsafe_allow_html=True
            )

            st.divider()

        # ------------------------------------------------------
        # Retrieval coverage stats
        # ------------------------------------------------------
        scores = [c.get("score") or 0 for c in chunks]
        avg_score = sum(scores) / len(scores) if scores else 0
        distinct_companies = len({
            (c.get("metadata", {}) or {}).get("company") for c in chunks
        })

        stat_cols = st.columns(3)
        with stat_cols[0]:
            st.metric("Avg. Retrieval Score", f"{avg_score:.3f}")
        with stat_cols[1]:
            st.metric("Companies Represented", distinct_companies)
        with stat_cols[2]:
            st.metric("Chunks Shown", f"{len(working)}/{len(chunks)}")