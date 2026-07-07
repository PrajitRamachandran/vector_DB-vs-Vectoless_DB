"""
Welcome screen shown when no chat history exists yet.
"""

import streamlit as st


_METHOD_OVERVIEW = [
    (
        "📚 Vector RAG",
        "Dense embedding search over chunked 10-K filings. "
        "Fast and strong on semantic/paraphrased questions."
    ),
    (
        "🔤 Vectorless RAG",
        "Keyword / structural retrieval without embeddings. "
        "Strong on exact figures, names, and section lookups."
    ),
    (
        "🔀 Hybrid RAG",
        "Fuses vector and keyword retrieval (RRF) then reranks. "
        "Best general-purpose default."
    ),
    (
        "🎲 Random",
        "Picks a retrieval method at random per question — "
        "useful for blind A/B comparisons."
    ),
    (
        "🤖 Auto",
        "Classifies the question and routes it to the retrieval "
        "method best suited to that query type."
    ),
]


def render_welcome_screen(starter_questions: list):
    """
    Renders the platform overview + starter questions.
    Returns the starter question text if the user clicked one,
    otherwise None.
    """

    st.markdown(
        """
        <div class="welcome-hero">
            <h3>💬 Welcome to the Financial RAG Benchmark</h3>
            <p style="opacity:0.85;">
            Ask questions about company 10-K filings and compare how
            different retrieval strategies answer them. Pick a method
            from the sidebar, or just start typing below.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("**Supported retrieval methods**")

    cols = st.columns(len(_METHOD_OVERVIEW))
    for col, (title, desc) in zip(cols, _METHOD_OVERVIEW):
        with col:
            st.markdown(
                f"""
                <div class="welcome-method-card">
                <b>{title}</b><br/>
                <span style="font-size:0.82rem; opacity:0.75;">{desc}</span>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("&nbsp;", unsafe_allow_html=True)
    st.markdown("**Try one of these to get started:**")

    clicked = None
    starter_cols = st.columns(2)

    for i, question in enumerate(starter_questions):
        with starter_cols[i % 2]:
            if st.button(question, key=f"starter_{i}", use_container_width=True):
                clicked = question

    return clicked