"""
Chat Interface

Financial RAG Benchmark

Supports:
- Vector RAG
- Vectorless RAG
- Hybrid RAG
"""

import time
import streamlit as st

from streamlit_app.services.rag_service import (
    ask_question,
    clear_pipeline_cache
)

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Chat",
    page_icon="💬",
    layout="wide"
)

# ============================================================
# SESSION STATE
# ============================================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ============================================================
# HEADER
# ============================================================

st.title("💬 Financial RAG Chat")

st.caption(
    "Ask questions about company 10-K reports."
)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.subheader("Retrieval Method")

    retrieval_method = st.radio(
        label="Choose Method",
        options=[
            "Hybrid",
            "Vector",
            "Vectorless"
        ],
        index=0
    )

    st.divider()

    st.subheader("Utilities")

    if st.button(
        "♻ Reload Pipelines",
        use_container_width=True
    ):

        clear_pipeline_cache()

        st.success(
            "Pipeline cache cleared."
        )

    if st.button(
        "🗑 Clear Chat",
        use_container_width=True
    ):

        st.session_state.chat_history = []

        st.rerun()

# ============================================================
# QUESTION INPUT
# ============================================================

question = st.chat_input(
    "Ask a financial question..."
)

# ============================================================
# DISPLAY HISTORY
# ============================================================

for msg in st.session_state.chat_history:

    with st.chat_message(msg["role"]):

        st.markdown(msg["content"])

# ============================================================
# ASK QUESTION
# ============================================================

if question:

    # USER MESSAGE

    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):

        st.markdown(question)

    # ASSISTANT MESSAGE

    with st.chat_message("assistant"):

        progress = st.progress(0)

        status = st.empty()

        start_time = time.time()

        status.info(
            f"Running {retrieval_method} retrieval..."
        )

        progress.progress(25)

        result = ask_question(
            question=question,
            method=retrieval_method.lower()
        )

        progress.progress(100)

        elapsed = round(
            time.time() - start_time,
            2
        )

        if result["success"]:

            rag_result = result["result"]

            answer = rag_result.get(
                "answer",
                "No answer generated."
            )

            st.markdown(answer)

            st.caption(
                f"Response Time: {elapsed}s"
            )

            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": answer
                }
            )

            # ====================================================
            # RETRIEVED CHUNKS
            # ====================================================

            retrieved = rag_result.get(
                "retrieved",
                []
            )

            if retrieved:

                with st.expander(
                    f"Retrieved Chunks ({len(retrieved)})"
                ):

                    for i, chunk in enumerate(
                        retrieved,
                        start=1
                    ):

                        st.markdown(
                            f"### Chunk {i}"
                        )

                        if isinstance(
                            chunk,
                            dict
                        ):

                            metadata = chunk.get(
                                "metadata",
                                {}
                            )

                            st.write(
                                f"Company: "
                                f"{metadata.get('company', '-')}"
                            )

                            st.write(
                                f"Score: "
                                f"{chunk.get('score', '-')}"
                            )

                            st.text_area(
                                label=f"Text {i}",
                                value=chunk.get(
                                    "text",
                                    ""
                                ),
                                height=150,
                                disabled=True
                            )

                            st.divider()

            # ====================================================
            # RAW RESPONSE
            # ====================================================

            with st.expander(
                "Raw Pipeline Output"
            ):

                st.json(
                    rag_result
                )

        else:

            st.error(
                result["error"]
            )

            with st.expander(
                "Traceback"
            ):

                st.code(
                    result["traceback"]
                )