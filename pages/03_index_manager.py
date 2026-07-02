"""
Index Manager

Responsibilities:
----------------
1. Build Vector Index
2. Build BM25 Index
3. Build All Indexes
4. Delete Indexes
5. Monitor Index Status
"""

import streamlit as st

from streamlit_app.services.indexing_service import (
    build_vector_index,
    build_vectorless_index,
    build_all_indexes,
    delete_vector_index,
    delete_bm25_index,
    delete_all_indexes,
    get_index_status
)


from streamlit_app.auth.protect_page import (
    require_login
)

require_login()
# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Index Manager",
    page_icon="🗂️",
    layout="wide"
)

# ============================================================
# HEADER
# ============================================================

st.title("🗂️ Index Manager")

st.caption(
    "Manage Vector, BM25 and Hybrid retrieval indexes."
)

# ============================================================
# STATUS
# ============================================================

status = get_index_status()

st.subheader("Current Status")

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Vector Index",
        "Available" if status["vector_index"]
        else "Missing"
    )

with col2:

    st.metric(
        "BM25 Index",
        "Available" if status["bm25_index"]
        else "Missing"
    )

with col3:

    hybrid_ready = (
        status["vector_index"]
        and status["bm25_index"]
    )

    st.metric(
        "Hybrid Ready",
        "Yes" if hybrid_ready
        else "No"
    )

# ============================================================
# BUILD INDEXES
# ============================================================

st.divider()

st.subheader("Build Indexes")

col1, col2, col3 = st.columns(3)

# ------------------------------------------------------------
# VECTOR
# ------------------------------------------------------------

if "indexing" not in st.session_state:
    st.session_state.indexing = False

with col1:

    st.markdown("### Vector RAG")

    st.caption(
        """
Uses:

- ChromaDB
- BGE Embeddings
- HNSW
- Cross Encoder
"""
    )

if st.button(
    "Build Vector Index",
    use_container_width=True,
    disabled=st.session_state.indexing
):

    st.session_state.indexing = True

    progress = st.progress(0)

    stage_box = st.empty()

    log_box = st.empty()

    logs = []

    def stage_callback(
        stage,
        current,
        total
    ):

        stage_box.markdown(
            f"""
### Current Stage

**Stage {current}/{total}**

{stage}
"""
        )

    def progress_callback(
        current,
        total
    ):

        percentage = int(
            (current / total) * 100
        )

        progress.progress(
            current / total
        )

        stage_box.markdown(
            f"""
### Current Stage

**Generating Embeddings**

Batch {current}/{total}

**Progress: {percentage}%**
"""
        )

    def log_callback(message):

        logs.append(message)

        log_box.code(
            "\n".join(logs[-20:]),
            language="text"
        )

    try:

        stage_callback(
            "Loading chunks.json",
            1,
            4
        )

        result = build_vector_index(
            progress_callback=progress_callback,
            stage_callback=stage_callback,
            log_callback=log_callback
        )

        progress.progress(1.0)

        if result["success"]:

            stage_box.success(
                result["message"]
            )

            if "vectors" in result:

                st.info(
                    f"Total vectors indexed: "
                    f"{result['vectors']}"
                )

            st.balloons()

            st.rerun()

        else:

            stage_box.error(
                result.get(
                    "error",
                    "Indexing failed."
                )
            )

    except Exception as e:

        stage_box.error(
            f"Error: {str(e)}"
        )

    finally:

        st.session_state.indexing = False
# ------------------------------------------------------------
# BM25
# ------------------------------------------------------------

with col2:

    st.markdown("### Vectorless RAG")

    st.caption(
        """
Uses:

- BM25
- Financial Tokenizer
- Cross Encoder
"""
    )

if st.button(
    "Build BM25 Index",
    use_container_width=True
):

    progress = st.progress(0)
    status = st.empty()

    status.info("Loading chunks.json...")
    progress.progress(30)

    status.info("Building BM25 index...")
    progress.progress(70)

    result = build_vectorless_index()

    progress.progress(100)

    if result["success"]:

        status.success(result["message"])
        st.rerun()

    else:

        status.error(result["error"])

# ------------------------------------------------------------
# ALL
# ------------------------------------------------------------

with col3:

    st.markdown("### Hybrid Ready")

    st.caption(
        """
Creates:

- Vector Index
- BM25 Index

Required for Hybrid RAG
"""
    )

if st.button(
    "Build All Indexes",
    use_container_width=True
):

    progress = st.progress(0)
    status = st.empty()

    status.info("Loading chunks.json...")
    progress.progress(10)

    status.info("Building ChromaDB vector index...")
    progress.progress(35)

    status.info("Generating embeddings...")
    progress.progress(60)

    status.info("Building BM25 index...")
    progress.progress(85)

    result = build_all_indexes()

    progress.progress(100)

    if result["success"]:

        status.success(result["message"])
        st.rerun()

    else:

        status.error(result["error"])

# ============================================================
# DELETE INDEXES
# ============================================================

st.divider()

st.subheader("Delete Indexes")

warning = st.warning(
    """
Deleting indexes does NOT remove PDFs.

Only retrieval indexes are removed.
"""
)

col1, col2, col3 = st.columns(3)

# ------------------------------------------------------------
# DELETE VECTOR
# ------------------------------------------------------------

with col1:

    if st.button(
        "Delete Vector Index",
        type="secondary",
        use_container_width=True
    ):

        result = delete_vector_index()

        if result["success"]:

            st.success(
                result["message"]
            )

            st.rerun()

        else:

            st.error(
                result["error"]
            )

# ------------------------------------------------------------
# DELETE BM25
# ------------------------------------------------------------

with col2:

    if st.button(
        "Delete BM25 Index",
        type="secondary",
        use_container_width=True
    ):

        result = delete_bm25_index()

        if result["success"]:

            st.success(
                result["message"]
            )

            st.rerun()

        else:

            st.error(
    result.get(
        "error",
        result.get(
            "message",
            "Unknown error"
        )
    )
)

# ------------------------------------------------------------
# DELETE ALL
# ------------------------------------------------------------

with col3:

    if st.button(
        "Delete All Indexes",
        type="primary",
        use_container_width=True
    ):

        progress = st.progress(0)
        status = st.empty()

        status.info("Removing ChromaDB...")
        progress.progress(50)

        result = delete_all_indexes()

        status.info("Removing BM25 files...")
        progress.progress(100)

        if result["success"]:

            status.success(
                "All indexes deleted successfully."
            )

            st.rerun()

        else:

            st.error("Delete All Failed")

            st.write("Vector Result:")
            st.json(result["vector"])

            st.write("BM25 Result:")
            st.json(result["bm25"])

# ============================================================
# ARCHITECTURE OVERVIEW
# ============================================================

st.divider()

with st.expander(
    "Retrieval Architecture"
):

    st.code(
"""
Vector RAG
------------------------------------
Question
    ↓
BGE Embedding
    ↓
ChromaDB Search
    ↓
Cross Encoder Rerank
    ↓
LLM

Vectorless RAG
------------------------------------
Question
    ↓
BM25 Search
    ↓
Cross Encoder Rerank
    ↓
LLM

Hybrid RAG
------------------------------------
Question
    ↓
Vector Search
    ↓
BM25 Search
    ↓
RRF Fusion
    ↓
Cross Encoder Rerank
    ↓
LLM
"""
    )

# ============================================================
# HEALTH CHECK
# ============================================================

st.divider()

st.subheader("Health Check")

if hybrid_ready:

    st.success(
        "System ready for Chat page."
    )

else:

    st.warning(
        """
Hybrid RAG is not available yet.

Build both Vector and BM25 indexes.
"""
    )