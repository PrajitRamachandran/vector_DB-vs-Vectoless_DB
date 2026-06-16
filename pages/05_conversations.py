"""
Conversation Analytics
"""

import pandas as pd
import streamlit as st

from streamlit_app.database.repository import (
    get_conversations,
    get_chunks_for_chat,
    get_logs,
    delete_chat
)

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Conversations",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Conversations")

# ============================================================
# LOAD DATA
# ============================================================

conversations = get_conversations()

if not conversations:

    st.warning(
        "No conversations found."
    )

    st.stop()

df = pd.DataFrame(
    conversations
)

# ============================================================
# METRICS
# ============================================================

st.subheader(
    "Conversation Metrics"
)

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Total Chats",
        len(df)
    )

with col2:

    st.metric(
        "Avg Total Latency",
        round(
            df["total_latency"].fillna(0).mean(),
            2
        )
    )

with col3:

    st.metric(
        "Success Rate",
        round(
            (
                len(
                    df[df["status"]=="SUCCESS"]
                )
                /
                len(df)
            ) * 100,
            1
        )
    )

with col4:

    st.metric(
        "Methods Used",
        df["method"].nunique()
    )

# ============================================================
# FILTERS
# ============================================================

st.divider()

col1, col2 = st.columns(2)

with col1:

    method_filter = st.selectbox(
        "Method",
        options=[
            "All"
        ] + sorted(
            df["method"]
            .dropna()
            .unique()
            .tolist()
        )
    )

with col2:

    search = st.text_input(
        "Search Question"
    )

filtered = df.copy()

if method_filter != "All":

    filtered = filtered[
        filtered["method"]
        == method_filter
    ]

if search:

    filtered = filtered[
        filtered["prompt"]
        .str.contains(
            search,
            case=False,
            na=False
        )
    ]

# ============================================================
# TABLE
# ============================================================

st.divider()

st.subheader(
    "Conversation History"
)

display_columns = [

    "timestamp",
    "method",

    "prompt",

    "retrieval_latency",
    "generation_latency",
    "total_latency",

    "status"
]

st.dataframe(
    filtered[
        display_columns
    ],
    use_container_width=True
)

# ============================================================
# DETAIL VIEW
# ============================================================

st.divider()

st.subheader(
    "Conversation Detail"
)

selected_chat = st.selectbox(

    "Select Chat",

    options=
        filtered["chat_id"]
        .tolist()
)

if selected_chat:

    selected = df[
        df["chat_id"]
        ==
        selected_chat
    ].iloc[0]

    st.markdown(
        "### Question"
    )

    st.write(
        selected["prompt"]
    )

    st.markdown(
        "### Answer"
    )

    st.write(
        selected["response"]
    )

    st.markdown(
        "### Metadata"
    )

    st.json(
        dict(selected)
    )

# ============================================================
# RETRIEVED CHUNKS
# ============================================================

    chunks = get_chunks_for_chat(
        selected_chat
    )

    if chunks:

        st.markdown(
            "### Retrieved Chunks"
        )

        for chunk in chunks:

            with st.expander(
                f"Rank {chunk['rank_position']}"
            ):

                st.write(
                    f"Company: {chunk['company']}"
                )

                st.write(
                    f"Page: {chunk['page']}"
                )

                st.write(
                    f"Score: {chunk['retrieval_score']}"
                )

                st.write(
                    f"Rerank: {chunk['rerank_score']}"
                )

                st.text_area(
                    "Chunk",
                    chunk["chunk_text"],
                    height=200
                )

# ============================================================
# DELETE
# ============================================================

st.divider()

if st.button(
    "Delete Selected Chat"
):

    delete_chat(
        selected_chat
    )

    st.success(
        "Chat deleted."
    )

    st.rerun()

# ============================================================
# EXPORT
# ============================================================

csv = filtered.to_csv(
    index=False
)

st.download_button(

    "Download CSV",

    csv,

    "conversations.csv",

    "text/csv"
)

# ============================================================
# LOGS
# ============================================================

st.divider()

st.subheader(
    "System Logs"
)

logs = get_logs()

if logs:

    st.dataframe(
        pd.DataFrame(logs),
        use_container_width=True
    )