# pages/dashboard.py
import plotly.graph_objects as go
from pathlib import Path
import json
import pandas as pd
import streamlit as st
from datetime import datetime
from streamlit_app.database.repository import (
    get_conversations,
    get_evaluations,
    get_dashboard_stats
)

# Page Config

st.set_page_config(
  page_title="Dashboard",
  page_icon="$$$",
  layout="wide"
)

# Paths Setting

ROOT_DIR = Path(__file__).resolve().parent.parent

MANIFEST_PATH = (
    ROOT_DIR /
    "data" /
    "processed" /
    "manifest.json"
)

CHUNKS_PATH = (
    ROOT_DIR /
    "data" /
    "processed" /
    "chunks.json"
)

RESULTS_DIR = (
    ROOT_DIR /
    "evaluation" /
    "results"
)

# Helpers

@st.cache_data
def load_manifest():
  # load manifest.json
  if not MANIFEST_PATH.exists():
    return json.load(f)
  with open(MANIFEST_PATH, "r" , encoding = "utf-8") as f:
    return json.load(f)
  
@st.cache_data
def load_chunks():
  #loads chunks.json
  if not CHUNKS_PATH.exists():
    return{
      "parents" :[],
      "children" : []
    }
  
  with open(CHUNKS_PATH,"r", encoding = "utf-8") as f:
    return json.load(f)
  
@st.cache_data
def get_company_stats(children):
  #Company distribution

  stats = {}

  for chunk in children:
    company = chunk.get(
      "company",
      "Unknown"
    )

    stats[company] = (
      stats.get(company,0) + 1
    )

  return stats

@st.cache_data
def get_latest_result_file():
  csv_files = list(
    RESULTS_DIR.glob("*.csv")
  )

  if not csv_files:
    return None
  return max(csv_files,key = lambda x: x.stat().st_mtime)


# Load Data

manifest = load_manifest()
chunk_data = load_chunks()
parents = chunk_data.get(
  "parents",[]
)
children = chunk_data.get(
  "children",[]
)

company_stats = get_company_stats(children)
latest_result = get_latest_result_file()

# Header

st.title("Dashboard")

st.caption(
    "Financial RAG Benchmark Monitoring"
)
# Top Metrics

stats = get_dashboard_stats()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Chats",
        stats["total_chats"]
    )

with col2:
    st.metric(
        "Methods Used",
        stats["methods_used"]
    )

with col3:
    st.metric(
        "Avg Latency",
        round(
            stats["avg_latency"] or 0,
            2
        )
    )

with col4:
    st.metric(
        "Evaluation Runs",
        stats["evaluation_runs"]
    )

pdf_count = len(manifest)
parent_count = len(parents)
child_count = len(children)

evaluation_count = len(
    list(RESULTS_DIR.glob("*.csv"))
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "PDF Documents",
        pdf_count
    )

with col2:
    st.metric(
        "Parent Chunks",
        parent_count
    )

with col3:
    st.metric(
        "Child Chunks",
        child_count
    )

with col4:
    st.metric(
        "Evaluation Files",
        evaluation_count
    )


# Dataset Section

st.divider()
st.subheader("Dataset Overview")

if manifest:
  rows = []
  for filename, data in manifest.items():
    rows.append({
    "PDF": filename,

    "Pages":
        data.get(
            "pages_count",
            "-"
        ),

    "Parent Chunks":
        data.get(
            "parents_count",
            0
        ),

    "Child Chunks":
        data.get(
            "children_count",
            0
        ),

    "Total Chunks":
        data.get(
            "parents_count",
            0
        ) +
        data.get(
            "children_count",
            0
        ),

    "Processed At":
        data.get(
            "processed_at",
            "-"
        )
})

  df = pd.DataFrame(rows)
  st.dataframe(
      df,
      use_container_width=True
  )

else:
  st.warning(
    "No processed documents found."
  )


# Company Distribution

st.divider()
st.subheader(
  "Company Chunk Distribution"
)

if company_stats:
  chart_df = pd.DataFrame(
  {
    "Company": list(
        company_stats.keys()
    ),
    "Chunks": list(
        company_stats.values()
    )
  }
)

  st.bar_chart(
    chart_df.set_index("Company")
  )

else:
  st.info(
    "No chunk data available."
  )


# Chunk Details

st.divider()
st.subheader(
  "Chunk Statistics"
)

left, right = st.columns(2)

with left:
  st.info(
    f"Parents: {parent_count}"
  )

with right:
  st.info(
    f"Children: {child_count}"
  )


# Evaluation Status

st.divider()
st.subheader(
  "Evaluation Status"
)

if latest_result:
  st.success(
    f"Latest Results: "
    f"{latest_result.name}"
  )
  st.caption(
    f"Modified: "
    f"{pd.to_datetime(latest_result.stat().st_mtime, unit='s')}"
  )

else:
  st.warning(
    "No evaluation results found."
  )

# ============================================================
# EVALUATION SUMMARY
# ============================================================

evaluations = get_evaluations()

if not evaluations:

    st.info(
        "No benchmark results available yet."
    )

else:

    leaderboard_df = pd.DataFrame(
        evaluations
    )

    leaderboard_df = (
        leaderboard_df
        .sort_values(
            "overall_score",
            ascending=False
        )
    )

        
    # Benchmark LeaderBoard

    st.subheader(
        "Method Comparison Radar"
    )

    metrics = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall"
    ]

    fig = go.Figure()

    for _, row in leaderboard_df.iterrows():

        fig.add_trace(
            go.Scatterpolar(
                r=[
                    row[m]
                    for m in metrics
                ],

                theta=[
                    "Faithfulness",
                    "Answer Relevancy",
                    "Context Precision",
                    "Context Recall"
                ],

                fill="toself",

                name=row["method"]
            )
        )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0,1]
            )
        ),
        showlegend=True
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # Recent Chats

    st.divider()

    st.subheader(
        "Recent Conversations"
    )

    recent = get_conversations()

    if len(recent) > 0:

        recent_df = pd.DataFrame(
            recent
        )

        display_cols = [

            col

            for col in [
                "timestamp",
                "method",
                "prompt"
            ]

            if col in recent_df.columns
        ]

        st.dataframe(
            recent_df[
                display_cols
            ].head(10),
            use_container_width=True
        )

    else:

        st.info(
            "No conversations found."
        )


# System Health

st.divider()
st.subheader(
  "System Health"
)

checks = {
  "Manifest":
    MANIFEST_PATH.exists(),

  "Chunks":
    CHUNKS_PATH.exists(),

  "Evaluation Directory":
    RESULTS_DIR.exists(),
}

for name, status in checks.items():
  if status:
    st.success(f"{name} ✓")
  else:
    st.error(f"{name} ✗")


# Quick Summary

st.divider()
st.subheader(
  "Benchmark Summary"
)

st.markdown(
f"""
**Documents Loaded:** {pdf_count}

**Parent Chunks:** {parent_count}

**Child Chunks:** {child_count}

**Companies:** {len(company_stats)}

**Evaluation Files:** {evaluation_count}
"""
)