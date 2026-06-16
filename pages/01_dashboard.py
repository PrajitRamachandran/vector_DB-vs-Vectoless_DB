# pages/dashboard.py

from pathlib import Path
import json
import pandas as pd
import streamlit as st

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
    rows.append(
       {
        "PDF": filename,
        "Pages": data.get(
          "pages_count",
          "-"
        ),
        "Chunks": data.get(
          "chunks_count",
          "-"
        ),
        "Processed At": data.get(
          "processed_at",
          "-"
        )
      }
      )

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