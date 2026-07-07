# pages/dashboard.py

from streamlit_app.auth.protect_page import (
    require_login
)

require_login()

import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import json
import pandas as pd
import streamlit as st
from datetime import datetime

from streamlit_app.database.repository import (
    get_conversations,
    get_evaluations,
    get_dashboard_stats,
    get_dashboard_stats_by_user,
    get_recent_conversations_by_user,
    get_best_method,
    get_latency_breakdown,
    get_top_queries,
    get_top_companies,
    get_user_analytics,
    get_evaluation_trend,
    get_recent_activity
)

from streamlit_app.auth.session_manager import (
    is_admin
)

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Dashboard",
    page_icon="📊",
    layout="wide"
)

# ============================================================
# PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

MANIFEST_PATH = ROOT_DIR / "data" / "processed" / "manifest.json"
CHUNKS_PATH = ROOT_DIR / "data" / "processed" / "chunks.json"
RESULTS_DIR = ROOT_DIR / "evaluation" / "results"

METRIC_LABELS = {
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer Relevancy",
    "context_precision": "Context Precision",
    "context_recall": "Context Recall"
}

ARCHITECTURES = [
    "Vector",
    "Vectorless",
    "Hybrid"
]

# ============================================================
# STYLES
# ============================================================

st.markdown(
    """
    <style>
    .hero-card {
        background: linear-gradient(135deg, rgba(99,102,241,0.12), rgba(59,130,246,0.06));
        border: 1px solid rgba(148,163,184,0.25);
        border-radius: 14px;
        padding: 18px 20px;
        text-align: left;
    }
    .hero-label {
        font-size: 0.8rem;
        opacity: 0.7;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 4px;
    }
    .hero-value {
        font-size: 1.9rem;
        font-weight: 700;
        line-height: 1.1;
    }
    .hero-sub {
        font-size: 0.8rem;
        opacity: 0.65;
        margin-top: 6px;
    }
    .status-card {
        border-radius: 12px;
        padding: 14px 16px;
        border: 1px solid rgba(148,163,184,0.25);
        margin-bottom: 8px;
    }
    .status-ok { background: rgba(34,197,94,0.10); border-color: rgba(34,197,94,0.35); }
    .status-fail { background: rgba(239,68,68,0.10); border-color: rgba(239,68,68,0.35); }
    .arch-card {
        border-radius: 12px;
        padding: 14px;
        border: 1px solid rgba(148,163,184,0.25);
        text-align: center;
        background: rgba(148,163,184,0.06);
    }
    .empty-state {
        border: 1px dashed rgba(148,163,184,0.4);
        border-radius: 14px;
        padding: 36px 20px;
        text-align: center;
        opacity: 0.85;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# HELPERS
# ============================================================

@st.cache_data(ttl=60)
def load_manifest():
    try:
        if not MANIFEST_PATH.exists():
            return {}
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading manifest: {e}")
        return {}


@st.cache_data(ttl=60)
def load_chunks():
    try:
        if not CHUNKS_PATH.exists():
            return {"parents": [], "children": []}
        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading chunks: {e}")
        return {"parents": [], "children": []}


@st.cache_data(ttl=60)
def get_company_stats(children):
    stats = {}
    for chunk in children:
        company = chunk.get("company", "Unknown")
        stats[company] = stats.get(company, 0) + 1
    return stats


@st.cache_data(ttl=60)
def get_latest_result_file():
    csv_files = list(RESULTS_DIR.glob("*.csv"))
    if not csv_files:
        return None
    return max(csv_files, key=lambda x: x.stat().st_mtime)


def fmt_seconds(value):
    try:
        return f"{value:.2f}s"
    except (TypeError, ValueError):
        return "-"


def fmt_bytes(num_bytes):
    for unit in ["B", "KB", "MB", "GB"]:
        if num_bytes < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TB"


def dataset_growth_by_date(manifest):
    """Derive growth trend from processed_at dates already in manifest."""
    rows = []
    for filename, data in manifest.items():
        processed_at = data.get("processed_at")
        if not processed_at:
            continue
        try:
            date = pd.to_datetime(processed_at).date()
        except Exception:
            continue
        rows.append({
            "date": date,
            "chunks": data.get("parents_count", 0) + data.get("children_count", 0)
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).groupby("date").agg(
        docs=("date", "count"),
        chunks=("chunks", "sum")
    ).sort_index()

    df["cumulative_docs"] = df["docs"].cumsum()
    df["cumulative_chunks"] = df["chunks"].cumsum()

    return df.reset_index()


# ============================================================
# LOAD DATA
# ============================================================

manifest = load_manifest()
chunk_data = load_chunks()
parents = chunk_data.get("parents", [])
children = chunk_data.get("children", [])

company_stats = get_company_stats(children)
latest_result = get_latest_result_file()
best_method = get_best_method()
evaluations = get_evaluations()
latency_breakdown = get_latency_breakdown()

pdf_count = len(manifest)
parent_count = len(parents)
child_count = len(children)
evaluation_count = len(list(RESULTS_DIR.glob("*.csv")))

if is_admin():
    stats = get_dashboard_stats()
else:
    stats = get_dashboard_stats_by_user(st.session_state.user_id)

# ============================================================
# HEADER + QUICK ACTIONS
# ============================================================

header_col, action_col = st.columns([3, 2])

with header_col:
    st.title("📊 Dashboard")
    st.caption("Financial RAG Benchmark Monitoring")

def _go_to_page(page_path: str, label: str):
    try:
        st.switch_page(page_path)
    except Exception:
        st.info(f"{label} page not found at '{page_path}'.")


with action_col:
    st.write("")
    a1, a2, a3, a4 = st.columns(4)
    with a1:
        if st.button("📤 Upload", use_container_width=True):
            _go_to_page("pages/upload.py", "Upload")
    with a2:
        if st.button("🚀 Run Benchmark", use_container_width=True):
            _go_to_page("pages/evaluation.py", "Evaluation")
    with a3:
        if st.button("📈 Analytics", use_container_width=True):
            _go_to_page("pages/analytics.py", "Analytics")
    with a4:
        if st.button("🗑️ Clear Results", use_container_width=True):
            st.session_state["confirm_clear"] = True

if st.session_state.get("confirm_clear"):
    st.warning("This will not delete files automatically. Please confirm manually in the Settings page.")
    st.session_state["confirm_clear"] = False

auto_refresh = st.toggle("🔄 Auto-refresh (30s)", value=False)
if auto_refresh:
    st.markdown(
        "<meta http-equiv='refresh' content='30'>",
        unsafe_allow_html=True
    )

st.caption(f"Last checked: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ============================================================
# EXECUTIVE SUMMARY
# ============================================================

st.divider()

with st.container(border=True):
    if best_method:
        summary = (
            f"Corpus contains **{sum(d.get('pages_count', 0) for d in manifest.values())} pages** "
            f"across **{len(company_stats)} companies**. "
            f"**{best_method['method']}** retrieval currently leads with an overall score of "
            f"**{best_method['overall_score']:.2f}**."
        )
    else:
        summary = (
            f"Corpus contains **{sum(d.get('pages_count', 0) for d in manifest.values())} pages** "
            f"across **{len(company_stats)} companies**. No benchmark results available yet."
        )
    st.markdown(f"### 📝 Executive Summary\n{summary}")

# ============================================================
# HERO KPI SECTION
# ============================================================

st.divider()
st.subheader("Hero Metrics")

hero_cols = st.columns(4)

hero_data = [
    ("Total Chats", stats["total_chats"], "All conversations logged"),
    ("Avg Latency", fmt_seconds(stats["avg_latency"] or 0), "Across all methods"),
    ("Best Method", best_method["method"] if best_method else "N/A",
     f"Score: {best_method['overall_score']:.2f}" if best_method else "No evaluations yet"),
    ("Corpus Size", f"{pdf_count} PDFs", f"{parent_count + child_count} total chunks"),
]

for col, (label, value, sub) in zip(hero_cols, hero_data):
    with col:
        st.markdown(
            f"""
            <div class="hero-card">
                <div class="hero-label">{label}</div>
                <div class="hero-value">{value}</div>
                <div class="hero-sub">{sub}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

# ============================================================
# BENCHMARK SCORE KPI CARDS
# ============================================================

st.divider()
st.subheader("Best Run Benchmark Scores")

if best_method:
    b1, b2, b3, b4 = st.columns(4)
    with b1:
        st.metric("Overall Score", round(best_method["overall_score"], 3))
    with b2:
        st.metric("Faithfulness", round(best_method["faithfulness"], 3))
    with b3:
        st.metric("Answer Relevancy", round(best_method["answer_relevancy"], 3))
    with b4:
        st.metric("Context Recall", round(best_method["context_recall"], 3))
else:
    st.markdown(
        """
        <div class="empty-state">
        📉 <b>No benchmark results yet</b><br>
        Run an evaluation to populate benchmark scores.
        </div>
        """,
        unsafe_allow_html=True
    )

# ============================================================
# SYSTEM HEALTH (moved near top, status cards)
# ============================================================

st.divider()
st.subheader("System Health")

checks = {
    "Manifest": MANIFEST_PATH.exists(),
    "Chunks": CHUNKS_PATH.exists(),
    "Evaluation Directory": RESULTS_DIR.exists(),
}

health_cols = st.columns(len(checks))
for col, (name, status) in zip(health_cols, checks.items()):
    css_class = "status-ok" if status else "status-fail"
    icon = "🟢" if status else "🔴"
    with col:
        st.markdown(
            f"""
            <div class="status-card {css_class}">
                <b>{icon} {name}</b><br>
                <span style="font-size:0.85rem;opacity:0.75;">
                    {"Operational" if status else "Missing / Unavailable"}
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )

# ============================================================
# RETRIEVAL ARCHITECTURE OVERVIEW
# ============================================================

st.divider()
st.subheader("Retrieval Architecture Overview")

methods_used_raw = {row.get("method") for row in evaluations} if evaluations else set()
methods_used = {m.strip().lower().replace(" ", "_") for m in methods_used_raw if m}

arch_cols = st.columns(len(ARCHITECTURES))
for col, arch in zip(arch_cols, ARCHITECTURES):
    normalized_arch = arch.strip().lower().replace(" ", "_")
    available = normalized_arch in methods_used
    icon = "✅" if available else "⚪"
    with col:
        st.markdown(
            f"""
            <div class="arch-card">
                <div style="font-size:1.4rem;">{icon}</div>
                <b>{arch}</b><br>
                <span style="font-size:0.8rem;opacity:0.7;">
                    {"Benchmarked" if available else "Not yet run"}
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )

# ============================================================
# TOP METRICS (secondary)
# ============================================================

st.divider()
st.subheader("Corpus Metrics")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("PDF Documents", pdf_count)
with col2:
    st.metric("Parent Chunks", parent_count)
with col3:
    st.metric("Child Chunks", child_count)
with col4:
    st.metric("Evaluation Files", evaluation_count)

# ============================================================
# DATASET OVERVIEW (searchable)
# ============================================================

st.divider()
st.subheader("Dataset Overview")

if manifest:
    rows = []
    for filename, data in manifest.items():
        rows.append({
            "PDF": filename,
            "Pages": data.get("pages_count", "-"),
            "Parent Chunks": data.get("parents_count", 0),
            "Child Chunks": data.get("children_count", 0),
            "Total Chunks": data.get("parents_count", 0) + data.get("children_count", 0),
            "Processed At": data.get("processed_at", "-"),
        })

    df = pd.DataFrame(rows)

    search_col, sort_col = st.columns([3, 1])
    with search_col:
        search_term = st.text_input("🔍 Search PDF filename", "")
    with sort_col:
        sort_by = st.selectbox("Sort by", df.columns.tolist(), index=0)

    filtered_df = df[df["PDF"].str.contains(search_term, case=False, na=False)] if search_term else df
    filtered_df = filtered_df.sort_values(sort_by, ascending=False)

    st.dataframe(filtered_df, use_container_width=True, hide_index=True)

    # Dataset growth (derived from processed_at dates)
    growth_df = dataset_growth_by_date(manifest)
    if not growth_df.empty and len(growth_df) > 1:
        st.markdown("**Dataset Growth Over Time**")
        growth_fig = px.line(
            growth_df, x="date", y=["cumulative_docs", "cumulative_chunks"],
            labels={"value": "Count", "date": "Date", "variable": "Metric"},
            markers=True
        )
        st.plotly_chart(growth_fig, use_container_width=True)
else:
    st.markdown(
        """
        <div class="empty-state">
        📂 <b>No processed documents found</b><br>
        Upload documents to begin building your corpus.
        </div>
        """,
        unsafe_allow_html=True
    )

# ============================================================
# COMPANY DISTRIBUTION (Plotly + coverage summary)
# ============================================================

st.divider()
st.subheader("Company Chunk Distribution")

if company_stats:
    total_chunks_company = sum(company_stats.values())
    chart_df = pd.DataFrame({
        "Company": list(company_stats.keys()),
        "Chunks": list(company_stats.values())
    }).sort_values("Chunks", ascending=False)
    chart_df["Percentage"] = (chart_df["Chunks"] / total_chunks_company * 100).round(1)

    bar_fig = px.bar(
        chart_df, x="Company", y="Chunks",
        text=chart_df["Percentage"].astype(str) + "%",
        hover_data={"Chunks": True, "Percentage": True},
        color="Chunks", color_continuous_scale="Blues"
    )
    bar_fig.update_traces(textposition="outside")
    st.plotly_chart(bar_fig, use_container_width=True)

    tree_fig = px.treemap(
        chart_df, path=["Company"], values="Chunks",
        color="Chunks", color_continuous_scale="Blues"
    )
    st.plotly_chart(tree_fig, use_container_width=True)

    largest = chart_df.iloc[0]
    smallest = chart_df.iloc[-1]
    avg_chunks = chart_df["Chunks"].mean()

    cov1, cov2, cov3 = st.columns(3)
    with cov1:
        st.metric("Largest Company", largest["Company"], f"{largest['Chunks']} chunks")
    with cov2:
        st.metric("Smallest Company", smallest["Company"], f"{smallest['Chunks']} chunks")
    with cov3:
        st.metric("Avg Chunks / Company", round(avg_chunks, 1))
else:
    st.markdown(
        """
        <div class="empty-state">
        🏢 <b>No chunk data available</b>
        </div>
        """,
        unsafe_allow_html=True
    )

# ============================================================
# CHUNK STATISTICS
# ============================================================

st.divider()
st.subheader("Chunk Statistics")

left, right = st.columns(2)
with left:
    st.info(f"Parents: {parent_count}")
with right:
    st.info(f"Children: {child_count}")

# ============================================================
# EVALUATION STATUS (improved)
# ============================================================

st.divider()
st.subheader("Evaluation Status")

if latest_result and best_method:
    e1, e2, e3 = st.columns(3)
    with e1:
        st.success(f"Latest Results: {latest_result.name}")
        st.caption(f"Modified: {pd.to_datetime(latest_result.stat().st_mtime, unit='s')}")
    with e2:
        st.metric("Latest Benchmark Score", round(best_method["overall_score"], 3))
    with e3:
        st.metric("Winning Method", best_method["method"])
else:
    st.markdown(
        """
        <div class="empty-state">
        🧪 <b>No evaluation results found</b><br>
        Run a benchmark to see results here.
        </div>
        """,
        unsafe_allow_html=True
    )

# ============================================================
# EVALUATION SUMMARY (radar + trend)
# ============================================================

if evaluations:
    leaderboard_df = pd.DataFrame(evaluations).sort_values("overall_score", ascending=False)

    st.divider()
    st.subheader("Method Comparison Radar")

    metrics = list(METRIC_LABELS.keys())
    fig = go.Figure()
    for _, row in leaderboard_df.iterrows():
        fig.add_trace(
            go.Scatterpolar(
                r=[row[m] for m in metrics],
                theta=list(METRIC_LABELS.values()),
                fill="toself",
                name=row["method"]
            )
        )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True)

    # Benchmark trend sparkline
    trend = get_evaluation_trend(limit=15)
    if len(trend) > 1:
        st.markdown("**Benchmark Score Trend (recent runs)**")
        trend_df = pd.DataFrame(trend)
        trend_df["timestamp"] = pd.to_datetime(trend_df["timestamp"])
        trend_fig = px.line(
            trend_df, x="timestamp", y="overall_score", color="method",
            markers=True
        )
        trend_fig.update_layout(height=250, margin=dict(t=10, b=10))
        st.plotly_chart(trend_fig, use_container_width=True)

    # ============================================================
    # RECENT CONVERSATIONS (searchable, paginated)
    # ============================================================

    st.divider()
    st.subheader("Recent Conversations")

    if is_admin():
        recent = get_conversations()
    else:
        recent = get_recent_conversations_by_user(st.session_state.user_id)

    if len(recent) > 0:
        recent_df = pd.DataFrame(recent)
        display_cols = [c for c in ["timestamp", "method", "prompt"] if c in recent_df.columns]

        conv_search = st.text_input("🔍 Search conversations", "", key="conv_search")
        filtered_recent = recent_df[display_cols]
        if conv_search:
            filtered_recent = filtered_recent[
                filtered_recent["prompt"].str.contains(conv_search, case=False, na=False)
            ]

        if "prompt" in filtered_recent.columns:
            filtered_recent = filtered_recent.copy()
            filtered_recent["prompt"] = filtered_recent["prompt"].apply(
                lambda p: (p[:80] + "…") if isinstance(p, str) and len(p) > 80 else p
            )

        page_size = 10
        total_rows = len(filtered_recent)
        total_pages = max(1, (total_rows - 1) // page_size + 1)
        page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)
        start = (page - 1) * page_size
        end = start + page_size

        st.dataframe(filtered_recent.iloc[start:end], use_container_width=True, hide_index=True)
        st.caption(f"Showing {min(end, total_rows)} of {total_rows} conversations")
    else:
        st.info("No conversations found.")
else:
    st.info("No benchmark results available yet.")

# ============================================================
# TOP QUERIES WIDGET
# ============================================================

st.divider()
st.subheader("Top Queries")

top_queries = get_top_queries(limit=10)
top_companies_queried = get_top_companies(limit=10)

tq1, tq2 = st.columns(2)
with tq1:
    st.markdown("**Most Frequent Questions**")
    if top_queries:
        tq_df = pd.DataFrame(top_queries)
        tq_df["prompt"] = tq_df["prompt"].apply(
            lambda p: (p[:60] + "…") if isinstance(p, str) and len(p) > 60 else p
        )
        st.dataframe(tq_df, use_container_width=True, hide_index=True)
    else:
        st.info("No query data available.")

with tq2:
    st.markdown("**Most Queried Companies**")
    if top_companies_queried:
        st.dataframe(pd.DataFrame(top_companies_queried), use_container_width=True, hide_index=True)
    else:
        st.info("No company filter data available.")

# ============================================================
# RETRIEVAL PERFORMANCE SUMMARY
# ============================================================

st.divider()
st.subheader("Retrieval Performance Summary")

p1, p2, p3, p4 = st.columns(4)
with p1:
    st.metric("Avg Retrieval Time", fmt_seconds(latency_breakdown["avg_retrieval_latency"]))
with p2:
    st.metric("Avg Reranker Time", fmt_seconds(latency_breakdown["avg_rerank_latency"]))
with p3:
    st.metric("Avg Generation Time", fmt_seconds(latency_breakdown["avg_generation_latency"]))
with p4:
    st.metric("Avg Total Time", fmt_seconds(latency_breakdown["avg_total_latency"]))

# ============================================================
# USER ANALYTICS (ADMIN ONLY)
# ============================================================

if is_admin():
    st.divider()
    st.subheader("User Analytics")

    user_analytics = get_user_analytics()
    if user_analytics:
        ua_df = pd.DataFrame(user_analytics)
        ua_df["avg_latency"] = ua_df["avg_latency"].apply(fmt_seconds)

        u1, u2, u3 = st.columns(3)
        with u1:
            st.metric("Active Users", len(ua_df))
        with u2:
            st.metric("Total Conversations", int(pd.DataFrame(user_analytics)["total_chats"].sum()))
        with u3:
            st.metric("Avg Chats / User", round(pd.DataFrame(user_analytics)["total_chats"].mean(), 1))

        st.dataframe(ua_df, use_container_width=True, hide_index=True)
    else:
        st.info("No user activity recorded yet.")

# ============================================================
# RECENT ACTIVITY FEED
# ============================================================

st.divider()
st.subheader("Recent Activity")

activity = get_recent_activity(limit=15)
if activity:
    for item in activity:
        ts = item.get("timestamp", "-")
        activity_type = item.get("activity_type", "event")
        detail = item.get("detail", "-")
        icon = "💬" if activity_type == "conversation" else "📊"
        st.markdown(f"{icon} **{ts}** — {activity_type.title()}: `{detail}`")
else:
    st.info("No recent activity.")

# ============================================================
# STORAGE STATISTICS
# ============================================================

st.divider()
st.subheader("Storage Statistics")

def safe_size(path: Path):
    try:
        return path.stat().st_size if path.exists() else 0
    except Exception:
        return 0

manifest_size = safe_size(MANIFEST_PATH)
chunks_size = safe_size(CHUNKS_PATH)
results_size = sum(safe_size(f) for f in RESULTS_DIR.glob("*.csv")) if RESULTS_DIR.exists() else 0
total_size = manifest_size + chunks_size + results_size

s1, s2, s3, s4 = st.columns(4)
with s1:
    st.metric("Manifest Size", fmt_bytes(manifest_size))
with s2:
    st.metric("Chunk Database Size", fmt_bytes(chunks_size))
with s3:
    st.metric("Evaluation Results Size", fmt_bytes(results_size))
with s4:
    st.metric("Total Corpus Size", fmt_bytes(total_size))

# ============================================================
# LAST DATA REFRESH
# ============================================================

st.divider()
st.subheader("Last Data Refresh")

r1, r2 = st.columns(2)
with r1:
    if MANIFEST_PATH.exists():
        st.info(f"Corpus last processed: {pd.to_datetime(MANIFEST_PATH.stat().st_mtime, unit='s')}")
    else:
        st.info("No corpus processing recorded.")
with r2:
    if latest_result:
        st.info(f"Evaluations last updated: {pd.to_datetime(latest_result.stat().st_mtime, unit='s')}")
    else:
        st.info("No evaluations recorded.")

# ============================================================
# QUICK SUMMARY
# ============================================================

st.divider()
st.subheader("Benchmark Summary")

st.markdown(
    f"""
**Documents Loaded:** {pdf_count}

**Parent Chunks:** {parent_count}

**Child Chunks:** {child_count}

**Companies:** {len(company_stats)}

**Evaluation Files:** {evaluation_count}
"""
)