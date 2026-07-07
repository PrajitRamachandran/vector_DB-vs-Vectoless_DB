"""
Evaluation Dashboard
A polished, real-time benchmark control center for the Vector / Vectorless / Hybrid
RAG pipelines. Combines persisted historical benchmark data (via get_evaluations())
with live, in-session, question-level detail captured during the current run.
"""

import json
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from streamlit_app.services.evaluation_service import (
    run_judge_benchmark,
    run_ragas_benchmark,
)

from streamlit_app.database.repository import (
    get_evaluations,
    clear_evaluations,
)

from streamlit_app.services.rag_service import (
    get_vector_pipeline,
    get_vectorless_pipeline,
    get_hybrid_pipeline,
)

from streamlit_app.auth.protect_page import require_login

require_login()

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Evaluation",
    page_icon="📊",
    layout="wide",
)

# ============================================================
# CONSTANTS
# ============================================================

METHOD_LABELS = {
    "vector": "Vector RAG",
    "vectorless": "Vectorless RAG",
    "hybrid": "Hybrid RAG",
}

METHOD_COLORS = {
    "vector": "#3B82F6",       # blue
    "vectorless": "#F59E0B",   # amber
    "hybrid": "#10B981",       # emerald
}

METRIC_LABELS = {
    "avg_judge_score": "Judge Score",
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer Relevancy",
    "context_precision": "Context Precision",
    "context_recall": "Context Recall",
    "company_accuracy": "Company Accuracy",
    "overall_score": "Overall Score",
}

# Metrics on a 0-1 scale (used as-is in the radar chart)
NORMALIZED_METRICS = [
    "faithfulness", "answer_relevancy",
    "context_precision", "context_recall", "company_accuracy",
]

RADAR_METRICS_DEFAULT = [
    "avg_judge_score", "faithfulness", "answer_relevancy",
    "context_precision", "context_recall", "company_accuracy",
]

NOTES_PATH = Path("data/benchmark_notes.json")
RAW_OUTPUT_DIR = Path("data/raw_runs")

# ============================================================
# STYLING — "LUXURY" DARK-MODE-FRIENDLY THEME
# ============================================================

def inject_css():
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.6rem; }

        /* KPI cards */
        .kpi-row { display: flex; gap: 14px; margin-bottom: 6px; flex-wrap: wrap; }
        .kpi-card {
            flex: 1; min-width: 200px;
            background: linear-gradient(135deg, rgba(212,175,55,0.10) 0%, rgba(120,120,140,0.06) 100%);
            border: 1px solid rgba(212,175,55,0.35);
            border-radius: 14px;
            padding: 18px 20px;
            box-shadow: 0 4px 18px rgba(0,0,0,0.12);
        }
        .kpi-label {
            font-size: 0.78rem; letter-spacing: 0.06em; text-transform: uppercase;
            opacity: 0.65; margin-bottom: 6px; font-weight: 600;
        }
        .kpi-value { font-size: 1.65rem; font-weight: 700; line-height: 1.15; }
        .kpi-sub { font-size: 0.78rem; opacity: 0.6; margin-top: 4px; }

        .delta-pos { color: #10B981; font-weight: 600; }
        .delta-neg { color: #EF4444; font-weight: 600; }
        .delta-flat { color: #9CA3AF; font-weight: 600; }

        /* Status badges */
        .status-row { display: flex; gap: 10px; margin: 4px 0 18px 0; flex-wrap: wrap; }
        .status-badge {
            display: inline-flex; align-items: center; gap: 6px;
            padding: 5px 12px; border-radius: 999px;
            font-size: 0.80rem; font-weight: 600;
            border: 1px solid rgba(255,255,255,0.10);
        }
        .status-ok { background: rgba(16,185,129,0.14); color: #10B981; }
        .status-pending { background: rgba(156,163,175,0.14); color: #9CA3AF; }
        .status-warn { background: rgba(239,68,68,0.14); color: #EF4444; }

        /* Insight box */
        .insight-box {
            background: rgba(212,175,55,0.08);
            border-left: 3px solid #D4AF37;
            border-radius: 8px;
            padding: 14px 18px;
            margin: 8px 0 18px 0;
            font-size: 0.94rem;
            line-height: 1.65;
        }

        .medal { font-size: 1.1rem; margin-right: 4px; }

        .section-title {
            font-size: 1.05rem; font-weight: 700; margin-top: 4px; margin-bottom: 2px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_css()

st.title("📊 Evaluation Dashboard")
st.caption("Live benchmark control center — Vector · Vectorless · Hybrid RAG")

# ============================================================
# SESSION STATE
# ============================================================

for key, default in [
    ("judge_results", None),
    ("judge_contexts", None),
    ("ragas_results", None),
    ("last_run_ts", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ============================================================
# SMALL HELPERS
# ============================================================

def safe_col(df: pd.DataFrame, col: str, default=np.nan):
    return df[col] if col in df.columns else pd.Series([default] * len(df))


def fmt_pct(x):
    try:
        v = float(x)
        if np.isnan(v):
            return "—"
        return f"{v * 100:.1f}%"
    except Exception:
        return "—"


def fmt_score(x, digits=3):
    try:
        v = float(x)
        if np.isnan(v):
            return "—"
        return f"{v:.{digits}f}"
    except Exception:
        return "—"


def latest_per_method(df: pd.DataFrame) -> pd.DataFrame:
    """Most recent row per method — used as the 'current standings'."""
    if df.empty or "timestamp" not in df.columns:
        return df
    d = df.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce")
    return d.sort_values("timestamp").groupby("method", as_index=False).tail(1)


def safe_best_row(df: pd.DataFrame, col: str):
    """
    Returns the row with the max value of `col`, or None if the column is
    missing / entirely NaN (idxmax raises ValueError on all-NA columns).
    """
    if df.empty or col not in df.columns:
        return None
    valid = df.dropna(subset=[col])
    if valid.empty:
        return None
    return valid.loc[valid[col].idxmax()]


def load_notes() -> dict:
    if NOTES_PATH.exists():
        try:
            return json.loads(NOTES_PATH.read_text())
        except Exception:
            return {}
    return {}


def save_note(run_key: str, note: str):
    NOTES_PATH.parent.mkdir(parents=True, exist_ok=True)
    notes = load_notes()
    notes[run_key] = {"note": note, "saved_at": datetime.now().isoformat()}
    NOTES_PATH.write_text(json.dumps(notes, indent=2))


def medal_for_rank(rank: int) -> str:
    return {0: "🥇", 1: "🥈", 2: "🥉"}.get(rank, f"#{rank + 1}")


def style_leaderboard(df: pd.DataFrame, cols: list[str]):
    numeric_cols = [c for c in cols if c not in ("method", "rank")]
    styler = df[cols].style
    for c in numeric_cols:
        styler = styler.background_gradient(subset=[c], cmap="RdYlGn")
    fmt = {c: "{:.3f}" for c in numeric_cols}
    return styler.format(fmt)


def compute_deltas(history_df: pd.DataFrame) -> dict:
    """Last-run vs previous-run overall_score delta, per method."""
    deltas = {}
    if history_df.empty or "overall_score" not in history_df.columns:
        return deltas
    d = history_df.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce")
    for method, g in d.sort_values("timestamp").groupby("method"):
        if len(g) >= 2:
            deltas[method] = g["overall_score"].iloc[-1] - g["overall_score"].iloc[-2]
        else:
            deltas[method] = None
    return deltas


def generate_insights(current_df: pd.DataFrame, history_df: pd.DataFrame) -> list[str]:
    insights = []
    if current_df.empty or "overall_score" not in current_df.columns:
        return ["Run a benchmark to generate insights."]

    ranked = current_df.dropna(subset=["overall_score"]).sort_values(
        "overall_score", ascending=False
    ).reset_index(drop=True)

    if ranked.empty:
        return [
            "Judge results are in, but overall scores need RAGAS metrics "
            "(faithfulness, relevancy, precision, recall). Run **RAGAS Evaluation** to unlock insights."
        ]

    best = ranked.iloc[0]
    insights.append(
        f"**{METHOD_LABELS.get(best['method'], best['method'])}** is the top performer "
        f"with an overall score of **{fmt_score(best['overall_score'])}**."
    )

    if len(ranked) > 1:
        gap = best["overall_score"] - ranked.iloc[1]["overall_score"]
        runner_up = METHOD_LABELS.get(ranked.iloc[1]["method"], ranked.iloc[1]["method"])
        if gap < 0.02:
            insights.append(
                f"It's a close race — only **{fmt_score(gap)}** separates it from "
                f"**{runner_up}**. Consider running more questions to confirm the ranking."
            )
        else:
            insights.append(
                f"It leads **{runner_up}** by a clear margin of **{fmt_score(gap)}** overall points."
            )

    best_company = safe_best_row(ranked, "company_accuracy")
    if best_company is not None:
        insights.append(
            f"**{METHOD_LABELS.get(best_company['method'], best_company['method'])}** has the "
            f"strongest company-detection accuracy at **{fmt_pct(best_company['company_accuracy'])}**."
        )

    deltas = compute_deltas(history_df)
    for method, delta in deltas.items():
        if delta is not None and abs(delta) > 0.005:
            direction = "improved" if delta > 0 else "regressed"
            insights.append(
                f"{METHOD_LABELS.get(method, method)} {direction} by **{fmt_score(abs(delta))}** "
                f"vs. its previous run."
            )
    return insights


# ============================================================
# LOAD PERSISTED HISTORY
# ============================================================

data_load_ok = True
try:
    evaluations = get_evaluations()
except Exception:
    evaluations = []
    data_load_ok = False

_history_df_persisted = pd.DataFrame(evaluations) if evaluations else pd.DataFrame()


def merge_live_results(persisted_df: pd.DataFrame):
    """
    Overlay the current in-session Judge/RAGAS results on top of persisted
    history so every KPI, chart, and tab reflects the newest run instantly —
    without waiting for a page reload or for the database write to land.
    """
    live_source = st.session_state.ragas_results if st.session_state.ragas_results is not None \
        else st.session_state.judge_results

    if live_source is None:
        return persisted_df.copy(), False

    raw = live_source if isinstance(live_source, pd.DataFrame) else pd.DataFrame(live_source)
    if raw.empty or "method" not in raw.columns:
        return persisted_df.copy(), False

    run_ts = st.session_state.last_run_ts or datetime.now().isoformat()
    is_ragas = st.session_state.ragas_results is not None
    ragas_metric_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall", "company_accuracy"]

    live_rows = []
    for method, g in raw.groupby("method"):
        row = {"method": method, "timestamp": run_ts, "num_questions": len(g)}
        if "judge_score" in g.columns:
            row["avg_judge_score"] = g["judge_score"].mean()
        for col in ragas_metric_cols:
            if col in g.columns:
                row[col] = g[col].mean()
        if is_ragas:
            present = [row[c] for c in ragas_metric_cols if c in row and pd.notna(row[c])]
            if present and pd.notna(row.get("avg_judge_score")):
                row["overall_score"] = float(np.mean([row["avg_judge_score"] / 5.0] + present))
        live_rows.append(row)

    live_df = pd.DataFrame(live_rows)
    if live_df.empty:
        return persisted_df.copy(), False

    merged = persisted_df.copy()
    if merged.empty:
        merged = live_df
    else:
        # Avoid double-counting once the DB write for this exact run lands —
        # drop any persisted row matching the same method + live run timestamp,
        # then layer the live rows on top.
        if "timestamp" in merged.columns:
            dup_mask = merged["method"].isin(live_df["method"]) & (merged["timestamp"].astype(str) == str(run_ts))
            merged = merged[~dup_mask]
        merged = pd.concat([merged, live_df], ignore_index=True, sort=False)

    return merged, True


history_df, has_live_overlay = merge_live_results(_history_df_persisted)
current_df = latest_per_method(history_df) if not history_df.empty else history_df


def compute_run_kpis(hist_df: pd.DataFrame) -> dict:
    """Benchmark-specific KPI figures: run counts, latest run timestamps, questions evaluated."""
    kpis = {
        "total_judge_runs": 0,
        "total_ragas_runs": 0,
        "latest_judge_run": None,
        "latest_ragas_run": None,
        "questions_evaluated": None,
    }
    if hist_df.empty:
        d = pd.DataFrame()
    else:
        d = hist_df.copy()
        d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce")

    if not d.empty and "avg_judge_score" in d.columns:
        judge_rows = d[d["avg_judge_score"].notna()]
        kpis["total_judge_runs"] = len(judge_rows)
        if not judge_rows.empty:
            latest_judge = judge_rows.sort_values("timestamp").iloc[-1]
            kpis["latest_judge_run"] = latest_judge["timestamp"]
            if "num_questions" in judge_rows.columns and pd.notna(latest_judge.get("num_questions")):
                kpis["questions_evaluated"] = int(latest_judge["num_questions"])

    ragas_signal_col = None
    if not d.empty:
        ragas_signal_col = "overall_score" if "overall_score" in d.columns else (
            "faithfulness" if "faithfulness" in d.columns else None
        )

    if ragas_signal_col:
        ragas_rows = d[d[ragas_signal_col].notna()]
        kpis["total_ragas_runs"] = len(ragas_rows)
        if not ragas_rows.empty:
            latest_ragas = ragas_rows.sort_values("timestamp").iloc[-1]
            kpis["latest_ragas_run"] = latest_ragas["timestamp"]
            if kpis["questions_evaluated"] is None and "num_questions" in ragas_rows.columns \
                    and pd.notna(latest_ragas.get("num_questions")):
                kpis["questions_evaluated"] = int(latest_ragas["num_questions"])

    if kpis["questions_evaluated"] is None:
        live_raw = st.session_state.ragas_results if st.session_state.ragas_results is not None \
            else st.session_state.judge_results
        if live_raw is not None:
            live_df = live_raw if isinstance(live_raw, pd.DataFrame) else pd.DataFrame(live_raw)
            if not live_df.empty:
                kpis["questions_evaluated"] = (
                    int(live_df["question"].nunique()) if "question" in live_df.columns else len(live_df)
                )

    return kpis


def fmt_ts(ts) -> str:
    if ts is None or pd.isna(ts):
        return "—"
    return pd.Timestamp(ts).strftime("%b %d, %Y · %H:%M")

# ============================================================
# STATUS INDICATORS
# ============================================================

def badge(label: str, ok: bool, warn: bool = False) -> str:
    cls = "status-warn" if warn else ("status-ok" if ok else "status-pending")
    icon = "⚠️" if warn else ("✅" if ok else "⏳")
    return f'<span class="status-badge {cls}">{icon} {label}</span>'

st.markdown(
    '<div class="status-row">'
    + badge("Data Store", data_load_ok, warn=not data_load_ok)
    + badge("Judge Evaluation", st.session_state.judge_results is not None)
    + badge("RAGAS Evaluation", st.session_state.ragas_results is not None)
    + "</div>",
    unsafe_allow_html=True,
)

# ============================================================
# KPI SUMMARY CARDS
# ============================================================

if history_df.empty:
    st.info(
        "No benchmark results yet — run **Judge Benchmark** then **RAGAS Evaluation** "
        "to populate scores (overall_score requires RAGAS metrics)."
    )
else:
    kpis = compute_run_kpis(history_df)

    questions_val = kpis["questions_evaluated"] if kpis["questions_evaluated"] is not None else "—"
    live_tag = ' <span class="delta-pos">● live</span>' if has_live_overlay else ""

    st.markdown(
        f"""
        <div class="kpi-row">
          <div class="kpi-card">
            <div class="kpi-label">🧑‍⚖️ Total Judge Runs</div>
            <div class="kpi-value">{kpis['total_judge_runs']}</div>
            <div class="kpi-sub">runs with judge scores{live_tag}</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-label">📐 Total RAGAS Runs</div>
            <div class="kpi-value">{kpis['total_ragas_runs']}</div>
            <div class="kpi-sub">runs with RAGAS metrics</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-label">Latest Judge Run</div>
            <div class="kpi-value" style="font-size:1.15rem;">{fmt_ts(kpis['latest_judge_run'])}</div>
            <div class="kpi-sub">most recent judge benchmark</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-label">Latest RAGAS Run</div>
            <div class="kpi-value" style="font-size:1.15rem;">{fmt_ts(kpis['latest_ragas_run'])}</div>
            <div class="kpi-sub">most recent RAGAS evaluation</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-label">❓ Questions Evaluated</div>
            <div class="kpi-value">{questions_val}</div>
            <div class="kpi-sub">in the latest run</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()

# ============================================================
# ACTIONS
# ============================================================

st.markdown('<div class="section-title">Run a Benchmark</div>', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

# ------------------------------------------------------------
# JUDGE BENCHMARK
# ------------------------------------------------------------

with col1:
    if st.button("▶ Run Judge Benchmark", use_container_width=True, type="primary"):
        progress_bar = st.progress(0.0, text="Loading pipelines…")

        def _judge_progress(fraction: float, message: str) -> None:
            progress_bar.progress(fraction, text=message)

        vector_pipeline = get_vector_pipeline()
        vectorless_pipeline = get_vectorless_pipeline()
        hybrid_pipeline = get_hybrid_pipeline()

        result = run_judge_benchmark(
            vector_pipeline=vector_pipeline,
            vectorless_pipeline=vectorless_pipeline,
            hybrid_pipeline=hybrid_pipeline,
            progress_callback=_judge_progress,
        )

        if result.get("success"):
            st.session_state.judge_results = result["results"]
            st.session_state.judge_contexts = result["contexts"]
            # A fresh judge run invalidates any RAGAS scores computed against the
            # previous answers — clear them so stale metrics never linger on screen.
            st.session_state.ragas_results = None
            st.session_state.last_run_ts = datetime.now().isoformat()
            progress_bar.progress(1.0, text="Judge benchmark completed")
            st.success("Judge benchmark completed.")
            st.cache_data.clear()
        else:
            progress_bar.progress(1.0, text="Judge benchmark failed")
            st.error(result.get("error", "Unknown error"))
        st.rerun()

# ------------------------------------------------------------
# RAGAS
# ------------------------------------------------------------

with col2:
    if st.button("▶ Run RAGAS Evaluation", use_container_width=True, type="primary"):
        if st.session_state.judge_results is None:
            st.warning("Run Judge Benchmark first.")
        else:
            progress_bar = st.progress(0.0, text="Preparing RAGAS evaluation…")

            def _ragas_progress(fraction: float, message: str) -> None:
                progress_bar.progress(fraction, text=message)

            try:
                result = run_ragas_benchmark(
                    st.session_state.judge_results,
                    st.session_state.judge_contexts,
                    progress_callback=_ragas_progress,
                )
                if result.get("success"):
                    st.session_state.ragas_results = result["combined"]
                    progress_bar.progress(1.0, text="RAGAS evaluation completed")
                    st.success("RAGAS evaluation completed.")
                    st.cache_data.clear()
                else:
                    progress_bar.progress(1.0, text="RAGAS evaluation failed")
                    st.error(result.get("error", "Unknown error"))
            except Exception as e:
                progress_bar.progress(1.0, text="RAGAS evaluation failed")
                st.error(f"RAGAS evaluation raised an exception: {e}")
                with st.expander("Show error details"):
                    st.code(traceback.format_exc())
            st.rerun()

# ------------------------------------------------------------
# CLEAR
# ------------------------------------------------------------

with col3:
    if st.button("🗑 Clear Evaluation History", use_container_width=True):
        clear_evaluations()
        st.session_state.judge_results = None
        st.session_state.judge_contexts = None
        st.session_state.ragas_results = None
        st.session_state.last_run_ts = None
        st.cache_data.clear()
        st.success("Evaluation history cleared.")
        st.rerun()

st.divider()

# ============================================================
# EMPTY STATE
# ============================================================

if history_df.empty:
    st.info(
        """
        **No benchmark results available yet.**

        1. Run **Judge Benchmark** to generate answers and score them.
        2. Run **RAGAS Evaluation** to add faithfulness, relevancy, precision & recall.
        3. Results, charts, and insights below update automatically.
        """
    )
    st.stop()

# ============================================================
# METHOD FILTER (sidebar)
# ============================================================

all_methods = sorted(history_df["method"].unique().tolist())
with st.sidebar:
    st.markdown("### ⚙️ Dashboard Controls")
    selected_methods = st.multiselect(
        "Compare methods",
        options=all_methods,
        default=all_methods,
        format_func=lambda m: METHOD_LABELS.get(m, m),
    )
    if not selected_methods:
        selected_methods = all_methods
    st.caption("Applies to the charts and leaderboard below.")

current_view = current_df[current_df["method"].isin(selected_methods)].copy()
history_view = history_df[history_df["method"].isin(selected_methods)].copy()

# ============================================================
# TABS
# ============================================================

tab_leader, tab_analytics, tab_detail, tab_compare, tab_history, tab_exec, tab_download = st.tabs(
    [
        "🏆 Leaderboard",
        "📈 Analytics",
        "🔍 Detailed Results",
        "⚖️ Compare Runs",
        "📅 History",
        "💼 Executive Summary",
        "📥 Downloads",
    ]
)

# ------------------------------------------------------------
# TAB: LEADERBOARD
# ------------------------------------------------------------

with tab_leader:
    st.subheader("🏆 Current Standings")

    ranked = current_view.sort_values("overall_score", ascending=False).reset_index(drop=True)
    ranked.insert(0, "rank", [medal_for_rank(i) for i in range(len(ranked))])
    ranked["method"] = ranked["method"].map(lambda m: METHOD_LABELS.get(m, m))

    leaderboard_columns = [
        "rank", "method", "overall_score", "avg_judge_score",
        "faithfulness", "answer_relevancy", "context_precision",
        "context_recall", "company_accuracy",
    ]
    available_cols = [c for c in leaderboard_columns if c in ranked.columns]

    st.dataframe(
        style_leaderboard(ranked, available_cols),
        use_container_width=True,
        hide_index=True,
    )

    if not ranked.empty:
        best = ranked.iloc[0]
        st.success(f"🥇 **Best Method:** {best['method']} — Score: **{fmt_score(best['overall_score'])}**")

    st.divider()
    st.markdown('<div class="section-title">🎯 Metric Winners</div>', unsafe_allow_html=True)

    metric_candidates = [m for m in RADAR_METRICS_DEFAULT if m in current_view.columns]
    if metric_candidates:
        winner_cols = st.columns(len(metric_candidates))
        for col, metric in zip(winner_cols, metric_candidates):
            sub = current_view.dropna(subset=[metric])
            if sub.empty:
                continue
            top = sub.loc[sub[metric].idxmax()]
            with col:
                st.markdown(
                    f"""
                    <div class="kpi-card">
                      <div class="kpi-label">{METRIC_LABELS.get(metric, metric)}</div>
                      <div class="kpi-value" style="font-size:1.05rem;">
                        {METHOD_LABELS.get(top['method'], top['method'])}
                      </div>
                      <div class="kpi-sub">{fmt_score(top[metric])}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.divider()
    st.markdown('<div class="section-title">💡 Automatic Insights</div>', unsafe_allow_html=True)
    for line in generate_insights(current_view, history_view):
        st.markdown(f'<div class="insight-box">{line}</div>', unsafe_allow_html=True)

# ------------------------------------------------------------
# TAB: ANALYTICS
# ------------------------------------------------------------

with tab_analytics:
    st.subheader("🏢 Company Detection Accuracy")
    st.caption("Company accuracy is the critical correctness signal for this system — tracked separately.")

    if "company_accuracy" in current_view.columns:
        acc_df = current_view[["method", "company_accuracy"]].copy()
        acc_df["method_label"] = acc_df["method"].map(lambda m: METHOD_LABELS.get(m, m))
        fig_acc = px.bar(
            acc_df.sort_values("company_accuracy", ascending=True),
            x="company_accuracy", y="method_label", orientation="h",
            color="method", color_discrete_map=METHOD_COLORS,
            text=acc_df.sort_values("company_accuracy", ascending=True)["company_accuracy"].map(fmt_pct),
        )
        fig_acc.update_layout(
            xaxis_tickformat=".0%", xaxis_title="Company Accuracy",
            yaxis_title="", showlegend=False, height=280,
        )
        st.plotly_chart(fig_acc, use_container_width=True, config={"displaylogo": False})
    else:
        st.info("Company accuracy not present in this dataset yet.")

    st.divider()
    st.subheader("📡 Method Performance Radar")

    radar_metrics = [m for m in RADAR_METRICS_DEFAULT if m in current_view.columns]
    if radar_metrics and not current_view.empty:
        fig_radar = go.Figure()
        for _, row in current_view.iterrows():
            values = []
            for m in radar_metrics:
                v = row.get(m, np.nan)
                if m == "avg_judge_score" and pd.notna(v):
                    v = v / 5.0  # normalize 1-5 scale to 0-1
                values.append(v if pd.notna(v) else 0)
            values.append(values[0])
            fig_radar.add_trace(
                go.Scatterpolar(
                    r=values,
                    theta=[METRIC_LABELS.get(m, m) for m in radar_metrics] + [METRIC_LABELS.get(radar_metrics[0], radar_metrics[0])],
                    fill="toself",
                    name=METHOD_LABELS.get(row["method"], row["method"]),
                    line_color=METHOD_COLORS.get(row["method"], "#999"),
                    opacity=0.75,
                )
            )
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title="Vector vs Vectorless vs Hybrid",
            height=500,
        )
        st.plotly_chart(fig_radar, use_container_width=True, config={"displaylogo": False})
    else:
        st.info("Not enough metric data to render the radar chart yet.")

    st.divider()
    st.subheader("📊 Metric Breakdown")
    metric_cols = ["method", "faithfulness", "answer_relevancy",
                   "context_precision", "context_recall", "overall_score"]
    available = [c for c in metric_cols if c in current_view.columns]
    display_df = current_view[available].copy()
    display_df["method"] = display_df["method"].map(lambda m: METHOD_LABELS.get(m, m))
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # Failure / latency diagnostics from in-session raw data, if available
    st.divider()
    st.subheader("🩺 Failure & Latency Diagnostics")
    st.caption("Computed from the current session's raw run — not yet persisted historically.")

    raw = st.session_state.ragas_results if st.session_state.ragas_results is not None \
        else st.session_state.judge_results

    if raw is not None and len(raw) > 0:
        raw_df = raw if isinstance(raw, pd.DataFrame) else pd.DataFrame(raw)
        diag_cols = st.columns(4)
        avg_latency = raw_df["total_time"].mean() if "total_time" in raw_df.columns else None
        fail_count = int((~raw_df["pass"]).sum()) if "pass" in raw_df.columns else None
        miss_count = int((raw_df["company_accuracy"] < 1.0).sum()) if "company_accuracy" in raw_df.columns else None
        halluc_count = int((raw_df["faithfulness"] < 0.5).sum()) if "faithfulness" in raw_df.columns else None

        with diag_cols[0]:
            st.metric("Avg Latency", f"{avg_latency:.2f}s" if avg_latency is not None else "—")
        with diag_cols[1]:
            st.metric("Failed Queries (score<3)", fail_count if fail_count is not None else "—")
        with diag_cols[2]:
            st.metric("Retrieval Misses", miss_count if miss_count is not None else "—")
        with diag_cols[3]:
            st.metric("Low-Faithfulness Answers", halluc_count if halluc_count is not None else "—")
    else:
        st.info("Run a benchmark this session to see failure and latency diagnostics.")

# ------------------------------------------------------------
# TAB: DETAILED RESULTS
# ------------------------------------------------------------

with tab_detail:
    st.subheader("🔍 Question-Level Results")
    st.caption("From the current session's run. Persist raw outputs in the Downloads tab to keep this across sessions.")

    raw = st.session_state.ragas_results if st.session_state.ragas_results is not None \
        else st.session_state.judge_results

    if raw is None:
        st.info("Run a Judge Benchmark (and optionally RAGAS) to see question-level detail.")
    else:
        detail_df = raw if isinstance(raw, pd.DataFrame) else pd.DataFrame(raw)
        detail_df = detail_df[detail_df["method"].isin(selected_methods)] if "method" in detail_df.columns else detail_df

        f1, f2, f3 = st.columns([2, 1, 1])
        with f1:
            search = st.text_input("🔎 Search question / company / answer", "")
        with f2:
            only_failures = st.checkbox("Only failures (score < 3)", value=False)
        with f3:
            method_filter = st.multiselect(
                "Method", options=sorted(detail_df["method"].unique()) if "method" in detail_df.columns else [],
                default=sorted(detail_df["method"].unique()) if "method" in detail_df.columns else [],
                format_func=lambda m: METHOD_LABELS.get(m, m),
            )

        view = detail_df.copy()
        if method_filter and "method" in view.columns:
            view = view[view["method"].isin(method_filter)]
        if only_failures and "pass" in view.columns:
            view = view[~view["pass"]]
        if search:
            mask = pd.Series(False, index=view.index)
            for col in ["question", "company", "answer"]:
                if col in view.columns:
                    mask |= view[col].astype(str).str.contains(search, case=False, na=False)
            view = view[mask]

        show_cols = [c for c in [
            "id", "company", "category", "method", "question", "judge_score",
            "judge_reason", "pass", "company_accuracy", "faithfulness",
            "answer_relevancy", "context_precision", "context_recall",
            "retrieval_time", "generation_time", "total_time", "answer",
        ] if c in view.columns]

        st.dataframe(view[show_cols], use_container_width=True, hide_index=True, height=450)
        st.caption(f"{len(view)} of {len(detail_df)} rows shown.")

# ------------------------------------------------------------
# TAB: COMPARE RUNS
# ------------------------------------------------------------

with tab_compare:
    st.subheader("⚖️ Compare Two Runs")

    compare_method = st.selectbox(
        "Method", options=all_methods, format_func=lambda m: METHOD_LABELS.get(m, m),
    )
    method_history = history_df[history_df["method"] == compare_method].copy()
    method_history["timestamp"] = pd.to_datetime(method_history["timestamp"], errors="coerce")
    method_history = method_history.sort_values("timestamp", ascending=False)

    if len(method_history) < 2:
        st.info("Need at least two runs of this method to compare. Run the benchmark again after making a change.")
    else:
        options = method_history["timestamp"].dt.strftime("%b %d, %Y · %H:%M").tolist()
        c1, c2 = st.columns(2)
        with c1:
            run_a_idx = st.selectbox("Run A (baseline)", options=range(len(options)),
                                      format_func=lambda i: options[i], index=1)
        with c2:
            run_b_idx = st.selectbox("Run B (comparison)", options=range(len(options)),
                                      format_func=lambda i: options[i], index=0)

        row_a = method_history.iloc[run_a_idx]
        row_b = method_history.iloc[run_b_idx]

        compare_metrics = [m for m in ["overall_score"] + RADAR_METRICS_DEFAULT if m in method_history.columns]
        rows = []
        for m in compare_metrics:
            a_val, b_val = row_a.get(m, np.nan), row_b.get(m, np.nan)
            delta = (b_val - a_val) if pd.notna(a_val) and pd.notna(b_val) else np.nan
            rows.append({
                "Metric": METRIC_LABELS.get(m, m),
                "Run A": fmt_score(a_val),
                "Run B": fmt_score(b_val),
                "Δ (B − A)": f"{'+' if pd.notna(delta) and delta >= 0 else ''}{fmt_score(delta)}" if pd.notna(delta) else "—",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ------------------------------------------------------------
# TAB: HISTORY
# ------------------------------------------------------------

with tab_history:
    st.subheader("📅 Historical Trends")

    trend_metric_options = [m for m in ["overall_score"] + RADAR_METRICS_DEFAULT if m in history_view.columns]
    trend_metric = st.selectbox(
        "Metric to track over time", trend_metric_options,
        format_func=lambda m: METRIC_LABELS.get(m, m),
    )

    hist_plot = history_view.copy()
    if len(hist_plot) > 1 and trend_metric in hist_plot.columns:
        hist_plot["timestamp"] = pd.to_datetime(hist_plot["timestamp"], errors="coerce")
        hist_plot["method_label"] = hist_plot["method"].map(lambda m: METHOD_LABELS.get(m, m))
        fig_hist = px.line(
            hist_plot.sort_values("timestamp"),
            x="timestamp", y=trend_metric, color="method_label",
            markers=True, color_discrete_map={METHOD_LABELS.get(k, k): v for k, v in METHOD_COLORS.items()},
            title=f"{METRIC_LABELS.get(trend_metric, trend_metric)} Over Time",
        )
        st.plotly_chart(fig_hist, use_container_width=True, config={"displaylogo": False})
    else:
        st.info("Run benchmarks multiple times to see trends.")

    st.divider()
    st.subheader("⚙️ Benchmark Configuration")
    config_cols = ["chunk_size", "chunk_overlap", "embedding_model", "reranker",
                   "top_k", "fetch_k", "llm_model", "num_questions", "corpus_size", "version"]
    present_config_cols = [c for c in config_cols if c in history_df.columns]
    if present_config_cols:
        st.dataframe(
            history_df.sort_values("timestamp", ascending=False)[["method", "timestamp"] + present_config_cols],
            use_container_width=True, hide_index=True,
        )
    else:
        st.warning(
            "No configuration metadata (chunk size, overlap, embedding model, reranker, "
            "Top-K, Fetch-K, LLM) is currently stored per run. To show this, have "
            "`run_judge_benchmark` / your repository layer persist these fields alongside "
            "each evaluation row — see the note at the end of this response."
        )

    st.divider()
    st.subheader("📝 Experiment Notes")
    notes = load_notes()
    hist_sorted = history_df.sort_values("timestamp", ascending=False)
    hist_sorted["timestamp"] = pd.to_datetime(hist_sorted["timestamp"], errors="coerce")
    run_keys = hist_sorted["timestamp"].dropna().dt.strftime("%Y-%m-%d %H:%M:%S").unique().tolist()

    if run_keys:
        selected_run_key = st.selectbox("Select run", run_keys)
        existing_note = notes.get(selected_run_key, {}).get("note", "")
        new_note = st.text_area("Notes (config changes, observations, etc.)", value=existing_note, height=100)
        if st.button("💾 Save Note"):
            save_note(selected_run_key, new_note)
            st.success("Note saved.")
            st.rerun()
    else:
        st.info("No runs available to annotate yet.")

    st.divider()
    st.subheader("📊 Full History Table")
    search_hist = st.text_input("🔎 Search history (method, etc.)", "")
    hist_display = history_df.copy()
    if search_hist:
        mask = pd.Series(False, index=hist_display.index)
        for col in hist_display.columns:
            mask |= hist_display[col].astype(str).str.contains(search_hist, case=False, na=False)
        hist_display = hist_display[mask]
    st.dataframe(hist_display, use_container_width=True, hide_index=True)

# ------------------------------------------------------------
# TAB: EXECUTIVE SUMMARY
# ------------------------------------------------------------

with tab_exec:
    st.subheader("💼 Executive Summary")
    st.caption("The essentials, for a non-technical audience.")

    best = safe_best_row(current_view, "overall_score")
    if best is not None:
        st.markdown(
            f"### Recommendation: use **{METHOD_LABELS.get(best['method'], best['method'])}**"
        )
        st.markdown(
            f"It currently scores **{fmt_score(best['overall_score'])}** overall"
            + (f", with **{fmt_pct(best['company_accuracy'])}** company-detection accuracy."
               if "company_accuracy" in best else ".")
        )

        fig_exec = px.bar(
            current_view.assign(method_label=current_view["method"].map(lambda m: METHOD_LABELS.get(m, m))),
            x="method_label", y="overall_score", color="method",
            color_discrete_map=METHOD_COLORS, text_auto=".3f",
        )
        fig_exec.update_layout(showlegend=False, xaxis_title="", yaxis_title="Overall Score", height=350)
        st.plotly_chart(fig_exec, use_container_width=True, config={"displaylogo": False})

        for line in generate_insights(current_view, history_view)[:3]:
            st.markdown(f'<div class="insight-box">{line}</div>', unsafe_allow_html=True)
    else:
        st.info("No data available for a summary yet.")

# ------------------------------------------------------------
# TAB: DOWNLOADS
# ------------------------------------------------------------

with tab_download:
    st.subheader("📥 Downloads")

    csv_history = history_df.to_csv(index=False)
    st.download_button(
        "Download Evaluation History (CSV)", csv_history,
        "evaluation_results.csv", "text/csv", use_container_width=True,
    )

    st.divider()
    st.markdown("**Save raw outputs from this session** (judge scores, retrieved contexts, generated answers)")
    if st.button("💾 Save Raw Session Outputs to Disk", use_container_width=True):
        if st.session_state.judge_results is None:
            st.warning("Nothing to save yet — run a benchmark first.")
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = RAW_OUTPUT_DIR / ts
            run_dir.mkdir(parents=True, exist_ok=True)

            jr = st.session_state.judge_results
            jr_df = jr if isinstance(jr, pd.DataFrame) else pd.DataFrame(jr)
            jr_df.to_csv(run_dir / "judge_results.csv", index=False)

            if st.session_state.judge_contexts is not None:
                try:
                    ctx_serializable = {
                        f"{k[0]}::{k[1]}" if isinstance(k, tuple) else str(k): v
                        for k, v in st.session_state.judge_contexts.items()
                    }
                    (run_dir / "retrieved_contexts.json").write_text(json.dumps(ctx_serializable, indent=2))
                except Exception:
                    pass

            if st.session_state.ragas_results is not None:
                rr = st.session_state.ragas_results
                rr_df = rr if isinstance(rr, pd.DataFrame) else pd.DataFrame(rr)
                rr_df.to_csv(run_dir / "ragas_results.csv", index=False)

            st.success(f"Saved raw outputs to `{run_dir}/` for auditing and debugging.")

    st.caption(
        "Chart images can be downloaded as PNG directly from each chart's toolbar (camera icon, top-right)."
    )