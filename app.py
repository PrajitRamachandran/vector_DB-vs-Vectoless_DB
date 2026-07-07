#app.py

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from streamlit_app.auth.session_manager import (
    logout_user,
    is_authenticated,
    get_current_username,
    get_current_role,
)

from streamlit_app.database.repository import (
    get_evaluations,
    get_best_method,
    get_dashboard_stats,
    get_dashboard_stats_by_user,
    get_recent_activity,
    get_system_health,
    get_evaluation_trend,
)

from streamlit_app.services.rag_service import pipeline_status

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Financial RAG System",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MANIFEST_PATH = PROCESSED_DIR / "manifest.json"

EVALUATION_DIR = PROJECT_ROOT / "evaluation"
RESULTS_DIR = EVALUATION_DIR / "results"

PAGES = {
    "dashboard": "pages/01_dashboard.py",
    "upload": "pages/02_upload_documents.py",
    "index_manager": "pages/03_index_manager.py",
    "chat": "pages/04_chat.py",
    "conversations": "pages/05_conversations.py",
    "evaluation": "pages/06_evaluations.py",
    "login": "pages/00_login.py",
}

# ============================================================
# CONSTANTS — shared visual language with the Evaluation dashboard
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

METHOD_ICONS = {
    "vector": "🧭",
    "vectorless": "🔤",
    "hybrid": "🧬",
}

METRIC_HELP = {
    "judge_score": "An LLM-as-judge rates each generated answer from 1–5 for correctness and quality against the source documents.",
    "faithfulness": "RAGAS metric (0–1): how well the generated answer is grounded in the retrieved context, with less hallucination scoring higher.",
    "answer_relevancy": "RAGAS metric (0–1): how directly the generated answer addresses the actual question asked.",
    "context_precision": "RAGAS metric (0–1): how much of the retrieved context was actually relevant and necessary to answer the question.",
    "context_recall": "RAGAS metric (0–1): how completely the retrieved context covers the information needed for a full answer.",
    "company_accuracy": "Share of questions where the pipeline correctly identified which company's filing the question refers to.",
    "overall_score": "A blended score combining the judge score with all four RAGAS metrics into a single comparable number.",
}

# ============================================================
# SESSION STATE
# ============================================================

for key, default in [
    ("selected_method", "Vectorless"),
    ("chat_history", []),
    ("uploaded_files", []),
    ("evaluation_running", False),
    ("latest_results", None),
    ("app_loaded", True),
    ("health_status", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ============================================================
# STYLING — "LUXURY" DARK-MODE-FRIENDLY THEME
# (kept visually consistent with the Evaluation dashboard)
# ============================================================

def inject_css():
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.4rem; }

        /* ---------------- HERO ---------------- */
        .hero {
            background: linear-gradient(135deg, rgba(59,130,246,0.16) 0%, rgba(16,185,129,0.10) 50%, rgba(245,158,11,0.10) 100%);
            border: 1px solid rgba(212,175,55,0.30);
            border-radius: 20px;
            padding: 32px 36px;
            margin-bottom: 18px;
            box-shadow: 0 6px 24px rgba(0,0,0,0.16);
        }
        .hero-eyebrow {
            font-size: 0.80rem; letter-spacing: 0.10em; text-transform: uppercase;
            opacity: 0.70; font-weight: 700; margin-bottom: 8px;
        }
        .hero-title { font-size: 2.3rem; font-weight: 800; line-height: 1.15; margin-bottom: 10px; }
        .hero-subtitle { font-size: 1.02rem; opacity: 0.80; max-width: 760px; line-height: 1.6; margin-bottom: 4px; }

        /* ---------------- KPI / METRIC CARDS ---------------- */
        .kpi-row { display: flex; gap: 14px; margin-bottom: 6px; flex-wrap: wrap; }
        .kpi-card {
            flex: 1; min-width: 190px;
            background: linear-gradient(135deg, rgba(212,175,55,0.10) 0%, rgba(120,120,140,0.06) 100%);
            border: 1px solid rgba(212,175,55,0.35);
            border-radius: 14px;
            padding: 16px 18px;
            box-shadow: 0 4px 18px rgba(0,0,0,0.12);
        }
        .kpi-label {
            font-size: 0.75rem; letter-spacing: 0.06em; text-transform: uppercase;
            opacity: 0.65; margin-bottom: 6px; font-weight: 600;
        }
        .kpi-value { font-size: 1.55rem; font-weight: 700; line-height: 1.15; }
        .kpi-sub { font-size: 0.76rem; opacity: 0.6; margin-top: 4px; }

        /* ---------------- STATUS BADGES ---------------- */
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

        /* ---------------- WINNER / METRIC BADGES ---------------- */
        .badge-pill {
            display: inline-flex; align-items: center; gap: 6px;
            padding: 6px 14px; border-radius: 999px;
            font-size: 0.82rem; font-weight: 700; margin: 3px 6px 3px 0;
            border: 1px solid rgba(255,255,255,0.12);
        }

        /* ---------------- INSIGHT / SUMMARY BOX ---------------- */
        .insight-box {
            background: rgba(212,175,55,0.08);
            border-left: 3px solid #D4AF37;
            border-radius: 8px;
            padding: 14px 18px;
            margin: 8px 0 18px 0;
            font-size: 0.94rem;
            line-height: 1.65;
        }
        .summary-card {
            border-radius: 16px; padding: 20px 24px; margin-bottom: 10px;
            border: 1px solid rgba(255,255,255,0.10);
            background: linear-gradient(135deg, rgba(16,185,129,0.10) 0%, rgba(120,120,140,0.05) 100%);
        }

        /* ---------------- WORKFLOW ---------------- */
        .workflow-row { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; margin: 10px 0 6px 0; }
        .workflow-step {
            flex: 1; min-width: 118px; text-align: center;
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.10);
            border-radius: 12px; padding: 12px 8px;
        }
        .workflow-step .icon { font-size: 1.4rem; }
        .workflow-step .label { font-size: 0.78rem; font-weight: 700; margin-top: 4px; }
        .workflow-arrow { opacity: 0.35; font-size: 1.2rem; padding: 0 2px; }

        /* ---------------- ACTIVITY FEED ---------------- */
        .activity-item {
            display: flex; gap: 10px; align-items: flex-start;
            padding: 8px 4px; border-bottom: 1px dashed rgba(255,255,255,0.08);
            font-size: 0.86rem;
        }
        .activity-item:last-child { border-bottom: none; }
        .activity-time { opacity: 0.55; font-size: 0.76rem; min-width: 92px; }

        /* ---------------- MISC ---------------- */
        .medal { font-size: 1.1rem; margin-right: 4px; }
        .section-title { font-size: 1.08rem; font-weight: 700; margin-top: 4px; margin-bottom: 2px; }
        .muted { opacity: 0.65; font-size: 0.85rem; }
        table.comparison-table { width: 100%; border-collapse: collapse; font-size: 0.90rem; }
        table.comparison-table th, table.comparison-table td {
            padding: 10px 12px; border-bottom: 1px solid rgba(255,255,255,0.08); text-align: left;
        }
        table.comparison-table th { font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.04em; opacity: 0.75; }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_css()

# ============================================================
# SIDEBAR — AUTH
# ============================================================

with st.sidebar:
    if is_authenticated():
        username = get_current_username() or "User"
        role = get_current_role() or "user"

        st.divider()
        st.markdown(
            f"""
            ### 👤 User

            **Username:** {username}

            **Role:** {role.title()}
            """
        )

        if st.button("🚪 Logout", use_container_width=True, key="logout_button"):
            logout_user()
            st.switch_page(PAGES["login"])
    else:
        st.divider()
        st.markdown("### 👋 Welcome")
        st.caption("Log in to start chatting and running benchmarks.")
        st.page_link(PAGES["login"], label="Log In", icon="🔐", use_container_width=True)

# ============================================================
# DATA LOADING (always fresh — no caching, so every rerun after an
# upload / index rebuild / benchmark run reflects the latest state)
# ============================================================

def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        return {}
    try:
        return json.loads(MANIFEST_PATH.read_text())
    except Exception:
        return {}


def summarize_manifest(manifest: dict) -> dict:
    if not manifest:
        return {
            "total_documents": 0, "companies": [], "total_pages": 0,
            "total_parents": 0, "total_children": 0, "per_company": {},
        }

    companies = sorted({doc.get("company", "Unknown") for doc in manifest.values()})
    total_pages = sum(doc.get("pages_count", 0) or 0 for doc in manifest.values())
    total_parents = sum(doc.get("parents_count", 0) or 0 for doc in manifest.values())
    total_children = sum(doc.get("children_count", 0) or 0 for doc in manifest.values())

    per_company = {}
    for fname, doc in manifest.items():
        c = doc.get("company", "Unknown")
        entry = per_company.setdefault(c, {"documents": 0, "pages": 0, "parents": 0, "children": 0})
        entry["documents"] += 1
        entry["pages"] += doc.get("pages_count", 0) or 0
        entry["parents"] += doc.get("parents_count", 0) or 0
        entry["children"] += doc.get("children_count", 0) or 0

    return {
        "total_documents": len(manifest),
        "companies": companies,
        "total_pages": total_pages,
        "total_parents": total_parents,
        "total_children": total_children,
        "per_company": per_company,
    }


def latest_per_method(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "timestamp" not in df.columns:
        return df
    d = df.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce")
    return d.sort_values("timestamp").groupby("method", as_index=False).tail(1)


def fmt_score(x, digits=3):
    try:
        v = float(x)
        if pd.isna(v):
            return "—"
        return f"{v:.{digits}f}"
    except Exception:
        return "—"


def fmt_pct(x):
    try:
        v = float(x)
        if pd.isna(v):
            return "—"
        return f"{v * 100:.1f}%"
    except Exception:
        return "—"


def fmt_ts(ts) -> str:
    if ts is None or (isinstance(ts, float) and pd.isna(ts)):
        return "—"
    try:
        return pd.Timestamp(ts).strftime("%b %d, %Y · %H:%M")
    except Exception:
        return str(ts)


def time_ago(ts) -> str:
    try:
        t = pd.Timestamp(ts)
        if pd.isna(t):
            return "—"
        delta = datetime.now() - t.to_pydatetime().replace(tzinfo=None)
        seconds = int(delta.total_seconds())
        if seconds < 0:
            return "just now"
        if seconds < 60:
            return "just now"
        if seconds < 3600:
            return f"{seconds // 60}m ago"
        if seconds < 86400:
            return f"{seconds // 3600}h ago"
        return f"{seconds // 86400}d ago"
    except Exception:
        return "—"


try:
    evaluations = get_evaluations()
    evaluations_ok = True
except Exception:
    evaluations = []
    evaluations_ok = False

evaluations_df = pd.DataFrame(evaluations) if evaluations else pd.DataFrame()
current_df = latest_per_method(evaluations_df) if not evaluations_df.empty else evaluations_df

try:
    best_overall = get_best_method()
except Exception:
    best_overall = None

try:
    dash_stats = (
        get_dashboard_stats_by_user(st.session_state.get("user_id"))
        if is_authenticated() and st.session_state.get("user_id") is not None
        else get_dashboard_stats()
    )
except Exception:
    dash_stats = {}

try:
    recent_activity = get_recent_activity(limit=8)
except Exception:
    recent_activity = []

try:
    db_health = get_system_health()
except Exception:
    db_health = {"database": False, "conversations_table": False, "evaluations_table": False}

manifest = load_manifest()
corpus = summarize_manifest(manifest)

# Recommended method = best overall_score among each method's *latest* run
# (recency-aware — distinct from best_overall, which is the best run ever recorded).
recommended = None
if not current_df.empty and "overall_score" in current_df.columns:
    valid = current_df.dropna(subset=["overall_score"])
    if not valid.empty:
        recommended = valid.loc[valid["overall_score"].idxmax()]

# Per-metric winners among current standings, for the badge row.
metric_winners = {}
if not current_df.empty:
    for metric in ["overall_score", "faithfulness", "context_recall", "company_accuracy"]:
        if metric in current_df.columns:
            valid = current_df.dropna(subset=[metric])
            if not valid.empty:
                row = valid.loc[valid[metric].idxmax()]
                metric_winners[metric] = (row["method"], row[metric])

# ============================================================
# PIPELINE / SYSTEM HEALTH (uses st.cache_resource internally, so
# repeat visits are effectively instant after the first check)
# ============================================================

def run_health_check(force: bool = False) -> dict:
    if force:
        try:
            from streamlit_app.services.rag_service import clear_pipeline_cache
            clear_pipeline_cache()
        except Exception:
            pass
    try:
        status = pipeline_status()
    except Exception:
        status = {"vector": False, "vectorless": False, "hybrid": False, "random": False, "auto": False}

    llm_configured = False
    try:
        import config
        llm_configured = bool(getattr(config, "LLM_MODEL_ID", None))
    except Exception:
        pass

    return {
        "vector": status.get("vector", False),
        "vectorless": status.get("vectorless", False),
        "hybrid": status.get("hybrid", False),
        "llm_configured": llm_configured,
        "database": db_health.get("database", False),
        "evaluations_table": db_health.get("evaluations_table", False),
    }


if st.session_state.health_status is None:
    with st.spinner("Running startup health checks…"):
        st.session_state.health_status = run_health_check()

health = st.session_state.health_status

# ============================================================
# HERO BANNER
# ============================================================

username = get_current_username() if is_authenticated() else None
role = get_current_role() if is_authenticated() else None

if username:
    greeting = f"Welcome back, {username} 👋"
    sub_line = (
        "You have admin access — you can manage the index, rerun benchmarks, and review system health below."
        if role == "admin"
        else "Pick up where you left off — chat with your filings or check the latest benchmark standings below."
    )
else:
    greeting = "Financial RAG Benchmark"
    sub_line = "Log in from the sidebar to chat with indexed filings and run benchmarks."

st.markdown(
    f"""
    <div class="hero">
        <div class="hero-eyebrow">📊 Retrieval-Augmented Generation · 10-K / 10-Q Benchmark Suite</div>
        <div class="hero-title">{greeting}</div>
        <div class="hero-subtitle">
            A side-by-side benchmark of <b>Vector</b>, <b>Vectorless (BM25)</b>, and <b>Hybrid (RRF Fusion)</b>
            retrieval over real financial filings — with an LLM judge and RAGAS scoring every run so you can see,
            with evidence, which retrieval strategy actually answers financial questions best.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

cta_cols = st.columns(4)
with cta_cols[0]:
    st.page_link(PAGES["chat"], label="Start Chat", icon="💬", use_container_width=True)
with cta_cols[1]:
    st.page_link(PAGES["dashboard"], label="Dashboard", icon="📈", use_container_width=True)
with cta_cols[2]:
    st.page_link(PAGES["evaluation"], label="Evaluation", icon="📊", use_container_width=True)
with cta_cols[3]:
    st.page_link(PAGES["upload"], label="Upload Documents", icon="📥", use_container_width=True)

# ============================================================
# DETAILED READINESS / STARTUP CHECKLIST
# ============================================================

checklist = []
checklist.append(("Database connection", health["database"], "SQLite database is unreachable — check your DB path/permissions." ))
checklist.append(("Evaluations table", health["evaluations_table"], "The evaluations table couldn't be read — try re-running schema initialization."))
checklist.append(("Vector pipeline (ChromaDB + BGE + Reranker)", health["vector"], "Vector pipeline failed to load — check ChromaDB path and embedding model."))
checklist.append(("Vectorless pipeline (BM25 + Reranker)", health["vectorless"], "Vectorless pipeline failed to load — check the BM25 index."))
checklist.append(("Hybrid pipeline (RRF Fusion)", health["hybrid"], "Hybrid pipeline failed to load — check that both Vector and Vectorless pipelines load first."))
checklist.append(("LLM model configured", health["llm_configured"], "No LLM_MODEL_ID found in config — set one before chatting or judging."))
checklist.append(("Documents indexed", corpus["total_documents"] > 0, f"No documents found in {MANIFEST_PATH.name} — upload filings to get started."))
checklist.append(("Benchmark history", not evaluations_df.empty, "No benchmark runs yet — run a Judge + RAGAS evaluation to populate the dashboard."))

failing = [c for c in checklist if not c[1]]

status_html = '<div class="status-row">'
for label, ok, _ in checklist:
    cls = "status-ok" if ok else "status-warn"
    icon = "✅" if ok else "⚠️"
    status_html += f'<span class="status-badge {cls}">{icon} {label}</span>'
status_html += "</div>"
st.markdown(status_html, unsafe_allow_html=True)

if failing:
    with st.expander(f"⚠️ {len(failing)} item(s) need attention", expanded=False):
        for label, _, hint in failing:
            st.warning(f"**{label}** — {hint}")
else:
    st.success("All systems initialized successfully — documents indexed, pipelines loaded, and benchmark history available.")

st.divider()

# ============================================================
# KPI ROW
# ============================================================

st.markdown('<div class="section-title">At a Glance</div>', unsafe_allow_html=True)

recommended_label = METHOD_LABELS.get(recommended["method"], recommended["method"]) if recommended is not None else "—"
recommended_icon = METHOD_ICONS.get(recommended["method"], "🏆") if recommended is not None else "🏆"

st.markdown(
    f"""
    <div class="kpi-row">
      <div class="kpi-card">
        <div class="kpi-label">📄 Documents Indexed</div>
        <div class="kpi-value">{corpus['total_documents']}</div>
        <div class="kpi-sub">{corpus['total_pages']} pages processed</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">🏢 Companies Covered</div>
        <div class="kpi-value">{len(corpus['companies'])}</div>
        <div class="kpi-sub">{", ".join(corpus['companies'][:3]) + ("…" if len(corpus['companies']) > 3 else "") if corpus['companies'] else "no filings yet"}</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">🧪 Retrieval Methods</div>
        <div class="kpi-value">3</div>
        <div class="kpi-sub">Vector · Vectorless · Hybrid</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">🧮 Benchmark Runs</div>
        <div class="kpi-value">{dash_stats.get('evaluation_runs', len(evaluations_df))}</div>
        <div class="kpi-sub">{dash_stats.get('total_chats', 0)} chat messages logged</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">{recommended_icon} Latest Winner</div>
        <div class="kpi-value" style="font-size:1.25rem;">{recommended_label}</div>
        <div class="kpi-sub">{fmt_score(recommended['overall_score']) if recommended is not None else "run a benchmark"} overall score</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()

# ============================================================
# LATEST BENCHMARK SUMMARY + RECOMMENDATION
# ============================================================

left, right = st.columns([1.4, 1])

with left:
    st.markdown('<div class="section-title">🏆 Latest Benchmark Summary</div>', unsafe_allow_html=True)
    if best_overall:
        m = best_overall.get("method")
        color = METHOD_COLORS.get(m, "#999")
        st.markdown(
            f"""
            <div class="summary-card">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
                    <div style="font-size:1.15rem; font-weight:700;">
                        {METHOD_ICONS.get(m, "🏆")} Best Run: {METHOD_LABELS.get(m, m)}
                    </div>
                    <span class="badge-pill" style="background:{color}22; color:{color}; border-color:{color}55;">
                        Overall {fmt_score(best_overall.get('overall_score'))}
                    </span>
                </div>
                <div class="kpi-row">
                  <div class="kpi-card">
                    <div class="kpi-label">Judge Score</div>
                    <div class="kpi-value">{fmt_score(best_overall.get('avg_judge_score'), 2)} / 5</div>
                  </div>
                  <div class="kpi-card">
                    <div class="kpi-label">Faithfulness</div>
                    <div class="kpi-value">{fmt_score(best_overall.get('faithfulness'))}</div>
                  </div>
                  <div class="kpi-card">
                    <div class="kpi-label">Company Accuracy</div>
                    <div class="kpi-value">{fmt_pct(best_overall.get('company_accuracy'))}</div>
                  </div>
                  <div class="kpi-card">
                    <div class="kpi-label">Run Timestamp</div>
                    <div class="kpi-value" style="font-size:1.05rem;">{fmt_ts(best_overall.get('timestamp'))}</div>
                  </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("No benchmark runs recorded yet. Head to the Evaluation page to run your first Judge + RAGAS benchmark.")

    # Metric winner badges (current standings — latest run per method)
    if metric_winners:
        st.markdown('<div class="section-title" style="margin-top:14px;">🎖️ Current Standings Badges</div>', unsafe_allow_html=True)
        badge_defs = [
            ("overall_score", "🏆", "Overall Winner"),
            ("faithfulness", "🛡️", "Best Faithfulness"),
            ("context_recall", "🔎", "Best Recall"),
            ("company_accuracy", "🏢", "Best Company Accuracy"),
        ]
        badges_html = ""
        for key, icon, label in badge_defs:
            if key in metric_winners:
                method, value = metric_winners[key]
                color = METHOD_COLORS.get(method, "#999")
                val_str = fmt_pct(value) if key in ("company_accuracy",) else fmt_score(value)
                badges_html += (
                    f'<span class="badge-pill" style="background:{color}22; color:{color}; border-color:{color}55;" '
                    f'title="{METRIC_HELP.get(key, "")}">{icon} {label}: {METHOD_LABELS.get(method, method)} ({val_str})</span>'
                )
        st.markdown(badges_html, unsafe_allow_html=True)

with right:
    st.markdown('<div class="section-title">🧭 Recommended Method</div>', unsafe_allow_html=True)
    if recommended is not None:
        color = METHOD_COLORS.get(recommended["method"], "#999")
        st.markdown(
            f"""
            <div class="insight-box" style="border-left-color:{color};">
                Based on the <b>latest</b> run of each method, <b>{METHOD_LABELS.get(recommended['method'], recommended['method'])}</b>
                currently leads with an overall score of <b>{fmt_score(recommended['overall_score'])}</b>.
                Use it as the default for new chats unless you're specifically testing keyword-heavy or long-context queries.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.caption("Run a benchmark to get a data-backed recommendation.")

    st.markdown('<div class="section-title" style="margin-top:10px;">📉 Benchmark Trend</div>', unsafe_allow_html=True)
    try:
        trend = get_evaluation_trend(limit=20)
    except Exception:
        trend = []

    if trend:
        trend_df = pd.DataFrame(trend)
        trend_df["timestamp"] = pd.to_datetime(trend_df["timestamp"], errors="coerce")
        trend_df["method_label"] = trend_df["method"].map(lambda m: METHOD_LABELS.get(m, m))
        fig = px.line(
            trend_df.sort_values("timestamp"),
            x="timestamp", y="overall_score", color="method_label",
            color_discrete_map={METHOD_LABELS.get(k, k): v for k, v in METHOD_COLORS.items()},
            markers=True, height=230,
        )
        fig.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", y=-0.3),
            xaxis_title="", yaxis_title="Overall Score",
        )
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
        st.caption("Full trend, per-run drilldowns, and comparisons live on the Evaluation page.")
    else:
        st.caption("No trend data yet — trend appears once at least one benchmark has been run.")

st.divider()

# ============================================================
# QUICK ACTIONS
# ============================================================

st.markdown('<div class="section-title">⚡ Quick Actions</div>', unsafe_allow_html=True)

qa_row1 = st.columns(3)
qa_row2 = st.columns(3)

quick_actions = [
    ("📥", "Upload Documents", "Add new 10-K / 10-Q filings to the corpus.", PAGES["upload"]),
    ("💬", "Start Chat", "Ask questions against the indexed filings.", PAGES["chat"]),
    ("🗂️", "Manage Index", "Rebuild or inspect the Vector / BM25 index.", PAGES["index_manager"]),
    ("📈", "Dashboard", "Usage analytics, latency, and top queries.", PAGES["dashboard"]),
    ("🧑‍⚖️", "Judge Evaluation", "Run the LLM-as-judge benchmark.", PAGES["evaluation"]),
    ("📐", "RAGAS Evaluation", "Score faithfulness, relevancy, precision & recall.", PAGES["evaluation"]),
]

for i, (icon, title, desc, target) in enumerate(quick_actions):
    col = qa_row1[i] if i < 3 else qa_row2[i - 3]
    with col:
        with st.container(border=True):
            st.markdown(f"**{icon} {title}**")
            st.caption(desc)
            st.page_link(target, label=f"Open {title}", icon="📂")

st.divider()

# ============================================================
# RETRIEVAL METHOD COMPARISON
# ============================================================

st.markdown('<div class="section-title">⚖️ Retrieval Method Comparison</div>', unsafe_allow_html=True)

st.markdown(
    f"""
    <table class="comparison-table">
        <tr>
            <th>Aspect</th>
            <th style="color:{METHOD_COLORS['vector']};">🧭 Vector RAG</th>
            <th style="color:{METHOD_COLORS['vectorless']};">🔤 Vectorless RAG</th>
            <th style="color:{METHOD_COLORS['hybrid']};">🧬 Hybrid RAG</th>
        </tr>
        <tr>
            <td>Retrieval basis</td>
            <td>Dense embeddings (BGE) + ChromaDB similarity search</td>
            <td>Sparse keyword matching (BM25)</td>
            <td>Vector + BM25 fused via Reciprocal Rank Fusion (RRF)</td>
        </tr>
        <tr>
            <td>Strength</td>
            <td>Understands semantic / paraphrased questions</td>
            <td>Exact terms, tickers, line-item names, numbers</td>
            <td>Best of both — semantic and exact-match recall</td>
        </tr>
        <tr>
            <td>Weakness</td>
            <td>Can miss exact numeric or rare-term matches</td>
            <td>Struggles with paraphrased or conceptual questions</td>
            <td>Slightly higher latency (two retrievers + fusion)</td>
        </tr>
        <tr>
            <td>Best for</td>
            <td>"What are Apple's main revenue drivers?"</td>
            <td>"What was the exact figure for total liabilities?"</td>
            <td>Mixed / unpredictable question sets</td>
        </tr>
        <tr>
            <td>Reranking</td>
            <td>Cross-encoder reranking</td>
            <td>Cross-encoder reranking</td>
            <td>Cross-encoder reranking after fusion</td>
        </tr>
    </table>
    """,
    unsafe_allow_html=True,
)

with st.expander("💡 Why Hybrid / Vector / Vectorless? — quick explainer"):
    e1, e2, e3 = st.columns(3)
    with e1:
        st.markdown(f"**{METHOD_ICONS['vector']} Vector RAG**")
        st.caption(
            "Embeds both the question and document chunks into the same vector space and retrieves by "
            "cosine similarity. Good when the question is phrased differently from the source text."
        )
    with e2:
        st.markdown(f"**{METHOD_ICONS['vectorless']} Vectorless RAG**")
        st.caption(
            "Uses BM25, a classic keyword-frequency ranking algorithm. No embeddings involved — excels at "
            "precise term matches like specific dollar figures or line-item names."
        )
    with e3:
        st.markdown(f"**{METHOD_ICONS['hybrid']} Hybrid RAG**")
        st.caption(
            "Runs both retrievers in parallel and fuses their rankings with Reciprocal Rank Fusion (RRF) "
            "before reranking — aiming to combine semantic recall with keyword precision."
        )

st.divider()

# ============================================================
# WORKFLOW VISUALIZATION
# ============================================================

st.markdown('<div class="section-title">🔄 End-to-End Pipeline</div>', unsafe_allow_html=True)

workflow_steps = [
    ("📥", "Upload"),
    ("⚙️", "Processing"),
    ("🗂️", "Indexing"),
    ("🔎", "Retrieval"),
    ("✍️", "Generation"),
    ("🧪", "Evaluation"),
    ("📊", "Analytics"),
]

workflow_html = '<div class="workflow-row">'
for i, (icon, label) in enumerate(workflow_steps):
    workflow_html += f'<div class="workflow-step"><div class="icon">{icon}</div><div class="label">{label}</div></div>'
    if i < len(workflow_steps) - 1:
        workflow_html += '<div class="workflow-arrow">➜</div>'
workflow_html += "</div>"
st.markdown(workflow_html, unsafe_allow_html=True)
st.caption(
    "Filings are parsed and chunked (parent/child hierarchy), embedded and indexed into ChromaDB + BM25, "
    "retrieved and reranked at query time, answered by the LLM, then scored by the Judge + RAGAS evaluators."
)

st.divider()

# ============================================================
# CORPUS OVERVIEW
# ============================================================

st.markdown('<div class="section-title">🏢 Corpus Overview</div>', unsafe_allow_html=True)

if corpus["total_documents"] == 0:
    st.info("No documents indexed yet. Use **Upload Documents** to add your first filing.")
else:
    corpus_rows = [
        {
            "Company": company,
            "Documents": v["documents"],
            "Pages": v["pages"],
            "Parent Chunks": v["parents"],
            "Child Chunks": v["children"],
        }
        for company, v in sorted(corpus["per_company"].items())
    ]
    st.dataframe(pd.DataFrame(corpus_rows), use_container_width=True, hide_index=True)

st.divider()

# ============================================================
# BENCHMARK HIGHLIGHTS
# ============================================================

st.markdown('<div class="section-title">📌 Benchmark Highlights</div>', unsafe_allow_html=True)

total_questions = None
if not evaluations_df.empty and "total_questions" in evaluations_df.columns:
    latest_rows = current_df if not current_df.empty else evaluations_df
    if "total_questions" in latest_rows.columns and not latest_rows.empty:
        total_questions = int(latest_rows["total_questions"].dropna().max()) if latest_rows["total_questions"].notna().any() else None

hl_cols = st.columns(5)
with hl_cols[0]:
    st.metric("Total Questions (latest run)", total_questions if total_questions is not None else "—",
               help=METRIC_HELP["judge_score"])
with hl_cols[1]:
    st.metric("Total Chunks", corpus["total_parents"] + corpus["total_children"],
               help="Sum of all parent and child chunks across the indexed corpus.")
with hl_cols[2]:
    st.metric("Parent Chunks", corpus["total_parents"],
               help="Larger, coarse-grained chunks used for context assembly.")
with hl_cols[3]:
    st.metric("Child Chunks", corpus["total_children"],
               help="Smaller, fine-grained chunks used for precise retrieval matching.")
with hl_cols[4]:
    st.metric("Indexed Pages", corpus["total_pages"],
               help="Total PDF pages processed across all indexed filings.")

st.divider()

# ============================================================
# SYSTEM HEALTH DASHBOARD
# ============================================================

st.markdown('<div class="section-title">🩺 System Health</div>', unsafe_allow_html=True)

health_items = [
    ("Vector Retrieval", "ChromaDB + BGE Embeddings + Reranker", health["vector"]),
    ("Vectorless Retrieval", "BM25 + Reranker", health["vectorless"]),
    ("Hybrid Fusion", "RRF over Vector + BM25", health["hybrid"]),
    ("LLM", "Configured in config.LLM_MODEL_ID", health["llm_configured"]),
    ("Database", "SQLite connection", health["database"]),
    ("Evaluation Store", "Evaluations table readable", health["evaluations_table"]),
]

hcols = st.columns(6)
for col, (name, detail, ok) in zip(hcols, health_items):
    with col:
        icon = "✅" if ok else "❌"
        st.markdown(f"**{icon} {name}**")
        st.caption(detail)

if st.button("🔄 Recheck System Health", key="recheck_health"):
    with st.spinner("Reloading pipelines and rechecking health…"):
        st.session_state.health_status = run_health_check(force=True)
    st.rerun()

st.divider()

# ============================================================
# RECENT ACTIVITY FEED
# ============================================================

act_col, doc_col = st.columns([1.3, 1])

with act_col:
    st.markdown('<div class="section-title">🕒 Recent Activity</div>', unsafe_allow_html=True)
    if not recent_activity:
        st.caption("No activity recorded yet — uploads, chats, and benchmark runs will show up here.")
    else:
        feed_html = ""
        type_icons = {"conversation": "💬", "evaluation": "🧪"}
        for item in recent_activity:
            icon = type_icons.get(item.get("activity_type"), "•")
            detail = item.get("detail") or "—"
            detail_label = METHOD_LABELS.get(detail, detail)
            feed_html += (
                '<div class="activity-item">'
                f'<div>{icon}</div>'
                f'<div class="activity-time">{time_ago(item.get("timestamp"))}</div>'
                f'<div>{item.get("activity_type", "activity").title()} · {detail_label}</div>'
                "</div>"
            )
        st.markdown(feed_html, unsafe_allow_html=True)

with doc_col:
    st.markdown('<div class="section-title">📘 Learn More</div>', unsafe_allow_html=True)
    with st.expander("📄 Documentation"):
        st.caption(
            "This app compares three retrieval strategies over a corpus of 10-K / 10-Q filings. "
            "See the Evaluation page for full metric definitions, and the Dashboard for usage analytics."
        )
    with st.expander("🧪 Benchmark Methodology"):
        st.caption(
            "Each method answers the same fixed question set. An LLM judge scores every answer 1–5 for "
            "correctness, then RAGAS independently scores faithfulness, answer relevancy, context precision, "
            "and context recall. Company-detection accuracy is tracked separately as a correctness signal."
        )
    with st.expander("🏗️ Architecture Details"):
        st.markdown(
            """
            - **Vector RAG:** ChromaDB · BGE Embeddings · Cross-Encoder Reranking
            - **Vectorless RAG:** BM25 · Cross-Encoder Reranking
            - **Hybrid RAG:** Vector + BM25 · RRF Fusion · Cross-Encoder Reranking
            """
        )

# ============================================================
# FOOTER
# ============================================================

st.divider()
st.caption("Financial RAG Benchmark · Vector · Vectorless · Hybrid")