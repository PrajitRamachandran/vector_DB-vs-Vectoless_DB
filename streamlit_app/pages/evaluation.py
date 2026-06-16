"""
Evaluation Page
===============
Run the three-way RAG benchmark and visualise results.
Supports:
  - Mistral judge scoring (evaluator.py)
  - RAGAS metrics (ragas_evaluator.py)
  - Per-method, per-company, per-category breakdowns
  - Latency charts
"""

from __future__ import annotations

import sys
import json
from pathlib import Path
from typing import Optional

import streamlit as st
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from services.pipeline_manager import get_pipeline_manager

_RESULTS_DIR = _ROOT / "evaluation" / "results"
_QUESTIONS_PATH = _ROOT / "evaluation" / "test_questions.json"

_METHOD_COLORS = {
    "vector":     "#1f77b4",
    "vectorless": "#ff7f0e",
    "hybrid":     "#2ca02c",
}
_METHOD_LABELS = {
    "vector":     "Vector RAG",
    "vectorless": "Vectorless RAG",
    "hybrid":     "Hybrid RAG",
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_questions() -> list[dict]:
    if not _QUESTIONS_PATH.exists():
        return []
    with open(_QUESTIONS_PATH, encoding="utf-8") as f:
        return json.load(f).get("questions", [])


def _list_result_csvs() -> list[Path]:
    if not _RESULTS_DIR.exists():
        return []
    return sorted(_RESULTS_DIR.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)


def _load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalise company column to upper-case so pivot tables work
    if "company" in df.columns:
        df["company"] = df["company"].str.upper()
    return df


# ── Chart helpers ──────────────────────────────────────────────────────────────

def _bar_chart(df: pd.DataFrame, x: str, y: str, color: str, title: str):
    import matplotlib.pyplot as plt
    methods = df[x].unique()
    colors  = [_METHOD_COLORS.get(m, "#888") for m in methods]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    bars = ax.bar(
        [_METHOD_LABELS.get(m, m) for m in methods],
        df.set_index(x)[y],
        color=colors,
        edgecolor="white",
        linewidth=0.8,
    )
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylabel(y.replace("_", " ").title())
    ax.set_ylim(0, max(df[y].max() * 1.2, 1))
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig


def _heatmap(pivot: pd.DataFrame, title: str):
    import matplotlib.pyplot as plt
    import numpy as np
    fig, ax = plt.subplots(figsize=(8, max(3, len(pivot) * 0.55 + 1)))
    data = pivot.values.astype(float)
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=1, vmax=5)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([_METHOD_LABELS.get(c, c) for c in pivot.columns], fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = data[i, j]
            ax.text(j, i, f"{v:.2f}" if not np.isnan(v) else "—",
                    ha="center", va="center", fontsize=9,
                    color="black" if 2 < v < 4.5 else "white")
    ax.set_title(title, fontsize=11, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.03)
    plt.tight_layout()
    return fig


def _latency_chart(df: pd.DataFrame):
    import matplotlib.pyplot as plt
    import numpy as np

    methods = df["method"].unique()
    labels  = [_METHOD_LABELS.get(m, m) for m in methods]
    r_times = [df[df["method"] == m]["retrieval_time"].mean() for m in methods]
    g_times = [df[df["method"] == m]["generation_time"].mean() for m in methods]

    x = np.arange(len(methods))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    b1 = ax.bar(x - w/2, r_times, w, label="Retrieval", color="#4e79a7")
    b2 = ax.bar(x + w/2, g_times, w, label="Generation", color="#f28e2b")
    ax.bar_label(b1, fmt="%.2fs", padding=2, fontsize=8)
    ax.bar_label(b2, fmt="%.2fs", padding=2, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Seconds")
    ax.set_title("Average Latency Breakdown", fontsize=11, fontweight="bold")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig


# ── Main render ────────────────────────────────────────────────────────────────

def render() -> None:
    st.title("🔬 Evaluation")
    st.caption("Run the benchmark and explore results across all three RAG pipelines.")

    pm = get_pipeline_manager()

    tabs = st.tabs(["▶ Run Benchmark", "📈 Results & Charts", "📋 Question Browser"])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 — Run benchmark
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[0]:
        st.subheader("Configure Run")

        questions = _load_questions()
        if not questions:
            st.error(f"No questions found at `{_QUESTIONS_PATH}`. Add a test_questions.json first.")
            return

        st.metric("Questions in test set", len(questions))

        col1, col2 = st.columns(2)
        with col1:
            run_vector     = st.checkbox("Vector RAG",     value=True)
            run_vectorless = st.checkbox("Vectorless RAG", value=True)
            run_hybrid     = st.checkbox("Hybrid RAG",     value=True)

        with col2:
            run_ragas = st.checkbox(
                "Also run RAGAS metrics",
                value=False,
                help="Requires reference_answer fields in test_questions.json",
            )
            results_filename = st.text_input(
                "Output filename (CSV)",
                value="streamlit_eval_results.csv",
            )

        # Validate at least one method is selected
        if not any([run_vector, run_vectorless, run_hybrid]):
            st.warning("Select at least one pipeline to evaluate.")

        st.divider()
        run_btn = st.button(
            "▶ Run Evaluation",
            type="primary",
            disabled=not any([run_vector, run_vectorless, run_hybrid]),
            use_container_width=False,
        )

        if run_btn:
            # Lazy-load selected pipelines
            pipelines: dict = {}
            progress = st.progress(0, text="Loading pipelines…")
            loaded = 0
            total_pipelines = sum([run_vector, run_vectorless, run_hybrid])

            if run_vector:
                with st.spinner("Loading Vector RAG…"):
                    try:
                        pipelines["vector"] = pm.get_vector()
                    except Exception as exc:
                        st.error(f"Vector RAG failed to load: {exc}")
                        run_vector = False
                loaded += 1
                progress.progress(loaded / total_pipelines, text=f"Loaded {loaded}/{total_pipelines}")

            if run_vectorless:
                with st.spinner("Loading Vectorless RAG…"):
                    try:
                        pipelines["vectorless"] = pm.get_vectorless()
                    except Exception as exc:
                        st.error(f"Vectorless RAG failed to load: {exc}")
                        run_vectorless = False
                loaded += 1
                progress.progress(loaded / total_pipelines, text=f"Loaded {loaded}/{total_pipelines}")

            if run_hybrid:
                with st.spinner("Loading Hybrid RAG…"):
                    try:
                        pipelines["hybrid"] = pm.get_hybrid()
                    except Exception as exc:
                        st.error(f"Hybrid RAG failed to load: {exc}")
                        run_hybrid = False
                loaded += 1
                progress.progress(1.0, text="Pipelines ready")

            if not pipelines:
                st.error("No pipelines loaded — cannot run evaluation.")
                return

            progress.empty()

            # Run evaluation
            from evaluation.evaluator import run_evaluation
            log_box = st.empty()
            log_box.info(f"Running evaluation on {len(questions)} questions × {len(pipelines)} methods…")

            try:
                result = run_evaluation(
                    vector_pipeline      = pipelines.get("vector"),
                    vectorless_pipeline  = pipelines.get("vectorless"),
                    hybrid_pipeline      = pipelines.get("hybrid"),
                    results_filename     = results_filename,
                    capture_contexts     = run_ragas,
                )

                if run_ragas and isinstance(result, tuple):
                    judge_df, ctx_map = result
                else:
                    judge_df = result if not isinstance(result, tuple) else result[0]
                    ctx_map  = {}

                log_box.success(f"✅ Evaluation complete — {len(judge_df)} rows saved to `{results_filename}`")

                # Optionally run RAGAS
                if run_ragas and ctx_map:
                    ragas_log = st.empty()
                    ragas_log.info("Running RAGAS metrics…")
                    try:
                        from evaluation.ragas_evaluator import run_ragas_evaluation, merge_results
                        ragas_df = run_ragas_evaluation(
                            judge_df               = judge_df,
                            retrieved_contexts_map = ctx_map,
                            questions_path         = _QUESTIONS_PATH,
                            results_filename       = results_filename.replace(".csv", "_ragas.csv"),
                        )
                        combined = merge_results(judge_df, ragas_df)
                        ragas_log.success("✅ RAGAS metrics done.")
                        st.session_state["eval_df"] = combined
                    except Exception as exc:
                        ragas_log.error(f"RAGAS failed: {exc}")
                        st.session_state["eval_df"] = judge_df
                else:
                    st.session_state["eval_df"] = judge_df

                st.info("Switch to the **Results & Charts** tab to explore the results.")

            except Exception as exc:
                log_box.error(f"Evaluation failed: {exc}")
                st.exception(exc)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 — Results & Charts
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[1]:
        # Load from session state or from disk
        df: Optional[pd.DataFrame] = st.session_state.get("eval_df")

        csv_files = _list_result_csvs()
        if df is None and csv_files:
            selected_csv = st.selectbox(
                "Load a saved result file",
                csv_files,
                format_func=lambda p: p.name,
            )
            if selected_csv:
                df = _load_csv(selected_csv)
                st.session_state["eval_df"] = df

        if df is None or df.empty:
            st.info("No results loaded. Run an evaluation first, or select a saved CSV above.")
            return

        st.success(f"Loaded **{len(df)} rows** · {df['method'].nunique()} methods · {df['company'].nunique()} companies")
        st.divider()

        # ── Summary metrics ───────────────────────────────────────────────────
        st.subheader("Summary Metrics")
        summary = (
            df.groupby("method")
            .agg(
                avg_judge_score   = ("judge_score",      "mean"),
                pass_rate         = ("pass",             "mean"),
                avg_retrieval_s   = ("retrieval_time",   "mean"),
                avg_generation_s  = ("generation_time",  "mean"),
                avg_total_s       = ("total_time",       "mean"),
            )
            .reset_index()
        )
        summary["pass_rate"] = (summary["pass_rate"] * 100).round(1)

        display_summary = summary.copy()
        display_summary["method"] = display_summary["method"].map(
            lambda m: _METHOD_LABELS.get(m, m)
        )
        display_summary.columns = [
            "Method", "Avg Judge Score", "Pass Rate %",
            "Avg Retrieval (s)", "Avg Generation (s)", "Avg Total (s)",
        ]
        st.dataframe(
            display_summary.style.highlight_max(
                subset=["Avg Judge Score", "Pass Rate %"], color="#d4f4d4"
            ).highlight_min(
                subset=["Avg Total (s)"], color="#d4f4d4"
            ).format(precision=3),
            use_container_width=True,
            hide_index=True,
        )

        # ── Score bars ────────────────────────────────────────────────────────
        st.divider()
        ch1, ch2 = st.columns(2)

        with ch1:
            st.subheader("Average Judge Score")
            fig = _bar_chart(summary, "method", "avg_judge_score", "blue", "Avg Judge Score (1–5)")
            st.pyplot(fig)

        with ch2:
            st.subheader("Latency Breakdown")
            fig2 = _latency_chart(df)
            st.pyplot(fig2)

        # ── Heatmaps ──────────────────────────────────────────────────────────
        st.divider()
        st.subheader("Score by Company")
        try:
            pivot_company = (
                df.groupby(["company", "method"])["judge_score"]
                .mean()
                .unstack()
                .reindex(columns=[m for m in ["vector", "vectorless", "hybrid"] if m in df["method"].unique()])
            )
            st.pyplot(_heatmap(pivot_company, "Avg Judge Score by Company"))
        except Exception as exc:
            st.warning(f"Could not render company heatmap: {exc}")

        st.subheader("Score by Category")
        try:
            pivot_cat = (
                df.groupby(["category", "method"])["judge_score"]
                .mean()
                .unstack()
                .reindex(columns=[m for m in ["vector", "vectorless", "hybrid"] if m in df["method"].unique()])
            )
            st.pyplot(_heatmap(pivot_cat, "Avg Judge Score by Category"))
        except Exception as exc:
            st.warning(f"Could not render category heatmap: {exc}")

        # ── RAGAS metrics ─────────────────────────────────────────────────────
        ragas_cols = [c for c in df.columns if c in
                      ["answer_relevancy", "faithfulness", "context_precision",
                       "context_recall", "contextual_relevancy"]]
        if ragas_cols:
            st.divider()
            st.subheader("RAGAS Metrics")
            ragas_summary = (
                df.groupby("method")[ragas_cols]
                .mean()
                .reset_index()
            )
            ragas_summary["method"] = ragas_summary["method"].map(
                lambda m: _METHOD_LABELS.get(m, m)
            )
            st.dataframe(
                ragas_summary.style.highlight_max(subset=ragas_cols, color="#d4f4d4").format(precision=4),
                use_container_width=True,
                hide_index=True,
            )

        # ── Raw data ──────────────────────────────────────────────────────────
        st.divider()
        with st.expander("Raw results table"):
            filter_method = st.multiselect(
                "Filter by method",
                df["method"].unique().tolist(),
                default=df["method"].unique().tolist(),
                key="raw_method_filter",
            )
            filtered = df[df["method"].isin(filter_method)]
            st.dataframe(filtered, use_container_width=True, hide_index=True)
            csv_bytes = filtered.to_csv(index=False).encode()
            st.download_button(
                "⬇ Download filtered CSV",
                csv_bytes,
                file_name="eval_filtered.csv",
                mime="text/csv",
            )

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3 — Question browser
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[2]:
        st.subheader("Test Question Browser")
        questions = _load_questions()
        if not questions:
            st.info(f"No questions found at `{_QUESTIONS_PATH}`.")
            return

        companies  = sorted({q.get("company",  "") for q in questions})
        categories = sorted({q.get("category", "") for q in questions})

        col_f1, col_f2 = st.columns(2)
        with col_f1:
            comp_filter = st.multiselect("Company", companies, default=companies)
        with col_f2:
            cat_filter = st.multiselect("Category", categories, default=categories)

        filtered_qs = [
            q for q in questions
            if q.get("company",  "") in comp_filter
            and q.get("category", "") in cat_filter
        ]
        st.markdown(f"**{len(filtered_qs)}** question(s)")

        for q in filtered_qs:
            with st.expander(f"[{q.get('id','?')}] {q.get('company','')} · {q['question'][:80]}…"):
                st.markdown(f"**Company:** {q.get('company', '—')}")
                st.markdown(f"**Category:** {q.get('category', '—')}")
                st.markdown(f"**Difficulty:** {q.get('difficulty', '—')}")
                st.markdown(f"**Question:**\n\n{q['question']}")
                ref = q.get("reference_answer")
                if ref:
                    st.markdown(f"**Reference answer:**\n\n{ref}")
                else:
                    st.caption("No reference answer set.")
