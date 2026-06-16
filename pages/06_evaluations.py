"""
Evaluation Dashboard
"""

import streamlit as st
import pandas as pd
import plotly.express as px

from streamlit_app.services.evaluation_service import (
    run_judge_benchmark,
    run_ragas_benchmark
)

from streamlit_app.database.repository import (
    get_evaluations,
    clear_evaluations,
    get_best_method
)

from streamlit_app.services.rag_service import (
    get_vector_pipeline,
    get_vectorless_pipeline,
    get_hybrid_pipeline
)

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Evaluation",
    page_icon="📊",
    layout="wide"
)

st.title(
    "📊 Evaluation Dashboard"
)

# ============================================================
# SESSION STATE
# ============================================================

if "judge_results" not in st.session_state:
    st.session_state.judge_results = None

if "judge_contexts" not in st.session_state:
    st.session_state.judge_contexts = None

if "ragas_results" not in st.session_state:
    st.session_state.ragas_results = None

# ============================================================
# ACTIONS
# ============================================================

col1, col2, col3 = st.columns(3)

# ------------------------------------------------------------
# JUDGE
# ------------------------------------------------------------

with col1:

    if st.button(
        "Run Judge Benchmark",
        use_container_width=True
    ):

        with st.spinner(
            "Running benchmark..."
        ):

            result = run_judge_benchmark(
                vector_pipeline=
                    get_vector_pipeline(),

                vectorless_pipeline=
                    get_vectorless_pipeline(),

                hybrid_pipeline=
                    get_hybrid_pipeline()
            )

            if result["success"]:

                st.session_state.judge_results = (
                    result["results"]
                )

                st.session_state.judge_contexts = (
                    result["contexts"]
                )

                st.success(
                    "Judge benchmark completed."
                )

            else:

                st.error(
                    result["error"]
                )

# ------------------------------------------------------------
# RAGAS
# ------------------------------------------------------------

with col2:

    if st.button(
        "Run RAGAS",
        use_container_width=True
    ):

        if (
            st.session_state.judge_results
            is None
        ):

            st.warning(
                "Run Judge Benchmark first."
            )

        else:

            with st.spinner(
                "Running RAGAS..."
            ):

                result = run_ragas_benchmark(
                    st.session_state.judge_results,
                    st.session_state.judge_contexts
                )

                if result["success"]:

                    st.session_state.ragas_results = (
                        result["combined"]
                    )

                    st.success(
                        "RAGAS completed."
                    )

                else:

                    st.error(
                        result["error"]
                    )

# ------------------------------------------------------------
# CLEAR
# ------------------------------------------------------------

with col3:

    if st.button(
        "Clear Evaluation History",
        use_container_width=True
    ):

        clear_evaluations()

        st.success(
            "Evaluation history cleared."
        )

        st.rerun()

# ============================================================
# LEADERBOARD + ANALYTICS
# ============================================================

st.divider()

evaluations = get_evaluations()

if not evaluations:

    st.info(
        """
        No benchmark results available yet.

        Steps:

        1. Run Judge Benchmark
        2. Run RAGAS Evaluation
        3. Results will appear automatically
        """
    )

else:

    # ========================================================
    # DATAFRAME
    # ========================================================

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

    chart_df = leaderboard_df.copy()

    # ========================================================
    # LEADERBOARD
    # ========================================================

    st.subheader(
        "🏆 Benchmark Leaderboard"
    )

    leaderboard_columns = [

        "method",

        "overall_score",

        "avg_judge_score",

        "faithfulness",

        "answer_relevancy",

        "context_precision",

        "context_recall",

        "company_accuracy"
    ]

    available_cols = [

        col

        for col in leaderboard_columns

        if col in leaderboard_df.columns
    ]

    st.dataframe(
        leaderboard_df[
            available_cols
        ],
        use_container_width=True
    )

    # ========================================================
    # BEST METHOD
    # ========================================================

    best = leaderboard_df.iloc[0]

    st.success(
        f"🥇 Best Method: "
        f"{best['method']} "
        f"(Score: {best['overall_score']:.3f})"
    )

    # ========================================================
    # PERFORMANCE PROFILE
    # ========================================================

    st.divider()

    st.subheader(
        "📈 Method Performance Profile"
    )

    metrics = [

        "avg_judge_score",

        "faithfulness",

        "answer_relevancy",

        "context_precision",

        "context_recall"
    ]

    available_metrics = [

        metric

        for metric in metrics

        if metric in chart_df.columns
    ]

    plot_df = chart_df[
        ["method"] + available_metrics
    ]

    long_df = plot_df.melt(
        id_vars="method",
        var_name="metric",
        value_name="score"
    )

    fig = px.line(
        long_df,
        x="metric",
        y="score",
        color="method",
        markers=True,
        title="Vector vs Vectorless vs Hybrid"
    )

    fig.update_layout(
        hovermode="x unified",
        xaxis_title="Metric",
        yaxis_title="Score"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ========================================================
    # HISTORICAL TREND
    # ========================================================

    st.divider()

    st.subheader(
        "📅 Historical Benchmark Trend"
    )

    if len(chart_df) > 1:

        history_df = chart_df.copy()

        history_df["timestamp"] = pd.to_datetime(
            history_df["timestamp"]
        )

        fig = px.line(
            history_df,
            x="timestamp",
            y="overall_score",
            color="method",
            markers=True,
            title="Benchmark Performance Over Time"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:

        st.info(
            "Run benchmarks multiple times to see trends."
        )

    # ========================================================
    # METRIC TABLE
    # ========================================================

    st.divider()

    st.subheader(
        "📊 Metric Breakdown"
    )

    metric_cols = [

        "method",

        "faithfulness",

        "answer_relevancy",

        "context_precision",

        "context_recall",

        "overall_score"
    ]

    available = [

        c

        for c in metric_cols

        if c in chart_df.columns
    ]

    st.dataframe(
        chart_df[
            available
        ],
        use_container_width=True
    )

    # ========================================================
    # SCORE MATRIX
    # ========================================================

    st.divider()

    st.subheader(
        "🔥 Benchmark Score Matrix"
    )

    matrix_columns = [

        "method",

        "avg_judge_score",

        "faithfulness",

        "answer_relevancy",

        "context_precision",

        "context_recall",

        "overall_score"
    ]

    available_cols = [

        col

        for col in matrix_columns

        if col in chart_df.columns
    ]

    matrix_df = chart_df[
        available_cols
    ].copy()

    st.dataframe(
        matrix_df,
        use_container_width=True
    )

# ============================================================
# DOWNLOAD
# ============================================================

if evaluations:

    st.divider()

    csv = pd.DataFrame(
        evaluations
    ).to_csv(
        index=False
    )

    st.download_button(

        "Download Evaluation Results",

        csv,

        "evaluation_results.csv",

        "text/csv"
    )