# """
# Evaluation Service
# """

# import uuid

# from evaluation.evaluator import (
#     run_evaluation
# )

# from streamlit_app.database.repository import (
#     save_evaluation,
#     save_log
# )

# # --------------------------------------------------
# # JUDGE EVALUATION
# # --------------------------------------------------

# def run_judge_benchmark(
#     vector_pipeline,
#     vectorless_pipeline,
#     hybrid_pipeline
# ):
#     try:

#         from evaluation.ragas_evaluator import (
#             run_ragas_evaluation,
#             merge_results
#         )

#     except Exception as e:

#         raise RuntimeError(
#             f"Failed to load RAGAS: {e}"
#         )

#     try:

#         df, contexts = run_evaluation(
#             vector_pipeline=vector_pipeline,
#             vectorless_pipeline=vectorless_pipeline,
#             hybrid_pipeline=hybrid_pipeline,
#             capture_contexts=True
#         )

#         save_log(
#             "INFO",
#             "EVALUATION",
#             "Judge benchmark completed"
#         )

#         return {
#             "success": True,
#             "results": df,
#             "contexts": contexts
#         }

#     except Exception as e:

#         save_log(
#             "ERROR",
#             "EVALUATION",
#             str(e)
#         )

#         return {
#             "success": False,
#             "error": str(e)
#         }

# # --------------------------------------------------
# # RAGAS
# # --------------------------------------------------

# def run_ragas_benchmark(
#     judge_df,
#     contexts
# ):

#     import pandas as pd

#     try:

#         ragas_df = run_ragas_evaluation(
#             judge_df,
#             contexts
#         )

#         combined_df = merge_results(
#             judge_df,
#             ragas_df
#         )

#         # ==================================================
#         # SAVE BENCHMARK SUMMARY TO SQLITE
#         # ==================================================

#         for method in combined_df["method"].unique():

#             subset = combined_df[
#                 combined_df["method"] == method
#             ]

#             # -----------------------------
#             # Judge Metrics
#             # -----------------------------

#             avg_judge_score = None

#             for col in [
#                 "judge_score",
#                 "score",
#                 "overall_score"
#             ]:

#                 if col in subset.columns:

#                     avg_judge_score = (
#                         subset[col]
#                         .mean(skipna=True)
#                     )

#                     break

#             pass_rate = None

#             for col in [
#                 "pass",
#                 "passed",
#                 "is_correct"
#             ]:

#                 if col in subset.columns:

#                     pass_rate = (
#                         subset[col]
#                         .mean(skipna=True)
#                         * 100
#                     )

#                     break

#             company_accuracy = None

#             if (
#                 "company_accuracy"
#                 in subset.columns
#             ):

#                 company_accuracy = (
#                     subset[
#                         "company_accuracy"
#                     ].mean(
#                         skipna=True
#                     )
#                 )

#             # -----------------------------
#             # RAGAS Metrics
#             # -----------------------------

#             faithfulness = (
#                 subset["faithfulness"]
#                 .mean(skipna=True)
#                 if "faithfulness"
#                 in subset.columns
#                 else None
#             )

#             answer_relevancy = (
#                 subset["answer_relevancy"]
#                 .mean(skipna=True)
#                 if "answer_relevancy"
#                 in subset.columns
#                 else None
#             )

#             context_precision = (
#                 subset["context_precision"]
#                 .mean(skipna=True)
#                 if "context_precision"
#                 in subset.columns
#                 else None
#             )

#             context_recall = (
#                 subset["context_recall"]
#                 .mean(skipna=True)
#                 if "context_recall"
#                 in subset.columns
#                 else None
#             )

#             contextual_relevancy = (
#                 subset["contextual_relevancy"]
#                 .mean(skipna=True)
#                 if "contextual_relevancy"
#                 in subset.columns
#                 else None
#             )

#             # -----------------------------
#             # Overall Benchmark Score
#             # -----------------------------

#             overall_score = pd.Series(
#                 [
#                     faithfulness,
#                     answer_relevancy,
#                     context_precision,
#                     context_recall
#                 ]
#             ).mean(
#                 skipna=True
#             )

#             save_evaluation(

#                 evaluation_id=
#                     str(uuid.uuid4()),

#                 method=
#                     method,

#                 evaluator=
#                     "RAGAS + Judge",

#                 total_questions=
#                     len(subset),

#                 avg_judge_score=
#                     avg_judge_score,

#                 pass_rate=
#                     pass_rate,

#                 company_accuracy=
#                     company_accuracy,

#                 faithfulness=
#                     faithfulness,

#                 answer_relevancy=
#                     answer_relevancy,

#                 context_precision=
#                     context_precision,

#                 context_recall=
#                     context_recall,

#                 contextual_relevancy=
#                     contextual_relevancy,

#                 overall_score=
#                     overall_score
#             )

#         save_log(
#             "INFO",
#             "EVALUATION",
#             "RAGAS benchmark completed"
#         )

#         return {
#             "success": True,
#             "ragas": ragas_df,
#             "combined": combined_df
#         }

#     except Exception as e:

#         save_log(
#             "ERROR",
#             "RAGAS",
#             str(e)
#         )

#         return {
#             "success": False,
#             "error": str(e)
#         }


"""
Evaluation Service
"""

import uuid
from typing import Callable, Optional

import pandas as pd

from evaluation.evaluator import run_evaluation

from streamlit_app.database.repository import (
    save_evaluation,
    save_log,
)

ProgressCallback = Callable[[float, str], None]


def _emit(progress_callback: Optional[ProgressCallback], fraction: float, message: str) -> None:
    if progress_callback is None:
        return
    try:
        progress_callback(min(max(fraction, 0.0), 1.0), message)
    except Exception:
        pass  # progress reporting must never break a benchmark run


# --------------------------------------------------
# JUDGE EVALUATION
# --------------------------------------------------

def run_judge_benchmark(
    vector_pipeline,
    vectorless_pipeline,
    hybrid_pipeline,
    progress_callback: Optional[ProgressCallback] = None,
):
    """
    Runs the judge benchmark across all three pipelines.

    progress_callback(fraction: float in [0, 1], message: str) is invoked in
    real time as each question is scored.
    """
    try:
        _emit(progress_callback, 0.0, "Preparing judge benchmark…")

        df, contexts = run_evaluation(
            vector_pipeline=vector_pipeline,
            vectorless_pipeline=vectorless_pipeline,
            hybrid_pipeline=hybrid_pipeline,
            capture_contexts=True,
            progress_callback=lambda frac, msg: _emit(progress_callback, frac, msg),
        )

        save_log("INFO", "EVALUATION", "Judge benchmark completed")
        _emit(progress_callback, 1.0, "Judge benchmark completed")

        return {
            "success": True,
            "results": df,
            "contexts": contexts,
        }

    except Exception as e:
        save_log("ERROR", "EVALUATION", str(e))
        _emit(progress_callback, 1.0, f"Judge benchmark failed: {e}")
        return {
            "success": False,
            "error": str(e),
        }


# --------------------------------------------------
# RAGAS
# --------------------------------------------------

def run_ragas_benchmark(
    judge_df,
    contexts,
    progress_callback: Optional[ProgressCallback] = None,
):
    """
    Runs RAGAS on top of judge results, merges the two, and persists a
    per-method benchmark summary.

    progress_callback(fraction: float in [0, 1], message: str) tracks:
      0.00 – 0.90  RAGAS batch scoring (delegated to run_ragas_evaluation)
      0.90 – 1.00  Aggregating and saving per-method summaries
    """
    try:
        from evaluation.ragas_evaluator import run_ragas_evaluation, merge_results
    except Exception as e:
        error = f"Failed to load RAGAS: {e}"
        save_log("ERROR", "RAGAS", error)
        return {"success": False, "error": error}

    try:
        _emit(progress_callback, 0.0, "Preparing RAGAS evaluation…")

        ragas_df = run_ragas_evaluation(
            judge_df,
            contexts,
            progress_callback=lambda frac, msg: _emit(progress_callback, frac * 0.90, msg),
        )

        combined_df = merge_results(judge_df, ragas_df)

        methods = list(combined_df["method"].unique())
        n_methods = max(1, len(methods))

        for idx, method in enumerate(methods):
            subset = combined_df[combined_df["method"] == method]

            # -----------------------------
            # Judge metrics
            # -----------------------------
            avg_judge_score = None
            for col in ["judge_score", "score", "overall_score"]:
                if col in subset.columns:
                    avg_judge_score = subset[col].mean(skipna=True)
                    break

            pass_rate = None
            for col in ["pass", "passed", "is_correct"]:
                if col in subset.columns:
                    pass_rate = subset[col].mean(skipna=True) * 100
                    break

            company_accuracy = (
                subset["company_accuracy"].mean(skipna=True)
                if "company_accuracy" in subset.columns else None
            )

            # -----------------------------
            # RAGAS metrics
            # -----------------------------
            faithfulness = (
                subset["faithfulness"].mean(skipna=True)
                if "faithfulness" in subset.columns else None
            )
            answer_relevancy = (
                subset["answer_relevancy"].mean(skipna=True)
                if "answer_relevancy" in subset.columns else None
            )
            context_precision = (
                subset["context_precision"].mean(skipna=True)
                if "context_precision" in subset.columns else None
            )
            context_recall = (
                subset["context_recall"].mean(skipna=True)
                if "context_recall" in subset.columns else None
            )
            contextual_relevancy = (
                subset["contextual_relevancy"].mean(skipna=True)
                if "contextual_relevancy" in subset.columns else None
            )

            overall_score = pd.Series(
                [faithfulness, answer_relevancy, context_precision, context_recall]
            ).mean(skipna=True)

            save_evaluation(
                evaluation_id=str(uuid.uuid4()),
                method=method,
                evaluator="RAGAS + Judge",
                total_questions=len(subset),
                avg_judge_score=avg_judge_score,
                pass_rate=pass_rate,
                company_accuracy=company_accuracy,
                faithfulness=faithfulness,
                answer_relevancy=answer_relevancy,
                context_precision=context_precision,
                context_recall=context_recall,
                contextual_relevancy=contextual_relevancy,
                overall_score=overall_score,
            )

            _emit(
                progress_callback,
                0.90 + 0.10 * ((idx + 1) / n_methods),
                f"Saving results — {method} ({idx + 1}/{n_methods})",
            )

        save_log("INFO", "EVALUATION", "RAGAS benchmark completed")
        _emit(progress_callback, 1.0, "RAGAS benchmark completed")

        return {
            "success": True,
            "ragas": ragas_df,
            "combined": combined_df,
        }

    except Exception as e:
        save_log("ERROR", "RAGAS", str(e))
        _emit(progress_callback, 1.0, f"RAGAS benchmark failed: {e}")
        return {
            "success": False,
            "error": str(e),
        }