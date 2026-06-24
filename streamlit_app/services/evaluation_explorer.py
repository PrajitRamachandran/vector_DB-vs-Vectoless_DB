import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

RESULTS_DIR = (
    ROOT_DIR /
    "evaluation" /
    "results"
)

def load_latest_results():

    csv_files = list(
        RESULTS_DIR.glob("*.csv")
    )

    if not csv_files:
        return None

    latest = max(
        csv_files,
        key=lambda x: x.stat().st_mtime
    )

    return pd.read_csv(latest)

def get_evaluation_summary():

    df = load_latest_results()

    if df is None:
        return "No benchmark results available."

    summary = []

    for method in df["method"].unique():

        subset = df[
            df["method"] == method
        ]

        score = (
            subset["judge_score"]
            .mean()
        )

        summary.append(
            f"{method}: {score:.2f}"
        )

    return "\n".join(summary)