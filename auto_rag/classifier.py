import re

NUMERICAL_KEYWORDS = [
    "revenue",
    "income",
    "profit",
    "loss",
    "cash",
    "assets",
    "liabilities",
    "debt",
    "margin",
    "earnings",
    "operating income",
    "net income",
    "percentage",
    "growth rate",
    "amount"
]

COMPARISON_KEYWORDS = [
    "compare",
    "difference",
    "versus",
    "vs",
    "relative to"
]

EXPLANATION_KEYWORDS = [
    "why",
    "how",
    "explain",
    "describe",
    "discuss",
    "strategy",
    "risk",
    "competition",
    "business model"
]


def classify_question(question: str) -> str:

    q = question.lower()

    if any(
        keyword in q
        for keyword in COMPARISON_KEYWORDS
    ):
        return "comparison"

    if any(
        keyword in q
        for keyword in NUMERICAL_KEYWORDS
    ):
        return "numerical"

    if any(
        keyword in q
        for keyword in EXPLANATION_KEYWORDS
    ):
        return "explanation"

    return "default"