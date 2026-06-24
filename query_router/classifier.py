# query_router/classifier.py

import re

from query_router.intents import (
    CHAT,
    GENERAL_KNOWLEDGE,
    DOCUMENT_QUESTION,
    DOCUMENT_METADATA,
    DOCUMENT_EXPLORATION,
    EVALUATION_EXPLORATION
)

# ==========================================================
# CHAT
# ==========================================================

CHAT_PATTERNS = [
    r"^(hi|hello|hey)$",
    r"^(good morning)$",
    r"^(good afternoon)$",
    r"^(good evening)$",
    r"^(how are you)$",
    r"^(thank you)$",
    r"^(thanks)$",
    r"^(bye)$",
    r"^(goodbye)$"
]

# ==========================================================
# DOCUMENT METADATA
# ==========================================================

DOCUMENT_METADATA_KEYWORDS = [
    "how many documents",
    "how many reports",
    "what documents",
    "which documents",
    "what companies",
    "which companies",
    "uploaded documents",
    "uploaded reports",
    "available reports",
    "available companies",
    "what data do you have",
    "what reports do you have",
    "which reports do you have",
    "which reports are available",
    "what reports can i query",
    "list reports",
    "list companies",
    "show reports",
    "show companies",
    "available documents",
    "loaded documents",
    "loaded reports"
]

# ==========================================================
# DOCUMENT EXPLORATION
# ==========================================================

DOCUMENT_EXPLORATION_KEYWORDS = [
    "random fact",
    "interesting fact",
    "fun fact",
    "overview",
    "summary",
    "summarize",
    "summarise",
    "highlights",
    "key highlights",
    "key findings",
    "tell me about",
    "give me an overview",
    "what is this company about"
]

# ==========================================================
# KNOWN COMPANIES
# ==========================================================

KNOWN_COMPANIES = [
    "amazon",
    "microsoft",
    "netflix",
    "nvidia",
    "asus",
    "reliance",
    "coca cola",
    "cocacola"
]

# ==========================================================
# FINANCIAL TERMS
# ==========================================================

FINANCIAL_TERMS = [
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
    "operating margin",
    "gross margin",
    "segment",
    "business segment",
    "cash flow",
    "guidance",
    "strategy",
    "risk",
    "competition",
    "growth",
    "expenses",
    "cost of revenue"
]


# ==========================================================
# EVALUATION EXPLORATION
# ==========================================================

EVALUATION_KEYWORDS = [
    "benchmark",
    "evaluation",
    "judge score",
    "ragas",
    "faithfulness",
    "answer relevancy",
    "context recall",
    "context precision",
    "best method",
    "compare methods",
    "retrieval performance",
    "which retrieval method",
    "performed best",
    "benchmark results",
    "compare vector",
    "compare hybrid",
    "vector vs hybrid",
    "hybrid vs vector",
    "vector vs vectorless",
    "compare retrieval methods"
]


# ==========================================================
# GENERAL KNOWLEDGE
# ==========================================================

GENERAL_KNOWLEDGE_PATTERNS = [
    "what is",
    "define",
    "meaning of",
    "explain the concept of",
    "how does",
    "difference between"
]

# ==========================================================
# CLASSIFIER
# ==========================================================

def classify(question: str):

    q = question.lower().strip()

    # ------------------------------------
    # CHAT
    # ------------------------------------

    for pattern in CHAT_PATTERNS:

        if re.fullmatch(
            pattern,
            q
        ):
            return CHAT

    # ------------------------------------
    # DOCUMENT METADATA
    # ------------------------------------

    if any(
        keyword in q
        for keyword in DOCUMENT_METADATA_KEYWORDS
    ):
        return DOCUMENT_METADATA

    # ------------------------------------
    # DOCUMENT EXPLORATION
    # ------------------------------------

    if any(
        keyword in q
        for keyword in DOCUMENT_EXPLORATION_KEYWORDS
    ):
        return DOCUMENT_EXPLORATION
    
        # ------------------------------------
    # EVALUATION EXPLORATION
    # ------------------------------------

    if any(
        keyword in q
        for keyword in EVALUATION_KEYWORDS
    ):
        return EVALUATION_EXPLORATION

    # ------------------------------------
    # DOCUMENT QUESTION
    # ------------------------------------

    has_company = any(
        company in q
        for company in KNOWN_COMPANIES
    )

    has_financial_term = any(
        term in q
        for term in FINANCIAL_TERMS
    )

    has_year = bool(
        re.search(
            r"\b20\d{2}\b",
            q
        )
    )

    if (
        has_company
        or has_financial_term
        or has_year
    ):
        return DOCUMENT_QUESTION

    # ------------------------------------
    # GENERAL KNOWLEDGE
    # ------------------------------------

    if any(
        pattern in q
        for pattern in GENERAL_KNOWLEDGE_PATTERNS
    ):
        return GENERAL_KNOWLEDGE

    # ------------------------------------
    # FALLBACK
    # ------------------------------------

    return GENERAL_KNOWLEDGE