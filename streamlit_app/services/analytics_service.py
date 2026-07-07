"""
Analytics Service

Lightweight, dependency-free helpers that turn a raw pipeline
result into the extra signals the chat UI surfaces: token/cost
estimates, confidence, hallucination risk, and auto-generated
conversation titles.

These are heuristics, not ground truth. Token counts are word-
based approximations (no tokenizer dependency), and hallucination
risk is a simple rule derived from retrieval scores + answer
length. They are meant to give users a directional signal, and
are labeled as such in the UI.
"""

import re

import config

# ============================================================
# TOKEN / COST ESTIMATION
# ============================================================

_WORDS_PER_TOKEN = 0.75  # ~1 token per 0.75 words (rough English average)


def estimate_tokens(text: str) -> int:

    if not text:
        return 0

    words = len(text.split())

    return max(1, round(words / _WORDS_PER_TOKEN))


def estimate_cost(
    prompt_text: str,
    completion_text: str,
    model_name: str
) -> dict:

    tokens_prompt = estimate_tokens(prompt_text)
    tokens_completion = estimate_tokens(completion_text)

    pricing = config.LLM_PRICING_PER_1K_TOKENS.get(
        model_name,
        # Fall back to the medium-tier price if the model
        # isn't in the pricing table.
        next(iter(config.LLM_PRICING_PER_1K_TOKENS.values()))
    )

    cost = (
        (tokens_prompt / 1000) * pricing["prompt"]
        + (tokens_completion / 1000) * pricing["completion"]
    )

    return {
        "tokens_prompt": tokens_prompt,
        "tokens_completion": tokens_completion,
        "tokens_total": tokens_prompt + tokens_completion,
        "estimated_cost_usd": round(cost, 6),
    }


# ============================================================
# QUERY CLASSIFICATION (fallback, if the pipeline doesn't
# already return a query_type)
# ============================================================

_NUMERICAL_PATTERNS = re.compile(
    r"\b(revenue|profit|margin|percent|%|growth|eps|ebitda|income|"
    r"how much|how many|total|net income)\b",
    re.IGNORECASE
)

_COMPARATIVE_PATTERNS = re.compile(
    r"\b(compare|versus|vs\.?|difference between|which is (higher|lower|better)|"
    r"more than|less than)\b",
    re.IGNORECASE
)

_FACTUAL_PATTERNS = re.compile(
    r"\b(who|when|where|what is|what are|list|name the)\b",
    re.IGNORECASE
)


def classify_query(question: str) -> str:

    if not question:
        return "unknown"

    if _COMPARATIVE_PATTERNS.search(question):
        return "comparative"

    if _NUMERICAL_PATTERNS.search(question):
        return "numerical"

    if _FACTUAL_PATTERNS.search(question):
        return "factual"

    return "semantic"


# ============================================================
# CONFIDENCE / HALLUCINATION RISK
# ============================================================

def estimate_confidence(retrieved_chunks: list) -> float:
    """
    Confidence is approximated from retrieval/rerank scores:
    the better and more consistent the top chunks score, the
    higher the confidence. Returns a 0-1 float.
    """

    if not retrieved_chunks:
        return 0.0

    scores = []

    for chunk in retrieved_chunks:
        score = (
            chunk.get("rerank_score")
            if chunk.get("rerank_score") is not None
            else chunk.get("score")
        )
        if isinstance(score, (int, float)):
            scores.append(float(score))

    if not scores:
        return 0.0

    top_scores = sorted(scores, reverse=True)[:3]
    avg_top = sum(top_scores) / len(top_scores)

    # Normalize defensively in case scores aren't already 0-1
    # (e.g. raw cosine distances or BM25 scores).
    normalized = max(0.0, min(1.0, avg_top))

    return round(normalized, 3)


def confidence_level(confidence: float) -> str:

    if confidence >= 0.7:
        return "high"
    if confidence >= 0.4:
        return "medium"
    return "low"


def estimate_hallucination_risk(
    answer: str,
    retrieved_chunks: list,
    confidence: float
) -> str:
    """
    A coarse heuristic: no retrieved context + a confident-
    sounding answer is the highest-risk combination; strong
    retrieval backing is the lowest-risk one.
    """

    if not answer:
        return "unknown"

    has_context = bool(retrieved_chunks)

    if not has_context:
        return "high"

    if confidence >= 0.7:
        return "low"

    if confidence >= 0.4:
        return "medium"

    return "high"


# ============================================================
# AUTO TITLES
# ============================================================

def generate_title(question: str, max_len: int = 60) -> str:

    if not question:
        return "New conversation"

    cleaned = " ".join(question.strip().split())

    if len(cleaned) <= max_len:
        return cleaned

    return cleaned[: max_len - 1].rstrip() + "…"