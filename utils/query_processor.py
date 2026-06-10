import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
import config


def _normalize_text(text: str) -> str:
    """Normalizes whitespace and apostrophes without changing meaning."""
    text = (text or "").replace("’", "'")
    return re.sub(r"\s+", " ", text).strip()


def detect_company(question: str) -> str | None:
    """
    Scans the question for a known company name.
    Returns the standardised company name (e.g. "NVIDIA") or None.
    """
    q_lower = (question or "").lower()

    # Check longer keywords first so longer names win.
    for keyword, company in sorted(
        config.KNOWN_COMPANIES.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        pattern = rf"(?<!\w){re.escape(keyword.lower())}(?!\w)"
        if re.search(pattern, q_lower):
            return company

    return None


def detect_year(question: str) -> str | None:
    """
    Extracts a 4-digit year from the question if present.
    """
    match = re.search(r"\b(20\d{2})\b", question or "")
    return match.group(1) if match else None


def preprocess_query(question: str) -> dict:
    """
    Returns structured metadata about the question.
    """
    question = _normalize_text(question)
    company = detect_company(question)
    year = detect_year(question)

    semantic_query = question
    clean_query = question

    if company:
        # Remove possessive company mentions cleanly so we do not leave behind
        # stray "'s" tokens that can hurt retrieval quality.
        company_pattern = rf"(?<!\w){re.escape(company)}(?:['’]s)?(?!\w)"
        clean_query = re.sub(company_pattern, "", clean_query, flags=re.IGNORECASE)

    if year:
        year_pattern = rf"(?<!\d){re.escape(year)}(?!\d)"
        clean_query = re.sub(year_pattern, "", clean_query)

    clean_query = re.sub(r"\s+", " ", clean_query).strip(" ,;:-_()[]{}'\"?!.,")

    return {
        "original": question,
        "clean_query": clean_query,
        "semantic_query": semantic_query,
        "company": company,
        "year": year,
    }
