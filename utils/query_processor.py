import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
import config


def detect_company(question: str) -> str | None:
    """
    Scans the question for a known company name.
    Returns the standardised company name (e.g. "NVIDIA") or None.

    Matching is done with word-boundary style regex checks so short keywords
    do not accidentally match inside other words.
    """
    q_lower = question.lower()

    # Check longer keywords first so "bank of america" wins over "america", etc.
    for keyword, company in sorted(
        config.KNOWN_COMPANIES.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        pattern = rf"(?<!\\w){re.escape(keyword.lower())}(?!\\w)"
        if re.search(pattern, q_lower):
            return company
    return None


def detect_year(question: str) -> str | None:
    """
    Extracts a 4-digit year from the question if present.
    Useful for filtering to the right fiscal year.
    """
    match = re.search(r"\b(20\d{2})\b", question)
    return match.group(1) if match else None


def preprocess_query(question: str) -> dict:
    """
    Analyses the question and returns structured metadata about it.
    Both retrievers use this to narrow their search before scoring.

    Example:
        "What was NVIDIA's revenue in 2024?"
        -> {"company": "NVIDIA", "year": "2024", "clean_query": "..."}
    """
    company = detect_company(question)
    year = detect_year(question)

    clean_query = question

    if company:
        company_pattern = rf"(?<!\\w){re.escape(company)}(?!\\w)"
        clean_query = re.sub(company_pattern, "", clean_query, flags=re.IGNORECASE)

    if year:
        clean_query = re.sub(rf"(?<!\\d){re.escape(year)}(?!\\d)", "", clean_query)

    # Remove leftover punctuation from deletion and collapse extra spaces.
    clean_query = re.sub(r"\s+", " ", clean_query).strip(" ,;:-_()[]{}")

    return {
        "original": question,
        "clean_query": clean_query,
        "company": company,
        "year": year,
    }
