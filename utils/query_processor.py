# utils/query_processor.py
import re
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import config


def detect_company(question: str) -> str | None:
    """
    Scans the question for a known company name.
    Returns the standardised company name (e.g. "NVIDIA") or None.

    This is the single most important fix — it prevents the retriever
    from pulling Amazon chunks when you ask about NVIDIA.
    """
    q_lower = question.lower()
    for keyword, company in config.KNOWN_COMPANIES.items():
        if keyword in q_lower:
            return company
    return None


def detect_year(question: str) -> str | None:
    """
    Extracts a 4-digit year from the question if present.
    Useful for filtering to the right fiscal year.
    """
    match = re.search(r'\b(20\d{2})\b', question)
    return match.group(1) if match else None


def preprocess_query(question: str) -> dict:
    """
    Analyses the question and returns structured metadata about it.
    Both retrievers use this to narrow their search before scoring.

    Example:
        "What was NVIDIA's revenue in 2024?"
        → {"company": "NVIDIA", "year": "2024", "clean_query": "..."}
    """
    company = detect_company(question)
    year    = detect_year(question)

    # Remove company name and year from query to reduce bias in BM25
    clean_query = question
    if company:
        clean_query = re.sub(company, '', clean_query, flags=re.IGNORECASE)
    if year:
        clean_query = re.sub(year, '', clean_query)
    clean_query = re.sub(r'\s+', ' ', clean_query).strip()

    return {
        "original"   : question,
        "clean_query": clean_query,
        "company"    : company,
        "year"       : year
    }