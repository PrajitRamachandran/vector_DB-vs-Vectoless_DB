import json
import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import config
from config import DATA_PROCESSED_DIR


# ============================================================
# Load companies once
# ============================================================

_COMPANIES_CACHE = None


def get_known_companies():
    """
    Loads all companies from chunks.json.

    Cached after first call so we don't repeatedly
    read chunks.json for every query.
    """

    global _COMPANIES_CACHE

    if _COMPANIES_CACHE is not None:
        return _COMPANIES_CACHE

    chunks_path = (
        Path(DATA_PROCESSED_DIR)
        / "chunks.json"
    )

    with open(
        chunks_path,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    companies = set()

    for chunk in data["parents"]:

        company = chunk.get("company")

        if company:
            companies.add(
                company.upper().strip()
            )

    _COMPANIES_CACHE = sorted(companies)

    return _COMPANIES_CACHE


# ============================================================
# Company Detection
# ============================================================

def detect_company(question: str) -> str | None:
    """
    Detect company mentioned in user query.
    Handles:
    - Microsofts
    - Microsoft's
    - Amazons
    - Amazon's
    - NVIDIAs
    - NVIDIA's
    - Coca-Colas
    - Coca-Cola's
    """

    if not question:
        return None

    q_lower = question.lower()

    # Remove apostrophe possessives
    q_lower = q_lower.replace("'s", "")
    q_lower = q_lower.replace("’s", "")

    print("\n===== COMPANY MATCH DEBUG =====")
    print("QUESTION:", q_lower)

    companies = get_known_companies()

    for company in companies:

        company_lower = company.lower()

        variants = {
            company_lower,
            company_lower + "s",          # microsofts
            company_lower.replace("-", " "),
            company_lower.replace("-", "") ,
            company_lower.replace(" ", ""),
        }

        for variant in variants:

            pattern = (
                rf"(?<!\w)"
                f"{re.escape(variant)}"
                rf"(?!\w)"
            )

            if re.search(
                pattern,
                q_lower,
                flags=re.IGNORECASE
            ):
                print(f"MATCHED COMPANY: {company}")
                print("==============================")

                return company

    print("NO COMPANY MATCH")
    print("==============================")

    return None

# ============================================================
# Year Detection
# ============================================================

def detect_year(question: str) -> str |None:
    """
    Extracts first year from query.

    Example:
    2024
    2025
    2026
    """

    if not question:
        return None

    match = re.search(
        r"\b(20\d{2})\b",
        question
    )

    if match:
        return match.group(1)

    return None


# ============================================================
# Query Preprocessing
# ============================================================

def preprocess_query(question: str) -> dict:
    """
    Extract structured retrieval metadata.

    Returns:
    {
        original,
        clean_query,
        company,
        year
    }
    """

    question = (question or "").strip()

    company = detect_company(question)

    year = detect_year(question)

    clean_query = question

    # Remove company name from retrieval query
    if company:

        company_patterns = [

            company,

            company.replace("-", " "),

            company.replace(" ", ""),

            company.replace("-", "")
        ]

        for variant in company_patterns:

            clean_query = re.sub(
                rf"(?<!\w){re.escape(variant)}(?!\w)",
                "",
                clean_query,
                flags=re.IGNORECASE
            )

    print("RAW:", question)
    print("PROCESSED:", clean_query)

    return {
        "original": question,
        "clean_query": clean_query,
        "company": company,
        "year": year,
    }