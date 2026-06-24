#utils/query_processor.py
import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
import config
import json

from config import DATA_PROCESSED_DIR


def get_known_companies():

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

        company = chunk.get(
            "company"
        )

        if company:
            companies.add(
                company.upper()
            )

    return sorted(companies)

def detect_company(question: str) -> str | None:

    q_lower = (question or "").lower()

    print("\n===== COMPANY MATCH DEBUG =====")
    print("QUESTION:", q_lower)

    for company in get_known_companies():

        company_variants = [

            company.lower(),

            company.lower().replace(
                "-",
                " "
            ),

            company.lower().replace(
                " ",
                ""
            )
        ]

        for keyword in company_variants:

            pattern = (
                rf"(?<!\w)"
                f"{re.escape(keyword)}"
                rf"(?!\w)"
            )

            if re.search(
                pattern,
                q_lower
            ):
                return company

    print("NO COMPANY MATCH")
    print("==============================\n")

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
    question = question or ""
    company = detect_company(question)
    year = detect_year(question)

    clean_query = question

    if company:
        company_pattern = rf"(?<!\w){re.escape(company)}(?!\w)"
        clean_query = re.sub(company_pattern, "", clean_query, flags=re.IGNORECASE)

    if year:
        year_pattern = rf"(?<!\d){re.escape(year)}(?!\d)"
        clean_query = re.sub(year_pattern, "", clean_query)

    clean_query = re.sub(r"\s+", " ", clean_query).strip(" ,;:-_()[]{}")

    return {
        "original": question,
        "clean_query": clean_query,
        "company": company,
        "year": year,
    }