# import json
# import re
# import sys
# from pathlib import Path

# sys.path.append(str(Path(__file__).parent.parent))

# import config
# from config import DATA_PROCESSED_DIR


# # ============================================================
# # Load companies once
# # ============================================================

# _COMPANIES_CACHE = None


# def get_known_companies():
#     """
#     Loads all companies from chunks.json.

#     Cached after first call so we don't repeatedly
#     read chunks.json for every query.
#     """

#     global _COMPANIES_CACHE

#     if _COMPANIES_CACHE is not None:
#         return _COMPANIES_CACHE

#     chunks_path = (
#         Path(DATA_PROCESSED_DIR)
#         / "chunks.json"
#     )

#     with open(
#         chunks_path,
#         "r",
#         encoding="utf-8"
#     ) as f:

#         data = json.load(f)

#     companies = set()

#     for chunk in data["parents"]:

#         company = chunk.get("company")

#         if company:
#             companies.add(
#                 company.upper().strip()
#             )

#     _COMPANIES_CACHE = sorted(companies)

#     return _COMPANIES_CACHE


# # ============================================================
# # Company Detection
# # ============================================================

# def detect_company(question: str) -> str | None:
#     """
#     Detect company mentioned in user query.
#     Handles:
#     - Microsofts
#     - Microsoft's
#     - Amazons
#     - Amazon's
#     - NVIDIAs
#     - NVIDIA's
#     - Coca-Colas
#     - Coca-Cola's
#     """

#     if not question:
#         return None

#     q_lower = question.lower()

#     # Remove apostrophe possessives
#     q_lower = q_lower.replace("'s", "")
#     q_lower = q_lower.replace("’s", "")

#     print("\n===== COMPANY MATCH DEBUG =====")
#     print("QUESTION:", q_lower)

#     companies = get_known_companies()

#     for company in companies:

#         company_lower = company.lower()

#         variants = {
#             company_lower,
#             company_lower + "s",          # microsofts
#             company_lower.replace("-", " "),
#             company_lower.replace("-", "") ,
#             company_lower.replace(" ", ""),
#         }

#         for variant in variants:

#             pattern = (
#                 rf"(?<!\w)"
#                 f"{re.escape(variant)}"
#                 rf"(?!\w)"
#             )

#             if re.search(
#                 pattern,
#                 q_lower,
#                 flags=re.IGNORECASE
#             ):
#                 print(f"MATCHED COMPANY: {company}")
#                 print(f"QUERY: {question} -> {company}")
#                 return company

#     print(f"QUERY: {question} -> None")
#     print("NO COMPANY MATCH")
#     print("==============================")

#     return None

# # ============================================================
# # Year Detection
# # ============================================================

# def detect_year(question: str) -> str |None:
#     """
#     Extracts first year from query.

#     Example:
#     2024
#     2025
#     2026
#     """

#     if not question:
#         return None

#     match = re.search(
#         r"\b(20\d{2})\b",
#         question
#     )

#     if match:
#         return match.group(1)

#     return None


# # ============================================================
# # Query Preprocessing
# # ============================================================

# def preprocess_query(question: str) -> dict:
#     """
#     Extract structured retrieval metadata.

#     Returns:
#     {
#         original,
#         clean_query,
#         company,
#         year
#     }
#     """

#     question = (question or "").strip()

#     company = detect_company(question)

#     year = detect_year(question)

#     clean_query = question

#     # Remove company name from retrieval query
#     if company:

#         company_patterns = [

#             company,

#             company.replace("-", " "),

#             company.replace(" ", ""),

#             company.replace("-", "")
#         ]

#         for variant in company_patterns:

#             clean_query = re.sub(
#                 rf"(?<!\w){re.escape(variant)}(?!\w)",
#                 "",
#                 clean_query,
#                 flags=re.IGNORECASE
#             )

#     print("RAW:", question)
#     print("PROCESSED:", clean_query)

#     return {
#         "original": question,
#         "clean_query": clean_query,
#         "company": company,
#         "year": year,
#     }

# # utils/query_processor.py

# import json
# import logging
# import re
# import sys
# from dataclasses import dataclass
# from pathlib import Path

# sys.path.append(str(Path(__file__).parent.parent))

# import config
# from config import DATA_PROCESSED_DIR

# logger = logging.getLogger(__name__)


# # ============================================================
# # Constants
# # ============================================================

# # Apostrophe-like characters to recognize as possessive markers (fix #6).
# _APOSTROPHE_CHARS = "'’‘`"

# # Matches a 20xx year.
# _YEAR_RE = re.compile(r"\b(20\d{2})\b")

# # Manually curated aliases for companies whose canonical stored form (as
# # indexed in chunk metadata, e.g. "COCACOLA") loses word-boundary
# # information that can't be recovered programmatically — there's no way to
# # infer "Coca-Cola" or "Coca Cola" from the smashed-together string alone.
# # Add an entry here whenever a new multi-word/hyphenated company is added
# # to the dataset and simple hyphen/space stripping won't cover it (fix #2).
# COMPANY_ALIASES: dict[str, list[str]] = {
#     "COCACOLA": ["coca-cola", "coca cola", "cocacola"],
# }


# # ============================================================
# # Known companies (loaded from chunks.json, cached)
# # ============================================================

# _COMPANIES_CACHE: list[str] | None = None


# def get_known_companies(force_reload: bool = False) -> list[str]:
#     """
#     Loads all known company names (as stored in chunk metadata) from
#     chunks.json, merged with any companies declared in config.KNOWN_COMPANIES
#     (so config and the indexed data can't silently drift apart — fix #2/#5).

#     Cached after first call. Pass force_reload=True to re-read chunks.json
#     (e.g. after re-ingesting new documents) without restarting the process.
#     """
#     global _COMPANIES_CACHE

#     if _COMPANIES_CACHE is not None and not force_reload:
#         return _COMPANIES_CACHE

#     chunks_path = Path(DATA_PROCESSED_DIR) / "chunks.json"

#     try:
#         with open(chunks_path, "r", encoding="utf-8") as f:
#             data = json.load(f)
#     except FileNotFoundError as exc:
#         raise FileNotFoundError(
#             f"Could not load known companies: {chunks_path} does not exist. "
#             f"Run the ingestion/indexing pipeline first."
#         ) from exc
#     except json.JSONDecodeError as exc:
#         raise ValueError(f"Could not parse {chunks_path} as JSON: {exc}") from exc

#     parents = data.get("parents")
#     if parents is None:
#         raise KeyError(
#             f"{chunks_path} is missing the expected 'parents' key — "
#             f"has the chunks.json schema changed?"
#         )

#     companies = set()
#     for chunk in parents:
#         company = chunk.get("company")
#         if company:
#             companies.add(company.upper().strip())

#     # Fallback: also include anything declared in config, so a company that's
#     # configured but not yet fully indexed is still detectable.
#     companies.update(v.upper().strip() for v in config.KNOWN_COMPANIES.values())

#     _COMPANIES_CACHE = sorted(companies)
#     logger.debug("Loaded %d known companies", len(_COMPANIES_CACHE))
#     return _COMPANIES_CACHE


# def _company_variants(company: str) -> set[str]:
#     """
#     Builds the set of lowercase text variants that should match a given
#     canonical company name in free-text user queries, plus any manually
#     curated aliases (COMPANY_ALIASES) for names that lose word-boundary
#     information once stored in smashed-together form.
#     """
#     company_lower = company.lower()
#     variants = {
#         company_lower,
#         company_lower + "s",              # informal plural / typo, e.g. "microsofts"
#         company_lower.replace("-", " "),
#         company_lower.replace("-", ""),
#         company_lower.replace(" ", ""),
#     }
#     variants.update(alias.lower() for alias in COMPANY_ALIASES.get(company, []))
#     return {v for v in variants if v}  # drop empty strings defensively


# def _variant_pattern(variant: str) -> re.Pattern:
#     """
#     Compiles a pattern that matches a company variant as a whole word,
#     optionally followed immediately by a possessive marker ('s, 's, or a
#     bare trailing apostrophe) — so "Amazon's" is matched and later removed
#     as a single unit instead of leaving a dangling "'s" behind (fix #1).
#     """
#     pattern = (
#         rf"(?<!\w){re.escape(variant)}"
#         rf"(?:[{_APOSTROPHE_CHARS}]s?)?"
#         rf"(?!\w)"
#     )
#     return re.compile(pattern, flags=re.IGNORECASE)


# # ============================================================
# # Company detection
# # ============================================================

# @dataclass(frozen=True)
# class _CompanyMatch:
#     company: str  # canonical company name, e.g. "AMAZON"
#     start:   int  # start offset in the original question text
#     end:     int  # end offset in the original question text (exclusive)


# def _find_company_matches(question: str) -> list[_CompanyMatch]:
#     """
#     Finds every known company mentioned in `question`, in the order they
#     appear. Longer variants are tried first, and overlapping matches are
#     resolved by keeping the longest/earliest span (fix #9) — the exact
#     spans found here are what preprocess_query() later deletes to build
#     clean_query, so detection and removal can never disagree (fix #10).
#     """
#     if not question:
#         return []

#     candidates: list[_CompanyMatch] = []
#     for company in get_known_companies():
#         for variant in sorted(_company_variants(company), key=len, reverse=True):
#             for m in _variant_pattern(variant).finditer(question):
#                 candidates.append(_CompanyMatch(company, m.start(), m.end()))

#     # Order by position of appearance; break ties by preferring the longer
#     # (more specific) match at the same start offset.
#     candidates.sort(key=lambda c: (c.start, -(c.end - c.start)))

#     matches: list[_CompanyMatch] = []
#     last_end = -1
#     seen_companies: set[str] = set()
#     for c in candidates:
#         if c.start < last_end:
#             continue  # overlaps an already-kept, higher-priority match
#         if c.company in seen_companies:
#             continue  # don't report the same company twice
#         matches.append(c)
#         seen_companies.add(c.company)
#         last_end = c.end

#     return matches


# def detect_companies(question: str) -> list[str]:
#     """
#     Detects every known company mentioned in the query, in the order they
#     appear (fix #3). Returns an empty list if none are found.
#     """
#     matches = _find_company_matches(question or "")
#     companies = [m.company for m in matches]
#     logger.debug("Company detection | question=%r -> %r", question, companies)
#     return companies


# def detect_company(question: str) -> str | None:
#     """
#     Backward-compatible single-company detector: returns the first company
#     mentioned (by position in the text), or None. Prefer detect_companies()
#     for queries that may mention more than one company (e.g. comparisons).
#     """
#     companies = detect_companies(question)
#     return companies[0] if companies else None


# # ============================================================
# # Year detection
# # ============================================================

# def detect_years(question: str) -> list[str]:
#     """Extracts every 20xx year mentioned in the query, in order of appearance."""
#     if not question:
#         return []
#     return _YEAR_RE.findall(question)


# def detect_year(question: str) -> str | None:
#     """Backward-compatible single-year detector: returns the first year found."""
#     years = detect_years(question)
#     return years[0] if years else None


# # ============================================================
# # Query preprocessing
# # ============================================================

# def _remove_spans(text: str, spans: list[tuple[int, int]]) -> str:
#     """Deletes the given (start, end) character spans from text."""
#     chars = list(text)
#     for start, end in sorted(spans, reverse=True):  # delete back-to-front
#         del chars[start:end]
#     return "".join(chars)


# def preprocess_query(question: str) -> dict:
#     """
#     Extract structured retrieval metadata from a raw user question.

#     Returns:
#     {
#         "original":    the raw, unmodified question,
#         "clean_query": question with every detected company mention (and its
#                        possessive, if any) removed and whitespace
#                        normalized — safe to embed for dense retrieval
#                        without company-name bias,
#         "company":     first detected company (str) or None — kept for
#                        backward compatibility with single-company callers,
#         "companies":   ALL detected companies, in order of appearance,
#         "year":        first detected year (str) or None — kept for
#                        backward compatibility,
#         "years":       ALL detected years, in order of appearance,
#     }
#     """
#     question = (question or "").strip()

#     matches   = _find_company_matches(question)
#     companies = [m.company for m in matches]
#     years     = detect_years(question)

#     if matches:
#         clean_query = _remove_spans(question, [(m.start, m.end) for m in matches])
#         clean_query = re.sub(r"\s+", " ", clean_query).strip()  # fix #4
#     else:
#         clean_query = question

#     logger.debug(
#         "preprocess_query | original=%r -> clean_query=%r | companies=%r | years=%r",
#         question, clean_query, companies, years,
#     )

#     return {
#         "original":    question,
#         "clean_query": clean_query,
#         "company":     companies[0] if companies else None,
#         "companies":   companies,
#         "year":        years[0] if years else None,
#         "years":       years,
#     }

















"""
query_processor.py

Extracts structured retrieval metadata (company mentions, years, and a
company/year-stripped "clean" query) from a raw user question, for use by
the retrieval pipeline (metadata filtering + bias-free embedding).

Known companies are sourced from chunks.json (as indexed by the ingestion
pipeline in company_detector.py) merged with any companies declared in
config.KNOWN_COMPANIES, so the two can never silently drift apart.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.append(str(_PROJECT_ROOT))

try:
    import config
except ImportError as exc:  # pragma: no cover - environment/config issue
    raise ImportError(
        f"Could not import 'config' from {_PROJECT_ROOT}. Make sure "
        f"config.py exists at the project root and defines at least "
        f"DATA_PROCESSED_DIR."
    ) from exc

logger = logging.getLogger(__name__)


# ============================================================
# Constants
# ============================================================

# Apostrophe-like characters to recognize as possessive markers.
_APOSTROPHE_CHARS = "'’‘`"

# Matches a 20xx year.
_YEAR_RE = re.compile(r"\b(20\d{2})\b")

# Manually curated aliases for companies whose canonical stored form (as
# indexed in chunk metadata, e.g. "COCACOLA") loses word-boundary
# information that can't be recovered programmatically — there's no way to
# infer "Coca-Cola" or "Coca Cola" from the smashed-together string alone.
# Add an entry here whenever a new multi-word/hyphenated company is added
# to the dataset and simple hyphen/space stripping won't cover it.
#
# Keys are matched case-insensitively against the canonical company name,
# so it doesn't matter whether chunks.json happens to store "COCACOLA" or
# "CocaCola" — lookups are normalized before comparison.
COMPANY_ALIASES: dict[str, list[str]] = {
    "COCACOLA": ["coca-cola", "coca cola", "cocacola"],
}


# ============================================================
# Known companies (loaded from chunks.json, cached)
# ============================================================

_COMPANIES_CACHE: list[str] | None = None


def get_known_companies(force_reload: bool = False) -> list[str]:
    """
    Loads all known company names (as stored in chunk metadata) from
    chunks.json, merged with any companies declared in
    config.KNOWN_COMPANIES, so the config and the indexed data can't
    silently drift apart.

    Cached after first call. Pass force_reload=True to re-read chunks.json
    (e.g. after re-ingesting new documents) without restarting the process.
    """
    global _COMPANIES_CACHE

    if _COMPANIES_CACHE is not None and not force_reload:
        return _COMPANIES_CACHE

    chunks_path = Path(config.DATA_PROCESSED_DIR) / "chunks.json"

    try:
        with open(chunks_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Could not load known companies: {chunks_path} does not exist. "
            f"Run the ingestion/indexing pipeline first."
        ) from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Could not parse {chunks_path} as JSON: {exc}") from exc

    parents = data.get("parents")
    if parents is None:
        raise KeyError(
            f"{chunks_path} is missing the expected 'parents' key — "
            f"has the chunks.json schema changed?"
        )

    companies = set()
    for chunk in parents:
        company = chunk.get("company")
        if company:
            companies.add(company.upper().strip())

    # Merge in anything declared in config, so a company that's configured
    # but not yet fully indexed is still detectable. Guarded with getattr
    # so a missing/renamed config attribute degrades gracefully instead of
    # crashing company detection entirely.
    known_companies_cfg = getattr(config, "KNOWN_COMPANIES", {}) or {}
    companies.update(str(v).upper().strip() for v in known_companies_cfg.values())

    _COMPANIES_CACHE = sorted(companies)
    logger.debug("Loaded %d known companies", len(_COMPANIES_CACHE))
    return _COMPANIES_CACHE


def _aliases_for(company: str) -> list[str]:
    """
    Case-insensitive lookup into COMPANY_ALIASES, so the alias table isn't
    silently ignored if the canonical stored form's casing ever changes.
    """
    company_upper = company.upper()
    for key, aliases in COMPANY_ALIASES.items():
        if key.upper() == company_upper:
            return aliases
    return []


def _company_variants(company: str) -> set[str]:
    """
    Builds the set of lowercase text variants that should match a given
    canonical company name in free-text user queries, plus any manually
    curated aliases for names that lose word-boundary information once
    stored in smashed-together form.
    """
    company_lower = company.lower()
    variants = {
        company_lower,
        company_lower + "s",              # informal plural / typo, e.g. "microsofts"
        company_lower.replace("-", " "),
        company_lower.replace("-", ""),
        company_lower.replace(" ", ""),
    }
    variants.update(alias.lower() for alias in _aliases_for(company))
    return {v for v in variants if v}  # drop empty strings defensively


def _variant_pattern(variant: str) -> re.Pattern:
    """
    Compiles a pattern that matches a company variant as a whole word,
    optionally followed immediately by a possessive marker ('s, 's, or a
    bare trailing apostrophe) — so "Amazon's" is matched and later removed
    as a single unit instead of leaving a dangling "'s" behind.
    """
    pattern = (
        rf"(?<!\w){re.escape(variant)}"
        rf"(?:[{_APOSTROPHE_CHARS}]s?)?"
        rf"(?!\w)"
    )
    return re.compile(pattern, flags=re.IGNORECASE)


# ============================================================
# Company detection
# ============================================================

@dataclass(frozen=True)
class _CompanyMatch:
    company: str  # canonical company name, e.g. "AMAZON"
    start:   int  # start offset in the original question text
    end:     int  # end offset in the original question text (exclusive)


def _find_company_matches(question: str) -> list[_CompanyMatch]:
    """
    Finds every mention of every known company in `question`, in order of
    appearance. If the same company is mentioned more than once (e.g. "compare
    Amazon's 2023 revenue to Amazon's 2024 revenue"), every occurrence is
    returned — not just the first — so that preprocess_query() can strip all
    of them out of clean_query. Overlapping candidate matches (e.g. a
    variant that is a substring of another company's variant) are resolved
    by keeping the longest match at the earliest position; the exact spans
    returned here are what preprocess_query() later deletes, so detection
    and removal can never disagree.
    """
    if not question:
        return []

    candidates: list[_CompanyMatch] = []
    for company in get_known_companies():
        for variant in sorted(_company_variants(company), key=len, reverse=True):
            for m in _variant_pattern(variant).finditer(question):
                candidates.append(_CompanyMatch(company, m.start(), m.end()))

    # Order by position of appearance; break ties by preferring the longer
    # (more specific) match at the same start offset.
    candidates.sort(key=lambda c: (c.start, -(c.end - c.start)))

    matches: list[_CompanyMatch] = []
    last_end = -1
    for c in candidates:
        if c.start < last_end:
            continue  # overlaps an already-kept, higher-priority match
        matches.append(c)
        last_end = c.end

    return matches


def detect_companies(question: str) -> list[str]:
    """
    Detects every known company mentioned in the query, in order of first
    appearance. Each company appears at most once in the returned list even
    if it's mentioned multiple times in the text (use _find_company_matches
    directly if you need every raw occurrence/span). Returns an empty list
    if none are found.
    """
    matches = _find_company_matches(question or "")
    # dict.fromkeys preserves first-seen order while de-duplicating.
    companies = list(dict.fromkeys(m.company for m in matches))
    logger.debug("Company detection | question=%r -> %r", question, companies)
    return companies


def detect_company(question: str) -> str | None:
    """
    Backward-compatible single-company detector: returns the first company
    mentioned (by position in the text), or None. Prefer detect_companies()
    for queries that may mention more than one company (e.g. comparisons).
    """
    companies = detect_companies(question)
    return companies[0] if companies else None


# ============================================================
# Year detection
# ============================================================

def detect_years(question: str) -> list[str]:
    """Extracts every 20xx year mentioned in the query, in order of appearance."""
    if not question:
        return []
    return _YEAR_RE.findall(question)


def detect_year(question: str) -> str | None:
    """Backward-compatible single-year detector: returns the first year found."""
    years = detect_years(question)
    return years[0] if years else None


# ============================================================
# Query preprocessing
# ============================================================

def _remove_spans(text: str, spans: list[tuple[int, int]]) -> str:
    """Deletes the given (start, end) character spans from text."""
    chars = list(text)
    for start, end in sorted(spans, reverse=True):  # delete back-to-front
        del chars[start:end]
    return "".join(chars)


def preprocess_query(question: str) -> dict:
    """
    Extract structured retrieval metadata from a raw user question.

    Returns:
    {
        "original":    the raw, unmodified question,
        "clean_query": question with EVERY detected company mention (and
                       its possessive, if any) removed — including repeat
                       mentions of the same company — and whitespace
                       normalized; safe to embed for dense retrieval
                       without company-name bias,
        "company":     first detected company (str) or None — kept for
                       backward compatibility with single-company callers,
        "companies":   ALL distinct detected companies, in order of first
                       appearance,
        "year":        first detected year (str) or None — kept for
                       backward compatibility,
        "years":       ALL detected years, in order of appearance,
    }
    """
    question = (question or "").strip()

    # Every occurrence of every company is found here (not de-duplicated),
    # so clean_query strips all of them, while `companies` below is
    # de-duplicated separately for the structured-metadata view.
    matches   = _find_company_matches(question)
    companies = list(dict.fromkeys(m.company for m in matches))
    years     = detect_years(question)

    if matches:
        clean_query = _remove_spans(question, [(m.start, m.end) for m in matches])
        clean_query = re.sub(r"\s+", " ", clean_query).strip()
    else:
        clean_query = question

    logger.debug(
        "preprocess_query | original=%r -> clean_query=%r | companies=%r | years=%r",
        question, clean_query, companies, years,
    )

    return {
        "original":    question,
        "clean_query": clean_query,
        "company":     companies[0] if companies else None,
        "companies":   companies,
        "year":        years[0] if years else None,
        "years":       years,
    }