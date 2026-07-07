# # import re

# # from mistralai.client.sdk import Mistral

# # import config


# # def regex_detect_company(text: str):

# #     patterns = [

# #         r"([A-Z][A-Z\s&.,'-]+?)\s+CORPORATION",
# #         r"([A-Z][A-Z\s&.,'-]+?)\s+INC",
# #         r"([A-Z][A-Z\s&.,'-]+?)\s+INC\.",
# #         r"([A-Z][A-Z\s&.,'-]+?)\s+LIMITED",
# #         r"([A-Z][A-Z\s&.,'-]+?)\s+LTD",
# #         r"([A-Z][A-Z\s&.,'-]+?)\s+PLC",
# #     ]

# #     text = text[:5000]

# #     for pattern in patterns:

# #         match = re.search(pattern, text)

# #         if match:

# #             company = match.group(1).strip()

# #             company = re.sub(r"\s+", " ", company)

# #             return company.title()

# #     return None


# # def llm_detect_company(text: str):

# #     client = Mistral(api_key=config.MISTRAL_API_KEY)

# #     prompt = f"""
# # Identify the company name from this annual report.

# # Rules:
# # - Return ONLY the company name.
# # - No explanation.
# # - No punctuation.
# # - No extra words.

# # Document:

# # {text[:8000]}
# # """

# #     response = client.chat.complete(
# #         model=config.LLM_MODEL_ID,
# #         messages=[
# #             {
# #                 "role": "user",
# #                 "content": prompt
# #             }
# #         ]
# #     )

# #     company = response.choices[0].message.content.strip()

# #     company = re.sub(
# #         r"\b(Inc|Inc\.|Corporation|Corp|Corp\.|Limited|Ltd)\b",
# #         "",
# #         company,
# #         flags=re.IGNORECASE
# #     )

# #     return company.strip()


# # def detect_company(text):

# #     company = regex_detect_company(text)

# #     if company:
# #         print(f"✅ Regex detected company: {company}")
# #         return company

# #     print("[WARN] Regex failed, falling back to LLM")

# #     return llm_detect_company(text)












# """
# company_detector.py

# Detects the company name that an annual-report / financial document belongs
# to. Two-stage strategy:

#     1. Fast, deterministic regex pass over common legal-suffix patterns
#        (CORPORATION, INC, LIMITED, LTD, PLC, LLC, ...).
#     2. LLM fallback (Mistral) when the regex pass finds nothing.

# The output of this module is later persisted as the "company" field in
# chunks.json and read back by query_processor.get_known_companies(), which
# uppercases and strips it before caching. To keep detection reliable
# end-to-end, every code path here funnels through a single
# `_normalize_company_name()` step so the stored value is always clean,
# whitespace-normalized, and free of dangling punctuation or legal suffixes.
# """

# from __future__ import annotations

# import logging
# from pydoc import text
# import re

# import config

# logger = logging.getLogger(__name__)


# # ============================================================
# # Mistral client import
# # ============================================================
# # The import path for the Mistral SDK has changed across versions:
# #   - modern SDKs (>=1.0):  `from mistralai import Mistral`
# #   - older SDKs   (<1.0):  `from mistralai.client import MistralClient`
# #
# # This import is deliberately lazy (deferred into _get_mistral_client()
# # below) rather than done at module load time. regex_detect_company() is
# # the hot path and succeeds for the vast majority of documents; a caller
# # who never needs the LLM fallback shouldn't have their import of this
# # module fail just because the mistralai package isn't installed or is on
# # an unexpected version.
# _LEGACY_MISTRAL_CLIENT = False


# def _get_mistral_client(api_key: str):
#     """Lazily imports and constructs a Mistral client, new-SDK first."""
#     global _LEGACY_MISTRAL_CLIENT

#     try:
#         from mistralai import Mistral  # modern SDK
#         _LEGACY_MISTRAL_CLIENT = False
#         return Mistral(api_key=api_key)
#     except ImportError:
#         pass

#     try:
#         from mistralai.client import MistralClient  # legacy SDK
#         _LEGACY_MISTRAL_CLIENT = True
#         return MistralClient(api_key=api_key)
#     except ImportError as exc:
#         raise ImportError(
#             "The 'mistralai' package is required for LLM-based company "
#             "detection but is not installed. Install it with "
#             "`pip install mistralai`, or ensure regex_detect_company() "
#             "succeeds so this fallback is never needed."
#         ) from exc


# # ============================================================
# # Constants
# # ============================================================

# # How many characters of the document to look at for each strategy.
# # The company name virtually always appears on the cover page / masthead,
# # so we only need the very start of the document.
# _REGEX_SCAN_CHARS = 5000
# _LLM_SCAN_CHARS = 8000

# # Legal-entity suffixes, longest/most-specific first so that a suffix like
# # "INCORPORATED" is preferred over a partial "INC" match at the same
# # position (the trailing \b already prevents "INC" from matching inside
# # "INCORPORATED", but ordering keeps intent explicit and the pattern easy
# # to extend).
# _SUFFIXES = [
#     "INCORPORATED",
#     "CORPORATION",
#     "LIMITED",
#     "HOLDINGS",
#     "HOLDING",
#     "COMPANY",
#     "GROUP",
#     r"CORP\.?",
#     r"INC\.?",
#     r"LTD\.?",
#     r"CO\.?",
#     "PLC",
#     "LLC",
#     "LLP",
# ]

# # A single compiled pattern: capture a run of capitalized "name-like" text
# # immediately followed by whitespace and one of the known legal suffixes,
# # with a trailing word boundary so the suffix can never match as a
# # substring of a longer word (fixes "INC" matching inside "INCORPORATED").
# _COMPANY_SUFFIX_RE = re.compile(
#     r"([A-Z][A-Z0-9&.,'-]*(?:\s+[A-Z0-9&.,'-]+)*?)\s+"
#     r"(?:" + "|".join(_SUFFIXES) + r")\b"
# )

# # Used to strip a legal suffix back out of an LLM response, anchored to the
# # *end* of the string only (a company whose real name happens to contain
# # one of these words in the middle, e.g. "Group Holdings Analytics", should
# # not be mangled).
# _TRAILING_SUFFIX_RE = re.compile(
#     r"[\s,.-]+(?:" + "|".join(_SUFFIXES) + r")\.?\s*$",
#     flags=re.IGNORECASE,
# )

# _WHITESPACE_RE = re.compile(r"\s+")

# # Characters that should never be left dangling at the edges of a detected
# # company name (stray commas/periods/hyphens left behind after suffix or
# # whitespace stripping).
# _EDGE_JUNK = " \t\n.,'-&"


# # ============================================================
# # Shared normalization
# # ============================================================

# def _normalize_company_name(name: str) -> str:
#     """
#     Applies the same cleanup rules regardless of which detection strategy
#     produced the raw name, so regex-detected and LLM-detected company names
#     end up in a consistent format:
#       - collapse internal whitespace
#       - strip legal suffixes trailing the name
#       - trim stray edge punctuation
#       - title-case for a clean, consistent display form
#     """
#     if not name:
#         return ""

#     name = _WHITESPACE_RE.sub(" ", name).strip()
#     name = _TRAILING_SUFFIX_RE.sub("", name).strip()
#     name = name.strip(_EDGE_JUNK)
#     name = _WHITESPACE_RE.sub(" ", name).strip()

#     return name.title() if name else ""


# # ============================================================
# # Regex-based detection
# # ============================================================

# def regex_detect_company(text: str) -> str | None:
#     """
#     Attempts to find a company name via common legal-suffix patterns
#     (CORPORATION, INC, LIMITED, LTD, PLC, LLC, ...).

#     Returns the normalized company name, or None if no confident match
#     was found.
#     """
#     if not text:
#         return None

#     snippet = text[:_REGEX_SCAN_CHARS]

#     match = _COMPANY_SUFFIX_RE.search(snippet)
#     if not match:
#         return None

#     company = _normalize_company_name(match.group(1))
#     return company or None

# # ============================================================
# # LLM-based detection (fallback)
# # ============================================================

# def llm_detect_company(text: str) -> str | None:
#     """
#     Falls back to an LLM (Mistral) to identify the company name when the
#     regex pass fails. Returns the normalized company name, or None if the
#     call fails or returns nothing usable.
#     """
#     if not text:
#         return None

#     if not getattr(config, "MISTRAL_API_KEY", None):
#         logger.error("MISTRAL_API_KEY is not configured; cannot run LLM fallback.")
#         return None

#     prompt = f"""Identify the company name from this annual report.

# Rules:
# - Return ONLY the company name.
# - No explanation.
# - No punctuation.
# - No extra words.

# Document:

# {text[:_LLM_SCAN_CHARS]}
# """

#     try:
#         client = _get_mistral_client(config.MISTRAL_API_KEY)

#         if _LEGACY_MISTRAL_CLIENT:
#             response = client.chat(
#                 model=config.LLM_MODEL_ID,
#                 messages=[{"role": "user", "content": prompt}],
#             )
#         else:
#             response = client.chat.complete(
#                 model=config.LLM_MODEL_ID,
#                 messages=[{"role": "user", "content": prompt}],
#             )
#     except Exception:
#         logger.exception("LLM company detection failed.")
#         return None

#     try:
#         choices = getattr(response, "choices", None)
#         if not choices:
#             logger.warning("LLM response contained no choices.")
#             return None

#         raw = (choices[0].message.content or "").strip()
#     except (AttributeError, IndexError):
#         logger.exception("Unexpected LLM response shape.")
#         return None

#     if not raw:
#         logger.warning("LLM returned an empty company name.")
#         return None

#     company = _normalize_company_name(raw)
#     return company or None


# # ============================================================
# # Public entry point
# # ============================================================

# def detect_company(text: str) -> str | None:
#     """
#     Detects the company name for a document: tries the fast regex pass
#     first, falling back to the LLM only when necessary.

#     Returns the normalized company name, or None if neither strategy
#     could identify one.
#     """
    
#     company = regex_detect_company(text)

#     print("=" * 50)
#     print("REGEX RESULT:", company)
#     print("TEXT SAMPLE:")
#     print(text[:1000])
#     print("=" * 50)

    
#     company = regex_detect_company(text)
#     if company:
#         logger.info("Regex detected company: %s", company)
#         return company

#     logger.info("Regex detection failed; falling back to LLM.")
#     company = llm_detect_company(text)

#     if company:
#         logger.info("LLM detected company: %s", company)
#     else:
#         logger.warning("Company detection failed via both regex and LLM.")

#     return company






















# """
# company_detector.py

# Detects the company name that an annual-report / financial document (10-K,
# 10-Q, integrated annual report, etc.) belongs to.

# Detection strategy, cheapest / most-reliable first:

#     1. Registrant-marker pass: SEC cover pages almost always print the
#        company name on the line(s) immediately above the boilerplate
#        "(Exact name of Registrant as specified in its charter)". When that
#        marker is present this is by far the most reliable signal, so it is
#        tried first.
#     2. Legal-suffix regex pass ("safe" suffixes): a deterministic scan for
#        a capitalized name run followed by an unambiguous legal-entity
#        suffix (CORPORATION, INC, LIMITED, LTD, PLC, LLC, LLP, CORP, CO,
#        INCORPORATED).
#     3. Legal-suffix regex pass ("generic" suffixes): the same scan for
#        generic words that are also common English words when used outside
#        a company name (COMPANY, GROUP, HOLDING, HOLDINGS). These require a
#        longer preceding name run to reduce false positives.
#     4. LLM fallback (Mistral) when every regex pass finds nothing.

# The output of this module is later persisted as the "company" field in
# chunks.json and read back by query_processor.get_known_companies(), which
# uppercases and strips it before caching. To keep detection reliable
# end-to-end, every code path here funnels through a single
# `_normalize_company_name()` step so the stored value is always clean,
# whitespace-normalized, and free of dangling punctuation or legal suffixes.
# """

# from __future__ import annotations

# import logging
# import re
# from typing import Iterator, Optional

# import config

# logger = logging.getLogger(__name__)


# # ============================================================
# # Mistral client import
# # ============================================================
# # The import path for the Mistral SDK has moved around across versions and,
# # unhelpfully, across releases of the *same* major version:
# #   - `from mistralai import Mistral`              -> most published 1.x SDKs
# #   - `from mistralai.client.sdk import Mistral`    -> some 2.x SDKs whose
# #                                                      top-level package has
# #                                                      no re-export at all
# #   - `from mistralai.client import MistralClient`  -> legacy pre-1.0 SDK
# #
# # All of these expose a class, but only the first two share the modern
# # calling convention (`client.chat.complete(...)`); the legacy client uses
# # `client.chat(...)`. Rather than relying on mutable global state to
# # remember which convention applies (which breaks under concurrent use),
# # `_get_mistral_client()` returns the calling convention alongside the
# # client instance.
# #
# # This import is deliberately lazy (deferred into _get_mistral_client()
# # below) rather than done at module load time. regex_detect_company() is
# # the hot path and succeeds for the vast majority of documents; a caller
# # who never needs the LLM fallback shouldn't have their import of this
# # module fail just because the mistralai package isn't installed or is on
# # an unexpected version.

# _MODERN = "modern"   # client.chat.complete(model=..., messages=...)
# _LEGACY = "legacy"    # client.chat(model=..., messages=...)

# # Ordered (import_path, attr_name, calling_convention) candidates.
# _MISTRAL_IMPORT_CANDIDATES = (
#     ("mistralai", "Mistral", _MODERN),
#     ("mistralai.client.sdk", "Mistral", _MODERN),
#     ("mistralai.client", "MistralClient", _LEGACY),
# )


# def _get_mistral_client(api_key: str) -> tuple[object, str]:
#     """
#     Lazily imports and constructs a Mistral client, trying each known SDK
#     layout in turn.

#     Returns a (client, calling_convention) tuple, where calling_convention
#     is one of _MODERN or _LEGACY.

#     Raises ImportError, listing every import path that was attempted, if
#     none of the known SDK layouts are importable.
#     """
#     import importlib

#     errors: list[str] = []

#     for module_path, attr_name, convention in _MISTRAL_IMPORT_CANDIDATES:
#         try:
#             module = importlib.import_module(module_path)
#             client_cls = getattr(module, attr_name)
#         except (ImportError, AttributeError) as exc:
#             errors.append(f"{module_path}.{attr_name}: {exc}")
#             continue

#         try:
#             return client_cls(api_key=api_key), convention
#         except Exception as exc:  # noqa: BLE001 - constructor failures vary by SDK
#             errors.append(f"{module_path}.{attr_name} (constructor failed): {exc}")
#             continue

#     raise ImportError(
#         "Could not construct a Mistral client with any known 'mistralai' "
#         "SDK layout. Install/upgrade with `pip install -U mistralai`, or "
#         "ensure regex_detect_company() succeeds so this fallback is never "
#         "needed. Attempts:\n  - " + "\n  - ".join(errors)
#     )


# # ============================================================
# # Constants
# # ============================================================

# # How many characters of the document to look at for each strategy.
# # The company name virtually always appears on the cover page / masthead,
# # so we only need the very start of the document.
# _REGEX_SCAN_CHARS = 5000
# _LLM_SCAN_CHARS = 8000

# # SEC filings (10-K, 10-Q, 8-K, ...) print the registrant's name directly
# # above this boilerplate line on the cover page. When present, this is the
# # single most reliable signal available and is checked before any
# # suffix-based heuristics.
# _REGISTRANT_MARKER_RE = re.compile(
#     r"\(Exact name of\s+Registrant\s+as specified in its charter\)",
#     re.IGNORECASE,
# )

# # Legal-entity suffixes that are essentially unambiguous when they appear
# # after a capitalized name run. "INCORPORATED" is included here but guarded
# # with a negative lookahead: SEC cover pages very commonly contain the
# # unrelated heading "Documents Incorporated by Reference", and without the
# # guard that phrase is mistaken for a company name on almost every 10-K.
# _SAFE_SUFFIXES = [
#     r"INCORPORATED(?!\s+(?:BY|HEREIN|HERETO|IN)\b)",
#     "CORPORATION",
#     "LIMITED",
#     r"CORP\.?",
#     r"INC\.?",
#     r"LTD\.?",
#     "PLC",
#     "LLC",
#     "LLP",
#     r"CO\.?",
# ]

# # Words that are unambiguous *legal* suffixes only in some contexts and are
# # otherwise ordinary English words ("the Company", "our Group"). Matches
# # against these require a longer preceding name run (see MIN_GENERIC_WORDS)
# # to cut down on false positives.
# _GENERIC_SUFFIXES = [
#     "HOLDINGS",
#     "HOLDING",
#     "GROUP",
#     "COMPANY",
# ]

# _MIN_GENERIC_WORDS = 2

# # A run of one-or-more capitalized "name-like" words: starts with an
# # uppercase letter, may contain further letters (upper or lower - real
# # filings mix "Apple Inc." with "NVIDIA CORPORATION"), digits, and a small
# # set of punctuation marks that legitimately appear inside company names
# # (&, ., ', -), with an optional trailing comma ("Netflix, Inc.").
# _NAME_RUN = r"(?:[A-Z][A-Za-z0-9&.'-]*,?\s+){1,6}?"


# def _make_suffix_pattern(suffixes: list[str]) -> re.Pattern:
#     # IGNORECASE: real-world filings mix "LIMITED" (all-caps cover pages)
#     # with "Limited" (mixed-case running prose, e.g. Indian annual
#     # reports), so the suffix keywords themselves must match either way.
#     return re.compile(
#         r"(" + _NAME_RUN + r")(?:" + "|".join(suffixes) + r")\b",
#         flags=re.IGNORECASE,
#     )


# _SAFE_SUFFIX_RE = _make_suffix_pattern(_SAFE_SUFFIXES)
# _GENERIC_SUFFIX_RE = _make_suffix_pattern(_GENERIC_SUFFIXES)

# # Used to strip a legal suffix back out of a raw candidate (from the
# # registrant-marker pass or an LLM response), anchored to the *end* of the
# # string only (a company whose real name happens to contain one of these
# # words in the middle, e.g. "Group Holdings Analytics", should not be
# # mangled).
# _ALL_SUFFIXES_FOR_STRIPPING = [s.split("(?!")[0] for s in _SAFE_SUFFIXES] + _GENERIC_SUFFIXES
# _TRAILING_SUFFIX_RE = re.compile(
#     r"[\s,.-]+(?:" + "|".join(_ALL_SUFFIXES_FOR_STRIPPING) + r")\.?\s*$",
#     flags=re.IGNORECASE,
# )

# _WHITESPACE_RE = re.compile(r"\s+")

# # Characters that should never be left dangling at the edges of a detected
# # company name (stray commas/periods/hyphens left behind after suffix or
# # whitespace stripping).
# _EDGE_JUNK = " \t\n.,'-&"

# # Boilerplate phrases that occasionally satisfy the suffix/marker patterns
# # but are never actual company names (SEC cover-page headings, section
# # labels, etc.). Checked case-insensitively against the fully normalized
# # candidate.
# _BLOCKLIST = {
#     "DOCUMENTS",
#     "DOCUMENT",
#     "TABLE OF CONTENTS",
#     "SECURITIES",
#     "REGISTRANT",
#     "EXACT NAME",
#     "STATE",
#     "PART",
#     "ITEM",
#     "NOTES",
#     "FORM",
#     "ANNUAL REPORT",
#     "QUARTERLY REPORT",
#     "TRANSITION REPORT",
#     "COMMISSION FILE NUMBER",
# }


# # ============================================================
# # Shared normalization
# # ============================================================

# def _normalize_company_name(name: str) -> str:
#     """
#     Applies the same cleanup rules regardless of which detection strategy
#     produced the raw name, so regex-detected and LLM-detected company names
#     end up in a consistent format:
#       - collapse internal whitespace
#       - strip legal suffixes trailing the name
#       - trim stray edge punctuation
#       - title-case for a clean, consistent display form
#     """
#     if not name:
#         return ""

#     name = _WHITESPACE_RE.sub(" ", name).strip()
#     name = _TRAILING_SUFFIX_RE.sub("", name).strip()
#     name = name.strip(_EDGE_JUNK)
#     name = _WHITESPACE_RE.sub(" ", name).strip()

#     return name.title() if name else ""


# def _is_valid_candidate(normalized: str) -> bool:
#     """Rejects normalized candidates that are empty or known boilerplate."""
#     if not normalized:
#         return False
#     return normalized.upper() not in _BLOCKLIST


# # ============================================================
# # Regex-based detection
# # ============================================================

# def _find_registrant_name(snippet: str) -> Optional[str]:
#     """
#     SEC cover pages print the company name directly above the line
#     "(Exact name of Registrant as specified in its charter)". This grabs
#     the one or two non-blank lines immediately preceding that marker.
#     """
#     match = _REGISTRANT_MARKER_RE.search(snippet)
#     if not match:
#         return None

#     preceding_text = snippet[: match.start()]
#     lines = [ln.strip() for ln in preceding_text.splitlines()]

#     # The company name is virtually always a single line directly above the
#     # marker. Only pull in a second line if the closest one is a single
#     # short token (e.g. a wrapped name split across two lines) rather than
#     # unconditionally grabbing two lines, which would also swallow
#     # unrelated boilerplate (commission file numbers, section labels, ...)
#     # that happens to sit just above it.
#     collected: list[str] = []
#     for line in reversed(lines):
#         if not line:
#             if collected:
#                 break
#             continue
#         collected.append(line)
#         if len(collected) == 1 and len(line.split()) > 1:
#             break
#         if len(collected) >= 2:
#             break

#     if not collected:
#         return None

#     collected.reverse()
#     candidate = _normalize_company_name(" ".join(collected))
#     return candidate if _is_valid_candidate(candidate) else None


# def _iter_suffix_candidates(
#     snippet: str, pattern: re.Pattern, min_words: int = 1
# ) -> Iterator[str]:
#     """Yields normalized, validated candidates for every suffix match, in order."""
#     for match in pattern.finditer(snippet):
#         raw = match.group(1)
#         if len(raw.split()) < min_words:
#             continue

#         normalized = _normalize_company_name(raw)
#         if _is_valid_candidate(normalized):
#             yield normalized


# def regex_detect_company(text: str) -> Optional[str]:
#     """
#     Attempts to find a company name via, in order of reliability:
#       1. the SEC "Exact name of Registrant" cover-page marker,
#       2. unambiguous legal-suffix patterns (CORPORATION, INC, LIMITED, ...),
#       3. generic legal-suffix patterns (COMPANY, GROUP, HOLDINGS, ...).

#     Returns the normalized company name, or None if no confident match
#     was found.
#     """
#     if not text:
#         return None

#     snippet = text[:_REGEX_SCAN_CHARS]

#     registrant_name = _find_registrant_name(snippet)
#     if registrant_name:
#         return registrant_name

#     for candidate in _iter_suffix_candidates(snippet, _SAFE_SUFFIX_RE):
#         return candidate

#     for candidate in _iter_suffix_candidates(
#         snippet, _GENERIC_SUFFIX_RE, min_words=_MIN_GENERIC_WORDS
#     ):
#         return candidate

#     return None


# # ============================================================
# # LLM-based detection (fallback)
# # ============================================================

# def llm_detect_company(text: str) -> Optional[str]:
#     """
#     Falls back to an LLM (Mistral) to identify the company name when the
#     regex pass fails. Returns the normalized company name, or None if the
#     call fails or returns nothing usable.
#     """
#     if not text:
#         return None

#     if not getattr(config, "MISTRAL_API_KEY", None):
#         logger.error("MISTRAL_API_KEY is not configured; cannot run LLM fallback.")
#         return None

#     prompt = f"""Identify the company name from this annual report.

# Rules:
# - Return ONLY the company name.
# - No explanation.
# - No punctuation.
# - No extra words.

# Document:

# {text[:_LLM_SCAN_CHARS]}
# """

#     try:
#         client, convention = _get_mistral_client(config.MISTRAL_API_KEY)
#     except ImportError:
#         logger.exception("LLM company detection unavailable: no usable Mistral SDK.")
#         return None

#     try:
#         if convention == _LEGACY:
#             response = client.chat(
#                 model=config.LLM_MODEL_ID,
#                 messages=[{"role": "user", "content": prompt}],
#             )
#         else:
#             response = client.chat.complete(
#                 model=config.LLM_MODEL_ID,
#                 messages=[{"role": "user", "content": prompt}],
#             )
#     except Exception:
#         logger.exception("LLM company detection call failed.")
#         return None

#     try:
#         choices = getattr(response, "choices", None)
#         if not choices:
#             logger.warning("LLM response contained no choices.")
#             return None

#         raw = (choices[0].message.content or "").strip()
#     except (AttributeError, IndexError):
#         logger.exception("Unexpected LLM response shape.")
#         return None

#     if not raw:
#         logger.warning("LLM returned an empty company name.")
#         return None

#     company = _normalize_company_name(raw)
#     if not _is_valid_candidate(company):
#         logger.warning("LLM returned an unusable company name: %r", raw)
#         return None

#     return company


# # ============================================================
# # Public entry point
# # ============================================================

# def detect_company(text: str) -> Optional[str]:
#     """
#     Detects the company name for a document: tries the fast regex pass
#     first, falling back to the LLM only when necessary.

#     Returns the normalized company name, or None if neither strategy
#     could identify one.
#     """
#     company = regex_detect_company(text)

#     logger.debug("Regex company-detection result: %s", company)
#     logger.debug("Text sample: %s", text[:1000] if text else "")

#     if company:
#         logger.info("Regex detected company: %s", company)
#         return company

#     logger.info("Regex detection failed; falling back to LLM.")
#     company = llm_detect_company(text)

#     if company:
#         logger.info("LLM detected company: %s", company)
#     else:
#         logger.warning("Company detection failed via both regex and LLM.")

#     return company


















































"""
company_detector.py

Detects the company name that an annual-report / financial document (10-K,
10-Q, integrated annual report, etc.) belongs to.

Detection strategy, cheapest / most-reliable first:

    1. Registrant-marker pass: SEC cover pages almost always print the
       company name on the line(s) immediately above the boilerplate
       "(Exact name of Registrant as specified in its charter)". When that
       marker is present this is by far the most reliable signal, so it is
       tried first.
    2. Legal-suffix regex pass ("safe" suffixes): a deterministic scan for
       a capitalized name run followed by an unambiguous legal-entity
       suffix (CORPORATION, INC, LIMITED, LTD, PLC, LLC, LLP, CORP, CO,
       INCORPORATED).
    3. Legal-suffix regex pass ("generic" suffixes): the same scan for
       generic words that are also common English words when used outside
       a company name (COMPANY, GROUP, HOLDING, HOLDINGS). These require a
       longer preceding name run to reduce false positives.
    4. LLM fallback (Mistral) when every regex pass finds nothing.

The output of this module is later persisted as the "company" field in
chunks.json and read back by query_processor.get_known_companies(), which
uppercases and strips it before caching. To keep detection reliable
end-to-end, every code path here funnels through a single
`_normalize_company_name()` step so the stored value is always clean,
whitespace-normalized, and free of dangling punctuation or legal suffixes.
"""

from __future__ import annotations

import logging
import re
from typing import Iterator, Optional

import config

logger = logging.getLogger(__name__)


# ============================================================
# Mistral client import
# ============================================================
# The import path for the Mistral SDK has moved around across versions and,
# unhelpfully, across releases of the *same* major version:
#   - `from mistralai import Mistral`              -> most published 1.x SDKs
#   - `from mistralai.client.sdk import Mistral`    -> some 2.x SDKs whose
#                                                      top-level package has
#                                                      no re-export at all
#   - `from mistralai.client import MistralClient`  -> legacy pre-1.0 SDK
#
# All of these expose a class, but only the first two share the modern
# calling convention (`client.chat.complete(...)`); the legacy client uses
# `client.chat(...)`. Rather than relying on mutable global state to
# remember which convention applies (which breaks under concurrent use),
# `_get_mistral_client()` returns the calling convention alongside the
# client instance.
#
# This import is deliberately lazy (deferred into _get_mistral_client()
# below) rather than done at module load time. regex_detect_company() is
# the hot path and succeeds for the vast majority of documents; a caller
# who never needs the LLM fallback shouldn't have their import of this
# module fail just because the mistralai package isn't installed or is on
# an unexpected version.

_MODERN = "modern"   # client.chat.complete(model=..., messages=...)
_LEGACY = "legacy"    # client.chat(model=..., messages=...)

# Ordered (import_path, attr_name, calling_convention) candidates.
_MISTRAL_IMPORT_CANDIDATES = (
    ("mistralai", "Mistral", _MODERN),
    ("mistralai.client.sdk", "Mistral", _MODERN),
    ("mistralai.client", "MistralClient", _LEGACY),
)


def _get_mistral_client(api_key: str) -> tuple[object, str]:
    """
    Lazily imports and constructs a Mistral client, trying each known SDK
    layout in turn.

    Returns a (client, calling_convention) tuple, where calling_convention
    is one of _MODERN or _LEGACY.

    Raises ImportError, listing every import path that was attempted, if
    none of the known SDK layouts are importable.
    """
    import importlib

    errors: list[str] = []

    for module_path, attr_name, convention in _MISTRAL_IMPORT_CANDIDATES:
        try:
            module = importlib.import_module(module_path)
            client_cls = getattr(module, attr_name)
        except (ImportError, AttributeError) as exc:
            errors.append(f"{module_path}.{attr_name}: {exc}")
            continue

        try:
            return client_cls(api_key=api_key), convention
        except Exception as exc:  # noqa: BLE001 - constructor failures vary by SDK
            errors.append(f"{module_path}.{attr_name} (constructor failed): {exc}")
            continue

    raise ImportError(
        "Could not construct a Mistral client with any known 'mistralai' "
        "SDK layout. Install/upgrade with `pip install -U mistralai`, or "
        "ensure regex_detect_company() succeeds so this fallback is never "
        "needed. Attempts:\n  - " + "\n  - ".join(errors)
    )


# ============================================================
# Constants
# ============================================================

# How many characters of the document to look at for each strategy.
# The company name virtually always appears on the cover page / masthead,
# but some documents (e.g. foreign-filer annual reports) front-load several
# pages of directory-style boilerplate - spokesperson, registrar, auditor -
# before the actual title page, so the candidate-discovery window is kept
# generous. _FREQUENCY_SCAN_CHARS is a separate, larger window used only to
# rank *already found* candidates by how often they recur in the document:
# the real company name is mentioned repeatedly (running headers, "Item 1.
# Business", etc.), while an incidental mention of a third party is not.
_REGEX_SCAN_CHARS = 12000
_FREQUENCY_SCAN_CHARS = 30000
_LLM_SCAN_CHARS = 8000

# SEC filings (10-K, 10-Q, 8-K, ...) print the registrant's name directly
# above this boilerplate line on the cover page. When present, this is the
# single most reliable signal available and is checked before any
# suffix-based heuristics.
_REGISTRANT_MARKER_RE = re.compile(
    r"\(Exact name of\s+Registrant\s+as specified in its charter\)",
    re.IGNORECASE,
)

# Legal-entity suffixes that are essentially unambiguous when they appear
# after a capitalized name run. "INCORPORATED" is included here but guarded
# with a negative lookahead: SEC cover pages very commonly contain the
# unrelated heading "Documents Incorporated by Reference", and without the
# guard that phrase is mistaken for a company name on almost every 10-K.
_SAFE_SUFFIXES = [
    r"INCORPORATED(?!\s+(?:BY|HEREIN|HERETO|IN)\b)",
    "CORPORATION",
    "LIMITED",
    r"CORP\.?",
    r"INC\.?",
    r"LTD\.?",
    "PLC",
    "LLC",
    "LLP",
    r"CO\.?",
]

# Words that are unambiguous *legal* suffixes only in some contexts and are
# otherwise ordinary English words ("the Company", "our Group"). Matches
# against these require a longer preceding name run (see MIN_GENERIC_WORDS)
# to cut down on false positives.
_GENERIC_SUFFIXES = [
    "HOLDINGS",
    "HOLDING",
    "GROUP",
    "COMPANY",
]

_MIN_GENERIC_WORDS = 2

# A run of one-or-more capitalized "name-like" words: starts with an
# uppercase letter (case-SENSITIVE - this is what stops the pattern from
# matching ordinary lowercase prose such as "...convinced me to interview
# at the consumer products company..."), may contain further letters
# (upper or lower - real filings mix "Apple Inc." with "NVIDIA
# CORPORATION"), digits, and a small set of punctuation marks that
# legitimately appear inside company names (&, ., ', -), with an optional
# trailing comma ("Netflix, Inc."). Restricted to "not a newline" so a
# match can never splice together words from unrelated, separately-wrapped
# lines (e.g. a marketing tagline printed one word per line above the real
# company name).
_NAME_RUN = r"(?:[A-Z][A-Za-z0-9&.'-]*,?[ \t]+){1,6}?"


def _make_suffix_pattern(suffixes: list[str]) -> re.Pattern:
    # Only the suffix keywords are case-insensitive (via the scoped inline
    # flag (?i:...)) - real-world filings mix "LIMITED" (all-caps cover
    # pages) with "Limited" (mixed-case running prose, e.g. Indian annual
    # reports). Applying re.IGNORECASE to the *whole* pattern instead would
    # also relax the name-run's required-uppercase-start check, letting it
    # match ordinary lowercase sentences - that was a real regression.
    suffix_group = "|".join(suffixes)
    return re.compile(r"(" + _NAME_RUN + r")(?:(?i:" + suffix_group + r"))\b")


_SAFE_SUFFIX_RE = _make_suffix_pattern(_SAFE_SUFFIXES)
_GENERIC_SUFFIX_RE = _make_suffix_pattern(_GENERIC_SUFFIXES)

# Used to strip a legal suffix back out of a raw candidate (from the
# registrant-marker pass or an LLM response), anchored to the *end* of the
# string only (a company whose real name happens to contain one of these
# words in the middle, e.g. "Group Holdings Analytics", should not be
# mangled).
_ALL_SUFFIXES_FOR_STRIPPING = [s.split("(?!")[0] for s in _SAFE_SUFFIXES] + _GENERIC_SUFFIXES
_TRAILING_SUFFIX_RE = re.compile(
    r"[\s,.-]+(?:" + "|".join(_ALL_SUFFIXES_FOR_STRIPPING) + r")\.?\s*$",
    flags=re.IGNORECASE,
)

_WHITESPACE_RE = re.compile(r"\s+")

# Characters that should never be left dangling at the edges of a detected
# company name (stray commas/periods/hyphens left behind after suffix or
# whitespace stripping).
_EDGE_JUNK = " \t\n.,'-&"

# Boilerplate phrases that occasionally satisfy the suffix/marker patterns
# but are never actual company names (SEC cover-page headings, section
# labels, etc.). Checked case-insensitively against the fully normalized
# candidate.
_BLOCKLIST = {
    "DOCUMENTS",
    "DOCUMENT",
    "TABLE OF CONTENTS",
    "SECURITIES",
    "REGISTRANT",
    "EXACT NAME",
    "STATE",
    "PART",
    "ITEM",
    "NOTES",
    "FORM",
    "ANNUAL REPORT",
    "QUARTERLY REPORT",
    "TRANSITION REPORT",
    "COMMISSION FILE NUMBER",
}

# Ordinary-English function words that legitimate company names essentially
# never contain. Guards against a Title-Case (or all-caps) phrase in a
# marketing tagline, pull-quote, or shareholder-letter anecdote being
# mistaken for a company name, e.g. "To Interview At The Consumer Products".
_INVALID_WORDS = {
    "TO", "AT", "FOR", "WITH", "FROM", "INTO", "ABOUT", "ONTO",
    "INTERVIEW", "SHOULD", "PLEASE", "CLICK", "APPLY", "CONTACT",
    "VISIT", "LEARN", "READ", "SEE", "DEAR", "SHAREHOLDERS",
}

# Phrases that mark a nearby capitalized name as a third-party service
# provider (registrar/transfer agent, broker, auditor, legal counsel, ...)
# rather than the filing company itself. Annual reports - especially
# foreign-filer ones - commonly list these in a corporate-directory section
# near the front of the document. Checked against the full line containing
# a candidate match, case-insensitively.
_THIRD_PARTY_INDICATORS = (
    "registrar and transfer",
    "transfer agent",
    "transfer services",
    "securities dealing institute",
    "stock transfer",
    "underwriter",
    "auditor",
    "cpa firm",
    "independent registered public accounting firm",
    "legal counsel",
    "depositary bank",
    "proxy solicitor",
)


# ============================================================
# Shared normalization
# ============================================================

def _normalize_company_name(name: str) -> str:
    """
    Applies the same cleanup rules regardless of which detection strategy
    produced the raw name, so regex-detected and LLM-detected company names
    end up in a consistent format:
      - collapse internal whitespace
      - strip legal suffixes trailing the name
      - trim stray edge punctuation
      - title-case for a clean, consistent display form
    """
    if not name:
        return ""

    name = _WHITESPACE_RE.sub(" ", name).strip()
    name = _TRAILING_SUFFIX_RE.sub("", name).strip()
    name = name.strip(_EDGE_JUNK)
    name = _WHITESPACE_RE.sub(" ", name).strip()

    return name.title() if name else ""


def _is_valid_candidate(normalized: str) -> bool:
    """Rejects normalized candidates that are empty or known boilerplate."""
    if not normalized:
        return False
    return normalized.upper() not in _BLOCKLIST


def _has_invalid_word(raw: str) -> bool:
    """True if raw contains an ordinary function word real company names don't."""
    return any(word.strip(",.").upper() in _INVALID_WORDS for word in raw.split())


def _looks_capitalized(raw: str) -> bool:
    """
    True if every alphabetic word (len > 1) in the ORIGINAL (pre-title-case)
    text already starts with an uppercase letter, as printed. This is what
    distinguishes a real cover-page heading ("Apple Inc.", "NVIDIA
    CORPORATION") from lowercase running prose that would only *look*
    capitalized after _normalize_company_name()'s cosmetic .title() call.
    """
    words = [w for w in re.findall(r"[A-Za-z][A-Za-z'&-]*", raw) if len(w) > 1]
    return bool(words) and all(word[0].isupper() for word in words)


def _is_third_party_context(context_line: str) -> bool:
    """True if the line containing a match names a service provider, not the filer."""
    lowered = context_line.lower()
    return any(indicator in lowered for indicator in _THIRD_PARTY_INDICATORS)


def _count_occurrences(candidate: str, haystack: str) -> int:
    """Counts recurrences of a candidate's most distinctive word in haystack."""
    if not candidate or not haystack:
        return 0
    core = max(candidate.split(), key=len)
    if len(core) < 3:
        core = candidate
    return len(re.findall(re.escape(core), haystack, flags=re.IGNORECASE))


def _rank_candidates(candidates: list[str], ranking_text: str) -> Optional[str]:
    """
    Picks the candidate that recurs most often in `ranking_text`. Real
    company names are repeated throughout a filing (running headers, "Item
    1. Business", ...); an incidental mention of a third party or a
    one-off phrase is not. Ties keep the first-found candidate.
    """
    if not candidates:
        return None

    seen_order = list(dict.fromkeys(candidates))
    if len(seen_order) == 1:
        return seen_order[0]

    scored = [(c, _count_occurrences(c, ranking_text)) for c in seen_order]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[0][0]


# ============================================================
# Regex-based detection
# ============================================================

def _find_registrant_name(snippet: str) -> Optional[str]:
    """
    SEC cover pages print the company name directly above the line
    "(Exact name of Registrant as specified in its charter)". This grabs
    the one or two non-blank lines immediately preceding that marker.
    """
    match = _REGISTRANT_MARKER_RE.search(snippet)
    if not match:
        return None

    preceding_text = snippet[: match.start()]
    lines = [ln.strip() for ln in preceding_text.splitlines()]

    # The company name is virtually always a single line directly above the
    # marker. Only pull in a second line if the closest one is a single
    # short token (e.g. a wrapped name split across two lines) rather than
    # unconditionally grabbing two lines, which would also swallow
    # unrelated boilerplate (commission file numbers, section labels, ...)
    # that happens to sit just above it.
    collected: list[str] = []
    for line in reversed(lines):
        if not line:
            if collected:
                break
            continue
        collected.append(line)
        if len(collected) == 1 and len(line.split()) > 1:
            break
        if len(collected) >= 2:
            break

    if not collected:
        return None

    collected.reverse()
    raw = " ".join(collected)

    if _has_invalid_word(raw) or not _looks_capitalized(raw):
        return None
    if _is_third_party_context(raw):
        return None

    candidate = _normalize_company_name(raw)
    return candidate if _is_valid_candidate(candidate) else None


def _iter_suffix_candidates(
    snippet: str, pattern: re.Pattern, min_words: int = 1
) -> Iterator[str]:
    """Yields normalized, validated candidates for every suffix match, in order."""
    lines = snippet.splitlines()
    line_ends = []
    pos = 0
    for line in lines:
        pos += len(line) + 1  # +1 for the stripped newline
        line_ends.append(pos)

    for match in pattern.finditer(snippet):
        raw = match.group(1)
        if len(raw.split()) < min_words:
            continue
        if _has_invalid_word(raw):
            continue

        # Find the full source line the match sits on, so context-sensitive
        # checks (third-party indicators) see the whole line, not just the
        # captured name run.
        line_idx = 0
        for line_idx, end in enumerate(line_ends):
            if match.start() < end:
                break
        context_line = lines[line_idx] if lines else raw
        if _is_third_party_context(context_line):
            continue

        normalized = _normalize_company_name(raw)
        if _is_valid_candidate(normalized):
            yield normalized


def regex_detect_company(text: str) -> Optional[str]:
    """
    Attempts to find a company name via, in order of reliability:
      1. the SEC "Exact name of Registrant" cover-page marker,
      2. unambiguous legal-suffix patterns (CORPORATION, INC, LIMITED, ...),
      3. generic legal-suffix patterns (COMPANY, GROUP, HOLDINGS, ...).

    Returns the normalized company name, or None if no confident match
    was found.
    """
    if not text:
        return None

    snippet = text[:_REGEX_SCAN_CHARS]
    ranking_text = text[:_FREQUENCY_SCAN_CHARS]

    registrant_name = _find_registrant_name(snippet)
    if registrant_name:
        return registrant_name

    safe_candidates = list(_iter_suffix_candidates(snippet, _SAFE_SUFFIX_RE))
    best = _rank_candidates(safe_candidates, ranking_text)
    if best:
        return best

    generic_candidates = list(
        _iter_suffix_candidates(snippet, _GENERIC_SUFFIX_RE, min_words=_MIN_GENERIC_WORDS)
    )
    best = _rank_candidates(generic_candidates, ranking_text)
    if best:
        return best

    return None


# ============================================================
# LLM-based detection (fallback)
# ============================================================

def llm_detect_company(text: str) -> Optional[str]:
    """
    Falls back to an LLM (Mistral) to identify the company name when the
    regex pass fails. Returns the normalized company name, or None if the
    call fails or returns nothing usable.
    """
    if not text:
        return None

    if not getattr(config, "MISTRAL_API_KEY", None):
        logger.error("MISTRAL_API_KEY is not configured; cannot run LLM fallback.")
        return None

    prompt = f"""Identify the company name from this annual report.

Rules:
- Return ONLY the company name.
- No explanation.
- No punctuation.
- No extra words.

Document:

{text[:_LLM_SCAN_CHARS]}
"""

    try:
        client, convention = _get_mistral_client(config.MISTRAL_API_KEY)
    except ImportError:
        logger.exception("LLM company detection unavailable: no usable Mistral SDK.")
        return None

    try:
        if convention == _LEGACY:
            response = client.chat(
                model=config.LLM_MODEL_ID,
                messages=[{"role": "user", "content": prompt}],
            )
        else:
            response = client.chat.complete(
                model=config.LLM_MODEL_ID,
                messages=[{"role": "user", "content": prompt}],
            )
    except Exception:
        logger.exception("LLM company detection call failed.")
        return None

    try:
        choices = getattr(response, "choices", None)
        if not choices:
            logger.warning("LLM response contained no choices.")
            return None

        raw = (choices[0].message.content or "").strip()
    except (AttributeError, IndexError):
        logger.exception("Unexpected LLM response shape.")
        return None

    if not raw:
        logger.warning("LLM returned an empty company name.")
        return None

    company = _normalize_company_name(raw)
    if not _is_valid_candidate(company):
        logger.warning("LLM returned an unusable company name: %r", raw)
        return None

    return company


# ============================================================
# Public entry point
# ============================================================

def detect_company(text: str) -> Optional[str]:
    """
    Detects the company name for a document: tries the fast regex pass
    first, falling back to the LLM only when necessary.

    Returns the normalized company name, or None if neither strategy
    could identify one.
    """
    company = regex_detect_company(text)

    logger.debug("Regex company-detection result: %s", company)
    logger.debug("Text sample: %s", text[:1000] if text else "")

    if company:
        logger.info("Regex detected company: %s", company)
        return company

    logger.info("Regex detection failed; falling back to LLM.")
    company = llm_detect_company(text)

    if company:
        logger.info("LLM detected company: %s", company)
    else:
        logger.warning("Company detection failed via both regex and LLM.")

    return company