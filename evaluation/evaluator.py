# evaluation/evaluator.py
"""
RAG benchmark evaluator.

Runs each active pipeline (vector / vectorless / hybrid) against a fixed
question set, scores the generated answers with an LLM judge, and computes
retrieval-quality metrics (company accuracy, correct-chunk count, etc.).

Known-issue fixes in this revision
-----------------------------------
1. Company-name matching in `compute_retrieval_metrics` used to require an
   exact, case-insensitive string match between the question's target
   company and each retrieved chunk's `metadata.company` field. Real-world
   metadata is rarely that clean (e.g. "Coca-Cola" vs "THE COCA-COLA
   COMPANY", "Reliance Industries" vs "RELIANCE INDUSTRIES LIMITED", ticker
   symbols like "KO"/"AAPL"/"TSLA"). The mismatch caused correct retrievals
   to be scored as 0% company accuracy / 0 correct chunks even when the
   generated answer contained the exact right figures. This is now handled
   by `_normalize_company_name` + `_company_matches`, which strips legal
   suffixes/punctuation and falls back to a small ticker/alias table.

2. The judge prompt had no explicit guidance for abstentions ("this
   information was not found..."). Because the judge only sees the
   (possibly irrelevant) retrieved context, an abstention is technically
   "faithful" to bad context and was being scored 5/5 - even when the
   context was for a completely different company. That produced
   logically-inconsistent rows: judge_score >= 4 with company_accuracy == 0
   and correct_chunks == 0. Two independent fixes address this:
     a) The rubric now explicitly tells the judge how to score abstentions
        relative to whether the context matches the target company/topic.
     b) A deterministic, code-level safety net (`_apply_abstention_guard`)
        detects abstention answers and caps their score when retrieval
        demonstrably failed (correct_chunks == 0), regardless of what the
        judge returned. The raw judge output is preserved alongside the
        corrected score for auditability.
"""

import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional
from datetime import datetime

ProgressCallback = Callable[[float, str], None]

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

import config
from utils.rate_limiter import RateLimiter

_HERE = Path(__file__).parent
QUESTIONS_PATH = _HERE / "test_questions.json"

RESULTS_DIR = _HERE / "benchmark_results" / "Judge Results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)

# Debug dumps of raw judge responses are extremely verbose. Off by default;
# enable by setting `JUDGE_DEBUG = True` in config.py.
_DEBUG = bool(getattr(config, "JUDGE_DEBUG", False))

# Number of attempts for a single judge call before giving up and recording
# a failure row. Transient network/API errors should not silently zero out
# a score without at least one retry.
_JUDGE_MAX_ATTEMPTS = int(getattr(config, "JUDGE_MAX_ATTEMPTS", 2))
_JUDGE_RETRY_BACKOFF_SECONDS = float(getattr(config, "JUDGE_RETRY_BACKOFF_SECONDS", 1.5))


# ─────────────────────────────────────────────────────────────────────────
# Judge client
# ─────────────────────────────────────────────────────────────────────────

def load_judge():
    """
    Creates a Mistral judge client for evaluation.
    """
    if not config.MISTRAL_JUDGE_API_KEY:
        raise ValueError(
            "MISTRAL_JUDGE_API_KEY not found in .env. "
            "Add your evaluation key before running the benchmark."
        )

    client = OpenAI(
        api_key=config.MISTRAL_JUDGE_API_KEY,
        base_url=config.MISTRAL_BASE_URL,
    )
    print(f"Mistral judge ready - model: {config.JUDGE_MODEL_ID}")
    return client


def _extract_text_content(content: Any) -> str:
    """
    Normalizes OpenAI/Mistral content formats into plain text.
    """
    if content is None:
        return ""

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    text = item.get("text", "")
                    if text:
                        parts.append(str(text))
            elif hasattr(item, "text"):
                text = getattr(item, "text", "")
                if text:
                    parts.append(str(text))
        return " ".join(parts).strip()

    return str(content).strip()


def _parse_judge_response(full_text: str) -> Optional[dict]:
    """
    Best-effort extraction of {"score": int, "reason": str} from model output.
    """
    text = (full_text or "").strip()
    if not text:
        return None

    # 1) Direct JSON.
    try:
        result = json.loads(text)
        return {
            "score": int(result["score"]),
            "reason": str(result["reason"]),
        }
    except Exception:
        pass

    # 2) JSON blob embedded in extra text.
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group(0))
            return {
                "score": int(result["score"]),
                "reason": str(result["reason"]),
            }
        except Exception:
            pass

    # 3) Regex fallback for truncated/partially formatted responses.
    score_match = re.search(r'"score"\s*:\s*(\d+)', text)
    reason_match = re.search(r'"reason"\s*:\s*"([^"]*)', text, re.DOTALL)

    if score_match:
        reason = "Judge reason truncated."
        if reason_match:
            partial_reason = reason_match.group(1).strip()
            if partial_reason:
                reason = partial_reason
        return {
            "score": int(score_match.group(1)),
            "reason": reason,
        }

    return None


def _clamp_score(score: int) -> int:
    """Guards against out-of-range scores from a misbehaving judge model."""
    return max(1, min(5, score))


def score_answer(judge_client, question: str, answer: str, context: str) -> dict:
    """
    Uses a higher-capability Mistral judge model to score an answer from 1 to 5.

    Retries on transient failures (network errors, malformed/empty
    responses) up to `_JUDGE_MAX_ATTEMPTS` times before returning a
    score of 0 with a diagnostic reason.
    """
    prompt = f"""You are an expert financial analyst evaluating an answer about a company 10-K annual report.

Question: {question}

Retrieved Context:
{context[:2200]}

Answer given: {answer}

Scoring rubric:
5 = Correct and specific - exact figures or facts supported by context
4 = Mostly correct - right conclusion with minor omission or imprecision
3 = Partially correct - some relevant truth but incomplete, vague, or weakly supported
2 = Mostly wrong - some overlap in topic but materially incorrect company, facts, or figures
1 = Completely wrong, unsupported, or hallucinated

Special case - the answer states the information could not be found:
- Score 1 if the Retrieved Context is about a different company, a different
  document, or an unrelated topic than the Question asks about. Do not give
  credit for correctly noticing irrelevant context; the system still failed
  to answer the question.
- Score 3 if the Retrieved Context is clearly about the right company/
  document but genuinely does not contain the specific requested fact.

Keep the reason under 18 words.

Return ONLY valid JSON in this exact schema:
{{"score": <integer 1-5>, "reason": "<one sentence>"}}"""

    last_error: Optional[str] = None

    for attempt in range(1, _JUDGE_MAX_ATTEMPTS + 1):
        try:
            response = judge_client.chat.completions.create(
                model=config.JUDGE_MODEL_ID,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a strict financial QA judge. "
                            "Return only compact JSON with no markdown."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=config.JUDGE_TEMPERATURE,
                max_tokens=config.JUDGE_MAX_TOKENS,
            )

            if not response.choices:
                last_error = "Judge response contained no choices."
                logger.warning(last_error)
                continue

            msg = response.choices[0].message

            if _DEBUG:
                print("\n===== DEBUG =====")
                print("MODEL:", config.JUDGE_MODEL_ID)
                print("MESSAGE:", msg)
                print("CONTENT TYPE:", type(msg.content))
                print("CONTENT:", repr(msg.content))
                if hasattr(msg, "reasoning_content"):
                    print("REASONING:", repr(msg.reasoning_content))
                print("=================\n")

            full_text = _extract_text_content(msg.content)
            parsed = _parse_judge_response(full_text)

            if parsed is not None:
                parsed["score"] = _clamp_score(parsed["score"])
                return parsed

            last_error = f"Parse failed - raw: {full_text[:80]!r}"
            logger.warning(
                "Could not extract judge JSON (attempt %d/%d). Raw response: %r",
                attempt, _JUDGE_MAX_ATTEMPTS, full_text[:300],
            )

        except Exception as e:
            last_error = f"Scoring failed: {e}"
            logger.warning(
                "Judge call raised an exception (attempt %d/%d): %s",
                attempt, _JUDGE_MAX_ATTEMPTS, e,
            )

        if attempt < _JUDGE_MAX_ATTEMPTS:
            time.sleep(_JUDGE_RETRY_BACKOFF_SECONDS * attempt)

    return {"score": 0, "reason": last_error or "Scoring failed for an unknown reason."}


# ─────────────────────────────────────────────────────────────────────────
# Company-name normalization
# ─────────────────────────────────────────────────────────────────────────

# Corporate-entity suffixes stripped from the tail of a normalized name.
# Applied repeatedly so "CO LTD" / "HOLDINGS INC" etc. are fully removed.
_CORP_SUFFIXES = {
    "INC", "INCORPORATED", "CORP", "CORPORATION", "CO", "COMPANY",
    "LTD", "LIMITED", "LLC", "LLP", "PLC", "GROUP", "HOLDINGS", "HOLDING",
    "SA", "NV", "AG", "SE",
}

# Leading articles stripped from the head of a normalized name
# (e.g. "THE COCA-COLA COMPANY" -> "COCA-COLA").
_LEADING_ARTICLES = {"THE"}

# Ticker symbols / short-form aliases that don't share a substring with the
# canonical company name and therefore can't be caught by suffix-stripping
# or containment matching alone. Extend this table (or override/merge via
# `config.COMPANY_ALIASES`) as new companies are added to the question set.
_DEFAULT_TICKER_ALIASES: dict[str, str] = {
    "AAPL": "APPLE",
    "TSLA": "TESLA",
    "KO": "COCA-COLA",
    "MSFT": "MICROSOFT",
    "GOOGL": "ALPHABET",
    "GOOG": "ALPHABET",
    "AMZN": "AMAZON",
    "META": "META",
    "RELIANCE": "RELIANCE INDUSTRIES",
    "RIL": "RELIANCE INDUSTRIES",
}


def _build_ticker_aliases() -> dict[str, str]:
    """
    Merges the built-in alias table with any project-specific overrides
    defined in config.py (`config.COMPANY_ALIASES: dict[str, str]`), so
    new companies can be added without touching this file.
    """
    aliases = dict(_DEFAULT_TICKER_ALIASES)
    extra = getattr(config, "COMPANY_ALIASES", None)
    if isinstance(extra, dict):
        aliases.update({str(k).upper(): str(v).upper() for k, v in extra.items()})
    return aliases


_TICKER_ALIASES = _build_ticker_aliases()

_PUNCTUATION_RE = re.compile(r"[.,'’]")
_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_company_name(name: str) -> str:
    """
    Normalizes a company name for comparison:
      - uppercases and trims
      - removes punctuation (periods, commas, apostrophes)
      - collapses whitespace
      - drops a leading "THE"
      - strips trailing corporate-entity suffixes (INC, LTD, CORP, ...)
      - resolves known ticker symbols / aliases to a canonical name

    Returns "" for empty/missing input so callers can short-circuit
    rather than accidentally matching two blank strings.
    """
    if not name:
        return ""

    cleaned = _PUNCTUATION_RE.sub("", name.upper().strip())
    cleaned = _WHITESPACE_RE.sub(" ", cleaned)
    tokens = cleaned.split(" ") if cleaned else []

    while tokens and tokens[0] in _LEADING_ARTICLES:
        tokens.pop(0)

    while tokens and tokens[-1] in _CORP_SUFFIXES:
        tokens.pop()

    normalized = " ".join(tokens).strip()
    return _TICKER_ALIASES.get(normalized, normalized)


def _company_matches(question_company: str, chunk_company: str) -> bool:
    """
    True if a retrieved chunk's company metadata refers to the same
    company as the question's target, tolerating legal-suffix and
    ticker/alias differences (e.g. "Coca-Cola" vs "THE COCA-COLA COMPANY",
    "Reliance Industries" vs "RELIANCE INDUSTRIES LIMITED", "AAPL" vs
    "Apple"). Falls back to substring containment for multi-word names.
    """
    q = _normalize_company_name(question_company)
    c = _normalize_company_name(chunk_company)

    if not q or not c:
        return False

    if q == c:
        return True

    # Guard against short/ambiguous tokens matching by accident via
    # containment (e.g. "CO" would match almost anything). Require the
    # shorter side to be a reasonably specific fragment before allowing
    # containment as a match.
    shorter, longer = (q, c) if len(q) <= len(c) else (c, q)
    if len(shorter) >= 4 and shorter in longer:
        return True

    return False


# ─────────────────────────────────────────────────────────────────────────
# Retrieval metrics
# ─────────────────────────────────────────────────────────────────────────

def compute_retrieval_metrics(question_company: str, retrieved_chunks: list[dict]) -> dict:
    """
    Computes company-match accuracy and average relevance score across a
    list of retrieved chunk dicts. Each chunk is expected to look like
    {"metadata": {"company": str, ...}, "score" or "rerank_score": float, ...}.
    """
    if not retrieved_chunks:
        return {
            "company_accuracy": 0.0,
            "avg_score": 0.0,
            "correct_chunks": 0,
            "total_chunks": 0,
        }

    companies = [
        str(c.get("metadata", {}).get("company", "")).strip()
        for c in retrieved_chunks
    ]
    correct = sum(1 for c in companies if _company_matches(question_company, c))
    total = len(retrieved_chunks)

    scores = []
    for c in retrieved_chunks:
        score = c.get("rerank_score", c.get("score", 0))
        try:
            scores.append(float(score))
        except (TypeError, ValueError):
            scores.append(0.0)

    avg_score = sum(scores) / total if total else 0.0

    return {
        "company_accuracy": round(correct / total, 3) if total else 0.0,
        "avg_score": round(avg_score, 4),
        "correct_chunks": correct,
        "total_chunks": total,
    }


def _extract_context_strings(retrieved_chunks: list[dict]) -> list[str]:
    """
    Converts a list of retrieved chunk dicts into a flat list of plain text
    strings suitable for RAGAS. Each chunk is expected to have a 'text' key
    (set by the parent-child swap) or falls back to 'content'.
    """
    texts: list[str] = []
    for chunk in retrieved_chunks:
        text = chunk.get("text") or chunk.get("content") or ""
        if isinstance(text, str) and text.strip():
            texts.append(text.strip())
    return texts if texts else ["No context retrieved."]


# ─────────────────────────────────────────────────────────────────────────
# Abstention safety net
# ─────────────────────────────────────────────────────────────────────────

# Phrases indicating the pipeline declined to answer because the retrieved
# context didn't contain the needed information. Matched case-insensitively
# against the full answer text.
_ABSTENTION_RE = re.compile(
    r"(not found in|could not find|cannot find|couldn't find|"
    r"no information (?:was|is) (?:found|available)|"
    r"does not contain|doesn't contain|"
    r"could not determine|cannot determine|"
    r"unable to find|unable to determine|"
    r"not (?:present|available) in the retrieved)",
    re.IGNORECASE,
)


def _is_abstention(answer: str) -> bool:
    """True if the answer text reads as a 'couldn't find the info' response."""
    return bool(_ABSTENTION_RE.search(answer or ""))


def _apply_abstention_guard(
    answer: str, judge_score: int, judge_reason: str, correct_chunks: int, total_chunks: int
) -> tuple[int, str, bool]:
    """
    Deterministic, code-level correction that runs after the LLM judge.

    Problem: an abstention ("this information was not found...") is
    technically faithful to whatever context it was given, so a judge can
    rate it 5/5 even when that context is for the wrong company entirely -
    producing rows where judge_score >= 4 but company_accuracy == 0 and
    correct_chunks == 0 (a logical inconsistency: a "perfect" answer from a
    retrieval that completely missed).

    Fix: if the answer is an abstention AND retrieval produced zero chunks
    matching the target company, the system failed end-to-end regardless of
    how gracefully it failed. Cap the score at 1 in that case. The judge's
    rubric already instructs it to do this (see `score_answer`), so this is
    a defense-in-depth guard for cases where the judge doesn't comply, not
    the primary fix.

    Returns (final_score, final_reason, was_adjusted).
    """
    if not _is_abstention(answer):
        return judge_score, judge_reason, False

    if correct_chunks > 0:
        # Right company/document was retrieved but the specific fact was
        # genuinely missing - a legitimate abstention. Leave the judge's
        # score as-is.
        return judge_score, judge_reason, False

    if judge_score >= 2:
        adjusted_reason = (
            f"[score capped: abstention + retrieval mismatch, "
            f"{total_chunks} chunk(s) retrieved, 0 matched target company] "
            f"original judge reason: {judge_reason}"
        )
        return 1, adjusted_reason, True

    return judge_score, judge_reason, False


# ─────────────────────────────────────────────────────────────────────────
# Main evaluation loop
# ─────────────────────────────────────────────────────────────────────────

def run_evaluation(
    vector_pipeline,
    vectorless_pipeline,
    hybrid_pipeline=None,          # ← optional; pass None to skip
    results_filename: str = "full_results.csv",
    capture_contexts: bool = False,
    progress_callback: Optional[ProgressCallback] = None,
) -> tuple[pd.DataFrame, dict] | pd.DataFrame:
    """
    Runs all active pipelines on every question with rate limiting.

    Per question flow (3-method run):
    1. [generation limiter] → vector_pipeline.ask()
    2. [generation limiter] → vectorless_pipeline.ask()
    3. [generation limiter] → hybrid_pipeline.ask()
    4. [judge limiter]      → score_answer() × 3

    Pass hybrid_pipeline=None to run only vector + vectorless (original behaviour).

    Parameters
    ----------
    capture_contexts : bool
        When True, also returns a dict keyed by (question_id, method) whose
        values are lists of plain-text context strings.  This dict is used by
        run_ragas_evaluation() to avoid re-running retrieval.
        Default is False to preserve the original return type (pd.DataFrame only).
    progress_callback : Optional[Callable[[float, str], None]]
        Invoked after every question with (fraction_complete in [0, 1], message).
        Safe to leave as None for headless/CLI use.

    Returns
    -------
    If capture_contexts=False (default):
        pd.DataFrame  — the judge results (original behaviour, unchanged)
    If capture_contexts=True:
        (pd.DataFrame, dict)  — judge results + retrieved_contexts_map
    """
    if not QUESTIONS_PATH.exists():
        raise FileNotFoundError(f"Questions file not found: {QUESTIONS_PATH}")

    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        data = json.load(f)
        questions = data.get("questions", [])

    judge_model = load_judge()

    generation_limiter = RateLimiter(max_rpm=config.MISTRAL_RPM,       name="Mistral generation")
    judge_limiter      = RateLimiter(max_rpm=config.MISTRAL_JUDGE_RPM, name="Mistral judge")

    # Build the active pipeline list dynamically
    active_pipelines = [
        ("vector",     vector_pipeline),
        ("vectorless", vectorless_pipeline),
    ]
    if hybrid_pipeline is not None:
        active_pipelines.append(("hybrid", hybrid_pipeline))

    n_methods = len(active_pipelines)
    records   = []
    # context capture: {(qid, method): [str, ...]}
    retrieved_contexts_map: dict[tuple[str, str], list[str]] = {}
    total     = len(questions)

    def _emit(fraction: float, message: str) -> None:
        if progress_callback is not None:
            try:
                progress_callback(min(max(fraction, 0.0), 1.0), message)
            except Exception:
                pass  # progress reporting must never break the benchmark run

    _emit(0.0, f"Starting judge benchmark — {total} questions × {n_methods} methods")

    print(f"\n{'=' * 55}")
    print(f"   PHASE 5 - EVALUATION ({total} questions × {n_methods} methods)")
    print(f"   Methods         : {', '.join(m for m, _ in active_pipelines)}")
    print(f"   Generation limit: {config.MISTRAL_RPM} RPM")
    print(f"   Judge limit     : {config.MISTRAL_JUDGE_RPM} RPM")
    if capture_contexts:
        print("   Context capture : ON  (for RAGAS)")
    print(f"{'=' * 55}\n")

    from llm import format_context

    for i, q in enumerate(tqdm(questions, desc="Questions"), 1):
        qid      = q.get("id", i)
        question = q.get("question", "")
        company  = q.get("company", "")
        category = q.get("category", "")

        print(f"\n[{i}/{total}] {qid} - {company}")

        # ── Generation pass ───────────────────────────────────────────────────
        pipeline_results = {}
        for method_name, pipeline in active_pipelines:
            generation_limiter.wait()
            try:
                pipeline_results[method_name] = pipeline.ask(question)
            except Exception as e:
                print(f"   {method_name} pipeline error: {e}")
                pipeline_results[method_name] = {
                    "answer"         : f"ERROR: {e}",
                    "retrieved"      : [],
                    "retrieval_time" : 0,
                    "rerank_time"    : 0,
                    "generation_time": 0,
                    "total_time"     : 0,
                }

        # ── Capture context strings for RAGAS ─────────────────────────────────
        if capture_contexts:
            for method_name, result in pipeline_results.items():
                key = (str(qid), method_name)
                retrieved_contexts_map[key] = _extract_context_strings(
                    result.get("retrieved", [])
                )

        # ── Judge pass ────────────────────────────────────────────────────────
        judge_scores = {}
        for method_name, result in pipeline_results.items():
            context = format_context(result.get("retrieved", []))
            judge_limiter.wait()
            judge_scores[method_name] = score_answer(
                judge_model, question, result.get("answer", ""), context
            )
            print(
                f"   {method_name:<12} score: "
                f"{judge_scores[method_name]['score']}/5 - "
                f"{judge_scores[method_name]['reason'][:60]}"
            )

        # ── Record rows ───────────────────────────────────────────────────────
        for method_name, result in pipeline_results.items():
            judge       = judge_scores[method_name]
            answer      = result.get("answer", "")
            ret_metrics = compute_retrieval_metrics(company, result.get("retrieved", []))

            final_score, final_reason, was_adjusted = _apply_abstention_guard(
                answer,
                judge["score"],
                judge["reason"],
                ret_metrics["correct_chunks"],
                ret_metrics["total_chunks"],
            )

            if was_adjusted:
                print(
                    f"   {method_name:<12} ⚠ score adjusted "
                    f"{judge['score']} → {final_score} (abstention + retrieval mismatch)"
                )

            records.append(
                {
                    "id"                : qid,
                    "company"           : company,
                    "category"          : category,
                    "method"            : method_name,
                    "question"          : question,
                    "answer"            : answer,
                    "judge_score"       : final_score,
                    "judge_reason"      : final_reason,
                    "judge_score_raw"   : judge["score"],
                    "judge_reason_raw"  : judge["reason"],
                    "score_adjusted"    : was_adjusted,
                    "pass"              : final_score >= 3,
                    "company_accuracy"  : ret_metrics["company_accuracy"],
                    "avg_chunk_score"   : ret_metrics["avg_score"],
                    "correct_chunks"    : ret_metrics["correct_chunks"],
                    "total_chunks"      : ret_metrics["total_chunks"],
                    "retrieval_time"    : result.get("retrieval_time", 0),
                    "rerank_time"       : result.get("rerank_time", result.get("rerank_latency", 0)),
                    "generation_time"   : result.get("generation_time", 0),
                    "total_time"        : result.get("total_time", 0),
                }
            )

        _emit(i / total, f"Judged question {i}/{total} — {company} ({qid})")

    _emit(1.0, "Judge benchmark complete")

    df = pd.DataFrame(records)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = Path(results_filename).stem
    extension = Path(results_filename).suffix or ".csv"
    timestamped_filename = f"{base_name}_{timestamp}{extension}"
    out_path = RESULTS_DIR / timestamped_filename
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\nResults saved → {out_path}")

    n_adjusted = int(df["score_adjusted"].sum()) if "score_adjusted" in df.columns else 0
    if n_adjusted:
        print(
            f"   Note: {n_adjusted} row(s) had their judge score capped due to "
            f"abstention + retrieval mismatch. See 'judge_score_raw' / "
            f"'judge_reason_raw' columns for the original judge output."
        )

    if capture_contexts:
        return df, retrieved_contexts_map
    return df


def print_summary(df: pd.DataFrame):
    if df.empty:
        print("No results to summarize.")
        return

    print(f"\n{'=' * 55}")
    print("   EVALUATION SUMMARY")
    print(f"{'=' * 55}")

    # Discover methods present in data (preserves insertion order in Python 3.7+)
    methods_in_data = list(dict.fromkeys(df["method"].tolist()))

    method_labels = {
        "vector"    : "Vector RAG",
        "vectorless": "Vectorless RAG",
        "hybrid"    : "Hybrid RAG",
    }

    for method in methods_in_data:
        label = method_labels.get(method, method.title())
        m = df[df["method"] == method]
        if m.empty:
            print(f"\n{label}: no rows")
            continue

        print(f"\n{'-' * 55}")
        print(f"  {label}")
        print(f"{'-' * 55}")
        print(f"  Avg judge score    : {m['judge_score'].mean():.2f} / 5")
        print(f"  Pass rate  (>=3)   : {m['pass'].mean() * 100:.1f}%")
        print(f"  Company accuracy   : {m['company_accuracy'].mean() * 100:.1f}%")
        if "score_adjusted" in m.columns and m["score_adjusted"].any():
            n = int(m["score_adjusted"].sum())
            print(f"  Scores auto-capped : {n} (abstention + retrieval mismatch)")
        print(f"  Avg retrieval time : {m['retrieval_time'].mean():.4f}s")
        rerank_mean = m["rerank_time"].mean() if "rerank_time" in m.columns else 0.0
        print(f"  Avg rerank time    : {rerank_mean:.4f}s")
        print(f"  Avg generation time: {m['generation_time'].mean():.2f}s")
        print(f"  Avg total latency  : {m['total_time'].mean():.2f}s")

    print(f"\n{'-' * 55}")
    print("  Score by category")
    print(f"{'-' * 55}")
    pivot = df.groupby(["category", "method"])["judge_score"].mean().unstack()
    print(pivot.round(2).to_string())

    print(f"\n{'-' * 55}")
    print("  Score by company")
    print(f"{'-' * 55}")
    pivot2 = df.groupby(["company", "method"])["judge_score"].mean().unstack()
    print(pivot2.round(2).to_string())
    print(f"\n{'=' * 55}\n")