# evaluation/ragas_evaluator.py
"""
RAGAS evaluation layer for the three-way RAG benchmark.

Adds five RAGAS metrics on top of the existing Mistral judge evaluation:
  - answer_relevancy      → ResponseRelevancy: is the answer on-topic for the question?
  - faithfulness          → Faithfulness: is the answer supported by retrieved context?
  - context_precision     → ContextualPrecision: are the most relevant chunks ranked first?
  - context_recall        → ContextualRecall: does the context cover the reference answer?
  - context_relevance     → ContextualRelevance: mapped to context_precision (no standalone
                            context_relevance metric in RAGAS 0.2.x; precision captures the
                            same signal of whether the retrieved set is relevant to the query)

NOTE ON DEPENDENCIES
  This module requires RAGAS 0.2.x with a pinned langchain ecosystem:
    pip install "ragas==0.2.15" "langchain-community==0.0.38" "langchain-openai==0.0.8"

  RAGAS is wired to Mistral via the OpenAI-compatible endpoint so no separate API key
  is needed beyond the existing MISTRAL_JUDGE_API_KEY already in your .env.

NOTE ON REFERENCE ANSWERS
  context_precision and context_recall both require a reference answer.
  If test_questions.json includes a "reference_answer" field per question, those are used.
  If not, both metrics are skipped for that question and the columns are set to NaN.

USAGE (from notebooks/05_three_way_comparison.ipynb)
  from evaluation.ragas_evaluator import run_ragas_evaluation, merge_results

  # After run_evaluation() returns judge_df:
  ragas_df   = run_ragas_evaluation(judge_df, questions_path=QUESTIONS_PATH)
  combined   = merge_results(judge_df, ragas_df)
  combined.to_csv(RESULTS_DIR / "three_way_combined_results.csv", index=False)
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Callable, Optional
from datetime import datetime

ProgressCallback = Callable[[float, str], None]

import langchain  # noqa: F401 – silence attribute warning before any import
langchain.verbose = False  # type: ignore[attr-defined]

import pandas as pd
from tqdm import tqdm

# ── Suppress RAGAS internal noise ─────────────────────────────────────────────
# ragas.executor logs "Exception raised in Job[N]: OutputParserException(...)"
# at ERROR level for every LLM response that isn't perfectly formatted JSON.
# These are handled internally (the metric retries or returns NaN) and are not
# actionable — suppressing them keeps notebook output readable.
logging.getLogger("ragas.executor").setLevel(logging.CRITICAL)
# Also suppress verbose langchain_core tracing lines that RAGAS triggers
logging.getLogger("langchain_core.tracers").setLevel(logging.CRITICAL)
logging.getLogger("langchain_core.callbacks").setLevel(logging.CRITICAL)

# ── langchain / RAGAS imports ─────────────────────────────────────────────────
try:

    from langchain_openai import ChatOpenAI

    from ragas import evaluate as ragas_evaluate

    from ragas.dataset_schema import (
        EvaluationDataset,
        SingleTurnSample
    )

    from ragas.embeddings import (
        BaseRagasEmbeddings
    )

    from ragas.llms import (
        LangchainLLMWrapper
    )

    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    from ragas.run_config import (
        RunConfig
    )

    _RAGAS_AVAILABLE = True

except Exception as _err:

    _RAGAS_AVAILABLE = False

    _RAGAS_IMPORT_ERROR = str(_err)

# ── project imports ───────────────────────────────────────────────────────────
sys.path.append(str(Path(__file__).parent.parent))
import config  # noqa: E402

_HERE = Path(__file__).parent
QUESTIONS_PATH = _HERE / "test_questions.json"

RESULTS_DIR = _HERE / "benchmark_results" / "RAGAS Results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _check_ragas() -> None:
    if not _RAGAS_AVAILABLE:
        raise ImportError(
            "RAGAS or its LangChain dependencies are not installed.\n"
            "Run: pip install 'ragas==0.2.15' 'langchain-community==0.0.38' "
            "'langchain-openai==0.0.8'\n"
            f"Original error: {_RAGAS_IMPORT_ERROR}"
        )


def _build_ragas_llm() -> LangchainLLMWrapper:
    """
    Returns a RAGAS-compatible LLM wrapper that calls Mistral via the
    OpenAI-compatible endpoint.  Uses MISTRAL_JUDGE_API_KEY from config.
    """
    if not config.MISTRAL_JUDGE_API_KEY:
        raise ValueError(
            "MISTRAL_JUDGE_API_KEY not set in .env. "
            "RAGAS needs a judge LLM to score metrics."
        )
    langchain_llm = ChatOpenAI(
        openai_api_key=config.MISTRAL_JUDGE_API_KEY,
        openai_api_base=config.MISTRAL_BASE_URL,   # "https://api.mistral.ai/v1"
        model_name=config.JUDGE_MODEL_ID,           # e.g. "mistral-medium-latest"
        temperature=0,
        max_retries=3,
    )
    return LangchainLLMWrapper(langchain_llm)


class _MistralEmbeddings(BaseRagasEmbeddings):
    """
    RAGAS-compatible embeddings wrapper that calls Mistral's /v1/embeddings
    endpoint using the openai client already present in the project.

    Mistral's embedding model is mistral-embed (1024-dim).  It accepts the
    same request format as OpenAI's embeddings API, so we can reuse the
    openai.OpenAI client with a custom base_url.

    No new dependencies — openai is already a project requirement via llm.py.
    """

    # Required by BaseRagasEmbeddings (pydantic dataclass field)
    run_config: RunConfig = None  # type: ignore[assignment]

    def __init__(self, api_key: str, base_url: str, model: str = "mistral-embed"):
        super().__init__()
        from openai import OpenAI as _OpenAI
        self._client = _OpenAI(api_key=api_key, base_url=base_url)
        self._model  = model
        self.run_config = RunConfig()

    def _embed(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(input=texts, model=self._model)
        return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]

    # ── sync interface ────────────────────────────────────────────────────────
    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    # ── async interface (RAGAS uses these internally) ─────────────────────────
    async def aembed_query(self, text: str) -> list[float]:
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_query, text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_documents, texts)

    def set_run_config(self, run_config: RunConfig) -> None:
        self.run_config = run_config

    def __repr__(self) -> str:
        return f"_MistralEmbeddings(model={self._model})"


def _build_ragas_embeddings() -> "_MistralEmbeddings":
    """
    Returns a RAGAS-compatible embeddings object backed by Mistral's
    mistral-embed model.  Uses MISTRAL_JUDGE_API_KEY from config.

    answer_relevancy is the only metric that requires embeddings — it computes
    cosine similarity between the question and a set of question variants
    generated from the answer.  Using Mistral embeddings instead of OpenAI
    embeddings keeps everything within the same API key already in .env.
    """
    if not config.MISTRAL_JUDGE_API_KEY:
        raise ValueError(
            "MISTRAL_JUDGE_API_KEY not set in .env. "
            "RAGAS needs this key for both the judge LLM and embeddings."
        )
    return _MistralEmbeddings(
        api_key  = config.MISTRAL_JUDGE_API_KEY,
        base_url = config.MISTRAL_BASE_URL,
        model    = "mistral-embed",
    )


def _load_reference_map(questions_path: Path) -> dict[str, str]:
    """
    Reads test_questions.json and returns {question_id: reference_answer}.
    Questions without a 'reference_answer' key are silently omitted.
    """
    if not questions_path.exists():
        return {}
    with open(questions_path, encoding="utf-8") as f:
        data = json.load(f)
    ref_map: dict[str, str] = {}
    for q in data.get("questions", []):
        ref = q.get("reference_answer") or q.get("reference") or ""
        if ref:
            qid = str(q.get("id", q.get("question", "")))
            ref_map[qid] = ref
    return ref_map


def _contexts_from_row(row: pd.Series) -> list[str]:
    """
    Extracts a flat list of context strings from the 'retrieved' payload stored
    in the judge DataFrame.

    The judge evaluator stores retrieved chunks as a list[dict] where each dict
    has at least a 'text' key.  The raw Python list is serialised as a string
    in the CSV; we try to re-parse it here.  Falls back to splitting on
    double-newlines if the value is already a plain string.
    """
    raw = row.get("retrieved_contexts_raw") or row.get("answer", "")

    if isinstance(raw, list):
        # Direct Python list of chunk dicts or strings
        texts: list[str] = []
        for item in raw:
            if isinstance(item, dict):
                texts.append(str(item.get("text", item.get("content", ""))))
            else:
                texts.append(str(item))
        return [t for t in texts if t.strip()]

    if isinstance(raw, str) and raw.startswith("["):
        try:
            parsed = json.loads(raw)
            return _contexts_from_row({"retrieved_contexts_raw": parsed})
        except json.JSONDecodeError:
            pass

    # Plain string fallback – not ideal but prevents hard failure
    return [raw] if raw.strip() else ["No context available."]


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def build_ragas_dataset(
    judge_df: pd.DataFrame,
    retrieved_contexts_map: dict[str, list[str]],
    questions_path: Path = QUESTIONS_PATH,
) -> tuple[EvaluationDataset, list[dict]]:
    """
    Converts judge results + the retrieved-context lookup into a RAGAS
    EvaluationDataset.

    Parameters
    ----------
    judge_df:
        The DataFrame returned by ``run_evaluation()``.  Must have columns:
        id, method, question, answer.
    retrieved_contexts_map:
        ``{(question_id, method): [context_str, ...]}`` built during the
        evaluation loop so that raw context strings are available here.
    questions_path:
        Path to test_questions.json.  Used to fetch reference answers when
        available.

    Returns
    -------
    dataset:
        A RAGAS EvaluationDataset ready to pass to ``ragas.evaluate()``.
    row_metadata:
        A list of dicts (one per sample) carrying id/method/question/company/
        category so scores can be joined back to the full DataFrame.
    """
    _check_ragas()
    ref_map = _load_reference_map(questions_path)
    samples: list[SingleTurnSample] = []
    row_metadata: list[dict] = []

    for _, row in judge_df.iterrows():
        qid    = str(row.get("id", ""))
        method = str(row.get("method", ""))
        key    = (qid, method)
        ctxs   = retrieved_contexts_map.get(key, [])

        # Graceful fallback when contexts were not captured
        if not ctxs:
            ctxs = ["No context was captured for this question."]

        reference = ref_map.get(qid) or None  # None → metrics that need it are skipped

        sample = SingleTurnSample(
            user_input=str(row.get("question", "")),
            response=str(row.get("answer", "")),
            retrieved_contexts=ctxs,
            reference=reference,
        )
        samples.append(sample)
        row_metadata.append(
            {
                "id"      : qid,
                "method"  : method,
                "question": str(row.get("question", "")),
                "company" : str(row.get("company", "")),
                "category": str(row.get("category", "")),
            }
        )

    dataset = EvaluationDataset(samples=samples)
    return dataset, row_metadata


def run_ragas_evaluation(
    judge_df: pd.DataFrame,
    retrieved_contexts_map: dict[str, list[str]],
    questions_path: Path = QUESTIONS_PATH,
    results_filename: str = "three_way_ragas_results.csv",
    batch_size: int = 10,
    progress_callback: Optional[ProgressCallback] = None,
) -> pd.DataFrame:
    """
    Runs RAGAS in batches of `batch_size` so progress can be reported in real
    time via `progress_callback(fraction_complete, message)`. Base metrics
    (answer_relevancy, faithfulness) run for every row; reference-dependent
    metrics (context_precision, context_recall) run only for rows whose
    question has a `reference_answer` in test_questions.json.
    """
    _check_ragas()

    def _emit(fraction: float, message: str) -> None:
        if progress_callback is not None:
            try:
                progress_callback(min(max(fraction, 0.0), 1.0), message)
            except Exception:
                pass  # progress reporting must never break the benchmark run

    ragas_llm  = _build_ragas_llm()
    ragas_embs = _build_ragas_embeddings()
    ref_map    = _load_reference_map(questions_path)
    has_refs   = bool(ref_map)

    print(f"\n{'=' * 55}")
    print("   RAGAS EVALUATION")
    print(f"   Total Evaluation Rows : {len(judge_df)}")
    print(f"   Batch size            : {batch_size}")
    print(f"   Judge LLM             : {config.JUDGE_MODEL_ID}")
    print(f"   Embeddings            : mistral-embed  (Mistral /v1/embeddings)")
    print(f"   Metrics (Base)        : answer_relevancy, faithfulness")
    if has_refs:
        print(f"   Metrics (With Refs)   : context_precision, context_recall")
    print(f"{'=' * 55}\n")

    base_metrics = [answer_relevancy, faithfulness]
    ref_metrics  = [context_precision, context_recall]

    samples: list[SingleTurnSample] = []
    meta: list[dict] = []

    for _, row in judge_df.iterrows():
        qid    = str(row.get("id", ""))
        method = str(row.get("method", ""))
        ctxs   = retrieved_contexts_map.get((qid, method), [])
        if not ctxs:
            ctxs = ["No context was captured for this question."]

        reference = ref_map.get(qid) or None

        samples.append(
            SingleTurnSample(
                user_input          = str(row.get("question", "")),
                response             = str(row.get("answer", "")),
                retrieved_contexts   = ctxs,
                reference            = reference,
            )
        )
        meta.append(
            {
                "id"       : qid,
                "method"   : method,
                "question" : str(row.get("question", "")),
                "company"  : str(row.get("company", "")),
                "category" : str(row.get("category", "")),
                "_has_ref" : reference is not None,
            }
        )

    total_rows = len(samples)
    ref_row_count = sum(1 for m in meta if m["_has_ref"]) if has_refs else 0
    # Weighted progress units: base pass counts once per row, ref pass counts
    # once more for rows that have a reference answer.
    total_units = total_rows + ref_row_count
    completed_units = 0

    _emit(0.0, f"Starting RAGAS evaluation — {total_rows} rows in batches of {batch_size}")

    base_scores: list[dict] = [None] * total_rows
    ref_scores: dict[int, dict] = {}

    n_batches = max(1, (total_rows + batch_size - 1) // batch_size)

    # ── Pass 1: base metrics (answer_relevancy, faithfulness) ────────────────
    for batch_idx, start in enumerate(range(0, total_rows, batch_size), 1):
        end = min(start + batch_size, total_rows)
        batch_samples = samples[start:end]
        batch_dataset = EvaluationDataset(samples=batch_samples)

        try:
            result = ragas_evaluate(
                dataset          = batch_dataset,
                metrics          = base_metrics,
                llm              = ragas_llm,
                embeddings       = ragas_embs,
                raise_exceptions = False,
                show_progress    = False,
            )
            batch_df = result.to_pandas()
        except Exception as exc:
            print(f"   Warning: base metrics failed for batch {batch_idx}/{n_batches}: {exc}")
            batch_df = pd.DataFrame(
                {
                    "answer_relevancy": [float("nan")] * len(batch_samples),
                    "faithfulness"    : [float("nan")] * len(batch_samples),
                }
            )

        for local_idx, global_idx in enumerate(range(start, end)):
            base_scores[global_idx] = {
                "answer_relevancy": float(batch_df.iloc[local_idx].get("answer_relevancy", float("nan"))),
                "faithfulness"    : float(batch_df.iloc[local_idx].get("faithfulness", float("nan"))),
            }

        completed_units += len(batch_samples)
        _emit(
            completed_units / total_units,
            f"Scoring base metrics — batch {batch_idx}/{n_batches} "
            f"({end}/{total_rows} rows)",
        )

    # ── Pass 2: reference-dependent metrics (context_precision/recall) ──────
    ref_indices = [i for i, m in enumerate(meta) if m["_has_ref"]]

    if ref_indices and has_refs:
        n_ref_batches = max(1, (len(ref_indices) + batch_size - 1) // batch_size)
        for batch_idx, start in enumerate(range(0, len(ref_indices), batch_size), 1):
            batch_global_idx = ref_indices[start:start + batch_size]
            batch_samples = [samples[i] for i in batch_global_idx]
            batch_dataset = EvaluationDataset(samples=batch_samples)

            try:
                result = ragas_evaluate(
                    dataset          = batch_dataset,
                    metrics          = ref_metrics,
                    llm              = ragas_llm,
                    embeddings       = ragas_embs,
                    raise_exceptions = False,
                    show_progress    = False,
                )
                batch_df = result.to_pandas()
                for local_idx, global_idx in enumerate(batch_global_idx):
                    ref_scores[global_idx] = {
                        "context_precision": float(batch_df.iloc[local_idx].get("context_precision", float("nan"))),
                        "context_recall"   : float(batch_df.iloc[local_idx].get("context_recall", float("nan"))),
                    }
            except Exception as exc:
                print(f"   Warning: reference metrics failed for batch {batch_idx}/{n_ref_batches}: {exc}")

            completed_units += len(batch_samples)
            _emit(
                completed_units / total_units,
                f"Scoring reference metrics — batch {batch_idx}/{n_ref_batches}",
            )

    # ── Compile final records ────────────────────────────────────────────────
    records: list[dict] = []
    for i, m in enumerate(meta):
        base = base_scores[i] or {"answer_relevancy": float("nan"), "faithfulness": float("nan")}
        ref  = ref_scores.get(i, {"context_precision": float("nan"), "context_recall": float("nan")})
        records.append(
            {
                "id"                  : m["id"],
                "method"              : m["method"],
                "question"            : m["question"],
                "company"             : m["company"],
                "category"            : m["category"],
                "answer_relevancy"    : base["answer_relevancy"],
                "faithfulness"        : base["faithfulness"],
                "context_precision"   : ref["context_precision"],
                "context_recall"      : ref["context_recall"],
                "contextual_relevancy": ref["context_precision"],
            }
        )

    df = pd.DataFrame(records)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = Path(results_filename).stem
    extension = Path(results_filename).suffix or ".csv"
    timestamped_filename = f"{base_name}_{timestamp}{extension}"
    out = RESULTS_DIR / timestamped_filename
    df.to_csv(out, index=False, encoding="utf-8")
    # latest copy
    latest_out = RESULTS_DIR / "latest_ragas_results.csv"
    df.to_csv(latest_out, index=False, encoding="utf-8")
    _emit(1.0, "RAGAS evaluation complete")
    print(f"\nRAGAS evaluation complete! Saved → {out}")
    return df

def merge_results(
    judge_df: pd.DataFrame,
    ragas_df: pd.DataFrame,
    combined_filename: str = "three_way_combined_results.csv",
) -> pd.DataFrame:
    """
    Merges judge scores and RAGAS scores into a single master DataFrame.

    Joins on (id, method).  Columns from both DataFrames are retained;
    RAGAS-only columns are appended to the right.

    The combined CSV is saved to evaluation/results/{combined_filename}.
    """
    ragas_cols = [
        "id", "method",
        "answer_relevancy", "faithfulness",
        "context_precision", "context_recall", "contextual_relevancy",
    ]
    ragas_slim = ragas_df[[c for c in ragas_cols if c in ragas_df.columns]]

    combined = judge_df.merge(ragas_slim, on=["id", "method"], how="left")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / combined_filename
    combined.to_csv(out, index=False, encoding="utf-8")
    print(f"Combined results saved → {out}")
    return combined


def print_ragas_summary(ragas_df: pd.DataFrame) -> None:
    """Prints a compact per-method RAGAS metric summary."""
    if ragas_df.empty:
        print("No RAGAS results to summarise.")
        return

    metric_cols = [
        "answer_relevancy", "faithfulness",
        "context_precision", "context_recall", "contextual_relevancy",
    ]
    cols = [c for c in metric_cols if c in ragas_df.columns]

    print(f"\n{'=' * 55}")
    print("   RAGAS SUMMARY")
    print(f"{'=' * 55}")
    for method in ragas_df["method"].unique():
        m = ragas_df[ragas_df["method"] == method]
        print(f"\n  {method.title()} RAG")
        print(f"  {'-' * 30}")
        for col in cols:
            val = m[col].mean(skipna=True)
            label = col.replace("_", " ").title()
            flag = "  (alias: context_precision)" if col == "contextual_relevancy" else ""
            print(f"  {label:<28}: {val:.4f}{flag}")
    print(f"\n{'=' * 55}\n")