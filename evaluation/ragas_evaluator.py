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
from typing import Optional

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
RESULTS_DIR = _HERE / "results"


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


# def run_ragas_evaluation(
#     judge_df: pd.DataFrame,
#     retrieved_contexts_map: dict[str, list[str]],
#     questions_path: Path = QUESTIONS_PATH,
#     results_filename: str = "three_way_ragas_results.csv",
#     batch_size: int = 10,
# ) -> pd.DataFrame:
#     """
#     Runs RAGAS on the same per-question data used for judge scoring.

#     Metrics computed
#     ----------------
#     answer_relevancy    – Is the answer on-topic for the question?
#                           (maps to "Answer Relevancy" in the spec)
#     faithfulness        – Is every claim in the answer grounded in the
#                           retrieved context?
#                           (maps to "Faithfulness")
#     context_precision   – Are the most relevant chunks ranked highest in the
#                           retrieved list?
#                           (maps to "Contextual Precision" AND
#                            "Contextual Relevancy" – no standalone
#                            context_relevancy metric exists in RAGAS 0.2.x)
#     context_recall      – Does the retrieved context contain enough information
#                           to produce the reference answer?
#                           (maps to "Contextual Recall"; requires reference_answer
#                           in test_questions.json – NaN when absent)

#     Parameters
#     ----------
#     judge_df:
#         DataFrame returned by ``run_evaluation()``.
#     retrieved_contexts_map:
#         ``{(question_id, method): [context_str, ...]}`` built during the loop.
#     questions_path:
#         Path to test_questions.json.
#     results_filename:
#         CSV filename saved under evaluation/results/.
#     batch_size:
#         How many samples to send to RAGAS per batch.  Lower values are safer
#         under strict rate limits.

#     Returns
#     -------
#     pd.DataFrame with columns:
#         id, method, question, company, category,
#         answer_relevancy, faithfulness,
#         context_precision, context_recall,
#         contextual_relevancy (alias of context_precision)
#     """
#     _check_ragas()

#     ragas_llm  = _build_ragas_llm()
#     ragas_embs = _build_ragas_embeddings()
#     ref_map    = _load_reference_map(questions_path)
#     has_refs  = bool(ref_map)

#     n_unique_questions = len(judge_df["id"].unique()) if "id" in judge_df.columns else "?"

#     print(f"\n{'=' * 55}")
#     print("   RAGAS EVALUATION")
#     print(f"   Rows          : {len(judge_df)}")
#     print(f"   Batch size    : {batch_size}")
#     print(f"   Judge LLM     : {config.JUDGE_MODEL_ID}")
#     print(f"   Embeddings    : mistral-embed  (Mistral /v1/embeddings)")
#     print(f"   Always on      : answer_relevancy, faithfulness")
#     if has_refs:
#         print(f"   With refs      : context_precision, context_recall")
#         print(f"   Refs loaded    : {len(ref_map)} / {n_unique_questions} questions")
#     else:
#         print(f"   With refs      : context_precision, context_recall  →  SKIPPED (NaN)")
#         print(f"   ⚠  No reference_answer fields found in test_questions.json.")
#         print(f"      Run first:  python evaluation/generate_references.py")
#         print(f"      Then re-run this cell to get real context metric scores.")
#     print(f"{'=' * 55}\n")

#     # Decide which metrics to run
#     # context_precision and context_recall need a reference → only when refs exist
#     base_metrics = [answer_relevancy, faithfulness]
#     ref_metrics  = [context_precision, context_recall]

#     records: list[dict] = []

#     # Process in batches to respect rate limits
#     rows = list(judge_df.iterrows())
#     n    = len(rows)

#     for batch_start in tqdm(range(0, n, batch_size), desc="RAGAS batches"):
#         batch_rows = rows[batch_start : batch_start + batch_size]
#         samples: list[SingleTurnSample] = []
#         meta: list[dict] = []

#         for _, row in batch_rows:
#             qid    = str(row.get("id", ""))
#             method = str(row.get("method", ""))
#             ctxs   = retrieved_contexts_map.get((qid, method), [])
#             if not ctxs:
#                 ctxs = ["No context was captured for this question."]

#             reference = ref_map.get(qid) or None

#             samples.append(
#                 SingleTurnSample(
#                     user_input       = str(row.get("question", "")),
#                     response         = str(row.get("answer", "")),
#                     retrieved_contexts = ctxs,
#                     reference        = reference,
#                 )
#             )
#             meta.append(
#                 {
#                     "id"         : qid,
#                     "method"     : method,
#                     "question"   : str(row.get("question", "")),
#                     "company"    : str(row.get("company", "")),
#                     "category"   : str(row.get("category", "")),
#                     "_has_ref"   : reference is not None,
#                 }
#             )

#         dataset = EvaluationDataset(samples=samples)

#         # ── Run base metrics (no reference required) ──────────────────────────
#         try:
#             base_result = ragas_evaluate(
#                 dataset          = dataset,
#                 metrics          = base_metrics,
#                 llm              = ragas_llm,
#                 embeddings       = ragas_embs,
#                 raise_exceptions = False,
#                 show_progress    = False,
#             )
#             base_df = base_result.to_pandas()
#         except Exception as exc:
#             print(f"   Warning: base metrics failed for batch {batch_start}: {exc}")
#             base_df = pd.DataFrame(
#                 {
#                     "answer_relevancy": [float("nan")] * len(samples),
#                     "faithfulness"    : [float("nan")] * len(samples),
#                 }
#             )

#         # ── Run reference metrics when refs are available ─────────────────────
#         ref_indices   = [i for i, m in enumerate(meta) if m["_has_ref"]]
#         ref_scores_by_idx: dict[int, dict] = {}

#         if ref_indices and has_refs:
#             ref_samples = [samples[i] for i in ref_indices]
#             ref_dataset = EvaluationDataset(samples=ref_samples)
#             try:
#                 ref_result = ragas_evaluate(
#                     dataset          = ref_dataset,
#                     metrics          = ref_metrics,
#                     llm              = ragas_llm,
#                     embeddings       = ragas_embs,
#                     raise_exceptions = False,
#                     show_progress    = False,
#                 )
#                 ref_df = ref_result.to_pandas()
#                 for local_idx, global_idx in enumerate(ref_indices):
#                     ref_scores_by_idx[global_idx] = {
#                         "context_precision": ref_df.iloc[local_idx].get("context_precision", float("nan")),
#                         "context_recall"   : ref_df.iloc[local_idx].get("context_recall",    float("nan")),
#                     }
#             except Exception as exc:
#                 print(f"   Warning: reference metrics failed for batch {batch_start}: {exc}")

#         # ── Assemble records ──────────────────────────────────────────────────
#         for i, m in enumerate(meta):
#             cp = ref_scores_by_idx.get(i, {}).get("context_precision", float("nan"))
#             cr = ref_scores_by_idx.get(i, {}).get("context_recall",    float("nan"))
#             records.append(
#                 {
#                     "id"                  : m["id"],
#                     "method"              : m["method"],
#                     "question"            : m["question"],
#                     "company"             : m["company"],
#                     "category"            : m["category"],
#                     "answer_relevancy"    : float(base_df.iloc[i].get("answer_relevancy", float("nan"))),
#                     "faithfulness"        : float(base_df.iloc[i].get("faithfulness",     float("nan"))),
#                     "context_precision"   : float(cp),
#                     "context_recall"      : float(cr),
#                     # Contextual Relevancy is context_precision in RAGAS 0.2.x
#                     # (no standalone context_relevancy metric exists in this version)
#                     "contextual_relevancy": float(cp),
#                 }
#             )

#     df = pd.DataFrame(records)
#     RESULTS_DIR.mkdir(parents=True, exist_ok=True)
#     out = RESULTS_DIR / results_filename
#     df.to_csv(out, index=False, encoding="utf-8")
#     print(f"\nRAGAS results saved → {out}")
#     return df

# evaluation/ragas_evaluator.py

def run_ragas_evaluation(
    judge_df: pd.DataFrame,
    retrieved_contexts_map: dict[str, list[str]],
    questions_path: Path = QUESTIONS_PATH,
    results_filename: str = "three_way_ragas_results.csv",
    batch_size: int = 10,  # Kept in signature for backward compatibility
) -> pd.DataFrame:
    """
    Runs RAGAS fully parallelized across the entire dataset in a single pass,
    completely bypassing the slow manual batching loop.
    """
    _check_ragas()

    ragas_llm  = _build_ragas_llm()
    ragas_embs = _build_ragas_embeddings()
    ref_map    = _load_reference_map(questions_path)
    has_refs   = bool(ref_map)

    n_unique_questions = len(judge_df["id"].unique()) if "id" in judge_df.columns else "?"

    print(f"\n{'=' * 55}")
    print("   🚀 HIGH-SPEED ASYNC RAGAS EVALUATION")
    print(f"   Total Evaluation Rows : {len(judge_df)}")
    print(f"   Judge LLM            : {config.JUDGE_MODEL_ID}")
    print(f"   Embeddings           : mistral-embed  (Mistral /v1/embeddings)")
    print(f"   Metrics (Base)       : answer_relevancy, faithfulness")
    if has_refs:
        print(f"   Metrics (With Refs)  : context_precision, context_recall")
    print(f"{'=' * 55}\n")

    base_metrics = [answer_relevancy, faithfulness]
    ref_metrics  = [context_precision, context_recall]

    samples: list[SingleTurnSample] = []
    meta: list[dict] = []

    # 1. Build the complete dataset layout at once
    for _, row in judge_df.iterrows():
        qid    = str(row.get("id", ""))
        method = str(row.get("method", ""))
        ctxs   = retrieved_contexts_map.get((qid, method), [])
        if not ctxs:
            ctxs = ["No context was captured for this question."]

        reference = ref_map.get(qid) or None

        samples.append(
            SingleTurnSample(
                user_input       = str(row.get("question", "")),
                response         = str(row.get("answer", "")),
                retrieved_contexts = ctxs,
                reference        = reference,
            )
        )
        meta.append(
            {
                "id"         : qid,
                "method"     : method,
                "question"   : str(row.get("question", "")),
                "company"    : str(row.get("company", "")),
                "category"   : str(row.get("category", "")),
                "_has_ref"   : reference is not None,
            }
        )

    full_dataset = EvaluationDataset(samples=samples)
    records: list[dict] = []

    # 2. Unified Pass 1: Evaluate base metrics across the ENTIRE dataset at once
    try:
        print("🔄 Running asynchronous evaluation for base metrics...")
        base_result = ragas_evaluate(
            dataset          = full_dataset,
            metrics          = base_metrics,
            llm              = ragas_llm,
            embeddings       = ragas_embs,
            raise_exceptions = False,
            show_progress    = True,  # Shows unified progress tracking
        )
        base_df = base_result.to_pandas()
    except Exception as exc:
        print(f"❌ Error: Base metrics calculation failed completely: {exc}")
        base_df = pd.DataFrame(
            {
                "answer_relevancy": [float("nan")] * len(samples),
                "faithfulness"    : [float("nan")] * len(samples),
            }
        )

    # 3. Unified Pass 2: Evaluate reference metrics for matching records at once
    ref_indices = [i for i, m in enumerate(meta) if m["_has_ref"]]
    ref_scores_by_idx: dict[int, dict] = {}

    if ref_indices and has_refs:
        print("🔄 Running asynchronous evaluation for reference metrics...")
        ref_samples = [samples[i] for i in ref_indices]
        ref_dataset = EvaluationDataset(samples=ref_samples)
        try:
            ref_result = ragas_evaluate(
                dataset          = ref_dataset,
                metrics          = ref_metrics,
                llm              = ragas_llm,
                embeddings       = ragas_embs,
                raise_exceptions = False,
                show_progress    = True,
            )
            ref_df = ref_result.to_pandas()
            for local_idx, global_idx in enumerate(ref_indices):
                ref_scores_by_idx[global_idx] = {
                    "context_precision": ref_df.iloc[local_idx].get("context_precision", float("nan")),
                    "context_recall"   : ref_df.iloc[local_idx].get("context_recall",    float("nan")),
                }
        except Exception as exc:
            print(f"❌ Error: Reference metrics calculation failed completely: {exc}")

    # 4. Compile and align records back into the required data structure format
    for i, m in enumerate(meta):
        cp = ref_scores_by_idx.get(i, {}).get("context_precision", float("nan"))
        cr = ref_scores_by_idx.get(i, {}).get("context_recall",    float("nan"))
        records.append(
            {
                "id"                  : m["id"],
                "method"              : m["method"],
                "question"            : m["question"],
                "company"             : m["company"],
                "category"            : m["category"],
                "answer_relevancy"    : float(base_df.iloc[i].get("answer_relevancy", float("nan"))),
                "faithfulness"        : float(base_df.iloc[i].get("faithfulness",     float("nan"))),
                "context_precision"   : float(cp),
                "context_recall"      : float(cr),
                "contextual_relevancy": float(cp),
            }
        )

    df = pd.DataFrame(records)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / results_filename
    df.to_csv(out, index=False, encoding="utf-8")
    print(f"\n🎉 RAGAS evaluation execution complete! Saved → {out}")
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