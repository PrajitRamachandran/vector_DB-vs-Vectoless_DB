# #vector_rag/pipeline.py
# import time

# start = time.time()
# print("Loading Vector_pipeline...")

# import sys
# import time
# import json
# from pathlib import Path

# sys.path.append(str(Path(__file__).parent.parent))

# from vector_rag.indexer import load_index, build_parent_lookup
# from vector_rag.retriever import retrieve
# from llm import load_llm, generate_answer, format_context


# class VectorRAGPipeline:
#     def __init__(self):
#         print("🔧 Initialising Vector RAG Pipeline...")
#         self.collection = load_index()
#         self.llm = load_llm()
#         self._data = self._load_data()
#         self.parent_lookup = build_parent_lookup(self._data)
#         print(f"✅ Vector RAG ready — {len(self.parent_lookup)} parents in lookup\n")

#     def _load_data(self) -> dict:
#         path = Path(__file__).resolve().parent.parent / "data" / "processed" / "chunks.json"
#         with open(path, encoding="utf-8") as f:
#             return json.load(f)

#     def ask(self, question: str, top_k: int = None) -> dict:
#         from config import TOP_K
#         top_k = top_k or TOP_K

#         if top_k < 15:
#             top_k = 15

#         ret = retrieve(question, self.collection, self.parent_lookup, top_k)
#         context = format_context(ret["chunks"])

#         if not context:
#             return {
#                 "question": question,
#                 "answer": "This information was not found in the retrieved sections.",
#                 "retrieved": ret["chunks"],
#                 "retrieval_time": ret["latency"],
#                 "generation_time": 0.0,
#                 "total_time": ret["latency"],
#                 "method": "vector",
#                 "retrieval_error": ret.get("error"),
#             }

#         gen_start = time.perf_counter()
#         answer = generate_answer(self.llm, context, question)
#         gen_time = round(time.perf_counter() - gen_start, 4)

#         return {
#             "question": question,
#             "answer": answer,
#             "retrieved": ret["chunks"],
#             "retrieval_time": ret["latency"],
#             "generation_time": gen_time,
#             "total_time": round(ret["latency"] + gen_time, 4),
#             "method": "vector",
#             "retrieval_error": ret.get("error"),
#         }

#     def show(self, result: dict):
#         print(f"\n{'='*55}")
#         print(f"  METHOD  : Vector RAG — BGE + HNSW + Parent-Child")
#         print(f"{'='*55}")
#         print(f"  Q: {result['question']}")
#         print(f"{'─'*55}")
#         print(f"  A: {result['answer']}")
#         print(f"{'─'*55}")
#         for c in result.get("retrieved", []):
#             m = c.get("metadata", {})
#             print(
#                 f"   • {m.get('company', 'Unknown')} | Page {m.get('page', '?')} "
#                 f"| Score {c.get('rerank_score', c.get('score', 0)):.4f}"
#             )
#         print(f"{'─'*55}")
#         print(
#             f"  Retrieval: {result.get('retrieval_time', 0)}s | "
#             f"Generation: {result.get('generation_time', 0)}s | "
#             f"Total: {result.get('total_time', 0)}s"
#         )
#         print(f"{'='*55}\n")


# print(
#     f"Vector_pipeline loaded in "
#     f"{time.time()-start:.2f}s"
# )



# vector_rag/pipeline.py
"""
Ties together indexing (vector_rag.indexer), retrieval (vector_rag.retriever),
and generation (llm) into a single VectorRAGPipeline.ask() call.

Construction is expensive (loads the embedding model, cross-encoder, and
LLM) — instantiate once per process and reuse across requests.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

_import_start = time.time()

sys.path.append(str(Path(__file__).parent.parent))

import config  # noqa: E402
from vector_rag.indexer import load_index, build_parent_lookup  # noqa: E402
from vector_rag.retriever import retrieve  # noqa: E402
from llm import load_llm, generate_answer, format_context  # noqa: E402

logger = logging.getLogger(__name__)
logger.info("vector_rag.pipeline imported in %.2fs", time.time() - _import_start)

# Floor on the *final* (post-rerank, parent-swapped) chunk count passed to
# the LLM as context — guards against a caller passing an unreasonably
# small top_k (e.g. 1) that would starve the LLM of context. This is
# deliberately small: it is NOT the ChromaDB candidate-pool size (that's
# config.FETCH_K, tuned separately in retriever.py), so raising it doesn't
# balloon context length/cost the way an oversized floor would.
MIN_TOP_K = getattr(config, "MIN_TOP_K", 3)

NOT_FOUND_ANSWER = "This information was not found in the retrieved sections."


class VectorRAGPipeline:
    """
    End-to-end Vector RAG query pipeline: loads the persisted ChromaDB index
    and parent lookup once at construction time, then serves ask() calls
    against them.
    """

    def __init__(self):
        logger.info("Initialising Vector RAG Pipeline...")

        try:
            self.collection = load_index()
        except Exception as exc:
            raise RuntimeError(
                "Failed to load the ChromaDB index. Run the indexing "
                "pipeline (index_chunks()) before starting VectorRAGPipeline."
            ) from exc

        try:
            self.llm = load_llm()
        except Exception as exc:
            raise RuntimeError(f"Failed to load the LLM: {exc}") from exc

        self._data = self._load_data()

        try:
            self.parent_lookup = build_parent_lookup(self._data)
        except KeyError as exc:
            raise RuntimeError(f"chunks.json is missing expected data: {exc}") from exc

        self._check_index_consistency()

        logger.info(
            "Vector RAG ready — %d parents in lookup, %d vectors indexed",
            len(self.parent_lookup), self.collection.count(),
        )

    @staticmethod
    def _load_data() -> dict:
        """
        Loads the same chunks.json the index was built from, so the parent
        lookup used at query time matches what's actually stored in
        ChromaDB. Sourced from config.DATA_PROCESSED_DIR — the same single
        source of truth used by utils.query_processor — rather than an
        independently hardcoded path, so the two locations can never
        silently drift apart.
        """
        path = Path(config.DATA_PROCESSED_DIR) / "chunks.json"
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Could not load {path}. Run the ingestion pipeline before "
                f"starting VectorRAGPipeline."
            ) from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"Could not parse {path} as JSON: {exc}") from exc

    def _check_index_consistency(self) -> None:
        """
        Best-effort sanity check that chunks.json roughly matches what's
        actually indexed in ChromaDB. A large mismatch is the classic
        symptom of a stale index — e.g. an embedding-model upgrade that
        silently left the old collection in place (see the stale-index
        guard notes in vector_rag/retriever.py). This only warns and never
        blocks startup, since some mismatch (e.g. mid-reindex) can be
        legitimate.
        """
        try:
            expected = len(self._data.get("children") or [])
            actual = self.collection.count()
            if expected and actual != expected:
                logger.warning(
                    "Index/data mismatch: chunks.json has %d children but "
                    "ChromaDB has %d vectors. The index may be stale — "
                    "consider re-running index_chunks().",
                    expected, actual,
                )
        except Exception:
            logger.exception("Index consistency check failed; continuing anyway.")

    def ask(self, question: str, top_k: Optional[int] = None) -> dict:
        """
        Runs the full retrieve -> rerank -> parent-swap -> generate pipeline
        for a single question.

        Args:
            question: the user's natural-language question.
            top_k: number of final (parent) chunks to retrieve and pass to
                the LLM as context. Defaults to config.TOP_K. A value of 0
                or an explicitly small value is honored (not silently
                replaced by the default) except for a small MIN_TOP_K floor.

        Returns:
            A result dict with question/answer/retrieved chunks, per-stage
            timings, method, and "error" (str or None) if retrieval or
            generation failed partway through.
        """
        question = (question or "").strip()
        if not question:
            return {
                "question": question,
                "answer": "Please ask a question.",
                "retrieved": [],
                "retrieval_time": 0.0,
                "rerank_time": 0.0,
                "generation_time": 0.0,
                "total_time": 0.0,
                "method": "vector",
                "error": None,
            }

        # `top_k or config.TOP_K` would silently discard an explicit top_k=0;
        # only fall back to the default when the caller didn't specify one.
        if top_k is None:
            top_k = config.TOP_K
        if top_k < MIN_TOP_K:
            logger.debug("Raising top_k from %d to MIN_TOP_K=%d", top_k, MIN_TOP_K)
            top_k = MIN_TOP_K

        try:
            ret = retrieve(question, self.collection, self.parent_lookup, top_k)
        except Exception as exc:
            logger.exception("Retrieval failed for question=%r", question)
            return {
                "question": question,
                "answer": "Something went wrong while searching the documents. Please try again.",
                "retrieved": [],
                "retrieval_time": 0.0,
                "rerank_time": 0.0,
                "generation_time": 0.0,
                "total_time": 0.0,
                "method": "vector",
                "error": str(exc),
            }

        # retriever.retrieve() reports retrieval and rerank latency
        # separately; "latency" is their sum. Prefer the split values so
        # timings reported here aren't misleadingly folded together.
        retrieval_time = ret.get("retrieval_latency", ret.get("latency", 0.0))
        rerank_time = ret.get("rerank_latency", 0.0)

        context = format_context(ret["chunks"])

        if not context:
            return {
                "question": question,
                "answer": NOT_FOUND_ANSWER,
                "retrieved": ret["chunks"],
                "retrieval_time": retrieval_time,
                "rerank_time": rerank_time,
                "generation_time": 0.0,
                "total_time": round(retrieval_time + rerank_time, 4),
                "method": "vector",
                "error": None,
            }

        gen_start = time.perf_counter()
        try:
            answer = generate_answer(self.llm, context, question)
        except Exception as exc:
            logger.exception("Answer generation failed for question=%r", question)
            return {
                "question": question,
                "answer": "Retrieved relevant sections, but answer generation failed. Please try again.",
                "retrieved": ret["chunks"],
                "retrieval_time": retrieval_time,
                "rerank_time": rerank_time,
                "generation_time": round(time.perf_counter() - gen_start, 4),
                "total_time": round(retrieval_time + rerank_time, 4),
                "method": "vector",
                "error": str(exc),
            }
        gen_time = round(time.perf_counter() - gen_start, 4)

        return {
            "question": question,
            "answer": answer,
            "retrieved": ret["chunks"],
            "retrieval_time": retrieval_time,
            "rerank_time": rerank_time,
            "generation_time": gen_time,
            "total_time": round(retrieval_time + rerank_time + gen_time, 4),
            "method": "vector",
            "error": None,
        }

    def show(self, result: dict) -> None:
        """Pretty-prints a result dict to the console (CLI/debug use)."""
        print(f"\n{'='*55}")
        print("  METHOD  : Vector RAG — BGE + HNSW + Parent-Child")
        print(f"{'='*55}")
        print(f"  Q: {result.get('question', '')}")
        print(f"{'─'*55}")
        print(f"  A: {result.get('answer', '')}")
        print(f"{'─'*55}")
        for c in result.get("retrieved", []):
            m = c.get("metadata") or {}
            score = c.get("rerank_score")
            if score is None:
                score = c.get("score", 0.0)
            print(
                f"   • {m.get('company', 'Unknown')} | Page {m.get('page', '?')} "
                f"| Score {float(score):.4f}"
            )
        print(f"{'─'*55}")
        print(
            f"  Retrieval: {result.get('retrieval_time', 0)}s | "
            f"Rerank: {result.get('rerank_time', 0)}s | "
            f"Generation: {result.get('generation_time', 0)}s | "
            f"Total: {result.get('total_time', 0)}s"
        )
        if result.get("error"):
            print(f"  ⚠️  Error: {result['error']}")
        print(f"{'='*55}\n")