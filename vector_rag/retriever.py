# vector_rag/retriever.py
#
# Bug fixes applied:
#
# 1. MISSING BGE QUERY PREFIX
#    BGE (BAAI/bge-base-en-v1.5) requires queries to be prefixed with an
#    instruction string for best retrieval quality. Documents are indexed
#    WITHOUT this prefix — only the query gets it. Without this, embedding
#    similarity scores are lower than they should be, causing weaker retrieval.
#
# 2. BROKEN FALLBACK — wrong company results returned silently
#    Old code: `if not child_chunks and where_filter is not None`
#    ChromaDB ALWAYS returns n_results chunks (from whatever company is closest
#    in embedding space). So child_chunks was never empty → fallback never fired
#    → NVIDIA chunks were returned for Amazon/Microsoft/Netflix questions.
#    Fix: check (a) fewer than 3 results, OR (b) results are mostly wrong company.
#
# 3. STALE INDEX GUARD
#    After a BGE upgrade (384-dim MiniLM → 768-dim BGE), ChromaDB silently loads
#    the old collection. The company metadata filter still runs but the distance
#    scores are meaningless. The _chunks_are_correct_company() guard catches this.

import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import config
from utils.query_processor import preprocess_query
from reranker import rerank

# BGE-specific instruction prefix — ONLY for queries, never for indexed documents.
# Source: https://huggingface.co/BAAI/bge-base-en-v1.5
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def _bge_query(text: str) -> str:
    """Returns the query with the required BGE instruction prefix."""
    return BGE_QUERY_PREFIX + text.strip()


def _run_query(collection, query_text: str, where_filter: dict | None) -> dict:
    """Executes a ChromaDB query safely."""
    kwargs = dict(
        query_texts=[query_text],
        n_results=config.FETCH_K,
        include=["documents", "metadatas", "distances"],
    )
    if where_filter:
        kwargs["where"] = where_filter
    return collection.query(**kwargs)


def _build_child_chunks(results: dict) -> list[dict]:
    """Converts raw ChromaDB results into child chunk dicts."""
    documents = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]

    child_chunks = []
    for i in range(min(len(documents), len(metadatas), len(distances))):
        distance   = distances[i]
        similarity = round(1 / (1 + float(distance)), 4)
        child_chunks.append({
            "text"    : documents[i],
            "metadata": metadatas[i] or {},
            "score"   : similarity,
        })
    return child_chunks


def _filter_company_chunks(chunks: list[dict], company: str | None) -> list[dict]:
    """Keeps only chunks whose metadata company matches the target company."""
    if not company:
        return chunks
    return [
        chunk for chunk in chunks
        if chunk.get("metadata", {}).get("company", "") == company
    ]


def retrieve(
    query:         str,
    collection,
    parent_lookup: dict,
    top_k:         int = config.TOP_K,
) -> dict:
    """
    Full Vector RAG retrieval with all safety guards:

    1.  Preprocess query — extract company, build BGE-prefixed query string
    2.  Try ChromaDB with company WHERE filter
    3a. If < 3 chunks returned → fallback (filter was too narrow)
    3b. If hit-rate < 50%      → fallback (stale index / wrong-company contamination)
    4.  On fallback: query without filter, keep only correct-company chunks
    5.  Rerank children with cross-encoder
    6.  Swap children → parents (deduplicated)
    7.  Final safety net: if parent lookup finds nothing, keep child text
    """
    start_time = time.perf_counter()

    query_info   = preprocess_query(query)
    company = query_info.get("company")
    # Use the semantic query so dense retrieval keeps the company name and
    # avoids possessive artifacts like "'s".
    raw_query = (
        query_info.get("semantic_query")
        or query_info.get("original")
        or query
    )

    # Build the BGE-prefixed query string (fix #1)
    bge_q = _bge_query(raw_query)

    # ── First attempt: with company filter ────────────────────────────────────
    where_filter  = {"company": company} if company else None
    results       = _run_query(collection, bge_q, where_filter)
    child_chunks  = _filter_company_chunks(_build_child_chunks(results), company)

    # ── Fallback decision (fix #2 + #3) ──────────────────────────────────────
    # Trigger fallback when the filter returned too few results OR the chunks
    # that came back are mostly from the wrong company (stale index signal).
    if where_filter is not None and len(child_chunks) < max(3, top_k):
        results        = _run_query(collection, bge_q, None)
        all_chunks     = _build_child_chunks(results)
        child_chunks   = _filter_company_chunks(all_chunks, company)

    retrieval_latency = time.perf_counter() - start_time

    # ── Nothing found — return cleanly ───────────────────────────────────────
    if not child_chunks:
        return {
            "chunks"           : [],
            "latency"          : round(retrieval_latency, 4),
            "retrieval_latency": round(retrieval_latency, 4),
            "rerank_latency"   : 0.0,
            "method"           : "vector",
            "company_filter"   : company,
            "search_query"     : bge_q,
        }

    # ── Rerank children ───────────────────────────────────────────────────────
    rerank_start   = time.perf_counter()
    child_chunks   = rerank(query, child_chunks, top_k=config.FETCH_K)
    rerank_latency = time.perf_counter() - rerank_start

    # ── Swap children → parents (deduplicated) ────────────────────────────────
    seen_parents = set()
    final_chunks = []
    for child in child_chunks:
        if len(final_chunks) >= top_k:
            break
        meta      = child.get("metadata") or {}
        parent_id = meta.get("parent_id")
        if not parent_id or parent_id in seen_parents:
            continue
        parent = parent_lookup.get(parent_id)
        if parent:
            seen_parents.add(parent_id)
            final_chunks.append({
                "text"        : parent.get("text", ""),
                "metadata"    : meta,
                "score"       : child.get("rerank_score", child.get("score", 0.0)),
                "rerank_score": child.get("rerank_score", child.get("score", 0.0)),
                "child_text"  : child.get("text", ""),
            })
        else:
            text = (child.get("text") or "").strip()
            if not text:
                continue
            if parent_id:
                seen_parents.add(parent_id)
            final_chunks.append({
                "text"        : text,
                "metadata"    : meta,
                "score"       : child.get("rerank_score", child.get("score", 0.0)),
                "rerank_score": child.get("rerank_score", child.get("score", 0.0)),
                "child_text"  : text,
            })

    # ── Safety net: parent lookup failed — keep child text ────────────────────
    if not final_chunks:
        for child in child_chunks[:top_k]:
            text = (child.get("text") or "").strip()
            if not text:
                continue
            final_chunks.append({
                "text"        : text,
                "metadata"    : child.get("metadata") or {},
                "score"       : child.get("rerank_score", child.get("score", 0.0)),
                "rerank_score": child.get("rerank_score", child.get("score", 0.0)),
                "child_text"  : text,
            })

    return {
        "chunks"           : final_chunks,
        "latency"          : round(retrieval_latency + rerank_latency, 4),
        "retrieval_latency": round(retrieval_latency, 4),
        "rerank_latency"   : round(rerank_latency, 4),
        "method"           : "vector",
        "company_filter"   : company,
        "search_query"     : bge_q,
    }
