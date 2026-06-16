# hybrid_rag/retriever.py
#
# Reciprocal Rank Fusion (RRF) over Vector + BM25 results.
#
# Bug fixes applied:
#
# 1. MISSING BGE QUERY PREFIX IN _vector_fetch()
#    BGE (BAAI/bge-base-en-v1.5) requires queries to be prefixed with a
#    retrieval instruction string. Without it, query embeddings are
#    sub-optimal and the vector side of the hybrid consistently underperforms.
#    The same fix is applied in vector_rag/retriever.py.
#
# 2. NO COMPANY SANITY-CHECK AFTER VECTOR FETCH
#    ChromaDB always returns n_results chunks even when the WHERE filter
#    produces fewer matches than requested — it silently pads with the
#    nearest neighbours from other companies. Without a post-fetch company
#    filter, wrong-company chunks enter the RRF fusion pool and pollute
#    the final answer.
#    Fix: after fetching from ChromaDB, strip any chunk whose metadata
#    company doesn't match the target. If that leaves < 3 chunks, retry
#    without the WHERE filter and filter manually from metadata.
#
# Why RRF works:
#   Vector search finds semantically similar chunks but misses exact terms.
#   BM25 nails exact financial keywords ($215B, "Data Center") but misses
#   paraphrases. RRF fuses the two ranked lists by rank position, not raw
#   scores (which are on incompatible scales), so both signals contribute
#   regardless of magnitude.
#
# Flow:
#   query
#   ├── vector retriever  → ranked list A  (FETCH_K children, correct company)
#   ├── BM25 retriever    → ranked list B  (FETCH_K children, correct company)
#   └── RRF fusion        → merged list → rerank → TOP_K → parent swap → LLM

import sys
import time
import numpy as np
from pathlib import Path
from rank_bm25 import BM25Okapi

sys.path.append(str(Path(__file__).parent.parent))

import config
from vectorless_rag.indexer import tokenize
from utils.query_processor  import preprocess_query
from reranker               import rerank

# ── BGE query prefix (fix #1) ─────────────────────────────────────────────────
# Applied to queries only — documents are indexed WITHOUT this prefix.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def _bge_query(text: str) -> str:
    return BGE_QUERY_PREFIX + text.strip()


# ── RRF constant ──────────────────────────────────────────────────────────────
# k=60 is the standard value from the original RRF paper (Cormack 2009).
# Higher k → more equal weighting across ranks.
# Lower  k → amplifies the advantage of top-ranked documents.
RRF_K = 60


def _rrf_score(rank: int, k: int = RRF_K) -> float:
    """Score for a document at a given rank position. Higher rank = higher score."""
    return 1.0 / (k + rank)


def _fuse_rrf(
    vector_children: list[dict],
    bm25_children:   list[dict],
    fetch_k:         int,
) -> list[dict]:
    """
    Merges two ranked child lists into one via Reciprocal Rank Fusion.

    Steps:
    1. Assign RRF scores by position in each list.
    2. Accumulate scores for chunks appearing in both lists (big boost).
    3. Sort by total RRF score descending.
    4. Return top fetch_k unique chunks.

    Deduplication key: parent_id (children of the same parent share context —
    we want to accumulate their votes rather than double-count the parent).
    """
    # parent_id → {"chunk": dict, "rrf_score": float, "sources": list}
    fused: dict[str, dict] = {}

    # --- Vector contributions (list A) ---
    for rank, chunk in enumerate(vector_children, start=1):
        pid   = chunk["metadata"].get("parent_id") or chunk["text"][:40]
        score = _rrf_score(rank)
        if pid not in fused:
            fused[pid] = {"chunk": chunk, "rrf_score": 0.0, "sources": []}
        fused[pid]["rrf_score"] += score
        fused[pid]["sources"].append("vector")

    # --- BM25 contributions (list B) ---
    for rank, chunk in enumerate(bm25_children, start=1):
        pid   = chunk["metadata"].get("parent_id") or chunk["text"][:40]
        score = _rrf_score(rank)
        if pid not in fused:
            fused[pid] = {"chunk": chunk, "rrf_score": 0.0, "sources": []}
        fused[pid]["rrf_score"] += score
        fused[pid]["sources"].append("bm25")

    # Sort by accumulated RRF score
    ranked = sorted(fused.values(), key=lambda x: x["rrf_score"], reverse=True)

    # Attach rrf_score + sources to each chunk dict and return top fetch_k
    result = []
    for entry in ranked[:fetch_k]:
        chunk = entry["chunk"].copy()
        chunk["rrf_score"]   = round(entry["rrf_score"], 6)
        chunk["rrf_sources"] = entry["sources"]  # ["vector"], ["bm25"], or both
        result.append(chunk)

    return result


# ── Vector first-stage ────────────────────────────────────────────────────────

def _vector_fetch(
    query:   str,
    collection,
    company: str | None,
    top_k:   int = config.TOP_K,
) -> list[dict]:
    """
    Runs a vector similarity search and returns up to FETCH_K child chunks
    that belong to the target company.

    Guards against ChromaDB silently returning wrong-company chunks when
    the WHERE filter produces fewer matches than n_results (fix #2).
    """
    bge_q        = _bge_query(query)                    # fix #1
    where_filter = None
    if company:
        where_filter = {
            "$and": [
                {"company": company},
                {"year": "2025"}
            ]
        }

    kwargs = dict(
        query_texts=[bge_q],
        n_results  = config.FETCH_K,
        include    = ["documents", "metadatas", "distances"],
    )
    if where_filter:
        kwargs["where"] = where_filter

    results  = collection.query(**kwargs)
    docs     = results["documents"][0]
    metas    = results["metadatas"][0]
    dists    = results["distances"][0]

    def _to_chunks(docs, metas, dists, company_filter):
        """Converts raw ChromaDB output to chunk dicts, filtering by company."""
        out = []
        for i in range(len(docs)):
            meta = metas[i]
            # Post-fetch company filter: drop any chunk from the wrong company (fix #2)
            if company_filter and meta.get("company", "") != company_filter:
                continue
            similarity = round(1 / (1 + dists[i]), 4)
            out.append({
                "text"    : docs[i],
                "metadata": meta,
                "score"   : similarity,
            })
        return out

    children = _to_chunks(docs, metas, dists, company)

    # Fallback: if the WHERE filter + post-filter left us with too few chunks,
    # query without WHERE and filter manually from metadata (fix #2 continued)
    if company and len(children) < max(3, top_k):
        kwargs_no_filter = {k: v for k, v in kwargs.items() if k != "where"}
        results2 = collection.query(**kwargs_no_filter)
        children = _to_chunks(
            results2["documents"][0],
            results2["metadatas"][0],
            results2["distances"][0],
            company,
        )

    return children


# ── BM25 first-stage ──────────────────────────────────────────────────────────

def _bm25_fetch(
    query_info:   dict,
    bm25_model,
    all_children: list[dict],
    company:      str | None,
) -> list[dict]:
    """
    Runs BM25 keyword search on the (optionally filtered) child corpus
    and returns FETCH_K children.

    BM25 is rebuilt on the company-filtered subset (same approach as
    vectorless_rag/retriever.py) so IDF scores reflect only the relevant
    documents rather than the whole corpus.
    """
    if company:
        search_children = [c for c in all_children if c["company"] == company]
        if not search_children:
            search_children = all_children   # fallback: search all
    else:
        search_children = all_children

    tokenized_corpus = [tokenize(c["text"]) for c in search_children]
    filtered_bm25    = BM25Okapi(tokenized_corpus)

    tokenized_query  = tokenize(
        query_info["clean_query"]
        or query_info["semantic_query"]
        or query_info["original"]
        or ""
    )
    scores           = filtered_bm25.get_scores(tokenized_query)
    top_indices      = np.argsort(scores)[::-1][: config.FETCH_K]

    children = []
    for idx in top_indices:
        children.append({
            "text"    : search_children[idx]["text"],
            "metadata": {
                "source"   : search_children[idx]["source"],
                "company"  : search_children[idx]["company"],
                "page"     : search_children[idx]["page"],
                "parent_id": search_children[idx].get("parent_id", ""),
            },
            "score": round(float(scores[idx]), 4),
        })
    return children


# ── Main retrieve function ─────────────────────────────────────────────────────

def retrieve(
    query:         str,
    collection,                 # ChromaDB collection
    bm25_model,                 # BM25Okapi instance
    all_children:  list[dict],  # full children list from BM25 pkl
    parent_lookup: dict,
    top_k:         int = config.TOP_K,
) -> dict:
    """
    Full Hybrid RAG retrieval:

    1. Preprocess query (company detection, cleaning)
    2. Vector first-stage   → up to FETCH_K correct-company children
    3. BM25 first-stage     → FETCH_K correct-company children
    4. RRF fusion           → merged list (up to 2×FETCH_K unique children)
    5. Cross-encoder rerank → TOP_K best children
    6. Parent swap          → return parent chunks to LLM
    """
    start_time = time.perf_counter()
    query_info = preprocess_query(query)
    company    = query_info["company"]
    semantic_query = (
        query_info.get("semantic_query")
        or query_info.get("original")
        or query
    )

    clean_query = (
        query_info.get("clean_query")
        or semantic_query
    )

    # ── First-stage: both retrievers ─────────────────────────────────────────
    vector_start    = time.perf_counter()
    vector_children = _vector_fetch(semantic_query, collection, company, top_k=top_k)
    vector_latency  = time.perf_counter() - vector_start

    bm25_start      = time.perf_counter()
    bm25_children   = _bm25_fetch(query_info, bm25_model, all_children, company)
    bm25_latency    = time.perf_counter() - bm25_start

    # ── RRF fusion ────────────────────────────────────────────────────────────
    fused_children    = _fuse_rrf(vector_children, bm25_children, config.FETCH_K)
    retrieval_latency = time.perf_counter() - start_time

    # ── Rerank fused list once ────────────────────────────────────────────────
    rerank_start   = time.perf_counter()
    reranked       = rerank(query, fused_children, top_k=config.FETCH_K)
    rerank_latency = time.perf_counter() - rerank_start

    # ── Swap children → parents (deduplicated) ────────────────────────────────
    seen_parents = set()
    final_chunks = []
    for child in reranked:
        if len(final_chunks) >= top_k:
            break
        parent_id = child["metadata"].get("parent_id")
        if parent_id and parent_id in seen_parents:
            continue

        parent = parent_lookup.get(parent_id) if parent_id else None
        if parent:
            seen_parents.add(parent_id)
            final_chunks.append({
                "text"        : parent["text"],
                "metadata"    : child["metadata"],
                "score"       : child.get("rerank_score", child["score"]),
                "rerank_score": child.get("rerank_score", child["score"]),
                "rrf_score"   : child.get("rrf_score", 0.0),
                "rrf_sources" : child.get("rrf_sources", []),
                "child_text"  : child["text"],    # kept for debugging
            })
        else:
            text = (child.get("text") or "").strip()
            if not text:
                continue
            if parent_id:
                seen_parents.add(parent_id)
            final_chunks.append({
                "text"        : text,
                "metadata"    : child["metadata"],
                "score"       : child.get("rerank_score", child["score"]),
                "rerank_score": child.get("rerank_score", child["score"]),
                "rrf_score"   : child.get("rrf_score", 0.0),
                "rrf_sources" : child.get("rrf_sources", []),
                "child_text"  : text,
            })

    return {
        "chunks"           : final_chunks,
        "latency"          : round(retrieval_latency + rerank_latency, 4),
        "retrieval_latency": round(retrieval_latency, 4),
        "vector_latency"   : round(vector_latency, 4),
        "bm25_latency"     : round(bm25_latency, 4),
        "rerank_latency"   : round(rerank_latency, 4),
        "method"           : "hybrid",
        "company_filter"   : company,
        "vector_candidates": len(vector_children),
        "bm25_candidates"  : len(bm25_children),
        "fused_candidates" : len(fused_children),
    }
