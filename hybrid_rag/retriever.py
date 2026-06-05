# hybrid_rag/retriever.py
#
# Reciprocal Rank Fusion (RRF) over Vector + BM25 results.
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
#   ├── vector retriever  → ranked list A  (FETCH_K children)
#   ├── BM25 retriever    → ranked list B  (FETCH_K children)
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

# ── RRF constant ──────────────────────────────────────────────────────────────
# k=60 is the standard value from the original RRF paper (Cormack 2009).
# Higher k → dampens the advantage of top ranks (more equal weighting).
# Lower  k → amplifies top-rank advantage (winner-takes-most).
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

    Deduplication key: chunk text (same child won't be added twice).
    """
    # chunk_id → {"chunk": dict, "rrf_score": float}
    fused: dict[str, dict] = {}

    # --- Vector contributions (list A) ---
    for rank, chunk in enumerate(vector_children, start=1):
        cid   = chunk["metadata"].get("parent_id", chunk["text"][:40])
        score = _rrf_score(rank)
        if cid not in fused:
            fused[cid] = {"chunk": chunk, "rrf_score": 0.0, "sources": []}
        fused[cid]["rrf_score"] += score
        fused[cid]["sources"].append("vector")

    # --- BM25 contributions (list B) ---
    for rank, chunk in enumerate(bm25_children, start=1):
        cid   = chunk["metadata"].get("parent_id", chunk["text"][:40])
        score = _rrf_score(rank)
        if cid not in fused:
            fused[cid] = {"chunk": chunk, "rrf_score": 0.0, "sources": []}
        fused[cid]["rrf_score"] += score
        fused[cid]["sources"].append("bm25")

    # Sort by accumulated RRF score
    ranked = sorted(fused.values(), key=lambda x: x["rrf_score"], reverse=True)

    # Attach rrf_score to each chunk dict and return top fetch_k
    result = []
    for entry in ranked[:fetch_k]:
        chunk = entry["chunk"].copy()
        chunk["rrf_score"]    = round(entry["rrf_score"], 6)
        chunk["rrf_sources"]  = entry["sources"]   # ["vector"], ["bm25"], or both
        result.append(chunk)

    return result


# ── Vector first-stage (inline — no ChromaDB import needed) ───────────────────

def _vector_fetch(query: str, collection, company: str | None) -> list[dict]:
    """
    Runs a vector similarity search and returns FETCH_K child chunks.
    No reranking here — that happens once after fusion.
    """
    where_filter = {"company": company} if company else None

    kwargs = dict(
        query_texts=[query],
        n_results  = config.FETCH_K,
        include    = ["documents", "metadatas", "distances"],
    )
    if where_filter:
        kwargs["where"] = where_filter

    results = collection.query(**kwargs)

    children = []
    for i in range(len(results["documents"][0])):
        distance   = results["distances"][0][i]
        similarity = round(1 / (1 + distance), 4)
        children.append({
            "text"    : results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "score"   : similarity,
        })
    return children


# ── BM25 first-stage ──────────────────────────────────────────────────────────

def _bm25_fetch(
    query_info: dict,
    bm25_model,
    all_children: list[dict],
    company:      str | None,
) -> list[dict]:
    """
    Runs BM25 keyword search on the (optionally filtered) child corpus
    and returns FETCH_K children. Rebuilds BM25 on filtered subset just
    like vectorless_rag/retriever.py does.
    """
    if company:
        search_children = [c for c in all_children if c["company"] == company]
    else:
        search_children = all_children

    if not search_children:
        search_children = all_children

    tokenized_corpus = [tokenize(c["text"]) for c in search_children]
    filtered_bm25    = BM25Okapi(tokenized_corpus)

    tokenized_query  = tokenize(query_info["clean_query"])
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
    collection,                  # ChromaDB collection
    bm25_model,                  # BM25Okapi instance
    all_children:  list[dict],   # full children list from BM25 pkl
    parent_lookup: dict,
    top_k:         int = config.TOP_K,
) -> dict:
    """
    Full Hybrid RAG retrieval:

    1. Preprocess query (company detection, cleaning)
    2. Vector first-stage   → FETCH_K children
    3. BM25 first-stage     → FETCH_K children
    4. RRF fusion           → merged list (up to 2×FETCH_K unique children)
    5. Cross-encoder rerank → TOP_K best children
    6. Parent swap          → return parent chunks to LLM
    """
    start_time = time.perf_counter()
    query_info = preprocess_query(query)
    company    = query_info["company"]

    # ── First-stage: both retrievers run in parallel (sequential here, same thread) ──
    vector_start    = time.perf_counter()
    vector_children = _vector_fetch(query, collection, company)
    vector_latency  = time.perf_counter() - vector_start

    bm25_start      = time.perf_counter()
    bm25_children   = _bm25_fetch(query_info, bm25_model, all_children, company)
    bm25_latency    = time.perf_counter() - bm25_start

    # ── RRF fusion ────────────────────────────────────────────────────────────
    fused_children  = _fuse_rrf(vector_children, bm25_children, config.FETCH_K)
    retrieval_latency = time.perf_counter() - start_time

    # ── Rerank fused list once ────────────────────────────────────────────────
    rerank_start   = time.perf_counter()
    reranked       = rerank(query, fused_children, top_k=top_k)
    rerank_latency = time.perf_counter() - rerank_start

    # ── Swap children → parents (deduplicated) ────────────────────────────────
    seen_parents = set()
    final_chunks = []
    for child in reranked:
        parent_id = child["metadata"].get("parent_id")
        if parent_id and parent_id not in seen_parents:
            parent = parent_lookup.get(parent_id)
            if parent:
                seen_parents.add(parent_id)
                final_chunks.append({
                    "text"        : parent["text"],
                    "metadata"    : child["metadata"],
                    "score"       : child.get("rerank_score", child["score"]),
                    "rerank_score": child.get("rerank_score", child["score"]),
                    "rrf_score"   : child.get("rrf_score", 0.0),
                    "rrf_sources" : child.get("rrf_sources", []),
                    "child_text"  : child["text"],    # keep for debugging
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