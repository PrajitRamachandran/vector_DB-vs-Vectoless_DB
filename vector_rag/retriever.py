import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import config
from utils.query_processor import preprocess_query
from reranker import rerank


def _run_query(collection, query_text: str, where_filter: dict | None):
    """
    Executes a Chroma query safely.
    """
    query_kwargs = dict(
        query_texts=[query_text],
        n_results=config.FETCH_K,
        include=["documents", "metadatas", "distances"],
    )

    if where_filter:
        query_kwargs["where"] = where_filter

    return collection.query(**query_kwargs)


def _build_child_chunks(results: dict) -> list[dict]:
    """
    Converts raw Chroma results into child chunk dicts.
    """
    documents = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]

    child_chunks = []
    for i in range(min(len(documents), len(metadatas), len(distances))):
        distance = distances[i]
        similarity = round(1 / (1 + float(distance)), 4)

        child_chunks.append(
            {
                "text": documents[i],
                "metadata": metadatas[i] or {},
                "score": similarity,
            }
        )

    return child_chunks


def retrieve(query: str, collection, parent_lookup: dict, top_k: int = config.TOP_K) -> dict:
    """
    1. Try query with company filter
    2. If that returns nothing, retry without the filter
    3. Rerank children
    4. Swap children → parents
    5. Deduplicate parents
    """
    start_time = time.perf_counter()

    query_info = preprocess_query(query)
    company = query_info["company"]
    search_query = query_info["original"] or query

    # First attempt: exact company filter if available
    where_filter = {"company": company} if company else None
    results = _run_query(collection, search_query, where_filter)
    child_chunks = _build_child_chunks(results)

    # Fallback: retry without company filter if exact filtering produced nothing
    if not child_chunks and where_filter is not None:
        results = _run_query(collection, search_query, None)
        child_chunks = _build_child_chunks(results)

    retrieval_latency = time.perf_counter() - start_time

    # If still nothing, return cleanly
    if not child_chunks:
        return {
            "chunks": [],
            "latency": round(retrieval_latency, 4),
            "retrieval_latency": round(retrieval_latency, 4),
            "rerank_latency": 0.0,
            "method": "vector",
            "company_filter": company,
            "search_query": search_query,
        }

    # Rerank children
    rerank_start = time.perf_counter()
    child_chunks = rerank(query, child_chunks, top_k=top_k)
    rerank_latency = time.perf_counter() - rerank_start

    # Swap children → parents (deduplicated)
    seen_parents = set()
    final_chunks = []

    for child in child_chunks:
        meta = child.get("metadata") or {}
        parent_id = meta.get("parent_id")

        if not parent_id or parent_id in seen_parents:
            continue

        parent = parent_lookup.get(parent_id)

        if parent:
            seen_parents.add(parent_id)
            final_chunks.append(
                {
                    "text": parent.get("text", ""),
                    "metadata": meta,
                    "score": child.get("rerank_score", child.get("score", 0.0)),
                    "rerank_score": child.get("rerank_score", child.get("score", 0.0)),
                    "child_text": child.get("text", ""),
                }
            )

    # Final fallback: if parent lookup failed, keep the child text
    if not final_chunks:
        for child in child_chunks[:top_k]:
            text = (child.get("text") or "").strip()
            if not text:
                continue
            final_chunks.append(
                {
                    "text": text,
                    "metadata": child.get("metadata") or {},
                    "score": child.get("rerank_score", child.get("score", 0.0)),
                    "rerank_score": child.get("rerank_score", child.get("score", 0.0)),
                    "child_text": text,
                }
            )

    return {
        "chunks": final_chunks,
        "latency": round(retrieval_latency + rerank_latency, 4),
        "retrieval_latency": round(retrieval_latency, 4),
        "rerank_latency": round(rerank_latency, 4),
        "method": "vector",
        "company_filter": company,
        "search_query": search_query,
    }