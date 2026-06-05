# vector_rag/retriever.py
import sys
import time
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import config
from utils.query_processor import preprocess_query
from reranker import rerank


def retrieve(query: str, collection, parent_lookup: dict,
             top_k: int = config.TOP_K) -> dict:
    """
    1. Embed query → search child chunks (precise match)
    2. Rerank children
    3. Swap each child for its parent chunk (rich context)
    4. Deduplicate parents (multiple children can share a parent)
    """
    start_time = time.perf_counter()
    query_info = preprocess_query(query)
    company    = query_info["company"]

    where_filter = {"company": company} if company else None

    query_kwargs = dict(
        query_texts=[query],
        n_results  = config.FETCH_K,
        include    = ["documents", "metadatas", "distances"]
    )
    if where_filter:
        query_kwargs["where"] = where_filter

    results = collection.query(**query_kwargs)
    retrieval_latency = time.perf_counter() - start_time

    # Build child results
    child_chunks = []
    for i in range(len(results["documents"][0])):
        distance   = results["distances"][0][i]
        similarity = round(1 / (1 + distance), 4)
        child_chunks.append({
            "text"    : results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "score"   : similarity
        })

    # Rerank children
    rerank_start  = time.perf_counter()
    child_chunks  = rerank(query, child_chunks, top_k=top_k)
    rerank_latency = time.perf_counter() - rerank_start

    # Swap children → parents (deduplicated)
    seen_parents = set()
    final_chunks = []
    for child in child_chunks:
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
                    "child_text"  : child["text"]   # keep for debugging
                })

    return {
        "chunks"           : final_chunks,
        "latency"          : round(retrieval_latency + rerank_latency, 4),
        "retrieval_latency": round(retrieval_latency, 4),
        "rerank_latency"   : round(rerank_latency, 4),
        "method"           : "vector",
        "company_filter"   : company
    }