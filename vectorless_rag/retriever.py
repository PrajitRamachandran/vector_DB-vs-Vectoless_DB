# vectorless_rag/retriever.py
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


def retrieve(query: str, bm25, children: list[dict],
             parent_lookup: dict,
             top_k: int = config.TOP_K) -> dict:
    """
    1. Filter children by company
    2. BM25 score filtered children
    3. Rerank top candidates
    4. Swap children → parents
    """
    start_time = time.perf_counter()
    query_info = preprocess_query(query)
    company    = query_info["company"]

    # Filter children to company
    if company:
        search_children = [c for c in children if c["company"] == company]
    else:
        search_children = children

    if not search_children:
        search_children = children

    # Rebuild BM25 on filtered subset
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

    retrieval_latency = time.perf_counter() - start_time

    child_results = []
    for idx in top_indices:
        child_results.append({
            "text"    : search_children[idx]["text"],
            "metadata": {
                "source"   : search_children[idx]["source"],
                "company"  : search_children[idx]["company"],
                "page"     : search_children[idx]["page"],
                "parent_id": search_children[idx].get("parent_id", "")
            },
            "score": round(float(scores[idx]), 4)
        })

    # Rerank
    rerank_start   = time.perf_counter()
    child_results  = rerank(query, child_results, top_k=config.FETCH_K)
    rerank_latency = time.perf_counter() - rerank_start

    # Swap children → parents
    seen_parents = set()
    final_chunks = []
    for child in child_results:
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
                "child_text"  : child["text"]
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
                "child_text"  : text
            })

    return {
        "chunks"           : final_chunks,
        "latency"          : round(retrieval_latency + rerank_latency, 4),
        "retrieval_latency": round(retrieval_latency, 4),
        "rerank_latency"   : round(rerank_latency, 4),
        "method"           : "bm25",
        "company_filter"   : company
    }
