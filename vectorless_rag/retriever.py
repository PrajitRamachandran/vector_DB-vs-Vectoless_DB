# vectorless_rag/retriever.py
import sys
import time
import numpy as np
from pathlib import Path
from rank_bm25 import BM25Okapi
sys.path.append(str(Path(__file__).parent.parent))

import config
from vectorless_rag.indexer  import load_bm25_index, tokenize
from utils.query_processor   import preprocess_query
from reranker                import rerank


def retrieve(query: str, bm25, chunks: list[dict],
             top_k: int = config.TOP_K) -> dict:
    """
    Improved BM25 retriever with:
    1. Company-aware chunk filtering before scoring
       — BM25 can't use metadata filters natively,
         so we filter the chunk list first, then run BM25 on the subset
    2. Reranking after retrieval
    """
    start_time = time.perf_counter()
    query_info = preprocess_query(query)
    company    = query_info["company"]

    # Filter chunks to the detected company
    if company:
        search_chunks = [c for c in chunks if c["company"] == company]
        print(f"   🎯 Company filter applied: {company} "
              f"({len(search_chunks)}/{len(chunks)} chunks)")
    else:
        search_chunks = chunks
        print("   ⚠️  No company detected — searching all documents")

    if not search_chunks:
        print(f"   ❌ No chunks found for {company}. Falling back to all chunks.")
        search_chunks = chunks

    # Build a fresh BM25 index on the filtered subset
    tokenized_corpus = [tokenize(c["text"]) for c in search_chunks]
    filtered_bm25    = BM25Okapi(tokenized_corpus)

    # Score with clean query
    tokenized_query = tokenize(query_info["clean_query"])
    scores          = filtered_bm25.get_scores(tokenized_query)
    top_indices     = np.argsort(scores)[::-1][: top_k * 2]  # 2x for reranking

    retrieval_latency = time.perf_counter() - start_time

    results = []
    for idx in top_indices:
        results.append({
            "text"    : search_chunks[idx]["text"],
            "metadata": {
                "source" : search_chunks[idx]["source"],
                "company": search_chunks[idx]["company"],
                "page"   : search_chunks[idx]["page"]
            },
            "score": round(float(scores[idx]), 4)
        })

    # Rerank and return top_k
    rerank_start   = time.perf_counter()
    results        = rerank(query, results, top_k=top_k)
    rerank_latency = time.perf_counter() - rerank_start

    return {
        "chunks"           : results,
        "latency"          : round(retrieval_latency + rerank_latency, 4),
        "retrieval_latency": round(retrieval_latency, 4),
        "rerank_latency"   : round(rerank_latency, 4),
        "method"           : "bm25",
        "company_filter"   : company
    }