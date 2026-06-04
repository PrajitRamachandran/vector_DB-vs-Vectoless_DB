# vector_rag/retriever.py
import sys
import time
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import config
from vector_rag.indexer      import load_index
from utils.query_processor   import preprocess_query
from reranker                import rerank


def retrieve(query: str, collection, top_k: int = config.TOP_K) -> dict:
    """
    Improved retriever with:
    1. Company-aware metadata filtering — no cross-document contamination
    2. Fetches 2x TOP_K candidates, then reranks to final TOP_K
    """
    start_time = time.perf_counter()
    query_info = preprocess_query(query)
    company    = query_info["company"]

    # Build ChromaDB where-filter if company detected
    where_filter = {"company": company} if company else None

    if company:
        print(f"   🎯 Company filter applied: {company}")
    else:
        print("   ⚠️  No company detected — searching all documents")

    # Fetch 2x candidates for reranking
    fetch_k = top_k * 2

    query_kwargs = dict(
        query_texts = [query],
        n_results   = fetch_k,
        include     = ["documents", "metadatas", "distances"]
    )
    if where_filter:
        query_kwargs["where"] = where_filter

    results = collection.query(**query_kwargs)

    retrieval_latency = time.perf_counter() - start_time

    # Build chunk list
    chunks = []
    for i in range(len(results["documents"][0])):
        distance   = results["distances"][0][i]
        similarity = round(1 / (1 + distance), 4)
        chunks.append({
            "text"    : results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "score"   : similarity
        })

    # Rerank and return top_k
    rerank_start  = time.perf_counter()
    chunks        = rerank(query, chunks, top_k=top_k)
    rerank_latency = time.perf_counter() - rerank_start

    return {
        "chunks"          : chunks,
        "latency"         : round(retrieval_latency + rerank_latency, 4),
        "retrieval_latency": round(retrieval_latency, 4),
        "rerank_latency"  : round(rerank_latency, 4),
        "method"          : "vector",
        "company_filter"  : company
    }