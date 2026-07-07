# # vector_rag/retriever.py
# #
# # Bug fixes applied:
# #
# # 1. MISSING BGE QUERY PREFIX
# #    BGE (BAAI/bge-base-en-v1.5) requires queries to be prefixed with an
# #    instruction string for best retrieval quality. Documents are indexed
# #    WITHOUT this prefix — only the query gets it. Without this, embedding
# #    similarity scores are lower than they should be, causing weaker retrieval.
# #
# # 2. BROKEN FALLBACK — wrong company results returned silently
# #    Old code: `if not child_chunks and where_filter is not None`
# #    ChromaDB ALWAYS returns n_results chunks (from whatever company is closest
# #    in embedding space). So child_chunks was never empty → fallback never fired
# #    → NVIDIA chunks were returned for Amazon/Microsoft/Netflix questions.
# #    Fix: check (a) fewer than 3 results, OR (b) results are mostly wrong company.
# #
# # 3. STALE INDEX GUARD
# #    After a BGE upgrade (384-dim MiniLM → 768-dim BGE), ChromaDB silently loads
# #    the old collection. The company metadata filter still runs but the distance
# #    scores are meaningless. The _chunks_are_correct_company() guard catches this.

# import sys
# import time
# from pathlib import Path

# sys.path.append(str(Path(__file__).parent.parent))

# import config
# from utils.query_processor import preprocess_query
# from reranker import rerank

# # BGE-specific instruction prefix — ONLY for queries, never for indexed documents.
# # Source: https://huggingface.co/BAAI/bge-base-en-v1.5
# BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


# def _bge_query(text: str) -> str:
#     """Returns the query with the required BGE instruction prefix."""
#     return BGE_QUERY_PREFIX + text.strip()


# def _run_query(collection, query_text: str, where_filter: dict | None) -> dict:
#     """Executes a ChromaDB query safely."""
#     kwargs = dict(
#         query_texts=[query_text],
#         n_results=config.FETCH_K,
#         include=["documents", "metadatas", "distances"],
#     )
#     if where_filter:
#         kwargs["where"] = where_filter
#     return collection.query(**kwargs)


# def _build_child_chunks(results: dict) -> list[dict]:
#     """Converts raw ChromaDB results into child chunk dicts."""
#     documents = (results.get("documents") or [[]])[0]
#     metadatas = (results.get("metadatas") or [[]])[0]
#     distances = (results.get("distances") or [[]])[0]

#     child_chunks = []
#     for i in range(min(len(documents), len(metadatas), len(distances))):
#         distance   = distances[i]
#         similarity = round(1 / (1 + float(distance)), 4)
#         child_chunks.append({
#             "text"    : documents[i],
#             "metadata": metadatas[i] or {},
#             "score"   : similarity,
#         })
#     return child_chunks


# def _filter_company_chunks(chunks: list[dict], company: str | None) -> list[dict]:
#     """Keeps only chunks whose metadata company matches the target company."""
#     if not company:
#         return chunks
#     return [
#         chunk for chunk in chunks
#         if chunk.get("metadata", {}).get("company", "") == company
#     ]


# def retrieve(
#     query:         str,
#     collection,
#     parent_lookup: dict,
#     top_k:         int = config.TOP_K,
# ) -> dict:
#     """
#     Full Vector RAG retrieval with all safety guards:

#     1.  Preprocess query — extract company, build BGE-prefixed query string
#     2.  Try ChromaDB with company WHERE filter
#     3a. If < 3 chunks returned → fallback (filter was too narrow)
#     3b. If hit-rate < 50%      → fallback (stale index / wrong-company contamination)
#     4.  On fallback: query without filter, keep only correct-company chunks
#     5.  Rerank children with cross-encoder
#     6.  Swap children → parents (deduplicated)
#     7.  Final safety net: if parent lookup finds nothing, keep child text
#     """
#     start_time = time.perf_counter()

#     query_info   = preprocess_query(query)
#     company = query_info.get("company")
#     # Use the semantic query so dense retrieval keeps the company name and
#     # avoids possessive artifacts like "'s".
#     raw_query = (
#         query_info.get("semantic_query")
#         or query_info.get("original")
#         or query
#     )

#     # Build the BGE-prefixed query string (fix #1)
#     bge_q = _bge_query(raw_query)
#     print("\n===================")
#     print("ORIGINAL :", query_info["original"])
#     print("COMPANY  :", company)
#     print("YEAR     :", query_info["year"])
#     print("USED FOR SEARCH:")
#     print(raw_query)
#     print("===================\n")

#     # ── First attempt: with company filter ────────────────────────────────────
#     where_filter  = {"company": company} if company else None
#     print("SEARCH QUERY:", bge_q)
#     results       = _run_query(collection, bge_q, where_filter)
#     child_chunks  = _filter_company_chunks(_build_child_chunks(results), company)

#     # ── Fallback decision (fix #2 + #3) ──────────────────────────────────────
#     # Trigger fallback when the filter returned too few results OR the chunks
#     # that came back are mostly from the wrong company (stale index signal).
#     if where_filter is not None and len(child_chunks) < max(3, top_k):
#         print("SEARCH QUERY:", bge_q)
#         results        = _run_query(collection, bge_q, None)
#         all_chunks     = _build_child_chunks(results)
#         child_chunks   = _filter_company_chunks(all_chunks, company)

#     retrieval_latency = time.perf_counter() - start_time

#     # ── Nothing found — return cleanly ───────────────────────────────────────
#     if not child_chunks:
#         return {
#             "chunks"           : [],
#             "latency"          : round(retrieval_latency, 4),
#             "retrieval_latency": round(retrieval_latency, 4),
#             "rerank_latency"   : 0.0,
#             "method"           : "vector",
#             "company_filter"   : company,
#             "search_query"     : bge_q,
#         }

#     # ── Rerank children ───────────────────────────────────────────────────────
#     rerank_start   = time.perf_counter()
#     child_chunks   = rerank(query, child_chunks, top_k=config.FETCH_K)
#     rerank_latency = time.perf_counter() - rerank_start

#     # ── Swap children → parents (deduplicated) ────────────────────────────────
#     seen_parents = set()
#     final_chunks = []
#     for child in child_chunks:
#         if len(final_chunks) >= top_k:
#             break
#         meta      = child.get("metadata") or {}
#         parent_id = meta.get("parent_id")
#         if not parent_id or parent_id in seen_parents:
#             continue
#         parent = parent_lookup.get(parent_id)
#         parent = parent_lookup.get(parent_id)

#         if parent:

#             print("\n========== DEBUG ==========")
#             print("PARENT ID:", parent_id)

#             print(
#                 "CHILD COMPANY:",
#                 meta.get("company")
#             )

#             print(
#                 "LOOKUP COMPANY:",
#                 parent.get("company")
#             )

#             print(
#                 "LOOKUP PAGE:",
#                 parent.get("page")
#             )

#             print(
#                 "LOOKUP CHUNK:",
#                 parent.get("chunk_id")
#             )

#             print("===========================")

#         if parent:
#             seen_parents.add(parent_id)
#             final_chunks.append({
#                 "text"        : parent.get("text", ""),
#                 "metadata"    : meta,
#                 "score"       : child.get("rerank_score", child.get("score", 0.0)),
#                 "rerank_score": child.get("rerank_score", child.get("score", 0.0)),
#                 "child_text"  : child.get("text", ""),
#             })
#         else:
#             text = (child.get("text") or "").strip()
#             if not text:
#                 continue
#             if parent_id:
#                 seen_parents.add(parent_id)
#             final_chunks.append({
#                 "text"        : text,
#                 "metadata"    : meta,
#                 "score"       : child.get("rerank_score", child.get("score", 0.0)),
#                 "rerank_score": child.get("rerank_score", child.get("score", 0.0)),
#                 "child_text"  : text,
#             })

#     # ── Safety net: parent lookup failed — keep child text ────────────────────
#     if not final_chunks:
#         for child in child_chunks[:top_k]:
#             text = (child.get("text") or "").strip()
#             if not text:
#                 continue
#             final_chunks.append({
#                 "text"        : text,
#                 "metadata"    : child.get("metadata") or {},
#                 "score"       : child.get("rerank_score", child.get("score", 0.0)),
#                 "rerank_score": child.get("rerank_score", child.get("score", 0.0)),
#                 "child_text"  : text,
#             })

#     return {
#         "chunks"           : final_chunks,
#         "latency"          : round(retrieval_latency + rerank_latency, 4),
#         "retrieval_latency": round(retrieval_latency, 4),
#         "rerank_latency"   : round(rerank_latency, 4),
#         "method"           : "vector",
#         "company_filter"   : company,
#         "search_query"     : bge_q,
#     }







# vector_rag/retriever.py

import logging
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import config
from utils.query_processor import preprocess_query
from reranker import rerank

logger = logging.getLogger(__name__)

# BGE-specific instruction prefix — ONLY for queries, never for indexed documents.
# Source: https://huggingface.co/BAAI/bge-large-en-v1.5
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Minimum length (chars) a cleaned query must have to be trusted; below this
# we fall back to the original question (protects against a query that was
# *only* a company name, which `clean_query` would reduce to near-nothing).
MIN_CLEAN_QUERY_LEN = 3

# How many candidates to pull on the *fallback* (no company filter) query.
# Wider than the primary fetch because we still need to find the target
# company's chunks somewhere in this pool before client-side filtering.
FALLBACK_FETCH_MULTIPLIER = 4
FALLBACK_MAX_N = 100

# Minimum fraction of returned chunks that must match the target company for
# the primary (filtered) query to be trusted. Below this, something is off
# (stale index, filter/schema mismatch, sparse data) and we fall back.
MIN_COMPANY_HIT_RATE = 0.5


def _bge_query(text: str) -> str:
    """Returns the query with the required BGE instruction prefix."""
    return BGE_QUERY_PREFIX + text.strip()


def _build_search_text(query_info: dict, original_query: str) -> str:
    """
    Picks the best available text to embed for dense retrieval.

    Prefers `clean_query` (company name + possessive artifacts already
    stripped by query_processor), but falls back to the raw question if
    cleaning left too little text to search on meaningfully.
    """
    clean = (query_info.get("clean_query") or "").strip()
    if len(clean) >= MIN_CLEAN_QUERY_LEN:
        return clean
    return (query_info.get("original") or original_query or "").strip()


def _normalize_company(value: str | None) -> str:
    """Normalizes a company string for safe equality comparisons."""
    return (value or "").strip().upper()


def _get_distance_metric(collection) -> str:
    """
    Reads the distance metric the collection was actually created with
    (Chroma stores this as `hnsw:space` in collection metadata), so we
    convert distances to similarity correctly instead of assuming L2.
    Defaults to "l2" (Chroma's own default) if metadata is unavailable.
    """
    try:
        meta = getattr(collection, "metadata", None) or {}
        return str(meta.get("hnsw:space", "l2")).lower()
    except Exception:
        return "l2"


def _distance_to_similarity(distance: float, metric: str) -> float:
    """
    Converts a raw ChromaDB distance into a display-friendly similarity
    score. The conversion is metric-aware:
      - cosine distance is bounded [0, 2] -> linearly mapped to [0, 1]
      - l2 / ip (unbounded) -> monotonic inverse, safe for ranking/display
    Note: this score is informational (pre-rerank). Final ordering of
    returned chunks is driven by the cross-encoder's rerank_score.
    """
    distance = float(distance)
    if metric == "cosine":
        return round(max(0.0, 1.0 - (distance / 2.0)), 4)
    return round(1.0 / (1.0 + distance), 4)


def _run_query(
    collection,
    query_text: str,
    where_filter: dict | None,
    n_results: int,
) -> dict:
    """
    Executes a ChromaDB query with error handling.

    Infra-level failures (e.g. a stale/incompatible collection after an
    embedding upgrade) are logged with full context and re-raised, rather
    than being swallowed — a broken pipeline should never silently look
    like "no relevant documents found".
    """
    kwargs = dict(
        query_texts=[query_text],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
    if where_filter:
        kwargs["where"] = where_filter

    try:
        return collection.query(**kwargs)
    except Exception as exc:
        logger.error(
            "ChromaDB query failed | query=%r | where=%r | n_results=%s | error=%s",
            query_text, where_filter, n_results, exc,
        )
        raise RuntimeError(
            f"Vector retrieval failed while querying collection "
            f"'{getattr(collection, 'name', '?')}': {exc}"
        ) from exc


def _build_child_chunks(results: dict, metric: str) -> list[dict]:
    """Converts raw ChromaDB results into child chunk dicts."""
    documents = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]

    child_chunks = []
    for i in range(min(len(documents), len(metadatas), len(distances))):
        child_chunks.append({
            "text"    : documents[i],
            "metadata": metadatas[i] or {},
            "score"   : _distance_to_similarity(distances[i], metric),
        })
    return child_chunks


def _filter_company_chunks(chunks: list[dict], company: str | None) -> list[dict]:
    """Keeps only chunks whose metadata company matches the target company."""
    if not company:
        return chunks
    target = _normalize_company(company)
    return [
        chunk for chunk in chunks
        if _normalize_company(chunk.get("metadata", {}).get("company")) == target
    ]


def _chunks_are_correct_company(
    chunks: list[dict],
    company: str | None,
    min_hit_rate: float = MIN_COMPANY_HIT_RATE,
) -> bool:
    """
    Sanity-checks that a set of chunks returned for a company-filtered query
    actually belong to that company. Even though ChromaDB's `where` filter
    should already guarantee this server-side, we re-verify client-side —
    this is what catches a stale/misaligned index or a metadata schema
    mismatch that would otherwise make the filter a silent no-op.

    Returns True if there's nothing to check (no company was requested) or
    if at least `min_hit_rate` of the chunks match the target company.
    """
    if not company:
        return True
    if not chunks:
        return False
    target = _normalize_company(company)
    matches = sum(
        1 for c in chunks
        if _normalize_company(c.get("metadata", {}).get("company")) == target
    )
    hit_rate = matches / len(chunks)
    if hit_rate < min_hit_rate:
        logger.warning(
            "Low company hit-rate for %r: %.0f%% of %d returned chunks matched "
            "(expected >= %.0f%%). Possible stale index or filter/schema mismatch.",
            company, hit_rate * 100, len(chunks), min_hit_rate * 100,
        )
    return hit_rate >= min_hit_rate


def retrieve(
    query:         str,
    collection,
    parent_lookup: dict,
    top_k:         int = config.TOP_K,
) -> dict:
    """
    Full Vector RAG retrieval with all safety guards:

    1.  Preprocess query — extract company, build a BGE-prefixed search string
        from the *cleaned* query text (not the raw question).
    2.  Try ChromaDB with a company WHERE filter.
    3.  Trigger fallback if too few chunks came back OR the company hit-rate
        on what came back is too low (stale index / filter mismatch signal).
    4.  On fallback: query a much wider candidate pool without the filter,
        then keep only correct-company chunks client-side.
    5.  Rerank children with a cross-encoder.
    6.  Swap children -> parents (deduplicated); a chunk with a missing or
        unresolvable parent degrades to its own child text rather than being
        dropped.
    7.  Final safety net: if the whole swap step somehow produces nothing,
        keep raw reranked child text.
    """
    start_time = time.perf_counter()

    query_info = preprocess_query(query)
    company    = query_info.get("company")
    raw_query  = _build_search_text(query_info, query)
    bge_q      = _bge_query(raw_query)

    logger.debug(
        "retrieve() | original=%r | company=%r | year=%r | search_text=%r",
        query_info.get("original"), company, query_info.get("year"), raw_query,
    )

    metric = _get_distance_metric(collection)

    # ── First attempt: with company filter ─────────────────────────────────
    where_filter = {"company": company} if company else None
    results      = _run_query(collection, bge_q, where_filter, config.FETCH_K)
    raw_chunks   = _build_child_chunks(results, metric)
    child_chunks = _filter_company_chunks(raw_chunks, company)

    # ── Fallback decision ────────────────────────────────────────────────
    # Trigger fallback when the filter returned too few results, OR the
    # results that came back don't actually look like the right company
    # (defensive re-check — see _chunks_are_correct_company docstring).
    used_fallback = False
    if where_filter is not None and (
        len(child_chunks) < max(3, top_k)
        or not _chunks_are_correct_company(raw_chunks, company)
    ):
        used_fallback = True
        fallback_n = min(config.FETCH_K * FALLBACK_FETCH_MULTIPLIER, FALLBACK_MAX_N)
        logger.debug(
            "Falling back to unfiltered search | company=%r | n_results=%d",
            company, fallback_n,
        )
        results      = _run_query(collection, bge_q, None, fallback_n)
        raw_chunks   = _build_child_chunks(results, metric)
        child_chunks = _filter_company_chunks(raw_chunks, company)

    retrieval_latency = time.perf_counter() - start_time

    # ── Nothing found — return cleanly ──────────────────────────────────────
    if not child_chunks:
        return {
            "chunks"           : [],
            "latency"          : round(retrieval_latency, 4),
            "retrieval_latency": round(retrieval_latency, 4),
            "rerank_latency"   : 0.0,
            "method"           : "vector",
            "company_filter"   : company,
            "search_query"     : bge_q,
            "used_fallback"    : used_fallback,
        }

    # ── Rerank children ──────────────────────────────────────────────────
    # top_k=config.FETCH_K here is intentional: we want the *entire* pool
    # re-sorted by the cross-encoder, not truncated yet, because the
    # parent-dedup step below may need to skip several children that map to
    # an already-seen parent before it fills up `top_k` unique parents.
    rerank_start   = time.perf_counter()
    child_chunks   = rerank(query, child_chunks, top_k=config.FETCH_K)
    rerank_latency = time.perf_counter() - rerank_start

    # ── Swap children → parents (deduplicated) ──────────────────────────
    seen_parents = set()
    final_chunks = []
    for child in child_chunks:
        if len(final_chunks) >= top_k:
            break

        meta      = child.get("metadata") or {}
        parent_id = meta.get("parent_id")

        # Skip only if we've already emitted this exact parent — do NOT
        # skip just because parent_id is missing; that case must still
        # fall through to the child-text fallback below.
        if parent_id and parent_id in seen_parents:
            continue

        parent = parent_lookup.get(parent_id) if parent_id else None

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
            # Parent missing entirely, or parent_id absent/unresolvable —
            # degrade gracefully to the child's own text instead of
            # dropping a potentially highly relevant chunk.
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

    # ── Final safety net (should rarely trigger given the fix above) ────────
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
        "used_fallback"    : used_fallback,
    }