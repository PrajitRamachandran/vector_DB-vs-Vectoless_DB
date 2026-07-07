# # vector_rag/indexer.py
# import sys
# from pathlib import Path
# from tqdm import tqdm
# import chromadb
# from chromadb.utils import embedding_functions
# sys.path.append(str(Path(__file__).parent.parent))
# import config  # noqa: E402

# def reset_collection():
#     client = chromadb.PersistentClient(
#         path=config.CHROMA_PERSIST_DIR
#     )

#     try:
#         client.delete_collection(
#             config.CHROMA_COLLECTION
#         )
#         print("✅ Existing collection deleted")
#     except Exception:
#         print("ℹ️ Collection does not exist")

#     return client

# def recreate_collection():
#     client = chromadb.PersistentClient(
#         path=config.CHROMA_PERSIST_DIR
#     )

#     try:
#         client.delete_collection(
#             config.CHROMA_COLLECTION
#         )
#         print("✅ Deleted existing collection")
#     except Exception as e:
#         print(f"No collection to delete: {e}")

#     embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
#         model_name=config.EMBEDDING_MODEL
#     )

#     collection = client.create_collection(
#         name=config.CHROMA_COLLECTION,
#         embedding_function=embedding_fn,
#         metadata={
#             "hnsw:space": "cosine",
#             "hnsw:M": config.HNSW_M,
#             "hnsw:construction_ef": config.HNSW_CONSTRUCTION_EF,
#             "hnsw:search_ef": config.HNSW_SEARCH_EF,
#         }
#     )

#     return collection

# def get_chroma_collection():
#     """
#     Creates or loads ChromaDB collection with tuned HNSW parameters.

#     HNSW (Hierarchical Navigable Small World) is the ANN algorithm
#     ChromaDB uses internally. Tuning it gives better recall with
#     similar query speed.

#     M=32:               more connections in the graph → better navigation
#     construction_ef=200: more candidates examined during build → better index
#     search_ef=100:       more candidates examined during query → better recall
#     """
#     client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)

#     embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
#         model_name = config.EMBEDDING_MODEL   # BAAI/bge-base-en-v1.5
#     )

#     collection = client.get_or_create_collection(
#         name               = config.CHROMA_COLLECTION,
#         embedding_function = embedding_fn,
#         metadata           = {
#             "hnsw:space"          : "cosine",
#             "hnsw:M"              : config.HNSW_M,
#             "hnsw:construction_ef": config.HNSW_CONSTRUCTION_EF,
#             "hnsw:search_ef"      : config.HNSW_SEARCH_EF,
#         }
#     )
#     return collection


# def get_indexed_sources(collection) -> set:
#     results = collection.get(include=["metadatas"])
#     if not results["metadatas"]:
#         return set()
#     return {m["source"] for m in results["metadatas"]}


# def index_chunks(
#     data: dict,
#     progress_callback=None,
#     stage_callback=None,
#     log_callback=None
# ):
#     """
#     Indexes CHILDREN into ChromaDB (small = precise retrieval).
#     Stores parent_id in metadata so retriever can look up parent context.
#     """
#     print("=" * 52)
#     print("   VECTOR RAG INDEXER — Parent-Child + BGE + HNSW")
#     print("=" * 52)

#     if stage_callback:
#         stage_callback(
#             "Loading ChromaDB",
#             1,
#             4
#         )

#     collection = recreate_collection()
#     if log_callback:
#         log_callback("ChromaDB initialized")

#     children = data["children"]

#     new_children = children

#     print(
#         f"\n[Full rebuild mode] "
#         f"({len(new_children)} child chunks)"
#     )

#     if not new_children:
#         print(f"\n[ChromaDB already up to date] {collection.count()} vectors\n")
#         return collection
    

#     if stage_callback:
#         stage_callback(
#             "Generating Embeddings",
#             2,
#             4
#         )

#     print(f"\n[Indexing] {len(new_children)} child chunks...")
#     print(f"   Embedding model: {config.EMBEDDING_MODEL}")

#     BATCH_SIZE = 64   # BGE base is slightly heavier — smaller batch

#     total_batches = (
#         len(new_children) + BATCH_SIZE - 1
#     ) // BATCH_SIZE

#     for batch_num, i in enumerate(
#         tqdm(
#             range(0, len(new_children), BATCH_SIZE),
#             desc="Embedding children"
#         )
#     ):
#         batch = new_children[i: i + BATCH_SIZE]

#         collection.upsert(
#             ids       = [c["chunk_id"] for c in batch],
#             documents = [c["text"]     for c in batch],
#             metadatas = [{
#                 "source"   : c["source"],
#                 "company"  : c["company"],
#                 "page"     : c["page"],
#                 "parent_id": c["parent_id"],   # ← key for context lookup
#                 "type"     : "child"
#             } for c in batch]
#         )

#         if progress_callback:
#             progress_callback(
#                 batch_num + 1,
#                 total_batches
#             )

#         if log_callback:
#             log_callback(
#                 f"Embedded batch "
#                 f"{batch_num + 1}/{total_batches}"
#             )

#         if stage_callback:
#             stage_callback(
#                 "Saving ChromaDB",
#                 3,
#                 4
#             )

#         if stage_callback:
#             stage_callback(
#                 "Completed",
#                 4,
#                 4
#             )

#     print(f"\n[ChromaDB updated] {collection.count()} total vectors\n")
#     return collection


# def build_parent_lookup(data: dict) -> dict:
#     """
#     Builds a fast dict: parent_id → parent chunk
#     Used by the retriever to swap children for their parent context.
#     """
#     return {p["chunk_id"]: p for p in data["parents"]}


# def load_index():
#     collection = get_chroma_collection()
#     if collection.count() == 0:
#         raise RuntimeError("ChromaDB is empty. Run index_chunks() first.")
#     print(f"✅ ChromaDB loaded — {collection.count()} child vectors")
#     return collection






































# vector_rag/indexer.py
"""
Builds and maintains the ChromaDB vector index for the child chunks
produced by the ingestion pipeline.

Always performs a full rebuild (delete + recreate) rather than incremental
upserts. This is deliberate, not a missing feature: a full rebuild is the
only way to guarantee the collection's embedding space and HNSW parameters
can never silently drift out of sync with the current config. This matters
because vector_rag/retriever.py trusts `collection.metadata["hnsw:space"]`
as ground truth for converting distances to similarity scores, and expects
every child chunk's metadata to carry `company` and `parent_id` — both of
which are written here.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Callable, Optional

import chromadb
from chromadb.utils import embedding_functions
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))
import config  # noqa: E402

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int], None]
StageCallback = Callable[[str, int, int], None]
LogCallback = Callable[[str], None]

# Fields index_chunks()/build_parent_lookup() require on every chunk. Missing
# any of these on a child chunk breaks either retrieval filtering (company),
# parent-swap (parent_id), or ChromaDB itself (chunk_id, text).
REQUIRED_CHILD_FIELDS = ("chunk_id", "text", "source", "company", "page", "parent_id")
REQUIRED_PARENT_FIELDS = ("chunk_id", "text")

DEFAULT_BATCH_SIZE = 64
DEFAULT_HNSW_M = 32
DEFAULT_HNSW_CONSTRUCTION_EF = 200
DEFAULT_HNSW_SEARCH_EF = 100


# ============================================================
# Internal helpers
# ============================================================

def _safe_call(callback: Optional[Callable], *args) -> None:
    """
    Invokes an optional, externally-supplied callback (progress/stage/log
    hooks, typically wired up to a UI). A broken or raising callback must
    never be allowed to abort indexing partway through a batch upsert.
    """
    if callback is None:
        return
    try:
        callback(*args)
    except Exception:
        logger.exception(
            "Callback %r raised an exception; continuing indexing.",
            getattr(callback, "__name__", callback),
        )


def _cfg(name: str, default):
    """Reads a config value, falling back to `default` with a warning if unset."""
    value = getattr(config, name, None)
    if value is None:
        logger.warning("config.%s not set; using default %r", name, default)
        return default
    return value


def _hnsw_metadata() -> dict:
    """
    Builds the HNSW tuning metadata used at collection-creation time.

    HNSW (Hierarchical Navigable Small World) is the ANN algorithm ChromaDB
    uses internally. Tuning it gives better recall at similar query speed:
      M:                more graph connections -> better navigation
      construction_ef:  more candidates examined at build time -> denser index
      search_ef:        more candidates examined per query -> better recall

    retriever._get_distance_metric() reads `hnsw:space` back out of this
    exact metadata dict at query time, so its presence here is load-bearing,
    not cosmetic — never omit it.
    """
    return {
        "hnsw:space": "cosine",
        "hnsw:M": _cfg("HNSW_M", DEFAULT_HNSW_M),
        "hnsw:construction_ef": _cfg("HNSW_CONSTRUCTION_EF", DEFAULT_HNSW_CONSTRUCTION_EF),
        "hnsw:search_ef": _cfg("HNSW_SEARCH_EF", DEFAULT_HNSW_SEARCH_EF),
    }


def _get_client() -> chromadb.PersistentClient:
    try:
        return chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)
    except Exception as exc:
        raise RuntimeError(
            f"Could not open ChromaDB persistent client at "
            f"'{config.CHROMA_PERSIST_DIR}': {exc}"
        ) from exc


def _get_embedding_function():
    try:
        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=config.EMBEDDING_MODEL
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not load embedding model '{config.EMBEDDING_MODEL}': {exc}"
        ) from exc


def _delete_collection_if_exists(client: chromadb.PersistentClient) -> bool:
    """
    Deletes the configured collection if present.

    Returns True if a collection was actually deleted, False if there was
    nothing to delete. ChromaDB raises when asked to delete a collection
    that doesn't exist, so "no collection yet" is treated as a normal,
    expected outcome rather than an error.
    """
    try:
        client.delete_collection(config.CHROMA_COLLECTION)
        logger.info("Deleted existing collection '%s'", config.CHROMA_COLLECTION)
        return True
    except Exception as exc:
        logger.info(
            "No existing collection '%s' to delete (%s)",
            config.CHROMA_COLLECTION, exc,
        )
        return False


def _validate_children(children: list[dict]) -> None:
    """
    Fails fast with a clear error if any child chunk is missing a required
    field or if chunk_ids collide — both would otherwise surface as a
    confusing KeyError or a silent upsert overwrite deep inside the
    embedding loop.
    """
    seen_ids: set = set()
    for idx, chunk in enumerate(children):
        missing = [f for f in REQUIRED_CHILD_FIELDS if chunk.get(f) in (None, "")]
        if missing:
            raise ValueError(
                f"Child chunk at index {idx} is missing required field(s) "
                f"{missing}: {chunk!r}"
            )
        chunk_id = chunk["chunk_id"]
        if chunk_id in seen_ids:
            raise ValueError(f"Duplicate chunk_id found in children: {chunk_id!r}")
        seen_ids.add(chunk_id)


def _chunk_metadata(chunk: dict) -> dict:
    """
    Builds ChromaDB-safe metadata for a child chunk. Chroma metadata values
    must be str/int/float/bool — never None — so anything missing is
    coerced to a safe default here rather than raising deep inside
    collection.upsert().
    """
    page = chunk.get("page")
    return {
        "source": chunk.get("source") or "",
        "company": chunk.get("company") or "",
        "page": page if page is not None else -1,
        "parent_id": chunk.get("parent_id") or "",
        "type": "child",
    }


# ============================================================
# Public API
# ============================================================

def reset_collection() -> chromadb.PersistentClient:
    """
    Deletes the configured collection if it exists, WITHOUT recreating it.
    Returns the underlying client so callers can create a fresh collection
    themselves (see recreate_collection() / get_chroma_collection()).
    """
    client = _get_client()
    _delete_collection_if_exists(client)
    return client


def recreate_collection():
    """
    Deletes the configured collection if it exists and creates a brand-new,
    empty one with the configured embedding function and HNSW parameters.
    Use this for a full index rebuild (this is what index_chunks() calls).
    """
    client = reset_collection()
    embedding_fn = _get_embedding_function()

    collection = client.create_collection(
        name=config.CHROMA_COLLECTION,
        embedding_function=embedding_fn,
        metadata=_hnsw_metadata(),
    )
    logger.info("Created fresh collection '%s'", config.CHROMA_COLLECTION)
    return collection


def get_chroma_collection():
    """
    Creates or loads the ChromaDB collection with tuned HNSW parameters.

    Note: get_or_create_collection() only applies the `metadata` argument
    when the collection doesn't already exist — calling this again after
    changing HNSW_* config values will NOT retune an existing collection.
    Use recreate_collection() (via index_chunks()) to apply new HNSW
    settings to an existing index.
    """
    client = _get_client()
    embedding_fn = _get_embedding_function()

    collection = client.get_or_create_collection(
        name=config.CHROMA_COLLECTION,
        embedding_function=embedding_fn,
        metadata=_hnsw_metadata(),
    )
    return collection


def get_indexed_sources(collection) -> set[str]:
    """Returns the set of distinct `source` values currently indexed."""
    results = collection.get(include=["metadatas"])
    metadatas = results.get("metadatas") or []
    return {m["source"] for m in metadatas if m and m.get("source")}


def index_chunks(
    data: dict,
    progress_callback: Optional[ProgressCallback] = None,
    stage_callback: Optional[StageCallback] = None,
    log_callback: Optional[LogCallback] = None,
):
    """
    Indexes CHILD chunks into ChromaDB (small = precise retrieval). Stores
    parent_id in each child's metadata so retriever.retrieve() can swap
    children for their parent context after reranking.

    Args:
        data: dict with a required "children" key — a list of chunk dicts,
            each carrying chunk_id, text, source, company, page, parent_id.
        progress_callback: called as (batch_num, total_batches) after each
            batch is upserted.
        stage_callback: called as (stage_name, step, total_steps) at each
            major phase transition (called exactly once per phase).
        log_callback: called as (message,) for human-readable progress logs.

    Returns:
        The populated ChromaDB collection.

    Raises:
        KeyError: `data` is missing the "children" key.
        ValueError: a child chunk is missing a required field, or two
            children share the same chunk_id.
    """
    logger.info("=" * 52)
    logger.info("   VECTOR RAG INDEXER — Parent-Child + BGE + HNSW")
    logger.info("=" * 52)

    if "children" not in data:
        raise KeyError("data is missing the required 'children' key")

    children = data["children"]
    _validate_children(children)

    _safe_call(stage_callback, "Loading ChromaDB", 1, 4)
    collection = recreate_collection()
    _safe_call(log_callback, "ChromaDB initialized")

    logger.info("[Full rebuild mode] (%d child chunks)", len(children))

    if not children:
        logger.info("[ChromaDB empty] no children to index; %d vectors", collection.count())
        _safe_call(stage_callback, "Completed", 4, 4)
        return collection

    _safe_call(stage_callback, "Generating Embeddings", 2, 4)
    logger.info("[Indexing] %d child chunks...", len(children))
    logger.info("   Embedding model: %s", config.EMBEDDING_MODEL)

    batch_size = _cfg("INDEX_BATCH_SIZE", DEFAULT_BATCH_SIZE)
    total_batches = (len(children) + batch_size - 1) // batch_size

    for batch_num, start in enumerate(
        tqdm(
            range(0, len(children), batch_size),
            desc="Embedding children",
            total=total_batches,
        )
    ):
        batch = children[start: start + batch_size]

        collection.upsert(
            ids=[c["chunk_id"] for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[_chunk_metadata(c) for c in batch],
        )

        _safe_call(progress_callback, batch_num + 1, total_batches)
        _safe_call(log_callback, f"Embedded batch {batch_num + 1}/{total_batches}")

    # Stage transitions fire exactly once, after all batches are done — not
    # once per batch, which would have falsely reported "Completed" while
    # embedding was still in progress.
    _safe_call(stage_callback, "Saving ChromaDB", 3, 4)
    _safe_call(stage_callback, "Completed", 4, 4)

    logger.info("[ChromaDB updated] %d total vectors", collection.count())
    return collection


def build_parent_lookup(data: dict) -> dict:
    """
    Builds a fast dict: parent_id -> parent chunk. Used by
    retriever.retrieve() to swap children for their parent context.

    Args:
        data: dict with a required "parents" key — a list of chunk dicts,
            each carrying at least chunk_id and text.

    Raises:
        KeyError: `data` is missing the "parents" key.
    """
    if "parents" not in data:
        raise KeyError("data is missing the required 'parents' key")

    lookup: dict = {}
    for idx, parent in enumerate(data["parents"]):
        missing = [f for f in REQUIRED_PARENT_FIELDS if not parent.get(f)]
        if missing:
            logger.warning(
                "Parent at index %d missing field(s) %s; skipping.", idx, missing
            )
            continue

        chunk_id = parent["chunk_id"]
        if chunk_id in lookup:
            logger.warning(
                "Duplicate parent chunk_id %r; keeping first occurrence.", chunk_id
            )
            continue

        lookup[chunk_id] = parent

    return lookup


def load_index():
    """
    Loads the existing ChromaDB collection.

    Raises:
        RuntimeError: the collection doesn't exist or is empty — the
            caller needs to run index_chunks() first.
    """
    collection = get_chroma_collection()
    if collection.count() == 0:
        raise RuntimeError(
            f"ChromaDB collection '{config.CHROMA_COLLECTION}' is empty. "
            f"Run index_chunks() first."
        )
    logger.info("ChromaDB loaded — %d child vectors", collection.count())
    return collection