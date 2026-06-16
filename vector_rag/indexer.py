# vector_rag/indexer.py
import sys
from pathlib import Path
from tqdm import tqdm
import chromadb
from chromadb.utils import embedding_functions
sys.path.append(str(Path(__file__).parent.parent))
import config


def get_chroma_collection():
    """
    Creates or loads ChromaDB collection with tuned HNSW parameters.

    HNSW (Hierarchical Navigable Small World) is the ANN algorithm
    ChromaDB uses internally. Tuning it gives better recall with
    similar query speed.

    M=32:               more connections in the graph → better navigation
    construction_ef=200: more candidates examined during build → better index
    search_ef=100:       more candidates examined during query → better recall
    """
    client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name = config.EMBEDDING_MODEL   # BAAI/bge-base-en-v1.5
    )

    collection = client.get_or_create_collection(
        name               = config.CHROMA_COLLECTION,
        embedding_function = embedding_fn,
        metadata           = {
            "hnsw:space"          : "cosine",
            "hnsw:M"              : config.HNSW_M,
            "hnsw:construction_ef": config.HNSW_CONSTRUCTION_EF,
            "hnsw:search_ef"      : config.HNSW_SEARCH_EF,
        }
    )
    return collection


def get_indexed_sources(collection) -> set:
    results = collection.get(include=["metadatas"])
    if not results["metadatas"]:
        return set()
    return {m["source"] for m in results["metadatas"]}


def index_chunks(
    data: dict,
    progress_callback=None,
    stage_callback=None,
    log_callback=None
):
    """
    Indexes CHILDREN into ChromaDB (small = precise retrieval).
    Stores parent_id in metadata so retriever can look up parent context.
    """
    print("=" * 52)
    print("   VECTOR RAG INDEXER — Parent-Child + BGE + HNSW")
    print("=" * 52)

    if stage_callback:
        stage_callback(
            "Loading ChromaDB",
            1,
            4
        )

    collection      = get_chroma_collection()
    if log_callback:
        log_callback("ChromaDB initialized")

    indexed_sources = get_indexed_sources(collection)
    children        = data["children"]

    new_children = [c for c in children
                    if c["source"] not in indexed_sources]

    if not new_children:
        print(f"\n✅ ChromaDB already up to date — {collection.count()} vectors\n")
        return collection
    

    if stage_callback:
        stage_callback(
            "Generating Embeddings",
            2,
            4
        )

    print(f"\n📤 Indexing {len(new_children)} child chunks...")
    print(f"   Embedding model: {config.EMBEDDING_MODEL}")

    BATCH_SIZE = 64   # BGE base is slightly heavier — smaller batch

    total_batches = (
    len(new_children) + BATCH_SIZE - 1
) // BATCH_SIZE

    for batch_num, i in enumerate(
        tqdm(
            range(0, len(new_children), BATCH_SIZE),
            desc="Embedding children"
        )
    ):
        batch = new_children[i: i + BATCH_SIZE]

        collection.upsert(
            ids       = [c["chunk_id"] for c in batch],
            documents = [c["text"]     for c in batch],
            metadatas = [{
                "source"   : c["source"],
                "company"  : c["company"],
                "page"     : c["page"],
                "parent_id": c["parent_id"],   # ← key for context lookup
                "type"     : "child"
            } for c in batch]
        )

        if progress_callback:
            progress_callback(
                batch_num + 1,
                total_batches
            )

        if log_callback:
            log_callback(
                f"Embedded batch "
                f"{batch_num + 1}/{total_batches}"
            )

        if stage_callback:
            stage_callback(
                "Saving ChromaDB",
                3,
                4
            )

        if stage_callback:
            stage_callback(
                "Completed",
                4,
                4
            )

    print(f"\n✅ ChromaDB updated — {collection.count()} total vectors\n")
    return collection


def build_parent_lookup(data: dict) -> dict:
    """
    Builds a fast dict: parent_id → parent chunk
    Used by the retriever to swap children for their parent context.
    """
    return {p["chunk_id"]: p for p in data["parents"]}


def load_index():
    collection = get_chroma_collection()
    if collection.count() == 0:
        raise RuntimeError("ChromaDB is empty. Run index_chunks() first.")
    print(f"✅ ChromaDB loaded — {collection.count()} child vectors")
    return collection