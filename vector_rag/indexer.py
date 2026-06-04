import json
import config
import sys
from tqdm import tqdm
import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))



def get_chroma_collection():
  """
  Returns the ChromaDB Colelction. Creates if it doesn't exist.
  chromaDB persist to disk automatically, no re-embedding on restart
  """

  client = chromadb.PersistentClient(path=config.CHROMA_PERSISTENT_DIR)

  embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=config.EMBEDDING_MODEL)

  collection = client.get_or_create_collection(
    name = config.CHROMA_COLLECTION,
    embedding_function = embedding_fn,
      metadata = {"hnsw:space":"cosine"}   #used for cosine similarity
  )

  return collection

def get_indexed_sources(collection)->set[str]:
  """
  Asks chormadb which source files are already indexed, to avoid re-embeddings
  """

  results = collection.get(include=["metadatas"])

  if not results["metadatas"]:
    return set()
  
  return {m["source"] for m in results["metadatas"]}

def index_chunks(chunks:list[dict]):
  """
  Embeds and stores chunks in chromadb. skips already indexed sources.
  """

  print("="*52)
  print("Phase 3A : Vector RAG INDEXER (CHROMADB")
  print("="*52)

  collection = get_chroma_collection()
  indexed_sources = get_indexed_sources(collection)

  new_chunks = [
    c for c in chunks
    if c["source"] not in indexed_sources
  ]

  if not new_chunks:
    total = collection.count()
    print(f"\n✅ ChromaDB already up to date.")
    print(f"   {total} vectors in index — nothing to add.\n")
    return collection
  
  print(f"\n📥 Already indexed: {indexed_sources or 'nothing yet'}")
  print(f"📤 Chunks to embed:  {len(new_chunks)}\n")

  batch_size = 50

  for i in tqdm(range(0, len(new_chunks), batch_size), desc="Embedding & Indexing"):
    batch = new_chunks[i:i + batch_size]

    collection.upsert(
    ids=[str(c["chunk_id"]) for c in batch],
    documents=[c["text"] for c in batch],
    metadatas=[
        {
            "source": c["source"],
            "company": c["company"],
            "page": c["page"],
            "chunk_id": c["chunk_id"]
        }for c in batch])
    
  print(f"\n✅ ChromaDB index updated.")
  print(f"   Total vectors in index: {collection.count()}\n")

  return collection


def load_index():
  """
  loads all existing chormadb index
  """

  collection = get_chroma_collection()
  count = collection.count()

  if count==0:
    raise RuntimeError(f"ChromaDB index is empty. Run the index_chunks() first.")
  
  print(f"✅ Loaded ChromaDB index: {count} vectors.")
  return collection