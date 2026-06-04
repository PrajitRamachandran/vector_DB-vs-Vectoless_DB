import json
import pickle
import hashlib
import config
import sys
from rank_bm25 import BM25Okapi
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

BM25_INDEX_PATH = Path("vectorless_rag/bm25_index.pkl")
BM25_MANIFEST_PATH = Path("vectorless_rag/bm25_manifest.json")

def tokenize(text:str)->list[str]:
  """
  Simple whitespace tokenizer.
  BM25 works on token lists — splitting on spaces is sufficient
  for English financial text.
  """

  return text.lower().split()

def get_chunks_hash(chunks:list[dict])->str:
  """
  Fingerprints the full chunk list.
  If chunks haven't changed, we skip rebuilding the BM25 index.
  """

  content = json.dumps(
    [c["chunk_id"] for c in chunks],
    sort_keys=True
  ).encode()
  return hashlib.md5(content).hexdigest()

def build_bm25_index(chunks:list[dict]):
  """
  Builds a BM25 index from all chunks and saves it to disk.

  Note: Unlike ChromaDB, BM25 cannot be updated incrementally —
  it needs all documents at once to compute term frequencies correctly.
  But it rebuilds in seconds (no GPU/embedding needed), so this is fine.
  """

  print("=" * 52)
  print("   PHASE 3B — VECTORLESS RAG INDEXER (BM25)")
  print("=" * 52)

  #check if rebuild is needed

  current_hash = get_chunks_hash(chunks)
  
  if BM25_MANIFEST_PATH.exists():
    with open (BM25_MANIFEST_PATH) as f:
      saved=json.load(f)
    if saved.get("chunks_hash") == current_hash:
      print(f"\n✅ BM25 index already up to date — skipping rebuild.")
      print(f"   {saved['chunks_count']} chunks indexed.\n")
      return
    
  print(f"\n🔨 Building BM25 index over {len(chunks)} chunks.....")

  tokenized_corpus = [tokenize(c["text"]) for c in chunks]
  bm25 = BM25Okapi(tokenized_corpus)

  payload = {
    "bm25" : bm25,
    "chunks": chunks
  }

  BM25_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
  with open(BM25_INDEX_PATH, "wb") as f:
    pickle.dump(payload, f)

  #save manifest
  with open(BM25_MANIFEST_PATH, "w") as f:
    json.dump({
      "chunks_hash": current_hash,
      "chunks_count":len(chunks)
    },f,indent=2)

  print(f"✅ BM25 index built and saved to {BM25_INDEX_PATH}")
  print(f"   Total chunks indexed: {len(chunks)}\n")

def load_bm25_index():
  """
  Loads BM25 index from disk.
  Returns (bm25_model, chunks_list).
  """

  if not BM25_INDEX_PATH.exists():
    raise FileNotFoundError(f"BM25 index not found at {BM25_INDEX_PATH}. Please run build_bm25_index() first.")
  
  with open(BM25_INDEX_PATH, "rb") as f:
    payload = pickle.load(f)

  print(f"✅ Loaded BM25 index: {len(payload['chunks'])} chunks")
  return payload["bm25"], payload["chunks"]