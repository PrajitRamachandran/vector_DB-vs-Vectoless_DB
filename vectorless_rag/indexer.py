# vectorless_rag/indexer.py
import re
import sys
import json
import pickle
import hashlib
from pathlib import Path
from rank_bm25 import BM25Okapi
sys.path.append(str(Path(__file__).parent.parent))
import config

BM25_INDEX_PATH    = Path("vectorless_rag/bm25_index.pkl")
BM25_MANIFEST_PATH = Path("vectorless_rag/bm25_manifest.json")


def tokenize(text: str) -> list[str]:
    """
    Financial-aware tokenizer.

    Standard whitespace split misses:
    - "$60.9B"  → should be one token
    - "122%"    → should be one token
    - "Q3-2024" → should be preserved

    This tokenizer preserves numbers, percentages, and
    dollar amounts that are critical for financial QA.
    """
    text   = text.lower()
    tokens = re.findall(
        r'\$[\d,.]+[bmt]?'     # $60.9B, $1,234M
        r'|\d+\.?\d*%'         # 122%, 14.3%
        r'|\d{4}'              # years: 2024, 2023
        r'|\d+\.?\d*[bmt]'     # 60.9b, 1.2t
        r'|[a-zA-Z]{2,}',      # regular words (min 2 chars)
        text
    )
    return tokens


def get_chunks_hash(children: list[dict]) -> str:
    content = json.dumps(
        [c["chunk_id"] for c in children],
        sort_keys=True
    ).encode()
    return hashlib.md5(content).hexdigest()


def build_bm25_index(children: list[dict] | dict):
    """
    Indexes CHILDREN in BM25 (same as ChromaDB — small precise chunks).
    Stores children alongside so retriever can look up parents.

    Accepts either:
    - a list of child chunk dicts, or
    - a dict containing a `children` key.
    """
    print("=" * 52)
    print("   VECTORLESS RAG INDEXER — BM25 + Financial Tokenizer")
    print("=" * 52)

    if isinstance(children, dict):
        children = children.get("children", [])

    current_hash = get_chunks_hash(children)

    if BM25_MANIFEST_PATH.exists():
        with open(BM25_MANIFEST_PATH) as f:
            saved = json.load(f)
        if saved.get("chunks_hash") == current_hash:
            print(f"\n✅ BM25 already up to date — "
                  f"{saved['chunks_count']} children indexed\n")
            return

    print(f"\n🔨 Tokenizing and indexing {len(children)} children...")

    tokenized = [tokenize(c["text"]) for c in children]
    bm25      = BM25Okapi(tokenized)

    BM25_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "children": children}, f)

    with open(BM25_MANIFEST_PATH, "w") as f:
        json.dump({
            "chunks_hash" : current_hash,
            "chunks_count": len(children)
        }, f, indent=2)

    print(f"✅ BM25 index built — {len(children)} children indexed\n")


def load_bm25_index() -> tuple:
    if not BM25_INDEX_PATH.exists():
        raise RuntimeError("BM25 index not found. Run build_bm25_index() first.")
    with open(BM25_INDEX_PATH, "rb") as f:
        payload = pickle.load(f)
    print(f"✅ BM25 loaded — {len(payload['children'])} children")
    return payload["bm25"], payload["children"]