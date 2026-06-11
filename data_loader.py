# data_loader.py
import fitz
import os
import re
import json
import hashlib
from pathlib import Path
from datetime import datetime
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm
import config

# ─────────────────────────────────────────────────────────
# MANIFEST — tracks which PDFs have been processed
# ─────────────────────────────────────────────────────────

MANIFEST_PATH = Path(config.DATA_PROCESSED_DIR) / "manifest.json"
CHUNKS_PATH   = Path(config.DATA_PROCESSED_DIR) / "chunks.json"


def get_file_hash(filepath: str) -> str:
    """
    Generates a unique fingerprint for a PDF file using its contents.
    If the file changes (re-downloaded, updated), the hash changes
    and it gets re-processed automatically.
    """
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()


def load_manifest() -> dict:
    """
    Loads the manifest file. Returns empty dict if it doesn't exist yet.
    Manifest structure:
    {
        "nvidia_10k.pdf": {
            "hash": "abc123...",
            "processed_at": "2024-01-01T12:00:00",
            "chunks_count": 1102,
            "pages_count": 190
        },
        ...
    }
    """
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_manifest(manifest: dict):
    Path(config.DATA_PROCESSED_DIR).mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def get_new_pdfs(raw_dir: str, manifest: dict) -> list[Path]:
    """
    Compares all PDFs in raw/ against the manifest.
    Returns only PDFs that are new or have changed since last run.
    """
    all_pdfs = list(Path(raw_dir).glob("*.pdf"))
    new_pdfs = []

    for pdf_path in all_pdfs:
        filename     = pdf_path.name
        current_hash = get_file_hash(str(pdf_path))

        if filename not in manifest:
            print(f"  🆕 New PDF detected: {filename}")
            new_pdfs.append(pdf_path)
        elif manifest[filename]["hash"] != current_hash:
            print(f"  🔄 Changed PDF detected: {filename} — will re-process")
            new_pdfs.append(pdf_path)
        else:
            print(f"  ✅ Already processed: {filename} — skipping")

    return new_pdfs


# ─────────────────────────────────────────────────────────
# EXTRACTION
# ─────────────────────────────────────────────────────────

import fitz
import pymupdf4llm
from pathlib import Path
import re

import pymupdf4llm
import re
from pathlib import Path

def extract_text_from_pdf(pdf_path: Path) -> list[dict]:
    """
    Extracts text page by page from a single PDF using Markdown 
    to preserve tabular structure.
    """
    company_name = re.sub(r'[_\-](10k|10K|annual|report).*', '', pdf_path.stem).upper()
    pages = []

    # Process the entire document at once to prevent file handle leaks
    # page_chunks=True returns a list of dicts: [{'text': '...', 'metadata': ...}, ...]
    md_pages = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True)

    for i, page_data in enumerate(md_pages):
        text = page_data.get("text", "")

        if len(text.strip()) < 50:   # skip blank/image-only pages
            continue

        pages.append({
            "text"        : text,
            "source"      : pdf_path.name,
            "company"     : company_name,
            "page"        : i + 1,
            "total_pages" : len(md_pages)
        })

    return pages

# ─────────────────────────────────────────────────────────
# CLEANING
# ─────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = re.sub(r'\n{3,}', '\n\n', text)           # collapse excess newlines
    text = re.sub(r' {2,}', ' ', text)               # collapse excess spaces
    text = re.sub(r'^\s*-?\s*\d+\s*-?\s*$', '',      # remove lone page numbers
                  text, flags=re.MULTILINE)
    text = re.sub(r'Table of Contents', '',
                  text, flags=re.IGNORECASE)
    return text.strip()


# ─────────────────────────────────────────────────────────
# CHUNKING
# ─────────────────────────────────────────────────────────

def chunk_pages_hierarchical(
    pages:            list[dict],
    start_parent_id:  int = 0,
    start_child_id:   int = 0,      # ← FIX: was hardcoded to 0 inside the fn
) -> tuple[list, list]:
    """
    Creates two levels of chunks from each page:

    PARENT chunks (1000 chars)
    └── CHILD chunks (300 chars) — each knows its parent ID

    Retrieval flow:
        query → find best children → return their parents to LLM
        Small children = precise match | Large parents = rich context

    Bug fixed: start_child_id must be passed in from the caller so that
    children from different PDFs receive globally unique IDs.
    Without this, every PDF restarts from child_0, causing ChromaDB upsert
    to silently overwrite earlier PDFs' data — leaving Amazon and Microsoft
    with zero vectors in the index while only Netflix/NVIDIA are retained.
    """
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size    = config.PARENT_CHUNK_SIZE,
        chunk_overlap = config.PARENT_CHUNK_OVERLAP,
        separators    = ["\n\n", "\n", ". ", " ", ""]
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size    = config.CHILD_CHUNK_SIZE,
        chunk_overlap = config.CHILD_CHUNK_OVERLAP,
        separators    = ["\n\n", "\n", ". ", " ", ""]
    )

    parents  = []
    children = []
    parent_id = start_parent_id
    child_id  = start_child_id     # ← FIX: was `child_id = 0`

    for page in pages:
        page["text"]   = clean_text(page["text"])
        parent_texts   = parent_splitter.split_text(page["text"])

        for parent_text in parent_texts:
            if len(parent_text.strip()) < 60:
                continue

            parent = {
                "chunk_id" : f"parent_{parent_id}",
                "text"     : parent_text,
                "source"   : page["source"],
                "company"  : page["company"],
                "page"     : page["page"],
                "type"     : "parent"
            }
            parents.append(parent)

            # Split each parent into children
            child_texts = child_splitter.split_text(parent_text)
            for child_text in child_texts:
                if len(child_text.strip()) < 30:
                    continue
                children.append({
                    "chunk_id" : f"child_{child_id}",
                    "text"     : child_text,
                    "parent_id": f"parent_{parent_id}",
                    "source"   : page["source"],
                    "company"  : page["company"],
                    "page"     : page["page"],
                    "type"     : "child"
                })
                child_id += 1

            parent_id += 1

    return parents, children


# ─────────────────────────────────────────────────────────
# LOAD / SAVE CHUNKS
# ─────────────────────────────────────────────────────────

def save_chunks(data: dict):
    """Saves parents and children separately."""
    Path(config.DATA_PROCESSED_DIR).mkdir(parents=True, exist_ok=True)
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_existing_chunks() -> dict:
    if not CHUNKS_PATH.exists():
        return {"parents": [], "children": []}
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Handle old flat format (upgrade path)
    if isinstance(data, list):
        print("⚠️  Old chunk format detected — rebuilding with parent-child structure")
        return {"parents": [], "children": []}
    return data


# ─────────────────────────────────────────────────────────
# MAIN PIPELINE — run this every time
# ─────────────────────────────────────────────────────────
def run_preprocessing_pipeline() -> dict:
    """
    Returns both parents and children so indexers can use them correctly:
    - Children → indexed in ChromaDB + BM25 (precise retrieval)
    - Parents  → stored for context lookup (rich LLM input)
    """
    print("=" * 52)
    print("   PHASE 2 — SCALABLE PREPROCESSING PIPELINE")
    print("=" * 52)

    manifest        = load_manifest()
    existing_data   = load_existing_chunks()  # now returns dict
    new_pdfs        = get_new_pdfs(config.DATA_RAW_DIR, manifest)

    if not new_pdfs:
        print(f"\n✅ Nothing new to process.")
        print(f"   Parents : {len(existing_data.get('parents', []))}")
        print(f"   Children: {len(existing_data.get('children', []))}\n")
        return existing_data

    print(f"\n📂 Processing {len(new_pdfs)} new PDF(s)...\n")

    new_pdf_names = [p.name for p in new_pdfs]

    # Remove stale chunks from re-processed PDFs
    old_parents  = [c for c in existing_data.get("parents",  [])
                    if c["source"] not in new_pdf_names]
    old_children = [c for c in existing_data.get("children", [])
                    if c["source"] not in new_pdf_names]

    # ── Compute globally unique starting IDs ──────────────────────────────────
    # Parent IDs: already correctly tracked in the original code
    existing_parent_ids = [
        int(p["chunk_id"].replace("parent_", ""))
        for p in old_parents if "parent_" in p.get("chunk_id", "")
    ]
    start_id = max(existing_parent_ids, default=-1) + 1

    # Child IDs: FIX — must also be globally unique across all PDFs.
    # Previously child_id always started at 0 inside chunk_pages_hierarchical,
    # so every PDF produced child_0, child_1, … causing ChromaDB upsert to
    # overwrite earlier companies' vectors with later companies' data.
    existing_child_ids = [
        int(c["chunk_id"].replace("child_", ""))
        for c in old_children if "child_" in c.get("chunk_id", "")
    ]
    start_child_id = max(existing_child_ids, default=-1) + 1

    all_new_parents  = []
    all_new_children = []

    for pdf_path in tqdm(new_pdfs, desc="Processing PDFs"):
        pages        = extract_text_from_pdf(pdf_path)
        new_p, new_c = chunk_pages_hierarchical(
            pages,
            start_parent_id = start_id,
            start_child_id  = start_child_id,   # ← FIX: pass unique start
        )

        manifest[pdf_path.name] = {
            "hash"          : get_file_hash(str(pdf_path)),
            "processed_at"  : datetime.now().isoformat(),
            "parents_count" : len(new_p),
            "children_count": len(new_c)
        }
        print(f"   ✅ {pdf_path.stem.upper()}: "
              f"{len(pages)} pages → {len(new_p)} parents, {len(new_c)} children")

        start_id         += len(new_p)
        start_child_id   += len(new_c)    # ← FIX: advance the child counter
        all_new_parents  += new_p
        all_new_children += new_c

    final = {
        "parents" : old_parents  + all_new_parents,
        "children": old_children + all_new_children
    }

    save_chunks(final)
    save_manifest(manifest)

    print(f"\n💾 Saved {len(final['parents'])} parents, "
          f"{len(final['children'])} children")
    print(f"📋 Manifest: {len(manifest)} PDFs tracked")
    print("\n🎉 Preprocessing complete!\n")
    return final