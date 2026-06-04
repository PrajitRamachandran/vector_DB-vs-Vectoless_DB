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

def extract_text_from_pdf(pdf_path: Path) -> list[dict]:
    """
    Extracts text page by page from a single PDF.
    """
    doc          = fitz.open(str(pdf_path))
    company_name = re.sub(r'[_\-](10k|10K|annual|report).*', '', pdf_path.stem).upper()
    pages        = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()

        if len(text.strip()) < 50:   # skip blank/image-only pages
            continue

        pages.append({
            "text"        : text,
            "source"      : pdf_path.name,
            "company"     : company_name,
            "page"        : page_num + 1,
            "total_pages" : len(doc)
        })

    doc.close()
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

def chunk_pages(pages: list[dict], start_chunk_id: int = 0) -> list[dict]:
    """
    Splits page text into overlapping chunks.
    start_chunk_id ensures chunk IDs are globally unique across all PDFs —
    so adding new PDFs never creates ID collisions.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size    = config.CHUNK_SIZE,
        chunk_overlap = config.CHUNK_OVERLAP,
        separators    = ["\n\n", "\n", ". ", " ", ""]
    )

    all_chunks = []
    chunk_id   = start_chunk_id

    for page in pages:
        page["text"] = clean_text(page["text"])
        raw_chunks   = splitter.split_text(page["text"])

        for i, chunk_text in enumerate(raw_chunks):
            if len(chunk_text.strip()) < 30:   # skip tiny meaningless chunks
                continue

            all_chunks.append({
                "chunk_id"  : chunk_id,
                "text"      : chunk_text,
                "source"    : page["source"],
                "company"   : page["company"],
                "page"      : page["page"],
                "chunk_num" : i
            })
            chunk_id += 1

    return all_chunks


# ─────────────────────────────────────────────────────────
# LOAD / SAVE CHUNKS
# ─────────────────────────────────────────────────────────

def load_existing_chunks() -> list[dict]:
    if CHUNKS_PATH.exists():
        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_chunks(chunks: list[dict]):
    Path(config.DATA_PROCESSED_DIR).mkdir(parents=True, exist_ok=True)
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)


# ─────────────────────────────────────────────────────────
# MAIN PIPELINE — run this every time
# ─────────────────────────────────────────────────────────

def run_preprocessing_pipeline() -> list[dict]:
    """
    Smart incremental pipeline:
    - First run  → processes all 4 PDFs
    - Later runs → only processes new or changed PDFs
    - Always returns the full chunk list
    """
    print("=" * 52)
    print("   PHASE 2 — SCALABLE PREPROCESSING PIPELINE")
    print("=" * 52)

    manifest       = load_manifest()
    existing_chunks = load_existing_chunks()
    new_pdfs       = get_new_pdfs(config.DATA_RAW_DIR, manifest)

    if not new_pdfs:
        print(f"\n✅ Nothing new to process.")
        print(f"   Loaded {len(existing_chunks)} existing chunks from cache.\n")
        return existing_chunks

    print(f"\n📂 Processing {len(new_pdfs)} new PDF(s)...\n")

    # Remove old chunks for any PDFs being re-processed (changed files)
    new_pdf_names   = [p.name for p in new_pdfs]
    existing_chunks = [
        c for c in existing_chunks
        if c["source"] not in new_pdf_names
    ]

    # Process each new PDF
    start_id   = max((c["chunk_id"] for c in existing_chunks), default=-1) + 1
    all_new_chunks = []

    for pdf_path in tqdm(new_pdfs, desc="Processing PDFs"):
        pages      = extract_text_from_pdf(pdf_path)
        new_chunks = chunk_pages(pages, start_chunk_id=start_id)

        # Update manifest
        manifest[pdf_path.name] = {
            "hash"         : get_file_hash(str(pdf_path)),
            "processed_at" : datetime.now().isoformat(),
            "chunks_count" : len(new_chunks),
            "pages_count"  : len(pages)
        }

        print(f"   ✅ {pdf_path.stem.upper()}: "
              f"{len(pages)} pages → {len(new_chunks)} chunks")

        start_id       += len(new_chunks)
        all_new_chunks += new_chunks

    # Merge old + new chunks and save
    final_chunks = existing_chunks + all_new_chunks
    save_chunks(final_chunks)
    save_manifest(manifest)

    print(f"\n💾 Saved {len(final_chunks)} total chunks to {CHUNKS_PATH}")
    print(f"📋 Manifest updated: {len(manifest)} PDFs tracked")
    print("\n🎉 Preprocessing complete!\n")

    return final_chunks