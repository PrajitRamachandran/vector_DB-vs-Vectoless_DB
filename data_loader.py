# data_loader.py
"""
PDF ingestion pipeline: discovers new/changed PDFs, extracts text,
cleans it, splits it into parent/child chunks, and persists both the
chunk store (chunks.json) and a manifest tracking what's been processed.

Downstream consumers:
    - vector_rag.indexer.index_chunks() indexes the "children" list.
    - vector_rag.indexer.build_parent_lookup() indexes the "parents" list
      by chunk_id.
    - vector_rag.retriever.retrieve() joins children back to parents via
      each child's "parent_id" field.

chunk_id uniqueness across the whole corpus (parent_<n> / child_<n>) is
load-bearing: indexer.index_chunks() upserts children by chunk_id, so a
collision between two PDFs silently overwrites one PDF's vectors with
another's. See _next_id() / run_preprocessing_pipeline() for how global
uniqueness is maintained across incremental runs.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF — used for fast PDF validation ahead of extraction
import pymupdf4llm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

import config

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────

MANIFEST_PATH = Path(config.DATA_PROCESSED_DIR) / "manifest.json"
CHUNKS_PATH = Path(config.DATA_PROCESSED_DIR) / "chunks.json"

# Bytes read per chunk when hashing files — avoids loading an entire large
# PDF into memory just to fingerprint it.
_HASH_READ_BLOCK_SIZE = 1024 * 1024  # 1 MiB

# Minimum characters a page must have (after extraction) to be kept.
MIN_PAGE_CHARS = 50
# Minimum characters a parent/child chunk must have to be kept.
MIN_PARENT_CHARS = 60
MIN_CHILD_CHARS = 30

_PARENT_ID_RE = re.compile(r"^parent_(\d+)$")
_CHILD_ID_RE = re.compile(r"^child_(\d+)$")


# ─────────────────────────────────────────────────────────
# ATOMIC JSON I/O
# ─────────────────────────────────────────────────────────

def _atomic_write_json(path: Path, data) -> None:
    """
    Writes JSON to `path` atomically: content is written to a temp file in
    the same directory and then moved into place with os.replace(), which
    is atomic on POSIX and Windows. This prevents a crash or Ctrl-C mid
    write from leaving a truncated/corrupt manifest.json or chunks.json —
    both of which the pipeline treats as its source of truth on the next
    run.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _load_json(path: Path, default):
    """Loads JSON from `path`, returning `default` if missing, raising a
    clear error if present but corrupt (rather than an opaque JSONDecodeError
    deep in the caller)."""
    if not path.exists():
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Could not parse {path} as JSON — it may be corrupted. "
            f"Restore from backup or delete it to rebuild from scratch: {exc}"
        ) from exc


# ─────────────────────────────────────────────────────────
# MANIFEST — tracks which PDFs have been processed
# ─────────────────────────────────────────────────────────

def get_file_hash(filepath: str) -> str:
    """
    Generates a fingerprint for a PDF file's contents (MD5 — used purely
    for change detection, not security). If the file changes (re-downloaded,
    updated), the hash changes and it gets re-processed automatically.

    Reads in fixed-size blocks rather than the whole file at once, so
    hashing a large PDF doesn't spike memory usage.
    """
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        for block in iter(lambda: f.read(_HASH_READ_BLOCK_SIZE), b""):
            hasher.update(block)
    return hasher.hexdigest()


def load_manifest() -> dict:
    """
    Loads the manifest file. Returns an empty dict if it doesn't exist yet.

    Manifest structure:
    {
        "nvidia_10k.pdf": {
            "company": "NVIDIA",
            "hash": "abc123...",
            "processed_at": "2024-01-01T12:00:00",
            "pages_count": 190,
            "parents_count": 240,
            "children_count": 1102
        },
        ...
    }
    """
    manifest = _load_json(MANIFEST_PATH, {})
    if not isinstance(manifest, dict):
        raise ValueError(f"{MANIFEST_PATH} does not contain a JSON object.")
    return manifest


def get_company_from_manifest(pdf_name: str, manifest: dict) -> str:
    """Returns the previously-recorded company for `pdf_name`, or 'UNKNOWN'."""
    entry = manifest.get(pdf_name, {})
    return entry.get("company", "UNKNOWN")


def save_manifest(manifest: dict) -> None:
    _atomic_write_json(MANIFEST_PATH, manifest)


def get_new_pdfs(raw_dir: str, manifest: dict) -> list[Path]:
    """
    Compares all PDFs in `raw_dir` against the manifest by content hash.
    Returns only PDFs that are new or have changed since last run, sorted
    by filename for deterministic, reproducible processing order (glob()
    order is filesystem-dependent and not guaranteed stable across runs
    or machines, which would otherwise make chunk_id assignment
    non-reproducible).
    """
    raw_path = Path(raw_dir)
    if not raw_path.is_dir():
        raise FileNotFoundError(f"Raw PDF directory does not exist: {raw_path}")

    all_pdfs = sorted(raw_path.glob("*.pdf"), key=lambda p: p.name)
    new_pdfs = []

    for pdf_path in all_pdfs:
        filename = pdf_path.name
        try:
            current_hash = get_file_hash(str(pdf_path))
        except OSError as exc:
            logger.error("Could not read %s to hash it; skipping. (%s)", filename, exc)
            continue

        prior = manifest.get(filename)
        if prior is None:
            logger.info("[NEW] New PDF detected: %s", filename)
            new_pdfs.append(pdf_path)
        elif prior.get("hash") != current_hash:
            logger.info("[CHANGED] Changed PDF detected: %s — will re-process", filename)
            new_pdfs.append(pdf_path)
        else:
            logger.debug("[OK] Already processed: %s — skipping", filename)

    return new_pdfs


# ─────────────────────────────────────────────────────────
# EXTRACTION
# ─────────────────────────────────────────────────────────

def _validate_pdf(pdf_path: Path) -> int:
    """
    Opens the PDF with PyMuPDF as a fast pre-flight check, so a corrupted
    or password-protected file raises a clear, actionable error here
    instead of an opaque failure inside pymupdf4llm.to_markdown().

    Returns the page count.
    """
    try:
        doc = fitz.open(str(pdf_path))
    except Exception as exc:
        raise ValueError(f"'{pdf_path.name}' could not be opened as a PDF: {exc}") from exc

    try:
        if doc.needs_pass:
            raise ValueError(f"'{pdf_path.name}' is password-protected; cannot extract text.")
        page_count = doc.page_count
        if page_count == 0:
            raise ValueError(f"'{pdf_path.name}' has zero pages.")
        return page_count
    finally:
        doc.close()


def extract_text_from_pdf(pdf_path: Path, company_name: str) -> list[dict]:
    """
    Extracts text page by page from a single PDF using Markdown conversion
    to preserve tabular structure.

    Raises:
        ValueError: the PDF is unreadable, encrypted, or empty.
    """
    _validate_pdf(pdf_path)

    pages = []

    # Process the entire document at once to avoid repeated file handle
    # churn. page_chunks=True returns a list of dicts: [{'text': ..., ...}]
    md_pages = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True)

    total_pages = len(md_pages)
    for i, page_data in enumerate(md_pages):
        text = page_data.get("text", "") or ""

        if len(text.strip()) < MIN_PAGE_CHARS:
            continue  # skip blank/image-only pages

        pages.append({
            "text": text,
            "source": pdf_path.name,
            "company": company_name,
            "page": i + 1,
            "total_pages": total_pages,
        })

    if not pages:
        logger.warning(
            "'%s': extracted 0 usable pages out of %d (scanned/image-only PDF?).",
            pdf_path.name, total_pages,
        )

    return pages


# ─────────────────────────────────────────────────────────
# CLEANING
# ─────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Normalizes extracted page text before chunking.

    The lone-page-number strip is intentionally conservative (1-5 digits
    only) so it can't accidentally delete a standalone financial figure
    that happens to sit alone on its own markdown-table line — a real risk
    on financial-statement PDFs where multi-digit values are common.
    """
    text = re.sub(r"\n{3,}", "\n\n", text)  # collapse excess newlines
    text = re.sub(r"[ \t]{2,}", " ", text)  # collapse excess spaces
    text = re.sub(
        r"^\s*-?\s*\d{1,5}\s*-?\s*$", "", text, flags=re.MULTILINE
    )  # remove lone page numbers (short, isolated numeric lines only)
    text = re.sub(
        r"^\s*Table of Contents\s*$", "", text, flags=re.IGNORECASE | re.MULTILINE
    )  # only strip when it's a standalone heading line, not mid-sentence
    return text.strip()


# ─────────────────────────────────────────────────────────
# CHUNKING
# ─────────────────────────────────────────────────────────

def chunk_pages_hierarchical(
    pages: list[dict],
    start_parent_id: int = 0,
    start_child_id: int = 0,
) -> tuple[list[dict], list[dict]]:
    """
    Creates two levels of chunks from each page:

        PARENT chunks (config.PARENT_CHUNK_SIZE chars)
        └── CHILD chunks (config.CHILD_CHUNK_SIZE chars) — each carries its
            parent's chunk_id in "parent_id"

    Retrieval flow: query -> find best-matching children -> return their
    parents to the LLM. Small children give a precise match signal; large
    parents give the LLM richer surrounding context.

    Both `start_parent_id` and `start_child_id` MUST be passed in by the
    caller as running counters across the whole corpus. If either resets
    to 0 per PDF, two PDFs will mint colliding chunk_ids (e.g. both produce
    "child_0"), and indexer.index_chunks()'s upsert-by-id will silently
    let the later PDF overwrite the earlier one's vectors in ChromaDB —
    the earlier company would then have zero retrievable chunks despite
    manifest.json claiming it was successfully processed.
    """
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.PARENT_CHUNK_SIZE,
        chunk_overlap=config.PARENT_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHILD_CHUNK_SIZE,
        chunk_overlap=config.CHILD_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    parents: list[dict] = []
    children: list[dict] = []
    parent_id = start_parent_id
    child_id = start_child_id

    for page in pages:
        cleaned = clean_text(page["text"])
        parent_texts = parent_splitter.split_text(cleaned)

        for parent_text in parent_texts:
            if len(parent_text.strip()) < MIN_PARENT_CHARS:
                continue

            parent_chunk_id = f"parent_{parent_id}"
            parents.append({
                "chunk_id": parent_chunk_id,
                "text": parent_text,
                "source": page["source"],
                "company": page["company"],
                "page": page["page"],
                "type": "parent",
            })

            child_texts = child_splitter.split_text(parent_text)
            for child_text in child_texts:
                if len(child_text.strip()) < MIN_CHILD_CHARS:
                    continue
                child_chunk_id = f"child_{child_id}"
                children.append({
                    "chunk_id": child_chunk_id,
                    "text": child_text,
                    "parent_id": parent_chunk_id,
                    "source": page["source"],
                    "company": page["company"],
                    "page": page["page"],
                    "type": "child",
                })
                child_id += 1

            parent_id += 1

    logger.debug(
        "chunk_pages_hierarchical: %d pages -> %d parents, %d children (ids %d..%d / %d..%d)",
        len(pages), len(parents), len(children),
        start_parent_id, parent_id - 1, start_child_id, child_id - 1,
    )

    return parents, children


def _max_indexed_id(items: list[dict], pattern: re.Pattern) -> int:
    """
    Returns the highest numeric suffix found among chunk_ids matching
    `pattern` (e.g. "parent_42" -> 42), or -1 if none match.

    Malformed chunk_ids (unexpected format) are logged and skipped rather
    than raising — a single bad historical record shouldn't prevent the
    pipeline from computing a safe next ID.
    """
    max_id = -1
    for item in items:
        chunk_id = item.get("chunk_id", "")
        match = pattern.match(chunk_id)
        if not match:
            logger.warning("Skipping chunk with unexpected chunk_id format: %r", chunk_id)
            continue
        max_id = max(max_id, int(match.group(1)))
    return max_id


# ─────────────────────────────────────────────────────────
# LOAD / SAVE CHUNKS
# ─────────────────────────────────────────────────────────

def save_chunks(data: dict) -> None:
    """Saves parents and children (atomically) to chunks.json."""
    _atomic_write_json(CHUNKS_PATH, data)


def load_existing_chunks() -> dict:
    """Loads chunks.json, upgrading/discarding an old flat-list format."""
    data = _load_json(CHUNKS_PATH, {"parents": [], "children": []})
    if isinstance(data, list):
        logger.warning("Old chunk format detected — rebuilding with parent-child structure")
        return {"parents": [], "children": []}
    if not isinstance(data, dict) or "parents" not in data or "children" not in data:
        raise ValueError(
            f"{CHUNKS_PATH} has an unexpected shape; expected keys "
            f"'parents' and 'children'."
        )
    return data


# ─────────────────────────────────────────────────────────
# COMPANY DETECTION
# ─────────────────────────────────────────────────────────

def _detect_company(pdf_path: Path) -> str:
    """Best-effort company-name detection from a PDF's opening pages."""
    from utils.company_detector import detect_company
    from utils.pdf_utils import extract_first_pages_text

    try:
        text = extract_first_pages_text(pdf_path)
        return detect_company(text) or "UNKNOWN"
    except Exception as exc:
        logger.error("Error detecting company for %s: %s", pdf_path.name, exc)
        return "UNKNOWN"


# ─────────────────────────────────────────────────────────
# MAIN PIPELINE — run this every time
# ─────────────────────────────────────────────────────────

def run_preprocessing_pipeline() -> dict:
    """
    Discovers new/changed PDFs, extracts + cleans + chunks them, and merges
    the result into the persisted chunk store and manifest.

    Returns both parents and children so downstream indexers can use them:
        - children -> indexed in ChromaDB (precise retrieval)
        - parents  -> looked up for full context (rich LLM input)

    A failure on any single PDF (corrupt file, extraction error, etc.) is
    logged and that PDF is skipped — it simply stays out of the manifest
    so it's retried on the next run — rather than aborting the whole batch
    and discarding progress already made on other PDFs in this run.
    """
    logger.info("=" * 52)
    logger.info("   PHASE 2 — SCALABLE PREPROCESSING PIPELINE")
    logger.info("=" * 52)

    manifest = load_manifest()
    existing_data = load_existing_chunks()
    new_pdfs = get_new_pdfs(config.DATA_RAW_DIR, manifest)

    if not new_pdfs:
        logger.info("[OK] Nothing new to process.")
        logger.info("   Parents : %d", len(existing_data.get("parents", [])))
        logger.info("   Children: %d", len(existing_data.get("children", [])))
        return existing_data

    logger.info("[Processing] %d new PDF(s)...", len(new_pdfs))

    new_pdf_names = {p.name for p in new_pdfs}

    # Drop stale chunks belonging to PDFs we're about to re-process, so a
    # changed PDF doesn't end up with both its old and new chunks present.
    old_parents = [c for c in existing_data.get("parents", []) if c["source"] not in new_pdf_names]
    old_children = [c for c in existing_data.get("children", []) if c["source"] not in new_pdf_names]

    # ── Compute globally unique starting IDs ────────────────────────────
    # Both counters must be tracked across the *entire* corpus (not reset
    # per PDF) — see chunk_pages_hierarchical()'s docstring for why a reset
    # silently corrupts the ChromaDB index via colliding upsert keys.
    next_parent_id = _max_indexed_id(old_parents, _PARENT_ID_RE) + 1
    next_child_id = _max_indexed_id(old_children, _CHILD_ID_RE) + 1

    all_new_parents: list[dict] = []
    all_new_children: list[dict] = []
    processed = 0
    failed: list[str] = []

    for pdf_path in tqdm(new_pdfs, desc="Processing PDFs"):
        try:
            company_name = get_company_from_manifest(pdf_path.name, manifest)
            if company_name == "UNKNOWN":
                company_name = _detect_company(pdf_path)

            logger.info("PREPROCESS: %s -> %s", pdf_path.name, company_name)

            pages = extract_text_from_pdf(pdf_path, company_name)
            new_parents, new_children = chunk_pages_hierarchical(
                pages,
                start_parent_id=next_parent_id,
                start_child_id=next_child_id,
            )

            file_hash = get_file_hash(str(pdf_path))

        except Exception as exc:
            logger.error("Failed to process '%s'; skipping. (%s)", pdf_path.name, exc)
            failed.append(pdf_path.name)
            continue

        manifest[pdf_path.name] = {
            "company": company_name,
            "hash": file_hash,
            "processed_at": datetime.now().isoformat(),
            "pages_count": len(pages),
            "parents_count": len(new_parents),
            "children_count": len(new_children),
        }

        logger.info(
            "   [OK] %s: %d pages -> %d parents, %d children",
            pdf_path.stem.upper(), len(pages), len(new_parents), len(new_children),
        )

        next_parent_id += len(new_parents)
        next_child_id += len(new_children)
        all_new_parents += new_parents
        all_new_children += new_children
        processed += 1

    final = {
        "parents": old_parents + all_new_parents,
        "children": old_children + all_new_children,
    }

    # Persist whatever succeeded even if some PDFs failed above, so a
    # single bad file doesn't cost progress made on the rest of the batch.
    save_chunks(final)
    save_manifest(manifest)

    logger.info("[Saved] %d parents, %d children", len(final["parents"]), len(final["children"]))
    logger.info("[Manifest] %d PDFs tracked", len(manifest))
    if failed:
        logger.warning(
            "[Done] Preprocessing complete with %d failure(s): %s",
            len(failed), ", ".join(failed),
        )
    else:
        logger.info("[Done] Preprocessing complete! (%d PDF(s) processed)", processed)

    return final