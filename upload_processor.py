# from pathlib import Path
# import json

# from utils.pdf_utils import (
#     extract_first_pages_text
# )

# from utils.company_detector import (
#     detect_company
# )

# MANIFEST_PATH = Path(
#     "data/processed/manifest.json"
# )



# def load_manifest():

#     if MANIFEST_PATH.exists():

#         with open(
#             MANIFEST_PATH,
#             "r",
#             encoding="utf-8"
#         ) as f:

#             return json.load(f)

#     return {}


# def save_manifest(manifest):

#     MANIFEST_PATH.parent.mkdir(
#         parents=True,
#         exist_ok=True
#     )

#     with open(
#         MANIFEST_PATH,
#         "w",
#         encoding="utf-8"
#     ) as f:

#         json.dump(
#             manifest,
#             f,
#             indent=2
#         )


# def process_uploaded_pdf(uploaded_file):

#     raw_dir = Path("data/raw")

#     raw_dir.mkdir(
#         parents=True,
#         exist_ok=True
#     )

#     save_path = (
#         raw_dir /
#         uploaded_file.name
#     )

#     with open(save_path, "wb") as f:

#         f.write(
#             uploaded_file.getbuffer()
#         )

#     text = extract_first_pages_text(
#         save_path
#     )

#     company = detect_company(
#         text
#     )

#     manifest = load_manifest()

#     manifest[
#         uploaded_file.name
#     ] = {

#         "company": company
#     }

#     save_manifest(
#         manifest
#     )

#     print(
#         f"UPLOAD: {uploaded_file.name} -> {company}"
#     )

#     return (
#         str(save_path),
#         company
#     )


# upload_processor.py
"""
Handles a single user-uploaded PDF: validates it, saves it into the raw
ingestion directory, and does a best-effort company-name detection so the
UI can show immediate feedback.

Important: this module intentionally does NOT write into the ingestion
pipeline's manifest.json (see data_loader.py). That manifest's "hash"
field is data_loader.get_new_pdfs()'s signal that a file has already been
fully extracted, chunked, and indexed — writing a partial entry here
would either:
    (a) omit "hash" entirely, causing a KeyError the next time
        data_loader.get_new_pdfs() runs (`manifest[filename]["hash"]`), or
    (b) include "hash", incorrectly telling the pipeline this upload was
        already processed, so it would silently skip chunking/indexing it.
Instead, uploads are tracked in a separate, upload-owned cache
(UPLOAD_CACHE_PATH) used only to avoid redundant company detection on
repeat uploads and to give the UI something to display immediately;
data_loader.run_preprocessing_pipeline() remains the sole owner of
manifest.json and does its own hashing/company-detection when it actually
processes the file.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from datetime import datetime
from pathlib import Path
from typing import NamedTuple, Optional

import config
from utils.company_detector import detect_company
from utils.pdf_utils import extract_first_pages_text

logger = logging.getLogger(__name__)

RAW_DIR = Path(config.DATA_RAW_DIR)
UPLOAD_CACHE_PATH = Path(config.DATA_PROCESSED_DIR) / "upload_cache.json"

ALLOWED_EXTENSIONS = {".pdf"}
MAX_UPLOAD_SIZE_MB = getattr(config, "MAX_UPLOAD_SIZE_MB", 200)
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024

_cache_lock = threading.Lock()


class UploadResult(NamedTuple):
    path: str
    company: str


# ─────────────────────────────────────────────────────────
# UPLOAD CACHE (upload-processor-owned; see module docstring)
# ─────────────────────────────────────────────────────────

def _atomic_write_json(path: Path, data) -> None:
    """Writes JSON atomically (temp file + os.replace) to avoid a crash
    mid-write leaving a truncated/corrupt cache file."""
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


def load_upload_cache() -> dict:
    """Loads the upload cache, returning {} if missing or corrupt (a
    corrupt cache is non-critical — it's only an optimization/UI hint — so
    it's logged and treated as empty rather than crashing the upload)."""
    if not UPLOAD_CACHE_PATH.exists():
        return {}
    try:
        with open(UPLOAD_CACHE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read upload cache (%s); starting fresh.", exc)
        return {}


def save_upload_cache(cache: dict) -> None:
    _atomic_write_json(UPLOAD_CACHE_PATH, cache)


# ─────────────────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────────────────

def _sanitize_filename(raw_name: str) -> str:
    """
    Strips any directory components from the uploaded filename so a
    maliciously crafted name (e.g. "../../etc/cron.d/evil.pdf") can't
    escape RAW_DIR — Path("data/raw") / "../../x" would otherwise resolve
    outside the intended upload directory.
    """
    name = Path(raw_name or "").name.strip()
    if not name or name in (".", ".."):
        raise ValueError(f"Invalid upload filename: {raw_name!r}")
    return name


def _validate_extension(filename: str) -> None:
    if Path(filename).suffix.lower() not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type for '{filename}'. "
            f"Only {sorted(ALLOWED_EXTENSIONS)} are accepted."
        )


def _read_upload_bytes(uploaded_file) -> bytes:
    """
    Reads the uploaded file's contents. Supports Streamlit's
    UploadedFile.getbuffer() as the primary path, falling back to a plain
    .read() for other file-like objects so this isn't hard-coupled to one
    web framework's upload API.
    """
    if hasattr(uploaded_file, "getbuffer"):
        return bytes(uploaded_file.getbuffer())
    if hasattr(uploaded_file, "read"):
        return uploaded_file.read()
    raise TypeError(
        f"Unsupported uploaded_file type: {type(uploaded_file)!r} "
        f"(expected an object with .getbuffer() or .read())."
    )


def _detect_company_safe(pdf_path: Path) -> str:
    """Best-effort company detection — never raises; a detection failure
    shouldn't block the upload itself."""
    try:
        text = extract_first_pages_text(pdf_path)
        return detect_company(text) or "UNKNOWN"
    except Exception as exc:
        logger.error("Company detection failed for %s: %s", pdf_path.name, exc)
        return "UNKNOWN"


# ─────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────

def process_uploaded_pdf(uploaded_file) -> UploadResult:
    """
    Validates and saves an uploaded PDF into the raw ingestion directory,
    then does best-effort company detection for immediate UI feedback.

    Args:
        uploaded_file: an object exposing `.name` and either
            `.getbuffer()` (e.g. Streamlit's UploadedFile) or `.read()`.

    Returns:
        UploadResult(path, company) — `path` is the absolute path the file
        was saved to; `company` is the detected company name, or
        "UNKNOWN" if detection failed.

    Raises:
        ValueError: the filename is invalid, the extension isn't allowed,
            or the file exceeds MAX_UPLOAD_SIZE_MB.
        OSError: the file could not be written to disk.

    Note: this does NOT index or chunk the PDF — it only stages it in the
    raw directory. Run data_loader.run_preprocessing_pipeline() (and the
    downstream indexer) to actually make it retrievable.
    """
    filename = _sanitize_filename(getattr(uploaded_file, "name", ""))
    _validate_extension(filename)

    content = _read_upload_bytes(uploaded_file)
    if len(content) == 0:
        raise ValueError(f"Uploaded file '{filename}' is empty.")
    if len(content) > MAX_UPLOAD_SIZE_BYTES:
        raise ValueError(
            f"Uploaded file '{filename}' is {len(content) / 1024 / 1024:.1f} MB, "
            f"which exceeds the {MAX_UPLOAD_SIZE_MB} MB limit."
        )

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    save_path = RAW_DIR / filename

    if save_path.exists():
        logger.info("'%s' already exists in %s and will be overwritten.", filename, RAW_DIR)

    # Write to a temp file first and rename into place, so a failure or
    # interruption mid-write can't leave a truncated PDF sitting at
    # save_path that a later indexing run would try (and fail) to parse.
    fd, tmp_path = tempfile.mkstemp(dir=str(RAW_DIR), prefix=f".{filename}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(content)
        os.replace(tmp_path, save_path)
    except OSError:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    company = _detect_company_safe(save_path)

    with _cache_lock:
        cache = load_upload_cache()
        cache[filename] = {
            "company": company,
            "size_bytes": len(content),
            "uploaded_at": datetime.now().isoformat(),
        }
        save_upload_cache(cache)

    logger.info("UPLOAD: %s -> %s", filename, company)

    return UploadResult(path=str(save_path), company=company)