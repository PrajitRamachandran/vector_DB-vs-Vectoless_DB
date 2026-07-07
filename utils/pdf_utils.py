# import fitz


# def extract_first_pages_text(pdf_path, pages=3):
#     """
#     Extract text from first few pages.
#     """

#     doc = fitz.open(pdf_path)

#     text = ""

#     for i in range(min(pages, len(doc))):
#         text += doc[i].get_text()

#     doc.close()

#     return text

"""
pdf_utils.py

PDF text extraction utilities used by the ingestion pipeline. Text
extracted here typically feeds directly into company_detector.detect_company()
and downstream chunking, so failures need to be specific and actionable
rather than raw PyMuPDF tracebacks — and a single bad page or a locked PDF
should never crash a whole batch ingestion run.
"""

from __future__ import annotations

import logging
from pathlib import Path

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


# ============================================================
# Exceptions
# ============================================================

class PDFExtractionError(Exception):
    """Base class for all PDF extraction failures raised by this module."""


class PDFNotFoundError(PDFExtractionError):
    """The given path does not exist or is not a file."""


class PDFCorruptError(PDFExtractionError):
    """The file exists but could not be parsed as a valid PDF."""


class PDFEncryptedError(PDFExtractionError):
    """The PDF is password-protected and cannot be opened without one."""


# ============================================================
# Internal helpers
# ============================================================

def _validate_pdf_path(pdf_path: str | Path) -> Path:
    """Resolves and validates that pdf_path points to an existing file."""
    path = Path(pdf_path)

    if not path.exists():
        raise PDFNotFoundError(f"PDF not found: {path}")
    if not path.is_file():
        raise PDFNotFoundError(f"Path exists but is not a file: {path}")

    return path


def _open_pdf(path: Path) -> fitz.Document:
    """
    Opens a PDF and validates it's readable and unencrypted, raising a
    specific, actionable exception on failure. Caller is responsible for
    closing the returned document (use as a context manager where possible).
    """
    try:
        doc = fitz.open(path)
    except (fitz.FileDataError, fitz.EmptyFileError) as exc:
        raise PDFCorruptError(f"Could not parse '{path}' as a valid PDF: {exc}") from exc
    except fitz.FileNotFoundError as exc:
        # fitz can raise its own FileNotFoundError variant during open()
        # even though we already checked existence (e.g. TOCTOU or a
        # permissions issue) — normalize it to our own exception type.
        raise PDFNotFoundError(f"PDF not found: {path}") from exc

    if doc.is_encrypted or doc.needs_pass:
        doc.close()
        raise PDFEncryptedError(
            f"'{path}' is password-protected; cannot extract text without "
            f"a password."
        )

    return doc


def _page_range(doc: fitz.Document, start_page: int, end_page: int | None) -> range:
    """
    Resolves and validates a 0-indexed [start_page, end_page) range against
    the document's actual page count, clamping end_page rather than raising
    if the caller asks for more pages than the document has.
    """
    page_count = doc.page_count

    if start_page < 0:
        raise ValueError(f"start_page must be >= 0, got {start_page}")

    if end_page is None:
        end_page = page_count
    elif end_page < start_page:
        raise ValueError(
            f"end_page ({end_page}) must be >= start_page ({start_page})"
        )

    return range(start_page, min(end_page, page_count))


# ============================================================
# Public API
# ============================================================

def get_page_count(pdf_path: str | Path) -> int:
    """
    Returns the number of pages in the PDF.

    Raises:
        PDFNotFoundError: path doesn't exist or isn't a file.
        PDFCorruptError: file isn't a valid/parseable PDF.
        PDFEncryptedError: file is password-protected.
    """
    path = _validate_pdf_path(pdf_path)
    doc = _open_pdf(path)
    try:
        return doc.page_count
    finally:
        doc.close()


def extract_text_range(
    pdf_path: str | Path,
    start_page: int = 0,
    end_page: int | None = None,
    *,
    skip_unreadable_pages: bool = True,
) -> str:
    """
    Extracts and concatenates text from pages [start_page, end_page)
    (0-indexed, end exclusive). If end_page is None, extracts through the
    last page.

    Args:
        pdf_path: path to the PDF file.
        start_page: 0-indexed first page to extract (inclusive).
        end_page: 0-indexed page to stop before (exclusive). None = end of
            document. Values beyond the document length are clamped rather
            than raising, so callers can safely request "first N pages" of
            a shorter-than-N-page document.
        skip_unreadable_pages: if True (default), a page that fails to
            extract (corrupted content stream, etc.) is logged and skipped
            rather than aborting the whole extraction. If False, the first
            such failure raises PDFCorruptError.

    Returns:
        Concatenated text of the requested pages. Returns "" if the range
        is empty or every page in range failed to extract.

    Raises:
        PDFNotFoundError: path doesn't exist or isn't a file.
        PDFCorruptError: file isn't a valid/parseable PDF (or, when
            skip_unreadable_pages=False, a page failed to extract).
        PDFEncryptedError: file is password-protected.
        ValueError: start_page/end_page are invalid.
    """
    path = _validate_pdf_path(pdf_path)
    doc = _open_pdf(path)

    try:
        pages = _page_range(doc, start_page, end_page)

        text_parts: list[str] = []
        for i in pages:
            try:
                text_parts.append(doc[i].get_text())
            except Exception as exc:
                if skip_unreadable_pages:
                    logger.warning(
                        "Skipping unreadable page %d in '%s': %s", i, path, exc
                    )
                    continue
                raise PDFCorruptError(
                    f"Failed to extract text from page {i} of '{path}': {exc}"
                ) from exc

        return "".join(text_parts)
    finally:
        doc.close()


def extract_first_pages_text(pdf_path: str | Path, pages: int = 3) -> str:
    """
    Extracts text from the first `pages` pages of a PDF. This is the
    primary entry point used by the ingestion pipeline to grab the
    cover-page text handed to company_detector.detect_company().

    Args:
        pdf_path: path to the PDF file.
        pages: number of pages to extract from the start of the document
            (must be >= 1). If the document has fewer pages than this,
            all available pages are extracted with no error.

    Returns:
        Concatenated text of the first `pages` pages (or all pages, if
        fewer exist). Returns "" if no text could be extracted.

    Raises:
        PDFNotFoundError: path doesn't exist or isn't a file.
        PDFCorruptError: file isn't a valid/parseable PDF.
        PDFEncryptedError: file is password-protected.
        ValueError: pages < 1.
    """
    if pages < 1:
        raise ValueError(f"pages must be >= 1, got {pages}")

    return extract_text_range(pdf_path, start_page=0, end_page=pages)


def extract_full_text(pdf_path: str | Path) -> str:
    """
    Extracts text from every page of the PDF. Convenience wrapper around
    extract_text_range() for full-document ingestion/chunking.
    """
    return extract_text_range(pdf_path, start_page=0, end_page=None)