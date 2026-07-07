"""
Indexing Service

Provides a clean interface between Streamlit UI
and the underlying indexing pipelines.

This service wraps:

- run_preprocessing_pipeline()
- index_chunks()
- build_bm25_index()

without modifying existing benchmark code.
"""

from pathlib import Path
import shutil
import traceback
import re
import fitz
from mistralai.client.sdk import Mistral
import config

# ============================================================
# COMPANY DETECTION
# ============================================================

def extract_first_pages_text(pdf_path, pages=3):

    doc = fitz.open(pdf_path)

    text = ""

    for i in range(min(pages, len(doc))):
        text += doc[i].get_text()

    doc.close()

    return text

def sanitize_company_name(company):

    company = company.lower()

    company = re.sub(
        r"[^a-z0-9]+",
        "_",
        company
    )

    company = company.strip("_")

    return company

def detect_company(text):

    client = Mistral(
        api_key=config.MISTRAL_API_KEY
    )

    prompt = f"""
Identify the company name from this annual report.

Rules:
- Return only the company name.
- No explanation.
- No punctuation.
- No extra words.

Document:

{text[:8000]}
"""

    response = client.chat.complete(
        model=config.LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    company = (
        response
        .choices[0]
        .message
        .content
        .strip()
    )

    return company

# ============================================================
# EXISTING PROJECT IMPORTS
# ============================================================

from data_loader import (
    run_preprocessing_pipeline
)

from vector_rag.indexer import (
    index_chunks
)

from vectorless_rag.indexer import (
    build_bm25_index
)

# ============================================================
# PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

RAW_DIR = (
    ROOT_DIR /
    "data" /
    "raw"
)

CHROMA_DIR = (
    ROOT_DIR /
    "vector_rag" /
    "chroma_db"
)

BM25_INDEX = (
    ROOT_DIR /
    "vectorless_rag" /
    "bm25_index.pkl"
)

BM25_MANIFEST = (
    ROOT_DIR /
    "vectorless_rag" /
    "bm25_manifest.json"
)

# ============================================================
# PDF MANAGEMENT
# ============================================================

def save_uploaded_pdf(uploaded_file):

    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    temp_path = RAW_DIR / uploaded_file.name

    with open(temp_path, "wb") as f:

        f.write(
            uploaded_file.getbuffer()
        )

    # -----------------------------------
    # Detect company
    # -----------------------------------

    text = extract_first_pages_text(
        temp_path
    )

    company = detect_company(
        text
    )

    print(
        f"Detected company: {company}"
    )

    return {
        "path": str(temp_path),
        "company": company
    }

    # ==========================================
    # HANDLE DUPLICATES
    # ==========================================

    counter = 1

    while final_path.exists():

        final_path = (
            RAW_DIR /
            f"{company_slug}_{counter}_10k.pdf"
        )

        counter += 1

    shutil.move(
        temp_path,
        final_path
    )

    print(
        f"Detected Company: {company}"
    )

    print(
        f"Saved As: {final_path.name}"
    )

    return final_path


def list_raw_pdfs():
    """
    Returns all PDFs
    currently in data/raw/
    """

    if not RAW_DIR.exists():
        return []

    return sorted(
        RAW_DIR.glob("*.pdf")
    )


def delete_pdf(filename):
    """
    Removes a PDF from raw data.
    """

    target = RAW_DIR / filename

    if target.exists():
        target.unlink()
        return True

    return False


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess_documents():
    """
    Executes existing preprocessing pipeline.

    Wrapper around:

        run_preprocessing_pipeline()

    Returns:
        dict
    """

    try:

        result = (
            run_preprocessing_pipeline()
        )

        return {
            "success": True,
            "data": result,
            "message":
                "Preprocessing completed successfully."
        }

    except Exception as e:

        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }


# ============================================================
# VECTOR INDEX
# ============================================================

def build_vector_index(
    progress_callback=None,
    stage_callback=None,
    log_callback=None
):
    """
    Builds ChromaDB index
    using processed chunks.
    """

    try:

        chunk_data = (
            run_preprocessing_pipeline()
        )

        collection = (
            index_chunks(chunk_data)
        )

        count = 0

        try:
            count = collection.count()
        except Exception:
            pass

        return {
            "success": True,
            "vectors": count,
            "message":
                "Vector index built successfully."
        }

    except Exception as e:

        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }


# ============================================================
# BM25 INDEX
# ============================================================

def build_vectorless_index():
    """
    Builds BM25 index.
    """

    try:

        chunk_data = (
            run_preprocessing_pipeline()
        )

        build_bm25_index(
            chunk_data
        )

        return {
            "success": True,
            "message":
                "BM25 index built successfully."
        }

    except Exception as e:

        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }


# ============================================================
# BUILD ALL INDEXES
# ============================================================

def build_all_indexes():
    """
    Full pipeline:

    1. preprocess
    2. vector index
    3. bm25 index
    """

    try:

        chunk_data = (
            run_preprocessing_pipeline()
        )

        index_chunks(chunk_data)

        build_bm25_index(
            chunk_data
        )

        return {
            "success": True,
            "message":
                "All indexes built successfully."
        }

    except Exception as e:

        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }


# ============================================================
# DELETE INDEXES
# ============================================================

def delete_vector_index():
    """
    Removes ChromaDB storage.
    """

    try:

        if CHROMA_DIR.exists():

            shutil.rmtree(
                CHROMA_DIR
            )

        return {
            "success": True,
            "message":
                "Vector index removed."
        }

    except Exception as e:

        return {
            "success": False,
            "error": str(e)
        }


def delete_bm25_index():
    """
    Removes BM25 artifacts.
    """

    try:

        if BM25_INDEX.exists():
            BM25_INDEX.unlink()

        if BM25_MANIFEST.exists():
            BM25_MANIFEST.unlink()

        return {
            "success": True,
            "message":
                "BM25 index removed."
        }

    except Exception as e:

        return {
            "success": False,
            "error": str(e)
        }


def delete_all_indexes():
    """
    Deletes all retrieval indexes.
    """

    vector_result = (
        delete_vector_index()
    )

    bm25_result = (
        delete_bm25_index()
    )

    return {
        "success":
            vector_result["success"]
            and bm25_result["success"],

        "vector":
            vector_result,

        "bm25":
            bm25_result
    }


# ============================================================
# STATUS HELPERS
# ============================================================

def get_index_status():
    """
    Returns index availability.
    """

    return {

        "vector_index":
            CHROMA_DIR.exists(),

        "bm25_index":
            BM25_INDEX.exists(),

        "bm25_manifest":
            BM25_MANIFEST.exists()
    }


def get_raw_document_count():

    return len(
        list_raw_pdfs()
    )