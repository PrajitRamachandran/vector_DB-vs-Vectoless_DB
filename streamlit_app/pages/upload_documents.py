"""
Upload Documents Page
=====================
Upload new 10-K PDFs into data/raw/ and trigger the preprocessing pipeline.
Displays manifest and corpus stats after processing.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from services.index_service import run_preprocessing, load_manifest, load_chunks

_RAW_DIR = _ROOT / "data" / "raw"


def _save_upload(uploaded_file) -> Path:
    """Save a Streamlit UploadedFile to data/raw/ and return the path."""
    _RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest = _RAW_DIR / uploaded_file.name
    with open(dest, "wb") as f:
        shutil.copyfileobj(uploaded_file, f)
    return dest


def render() -> None:
    st.title("📂 Upload Documents")
    st.caption("Add new 10-K PDFs to the corpus. Existing files are skipped automatically (manifest-based).")

    # ── Current corpus ────────────────────────────────────────────────────────
    st.subheader("Current Corpus")
    manifest = load_manifest()

    if manifest:
        for fname, meta in manifest.items():
            with st.expander(fname, expanded=False):
                col1, col2, col3 = st.columns(3)
                col1.metric("Parents",  meta.get("parents_count",  "—"))
                col2.metric("Children", meta.get("children_count", "—"))
                col3.metric("Processed", meta.get("processed_at",  "—")[:10])
    else:
        st.info("No documents processed yet.")

    st.divider()

    # ── File uploader ─────────────────────────────────────────────────────────
    st.subheader("Upload New PDFs")
    uploaded = st.file_uploader(
        "Drop 10-K PDF files here",
        type=["pdf"],
        accept_multiple_files=True,
        help="Files are saved to data/raw/. Only new or changed files will be reprocessed.",
    )

    if uploaded:
        saved_paths = []
        for f in uploaded:
            dest = _save_upload(f)
            saved_paths.append(dest.name)
        st.success(f"Saved {len(saved_paths)} file(s): {', '.join(saved_paths)}")

    st.divider()

    # ── Preprocessing trigger ─────────────────────────────────────────────────
    st.subheader("Run Preprocessing")
    st.markdown(
        "Preprocessing extracts text from each PDF, cleans it, and builds "
        "the **parent-child chunk hierarchy** used by all three RAG pipelines."
    )

    col_run, col_info = st.columns([1, 3])
    with col_run:
        run_btn = st.button("▶ Run Preprocessing", type="primary", use_container_width=True)

    with col_info:
        st.caption(
            "Only new/changed PDFs will be processed. "
            "A manifest tracks which files have already been chunked."
        )

    if run_btn:
        log_area = st.empty()
        log_area.info("Running preprocessing pipeline…")
        try:
            result = run_preprocessing()
            parents  = len(result.get("parents",  []))
            children = len(result.get("children", []))
            log_area.success(
                f"✅ Done — {parents:,} parent chunks, {children:,} child chunks in corpus."
            )
        except Exception as exc:
            log_area.error(f"❌ Preprocessing failed: {exc}")
            st.exception(exc)
            return

        # ── Refresh manifest view ─────────────────────────────────────────────
        st.divider()
        st.subheader("Updated Corpus")
        manifest = load_manifest()
        rows = [
            {
                "File":         fname,
                "Parents":      m.get("parents_count",  "—"),
                "Children":     m.get("children_count", "—"),
                "Processed At": m.get("processed_at",   "—")[:19],
            }
            for fname, m in manifest.items()
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)
