"""
Upload Documents Page

Responsibilities:
----------------
1. Upload PDF reports
2. Store in data/raw/
3. List available PDFs
4. Delete PDFs
5. Run preprocessing

Index creation is handled separately
in Index Manager.
"""
from pathlib import Path
import streamlit as st
from upload_processor import process_uploaded_pdf
from streamlit_app.services.indexing_service import (
    save_uploaded_pdf,
    list_raw_pdfs,
    delete_pdf,
    preprocess_documents
)


from streamlit_app.auth.protect_page import (
    require_login
)

require_login()
# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Upload Documents",
    page_icon="📄",
    layout="wide"
)

# ============================================================
# HEADER
# ============================================================

st.title("📄 Upload Documents")

st.caption(
    "Upload company 10-K reports and preprocess them."
)

# ============================================================
# PDF UPLOADER
# ============================================================

st.subheader("Upload PDF Reports")

uploaded_files = st.file_uploader(
    label="Select one or more PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

# ============================================================
# SAVE FILES
# ============================================================

if uploaded_files:

    if st.button(
        "💾 Save Uploaded Files",
        use_container_width=True
    ):

        saved_count = 0

        with st.spinner(
            "Saving files..."
        ):

            for file in uploaded_files:
                try:
                    pdf_path, company = process_uploaded_pdf(file)
                    st.success(
                        f"Detected company: {company}"
                    )
                    st.info(
                        f"Saved as: {Path(pdf_path).name}"
                    )
                    saved_count += 1
                except Exception as e:
                    st.error(
                        f"{file.name}: {e}"
                    )
        st.success(
            f"{saved_count} file(s) saved successfully."
        )

        st.rerun()

# ============================================================
# DOCUMENT LIBRARY
# ============================================================

st.divider()

st.subheader("Available Documents")

pdfs = list_raw_pdfs()

if not pdfs:

    st.warning(
        "No PDF documents found."
    )

else:

    for pdf in pdfs:

        col1, col2 = st.columns(
            [8, 1]
        )

        with col1:

            st.write(
                f"📄 {pdf.name}"
            )

        with col2:

            if st.button(
                "🗑️",
                key=f"delete_{pdf.name}"
            ):

                success = delete_pdf(
                    pdf.name
                )

                if success:

                    st.success(
                        f"{pdf.name} deleted."
                    )

                    st.rerun()

                else:

                    st.error(
                        "Delete failed."
                    )

# ============================================================
# DOCUMENT SUMMARY
# ============================================================

st.divider()

st.subheader("Dataset Summary")

col1, col2 = st.columns(2)

with col1:

    st.metric(
        "Total PDFs",
        len(pdfs)
    )

with col2:

    total_size_mb = round(
        sum(
            pdf.stat().st_size
            for pdf in pdfs
        ) / (1024 * 1024),
        2
    )

    st.metric(
        "Storage Used (MB)",
        total_size_mb
    )

# ============================================================
# PREPROCESSING
# ============================================================

st.divider()

st.subheader(
    "Preprocess Documents"
)

st.markdown(
    """
This step will:

- Read PDFs
- Extract text
- Create parent chunks
- Create child chunks
- Generate chunks.json
- Generate manifest.json
"""
)

if st.button(
    "⚙️ Run Preprocessing",
    use_container_width=True
):

    progress = st.progress(0)

    with st.spinner(
        "Processing documents..."
    ):

        progress.progress(20)

        result = preprocess_documents()

        progress.progress(100)

    if result["success"]:

        st.success(
            result["message"]
        )

    else:

        st.error(
            result["error"]
        )

        with st.expander(
            "View Traceback"
        ):
            st.code(
                result["traceback"]
            )

# ============================================================
# INFO PANEL
# ============================================================

st.divider()

with st.expander(
    "ℹ️ Processing Workflow"
):

    st.code(
        """
PDF Upload
    ↓
data/raw/
    ↓
Preprocessing
    ↓
manifest.json
chunks.json
    ↓
Index Manager
    ↓
Vector / BM25 / Hybrid
"""
    )