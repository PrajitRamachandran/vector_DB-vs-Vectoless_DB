"""
Export Service

Turns a list of conversation rows (as returned by the
repository layer) into downloadable bytes in CSV, Markdown,
JSON, or PDF format for the chat page's export buttons.

PDF export uses fpdf2 (pure-Python, no system dependencies).
If fpdf2 isn't installed, `export_pdf` raises a clear
ImportError with the install instructions rather than failing
silently.
"""

import csv
import io
import json
from datetime import datetime


# ============================================================
# CSV
# ============================================================

def export_csv(conversations: list) -> bytes:

    if not conversations:
        conversations = []

    fieldnames = [
        "timestamp", "title", "method", "model_name",
        "prompt", "response", "company_filter",
        "total_latency", "status"
    ]

    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer, fieldnames=fieldnames, extrasaction="ignore"
    )
    writer.writeheader()

    for row in conversations:
        writer.writerow(row)

    return buffer.getvalue().encode("utf-8")


# ============================================================
# MARKDOWN
# ============================================================

def export_markdown(conversations: list) -> bytes:

    lines = [
        "# Financial RAG Chat Export",
        f"_Exported: {datetime.now().isoformat(timespec='seconds')}_",
        "",
    ]

    for row in conversations:
        lines.append(f"## {row.get('title') or row.get('prompt', 'Untitled')}")
        lines.append(f"- **Timestamp:** {row.get('timestamp', '-')}")
        lines.append(f"- **Method:** {row.get('method', '-')}")
        lines.append(f"- **Model:** {row.get('model_name', '-')}")
        lines.append(f"- **Latency:** {row.get('total_latency', '-')}s")
        lines.append("")
        lines.append(f"**Question:** {row.get('prompt', '')}")
        lines.append("")
        lines.append(f"**Answer:** {row.get('response', '')}")
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines).encode("utf-8")


# ============================================================
# JSON
# ============================================================

def export_json(conversations: list) -> bytes:

    payload = {
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "count": len(conversations),
        "conversations": conversations,
    }

    return json.dumps(payload, indent=2, default=str).encode("utf-8")


# ============================================================
# PDF
# ============================================================

def export_pdf(conversations: list) -> bytes:

    try:
        from fpdf import FPDF
    except ImportError as e:
        raise ImportError(
            "PDF export requires the 'fpdf2' package. "
            "Install it with: pip install fpdf2"
        ) from e

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Financial RAG Chat Export", ln=True)

    pdf.set_font("Helvetica", "", 9)
    pdf.cell(
        0, 6,
        f"Exported: {datetime.now().isoformat(timespec='seconds')}",
        ln=True
    )
    pdf.ln(4)

    for row in conversations:

        pdf.set_font("Helvetica", "B", 12)
        title = str(row.get("title") or row.get("prompt", "Untitled"))
        pdf.multi_cell(0, 7, _ascii_safe(title))

        pdf.set_font("Helvetica", "", 9)
        meta = (
            f"{row.get('timestamp', '-')}  |  "
            f"Method: {row.get('method', '-')}  |  "
            f"Latency: {row.get('total_latency', '-')}s"
        )
        pdf.multi_cell(0, 6, _ascii_safe(meta))
        pdf.ln(1)

        pdf.set_font("Helvetica", "B", 10)
        pdf.multi_cell(0, 6, "Question:")
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 6, _ascii_safe(str(row.get("prompt", ""))))
        pdf.ln(1)

        pdf.set_font("Helvetica", "B", 10)
        pdf.multi_cell(0, 6, "Answer:")
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 6, _ascii_safe(str(row.get("response", ""))))

        pdf.ln(4)
        pdf.set_draw_color(200, 200, 200)
        y = pdf.get_y()
        pdf.line(10, y, 200, y)
        pdf.ln(4)

    return pdf.output(dest="S").encode("utf-8")


def _ascii_safe(text: str) -> str:
    """
    fpdf2's core Helvetica font is latin-1 only; degrade
    gracefully instead of raising on unusual characters.
    """

    return text.encode("latin-1", errors="replace").decode("latin-1")


# ============================================================
# DISPATCH
# ============================================================

_EXPORTERS = {
    "csv": (export_csv, "text/csv", "chat_export.csv"),
    "markdown": (export_markdown, "text/markdown", "chat_export.md"),
    "json": (export_json, "application/json", "chat_export.json"),
    "pdf": (export_pdf, "application/pdf", "chat_export.pdf"),
}


def export(conversations: list, fmt: str):
    """
    Returns (bytes, mime_type, filename) for the requested format.
    """

    fmt = fmt.lower()

    if fmt not in _EXPORTERS:
        raise ValueError(f"Unsupported export format: {fmt}")

    fn, mime, filename = _EXPORTERS[fmt]

    return fn(conversations), mime, filename