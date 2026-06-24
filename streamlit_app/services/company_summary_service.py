import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

CHUNKS_PATH = (
    ROOT_DIR /
    "data" /
    "processed" /
    "chunks.json"
)


def get_company_summary_context(
    company: str,
    max_chunks: int = 50
):

    if not CHUNKS_PATH.exists():
        return ""

    with open(
        CHUNKS_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        chunks = json.load(f)

        parent_chunk_objects = chunks.get(
            "parents",
            []
        )

        parent_chunks = []

        for chunk in parent_chunk_objects:

            if (
                chunk.get("company", "").upper()
                == company.upper()
            ):
                parent_chunks.append(
                    chunk["text"]
                )

    print("\n========== SUMMARY DEBUG ==========")
    print(f"Company: {company}")
    print(f"Parent Chunks Found: {len(parent_chunks)}")
    print("===================================\n")

    if not parent_chunks:
        return ""

    overview_chunks = []

    keywords = [

        "company",

        "overview",

        "business",

        "segment",

        "products",

        "services",

        "operations",

        "strategy",

        "customers",

        "markets"
    ]

    for chunk in parent_chunks:

        text_lower = chunk.lower()

        if any(
            k in text_lower
            for k in keywords
        ):
            overview_chunks.append(
                chunk
            )

    if overview_chunks:

        parent_chunks = (
            overview_chunks
            +
            parent_chunks
        )

    return "\n\n".join(
        parent_chunks[:max_chunks]
    )