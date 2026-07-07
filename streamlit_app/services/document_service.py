from pathlib import Path
import json

ROOT_DIR = Path(__file__).resolve().parents[2]

MANIFEST_PATH = (
    ROOT_DIR /
    "data" /
    "processed" /
    "manifest.json"
)

CHUNKS_PATH = (
    ROOT_DIR /
    "data" /
    "processed" /
    "chunks.json"
)

def get_document_metadata():

    if not MANIFEST_PATH.exists():

        return {
            "documents": [],
            "companies": [],
            "total_documents": 0
        }

    with open(
        MANIFEST_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        manifest = json.load(f)

    companies = []

    for pdf, metadata in manifest.items():

        company = metadata.get(
            "company",
            "UNKNOWN"
        )

        companies.append(
            company.upper()
        )

    return {

        "documents":
            list(manifest.keys()),

        "companies":
            sorted(list(set(companies))),

        "total_documents":
            len(manifest),

        "manifest":
            manifest
    }