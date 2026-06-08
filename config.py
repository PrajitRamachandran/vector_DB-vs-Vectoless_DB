# config.py
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
BASE_DIR = Path(__file__).resolve().parent


#_____________________________LLM CONFIGS____________________________________________________________________
# Mistral answer generation
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_BASE_URL = "https://api.mistral.ai/v1"
LLM_MODEL_ID = "mistral-medium-latest"

# Mistral evaluation / judging
MISTRAL_JUDGE_API_KEY = os.getenv("MISTRAL_JUDGE_API_KEY")
JUDGE_MODEL_ID = "mistral-medium-latest"

# Rate limits
MISTRAL_RPM = 18
MISTRAL_JUDGE_RPM = 12
#______________________________________________________________________________________________________________


# ── Embedding — upgraded to BGE base ────────────────────────────
# BGE models are trained specifically for retrieval tasks
# base (768 dims) vs MiniLM (384 dims) — 2x richer representation
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
HF_TOKEN = os.getenv("HF_TOKEN")
#________________________________________________________________

# ChromaDB
CHROMA_PERSIST_DIR = BASE_DIR / "vector_rag" / "chroma_db"
CHROMA_COLLECTION = "financial_docs"


# HNSW tuning — these are set at collection creation time
# M=32: more graph connections → better recall (default is 16)
# construction_ef=200: more candidates at build time → better index quality
# search_ef=100: more candidates at query time → better recall vs speed tradeoff
HNSW_M               = 32
HNSW_CONSTRUCTION_EF = 200
HNSW_SEARCH_EF       = 100


# ── Parent-Child Chunking ──────────────────────────────
# Children: small, precise → indexed in ChromaDB and BM25
# Parents:  large, rich   → retrieved and sent to LLM as context
CHILD_CHUNK_SIZE    = 300
CHILD_CHUNK_OVERLAP = 50
PARENT_CHUNK_SIZE   = 1000
PARENT_CHUNK_OVERLAP= 100

# ── Retrieval ──────────────────────────────────────────
TOP_K = 5   # final chunks returned (after reranking)
FETCH_K = 15  # candidates fetched before reranking (3x TOP_K)

# Paths
_ROOT = Path(__file__).parent
DATA_RAW_DIR = str(_ROOT / "data" / "raw")
DATA_PROCESSED_DIR = str(_ROOT / "data" / "processed")
RESULTS_DIR = str(_ROOT / "evaluation" / "results")
QUESTIONS_DIR = str(_ROOT / "evaluation" / "test_questions.json")

# Generation
MAX_NEW_TOKENS = 512
# Keep generation deterministic for benchmark-style QA.
TEMPERATURE = 0.0
DO_SAMPLE = False

# Judge generation
JUDGE_MAX_TOKENS = 180
JUDGE_TEMPERATURE = 0.0

# Known companies in your dataset
KNOWN_COMPANIES = {
    "nvidia": "NVIDIA",
    "microsoft": "MICROSOFT",
    "netflix": "NETFLIX",
    "amazon": "AMAZON",
}
