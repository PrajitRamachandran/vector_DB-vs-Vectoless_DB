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

# Selectable models exposed to admins in the chat UI
# ("Model selection controls" / "admin retrieval parameter controls").
AVAILABLE_LLM_MODELS = [
    "mistral-medium-latest",
    "mistral-small-latest",
    "mistral-large-latest",
]

# Approximate USD price per 1K tokens, used only for the
# in-app cost estimator shown in the chat UI. These are
# indicative figures, not billing-accurate — verify against
# your Mistral account/plan before treating them as exact.
LLM_PRICING_PER_1K_TOKENS = {
    "mistral-medium-latest": {"prompt": 0.0027, "completion": 0.0081},
    "mistral-small-latest":  {"prompt": 0.0010, "completion": 0.0030},
    "mistral-large-latest":  {"prompt": 0.0040, "completion": 0.0120},
}
#______________________________________________________________________________________________________________


# ── Embedding — upgraded to BGE base ────────────────────────────
# BGE models are trained specifically for retrieval tasks
# base (768 dims) vs MiniLM (384 dims) — 2x richer representation
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
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

# Bounds admins can tune live from the chat sidebar without
# risking a degenerate/empty retrieval configuration.
ADMIN_TOP_K_RANGE = (1, 20)
ADMIN_FETCH_K_RANGE = (5, 50)
ADMIN_RERANK_THRESHOLD_RANGE = (0.0, 1.0)
DEFAULT_RERANK_THRESHOLD = 0.0

# Paths
_ROOT = Path(__file__).parent
DATA_RAW_DIR = str(_ROOT / "data" / "raw")
DATA_PROCESSED_DIR = str(_ROOT / "data" / "processed")
RESULTS_DIR = str(_ROOT / "evaluation" / "benchmark_results")
QUESTIONS_DIR = str(_ROOT / "evaluation" / "test_questions.json")
EXPORTS_DIR = str(_ROOT / "data" / "exports")

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
    "reliance": "RELIANCE",
    "asus": "ASUS",
    "cocacola": "COCACOLA",
}

# Query classification labels surfaced in the chat UI
QUERY_TYPE_LABELS = {
    "numerical": "🔢 Numerical",
    "comparative": "⚖️ Comparative",
    "factual": "📌 Factual",
    "semantic": "🧠 Semantic",
    "unknown": "❓ Unclassified",
}

# Starter questions shown on the chat welcome screen
STARTER_QUESTIONS = [
    "What were NVIDIA's total revenues in the last fiscal year?",
    "Compare Microsoft and Amazon's R&D spending.",
    "What are the key risk factors mentioned in Netflix's 10-K?",
    "Summarize Reliance's business segments.",
]