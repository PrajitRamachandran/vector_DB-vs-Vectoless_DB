# config.py
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
BASE_DIR = Path(__file__).resolve().parent

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

# HuggingFace
HF_TOKEN = os.getenv("HF_TOKEN")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ChromaDB
CHROMA_PERSISTENT_DIR = BASE_DIR / "vector_rag" / "chroma_db"
CHROMA_COLLECTION = "financial_docs"

# Chunking - increased for better context
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# Retrieval - increased for better coverage
TOP_K = 8

# Paths
_ROOT = Path(__file__).parent
DATA_RAW_DIR = str(_ROOT / "data" / "raw")
DATA_PROCESSED_DIR = str(_ROOT / "data" / "processed")
RESULTS_DIR = str(_ROOT / "evaluation" / "results")
QUESTIONS_DIR = str(_ROOT / "evaluation" / "test_questions.json")

# Generation
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.1
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
