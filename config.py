# config.py
import sys
import os
from dotenv import load_dotenv
from pathlib import Path
from pathlib import Path

load_dotenv()
BASE_DIR = Path(__file__).resolve().parent

#Groq Cloud LLM 
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL_ID = "llama-3.3-70b-versatile"

#Gemini Cloud LLM
GOOGLE_API_KEY = os.getenv("GEMINI_KEY")
GEMINI_MODEL_ID = "gemini-2.5-flash"

# Rate limits (slightly under the real limits for safety margin)
GROQ_RPM   = 18    # real limit is 20
GEMINI_RPM = 13    # real limit is 15

# HuggingFace
HF_TOKEN        = os.getenv("HF_TOKEN")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ChromaDB
CHROMA_PERSISTENT_DIR = BASE_DIR / "vector_rag" / "chroma_db"
CHROMA_COLLECTION = "financial_docs"

# Chunking — INCREASED for better context
CHUNK_SIZE    = 1000   # was 500 — financial sentences need more room
CHUNK_OVERLAP = 150    # was 50  — more overlap prevents cut-off context

# Retrieval — INCREASED for better coverage
TOP_K = 8              # was 5

# Paths
# NEW — anchor everything to project root
_ROOT              = Path(__file__).parent
DATA_RAW_DIR       = str(_ROOT / "data" / "raw")
DATA_PROCESSED_DIR = str(_ROOT / "data" / "processed")
RESULTS_DIR        = str(_ROOT / "evaluation" / "results")
QUESTIONS_DIR = str(_ROOT / "evaluation" / "test_questions.json")

# Generation
MAX_NEW_TOKENS = 512
TEMPERATURE    = 0.1
DO_SAMPLE      = False

# Known companies in your dataset
# Add new company names here when you add new PDFs
KNOWN_COMPANIES = {
    "nvidia"   : "NVIDIA",
    "microsoft": "MICROSOFT",
    "netflix"  : "NETFLIX",
    "amazon"   : "AMAZON"
}