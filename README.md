# Financial RAG Benchmark

A full-stack retrieval-augmented generation benchmark for comparing three financial QA pipelines over 10-K reports.

This repository isolates retrieval quality while using the same answer generation and reranking stack across methods.

## What this project does

- Ingests financial documents from `data/raw/`
- Extracts markdown-aware text from annual reports
- Builds parent/child hierarchical chunks for precise retrieval
- Indexes the same corpus via:
  - semantic vector retrieval with ChromaDB + BGE embeddings
  - lexical BM25 retrieval over child chunks
  - hybrid Reciprocal Rank Fusion (RRF) of vector and BM25 candidates
- Reranks retrieved answers with a cross-encoder
- Generates responses with Mistral
- Benchmarks retrieval quality with judge scoring and optional RAGAS metrics
- Provides a Streamlit UI for upload, indexing, chat, conversation history, and evaluation dashboards

## Core retrieval methods

- **Vector RAG**: ChromaDB semantic search using `BAAI/bge-large-en-v1.5` and HNSW
- **Vectorless RAG**: BM25 lexical search over child chunks
- **Hybrid RAG**: RRF fusion of vector and BM25 candidates followed by shared reranking

## Architecture overview

### Data ingestion and preprocessing

- `data_loader.py` handles PDF detection, hashing, page extraction, and text cleaning.
- Raw document text is extracted into markdown-aware pages that preserve tables and structure.
- Text is broken into hierarchical chunks:
  - `parent` chunks (~1000 chars) preserve broader context
  - `child` chunks (~300 chars) capture precise retrieval targets
- `manifest.json` tracks files, hashes, pages, and chunk metadata.
- `chunks.json` stores the processed chunk corpus.

### Vector RAG pipeline

- `vector_rag/indexer.py` creates or loads a ChromaDB collection at `vector_rag/chroma_db/`.
- ChromaDB uses recall-tuned HNSW settings: `M=32`, `construction_ef=200`, `search_ef=100`.
- `vector_rag/retriever.py` uses a BGE query prefix, company-aware filters, stale index fallback, reranking, and parent lookup.
- `vector_rag/pipeline.py` executes retrieval, formats context, generates an answer, and returns latency and retrieval metadata.

### Vectorless RAG pipeline

- `vectorless_rag/indexer.py` builds a BM25 index on child chunks and saves it with a manifest.
- The tokenizer preserves finance-specific tokens like `$60.9B`, `122%`, `Q3-2024`, and year references.
- `vectorless_rag/retriever.py` filters candidates by company, ranks with BM25, reranks the top set, and returns parent-context chunks.
- `vectorless_rag/pipeline.py` loads the BM25 index, builds the parent lookup, generates answers, and returns retrieval metrics.

### Hybrid RAG pipeline

- `hybrid_rag/retriever.py` merges vector and BM25 candidate lists using Reciprocal Rank Fusion.
- `hybrid_rag/pipeline.py` loads both retrieval assets, retrieves fused candidates, reranks them, and returns the final answer.

### Shared components

- `reranker.py` uses `sentence-transformers.CrossEncoder` for reranking.
- `llm.py` wraps the OpenAI-compatible Mistral client and formats retrieved chunks into a labeled prompt.
- `config.py` centralizes model IDs, chunk sizes, retrieval parameters, paths, known companies, and rate limits.
- `utils/query_processor.py` detects company names and years and cleans retrieval queries.

## Streamlit application

The Streamlit app is served through `app.py` and organized via pages in the `pages/` folder.

### Streamlit pages

- `pages/01_dashboard.py`: dashboard, dataset stats, and evaluation summary
- `pages/02_upload_documents.py`: upload PDFs, delete files, and preprocess documents
- `pages/03_index_manager.py`: build/delete indexes and view readiness
- `pages/04_chat.py`: interactive QA with method selection and retrieval debugging
- `pages/05_conversations.py`: conversation history storage and review
- `pages/06_evaluations.py`: run benchmarks and inspect RAGAS/judge results

### Database and persistence

- SQLite persistence at `storage/benchmark.db`
- `streamlit_app/database/schema.py` defines tables for conversations, retrieved chunks, evaluations, and logs
- `streamlit_app/database/repository.py` manages database CRUD operations
- `streamlit_app/services/indexing_service.py`, `rag_service.py`, and `evaluation_service.py` provide UI service abstractions

## Evaluation

### Judge benchmark

- `evaluation/evaluator.py` runs benchmark questions from `evaluation/test_questions.json`
- Each question is executed through Vector, Vectorless, and Hybrid pipelines
- Answers are scored by a Mistral judge model
- Results are saved as CSV files in `evaluation/results/`

### RAGAS evaluation

- `evaluation/ragas_evaluator.py` computes RAGAS metrics over judge evaluation output
- Metrics include faithfulness, answer relevancy, context precision, context recall, and contextual relevancy
- RAGAS evaluation requires `ragas` and compatible LangChain packages

## Setup

### 1. Create a Python virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Create a `.env` file

```text
MISTRAL_API_KEY=your_mistral_generation_key_here
MISTRAL_JUDGE_API_KEY=your_mistral_judge_key_here
HF_TOKEN=your_huggingface_token_here
```

### 4. Initialize the database schema

```powershell
python -c "from streamlit_app.database.schema import initialize_schema; initialize_schema()"
```

## Running the app

```powershell
streamlit run app.py
```

Then open the Streamlit app and use sidebar navigation for dashboard, uploads, indexing, chat, conversations, and evaluation.

## Programmatic usage

```python
from data_loader import run_preprocessing_pipeline
from vector_rag.indexer import index_chunks
from vectorless_rag.indexer import build_bm25_index
from vector_rag.pipeline import VectorRAGPipeline
from vectorless_rag.pipeline import VectorlessRAGPipeline
from hybrid_rag.pipeline import HybridRAGPipeline

chunks = run_preprocessing_pipeline()
index_chunks(chunks)
build_bm25_index(chunks)

vector = VectorRAGPipeline()
vectorless = VectorlessRAGPipeline()
hybrid = HybridRAGPipeline()

result = vector.ask("What was NVIDIA's total revenue for the most recent fiscal year?")
print(result["answer"])
```

## Data flow

1. Add PDFs to `data/raw/`
2. Run preprocessing to generate `data/processed/manifest.json` and `data/processed/chunks.json`
3. Build the Vector index under `vector_rag/chroma_db/`
4. Build the BM25 index under `vectorless_rag/bm25_index.pkl`
5. Query via Streamlit chat or pipeline classes
6. Run benchmarks and review results in `evaluation/results/`

## Notes

- `config.py` centralizes runtime paths and model settings
- Streamlit persistence is backed by `storage/benchmark.db`
- Known companies currently tracked are NVIDIA, Microsoft, Netflix, and Amazon
- The Streamlit app uses `st.cache_resource` for pipeline caching and `st.cache_data` for manifest/chunk loading
- Clear the pipeline cache after rebuilding indexes using the chat page utility
- The project is built for financial 10-K benchmarking, but the retrieval pattern can be adapted to other document collections
