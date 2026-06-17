# Financial RAG Benchmark

A full-stack benchmark and Streamlit application for comparing three retrieval-first question answering pipelines over financial 10-K reports.

This repository is designed to isolate retrieval quality while using the same answer generation and reranking components across methods.

## Core Retrieval Methods

- **Vector RAG**: semantic retrieval using ChromaDB with `BAAI/bge-large-en-v1.5` embeddings, HNSW, and cross-encoder reranking.
- **Vectorless RAG**: lexical retrieval using BM25 over the same child chunk corpus with a financial-aware tokenizer.
- **Hybrid RAG**: reciprocal rank fusion (RRF) of Vector and BM25 candidate sets followed by a shared cross-encoder reranker.

## What This Project Does

The benchmark supports the full pipeline from raw PDF ingestion to evaluation:

- PDF upload and storage in `data/raw/`
- markdown-aware text extraction from annual reports
- hierarchical chunking into parent and child passages
- dual indexing with ChromaDB and BM25
- company-aware query preprocessing
- reranking with a cross-encoder
- answer generation with Mistral
- judge scoring plus optional RAGAS metrics
- Streamlit UI for upload, indexing, chat, conversation history, and benchmark dashboards

## Updated Architecture Overview

### Data ingestion and preprocessing

- `data_loader.py` manages PDF detection, hashing, reprocessing, and page extraction.
- PDF pages are extracted using `pymupdf4llm.to_markdown()` to preserve tables and structured text.
- Cleaned text is split into hierarchical chunks:
  - `parent` chunks (~1000 characters) preserve broader context.
  - `child` chunks (~300 characters) provide precise retrieval targets.
- `manifest.json` tracks processed PDFs, hashes, page counts, and chunk counts.
- `chunks.json` contains the full parent/child chunk corpus.

### Vector RAG pipeline

- `vector_rag/indexer.py` creates or loads a persistent ChromaDB collection under `vector_rag/chroma_db/`.
- ChromaDB uses HNSW parameters tuned for recall: `M=32`, `construction_ef=200`, `search_ef=100`.
- `vector_rag/retriever.py` applies a BGE query prefix, company filtering, stale-index fallback, reranking, and parent lookup.
- `vector_rag/pipeline.py` loads the index, resolves parent chunks, generates answers, and returns structured latency and retrieval metadata.

### Vectorless RAG pipeline

- `vectorless_rag/indexer.py` builds a BM25 index on child chunks and stores it with a manifest.
- The tokenizer preserves finance-specific tokens such as `$60.9B`, `122%`, `Q3-2024`, and years.
- `vectorless_rag/retriever.py` filters children by company, scores them with BM25, reranks the top results, and swaps child chunks for parent context.
- `vectorless_rag/pipeline.py` loads the BM25 index, builds the parent lookup, generates answers, and returns retrieval metrics.

### Hybrid RAG pipeline

- `hybrid_rag/retriever.py` merges vector and BM25 candidate lists using Reciprocal Rank Fusion.
- `hybrid_rag/pipeline.py` loads both vector and BM25 assets, retrieves fused candidates, reranks them, and generates the final answer.

### Shared components

- `reranker.py` uses `sentence-transformers.CrossEncoder` with `BAAI/bge-reranker-large`.
- `llm.py` wraps the OpenAI-compatible Mistral client and formats retrieved chunks into a labeled context prompt.
- `config.py` centralizes model IDs, chunk sizes, retrieval parameters, paths, known company names, and rate limits.
- `utils/query_processor.py` detects company names and years in questions and cleans the query for retrieval.

## Streamlit Application Features

The Streamlit app is served through `app.py` and the `pages/` folder.

### Pages and capabilities

- `pages/01_dashboard.py`: monitoring overview, dataset stats, chunk distribution, evaluation status, and recent interactions.
- `pages/02_upload_documents.py`: upload PDFs, delete stored files, and run preprocessing.
- `pages/03_index_manager.py`: build Vector, BM25, or all indexes; delete indexes; view index readiness.
- `pages/04_chat.py`: interactive QA with method selection, live response latency, retrieved chunk inspection, and raw response output.
- `pages/05_conversations.py`: conversation history, filtering, detail view, chunk inspection, deletion, and export.
- `pages/06_evaluations.py`: run judge benchmark, run RAGAS evaluation, display leaderboard, and plot tracker charts.

### Database and persistence

- SQLite database at `storage/benchmark.db`.
- `streamlit_app/database/schema.py` defines tables for conversations, retrieved chunks, evaluations, and logs.
- `streamlit_app/database/repository.py` manages all database CRUD operations.
- `streamlit_app/services/indexing_service.py` wraps preprocessing and indexing for the UI.
- `streamlit_app/services/rag_service.py` exposes cached pipeline loading and online QA.
- `streamlit_app/services/evaluation_service.py` orchestrates judge and RAGAS benchmark runs and saves results.

## Evaluation and Benchmarking

### Judge benchmark

- `evaluation/evaluator.py` loads benchmark questions from `evaluation/test_questions.json`.
- Each question is executed through Vector, Vectorless, and Hybrid pipelines.
- Answers are scored by a Mistral judge model from 1 to 5.
- Output includes answer text, judge score, reason, pass status, company accuracy, rerank metrics, and latency.
- Results are written to CSV in `evaluation/results/`.

### RAGAS evaluation

- `evaluation/ragas_evaluator.py` computes RAGAS metrics on top of judge evaluation data.
- It produces faithfulness, answer relevancy, context precision, context recall, and contextual relevancy.
- RAGAS evaluation is optional and requires `ragas` + compatible LangChain ecosystem packages.
- Aggregated benchmark summaries are persisted in the SQLite `evaluations` table.

## Setup Instructions

### 1. Create a Python virtual environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Create a `.env` file

```
MISTRAL_API_KEY=your_mistral_generation_key_here
MISTRAL_JUDGE_API_KEY=your_mistral_judge_key_here
HF_TOKEN=your_huggingface_token_here
```

### 4. Initialize the database schema

```powershell
python -c "from streamlit_app.database.schema import initialize_schema; initialize_schema()"
```

## Running the Project

### Start the Streamlit app

```powershell
streamlit run app.py
```

Then open the app in your browser and use the sidebar pages to upload reports, preprocess, build indexes, chat, and benchmark.

### Programmatic usage

```python
from data_loader import run_preprocessing_pipeline
from vector_rag.indexer import index_chunks
from vectorless_rag.indexer import build_bm25_index

chunks = run_preprocessing_pipeline()
index_chunks(chunks)
build_bm25_index(chunks)

from vector_rag.pipeline import VectorRAGPipeline
from vectorless_rag.pipeline import VectorlessRAGPipeline
from hybrid_rag.pipeline import HybridRAGPipeline

vector = VectorRAGPipeline()
vectorless = VectorlessRAGPipeline()
hybrid = HybridRAGPipeline()

result = vector.ask("What was NVIDIA's total revenue for the most recent fiscal year?")
print(result["answer"])
```

## File and Data Flow

1. Upload PDFs to `data/raw/`.
2. Run preprocessing to generate `data/processed/manifest.json` and `data/processed/chunks.json`.
3. Build the Vector index under `vector_rag/chroma_db/`.
4. Build the BM25 index under `vectorless_rag/bm25_index.pkl`.
5. Query pipelines in the Streamlit chat page or via pipeline classes.
6. Run benchmarks and inspect evaluation results.

## Retrieval and Generation Details

### Parent-child chunking

- Child chunks are indexed because they are compact and precise.
- Parent chunks are returned to the LLM because they preserve richer context.
- `build_parent_lookup()` maps child `parent_id` metadata back to the parent text.

### Query preprocessing

- `utils/query_processor.py` detects company names and years in the question.
- Company references are removed from the retrieval query to avoid overwhelming the match score.
- Vector retrieval applies a BGE query prefix for best performance.
- BM25 retrieval uses a tokenizer that preserves dollar amounts, percentages, years, and finance-specific symbols.

### Safety and fallback logic

- Vector retrieval first queries with a company filter, then falls back to an unfiltered search if too few or wrong-company chunks appear.
- Hybrid retrieval merges both retrieval sources with RRF.
- The reranker rescoring step is the same for all pipelines.
- If no context is retrieved, the LLM returns a safe fallback instead of hallucinating.

## Current Results and Notes

- The benchmark currently uses 4 company 10-K reports: Amazon, Microsoft, Netflix, and NVIDIA.
- The evaluation dataset is stored in `evaluation/test_questions.json`.
- The Streamlit app saves conversation history, retrieved chunks, evaluation runs, and logs in SQLite.
- There is an existing `company_accuracy` artifact caused by uppercase company metadata compared to mixed-case question company labels.

## Project Structure

```text
rag-benchmark/
├── app.py
├── config.py
├── data_loader.py
├── llm.py
├── reranker.py
├── evaluation/
│   ├── evaluator.py
│   ├── ragas_evaluator.py
│   ├── results/
│   └── test_questions.json
├── hybrid_rag/
│   ├── pipeline.py
│   └── retriever.py
├── pages/
│   ├── 01_dashboard.py
│   ├── 02_upload_documents.py
│   ├── 03_index_manager.py
│   ├── 04_chat.py
│   ├── 05_conversations.py
│   └── 06_evaluations.py
├── streamlit_app/
│   ├── database/
│   │   ├── db.py
│   │   ├── repository.py
│   │   └── schema.py
│   └── services/
│       ├── evaluation_service.py
│       ├── indexing_service.py
│       └── rag_service.py
├── vector_rag/
│   ├── chroma_db/
│   ├── indexer.py
│   ├── pipeline.py
│   └── retriever.py
├── vectorless_rag/
│   ├── bm25_index.pkl
│   ├── bm25_manifest.json
│   ├── indexer.py
│   ├── pipeline.py
│   └── retriever.py
├── data/
│   ├── raw/
│   └── processed/
│       ├── chunks.json
│       └── manifest.json
├── storage/
│   └── benchmark.db
├── notebooks/
└── requirements.txt
```

## Notes

- The Streamlit app uses `st.cache_resource` for pipeline caching and `st.cache_data` for manifest/chunk loading.
- Clear the pipeline cache after rebuilding indexes using the chat page utility.
- The system is built for financial 10-K QA but can be adapted to other document sets using the same parent-child retrieval pattern.
