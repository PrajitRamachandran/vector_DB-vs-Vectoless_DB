# RAG Benchmark for Financial 10-K Reports

A notebook-driven benchmark comparing three first-stage retrieval strategies for question answering over company 10-K annual reports.

- `Vector RAG`: semantic retrieval with embeddings stored in ChromaDB
- `Vectorless RAG`: lexical retrieval with BM25
- `Hybrid RAG`: fused Vector + BM25 retrieval using Reciprocal Rank Fusion (RRF)

All pipelines share the same preprocessing, company-aware query handling, reranking stage, and answer-generation model, making the benchmark a focused comparison of first-stage retrieval.

## What this project does

This project evaluates whether dense embedding retrieval adds value over a strong keyword baseline for financial 10-K documents.

It does that by:

- processing the same set of PDF 10-K reports into overlapping text chunks
- indexing the corpus using both ChromaDB embeddings and BM25 keyword search
- applying a shared cross-encoder reranker to both retrieval outputs
- generating final answers from the same Mistral-based LLM prompt
- scoring the results with a judge model and saving benchmark metrics

## Dataset scope

The repository currently contains 4 public company 10-K reports in `data/raw/`:

- `amazon_10k.pdf`
- `microsoft_10k.pdf`
- `netflix_10k.pdf`
- `nvidia_10k.pdf`

Processed corpus totals in the repo include:

- `4` PDFs
- `460` extracted pages
- `2,108` text chunks

The benchmark question set is stored in `evaluation/test_questions.json` and contains `20` labeled questions across categories such as:

- `financial_metrics`
- `risk_factors`
- `business_segments`
- `strategy`

## Key design decisions

- `data_loader.py` performs PDF hashing, extraction, cleaning, and overlapping chunk creation.
- `utils/query_processor.py` detects the target company and cleans the query before retrieval.
- `vector_rag/` stores dense embeddings in ChromaDB and performs semantic nearest-neighbor retrieval.
- `vectorless_rag/` uses BM25 keyword search as the first-stage retriever.
- `hybrid_rag/` fuses vector and BM25 candidate lists via Reciprocal Rank Fusion, then reranks the merged candidates.
- `reranker.py` uses a cross-encoder to rerank candidates from all pipelines.
- `llm.py` formats the retrieved context and calls Mistral for answer generation.
- `evaluation/evaluator.py` scores generated answers with a judge model and writes evaluation CSV files.

## How the methods differ

Shared across all methods:

- same processed chunks
- same company-aware query preprocessing
- same cross-encoder reranker
- same answer-generation model and prompt

Different first-stage retrieval:

- `Vector RAG`: embedding similarity via ChromaDB + `sentence-transformers/all-MiniLM-L6-v2`
- `Vectorless RAG`: BM25 lexical search via `rank-bm25`
- `Hybrid RAG`: fuses vector and BM25 ranked children with Reciprocal Rank Fusion (RRF) before reranking

The benchmark is intentionally designed to isolate first-stage retrieval differences while measuring the effect of hybrid fusion.

## Setup

### 1. Create and activate a Python virtual environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Create a `.env` file

Create `.env` in the repo root and add:

```env
MISTRAL_API_KEY=your_mistral_generation_key_here
MISTRAL_JUDGE_API_KEY=your_mistral_judge_key_here
HF_TOKEN=your_huggingface_token_here
```

Notes:

- `MISTRAL_API_KEY` is required for answer generation.
- `MISTRAL_JUDGE_API_KEY` is required for judge scoring.
- `HF_TOKEN` is optional but recommended for Hugging Face-backed embedding access.

## Running the project

### Recommended flow

1. Run `notebooks/01_data_exploration.ipynb` to preprocess the PDF corpus and build both indexes.
2. Use `notebooks/02_vector_rag.ipynb` to demo the vector retrieval flow.
3. Use `notebooks/03_vectorless_rag.ipynb` to demo the BM25 retrieval flow.
4. Use `notebooks/04_comparison.ipynb` to compare the two original pipelines.
5. Use `notebooks/05_three_way_comparison.ipynb` to benchmark Vector RAG, Vectorless RAG, and Hybrid RAG together.

### Programmatic usage

Build chunks and indexes:

```python
from data_loader import run_preprocessing_pipeline
from vector_rag.indexer import index_chunks
from vectorless_rag.indexer import build_bm25_index

chunks = run_preprocessing_pipeline()
index_chunks(chunks)
build_bm25_index(chunks)
```

Ask a question with the vector pipeline:

```python
from vector_rag.pipeline import VectorRAGPipeline
pipe = VectorRAGPipeline()
result = pipe.ask("What was NVIDIA's total revenue for the most recent fiscal year?")
pipe.show(result)
```

Ask a question with the vectorless pipeline:

```python
from vectorless_rag.pipeline import VectorlessRAGPipeline
pipe = VectorlessRAGPipeline()
result = pipe.ask("What was Amazon Web Services revenue for the most recent fiscal year?")
pipe.show(result)
```

## Current benchmark results

The repository includes both the original two-way benchmark and the newer three-way benchmark artifacts.

- `evaluation/results/full_results.csv`: original Vector vs Vectorless benchmark
- `evaluation/results/three_way_results.csv`: Vector, Vectorless, and Hybrid benchmark details
- `evaluation/results/three_way_summary.csv`: aggregated summary for all three methods

Key summary metrics from `evaluation/results/three_way_summary.csv`:

- `Vector RAG` average judge score: **3.70**
- `Vector RAG` pass rate: **85%**
- `Vector RAG` company accuracy: **100%**
- `Vector RAG` average total time: **3.05s**

- `Vectorless RAG` average judge score: **4.15**
- `Vectorless RAG` pass rate: **100%**
- `Vectorless RAG` company accuracy: **100%**
- `Vectorless RAG` average total time: **2.21s**

- `Hybrid RAG` average judge score: **3.75**
- `Hybrid RAG` pass rate: **90%**
- `Hybrid RAG` company accuracy: **100%**
- `Hybrid RAG` average total time: **3.02s**

These are snapshot metrics from the current saved results and may change when rerunning the benchmark.

## Project structure

```text
rag-benchmark/
|-- .env
|-- .gitignore
|-- config.py
|-- data_loader.py
|-- llm.py
|-- reranker.py
|-- requirements.txt
|-- data/
|   |-- raw/
|   |-- processed/
|-- vector_rag/
|   |-- chroma_db/
|   |-- indexer.py
|   |-- pipeline.py
|   |-- retriever.py
|-- vectorless_rag/
|   |-- indexer.py
|   |-- pipeline.py
|   |-- retriever.py
|-- hybrid_rag/
|   |-- pipeline.py
|   |-- retriever.py
|-- evaluation/
|   |-- evaluator.py
|   |-- test_questions.json
|   |-- results/
|-- notebooks/
|-- utils/
|   |-- query_processor.py
```

## File summary

- `config.py`: central configuration, model IDs, environment keys, path settings, retrieval/chunking constants.
- `data_loader.py`: extracts text from PDFs, cleans it, chunks it, manages incremental processing via manifest.
- `llm.py`: wraps Mistral chat completion calls and formats retrieved chunks into a prompt context.
- `reranker.py`: loads the cross-encoder reranker and rescoring logic.
- `vector_rag/indexer.py`: creates and updates the ChromaDB index.
- `vector_rag/retriever.py`: performs vector retrieval + company filtering + reranking.
- `vector_rag/pipeline.py`: pipeline wrapper for the vector flow.
- `vectorless_rag/indexer.py`: builds and loads a BM25 index, with chunk-list fingerprinting.
- `vectorless_rag/retriever.py`: performs BM25 retrieval with company filtering and reranking.
- `vectorless_rag/pipeline.py`: pipeline wrapper for the BM25 flow.
- `hybrid_rag/pipeline.py`: loads vector and BM25 indexes, runs RRF fusion, then reranks and answers.
- `hybrid_rag/retriever.py`: fuses vector and BM25 ranked children using Reciprocal Rank Fusion (RRF).
- `utils/query_processor.py`: extracts company and year information from questions.
- `evaluation/evaluator.py`: runs the benchmark on the question set and scores answers.

## Notes on current implementation

- `config.py` currently sets both `LLM_MODEL_ID` and `JUDGE_MODEL_ID` to `mistral-medium-latest`.
- `vectorless_rag/indexer.py` persists BM25 state to `vectorless_rag/bm25_index.pkl`.
- `vector_rag/chroma_db/` holds the persisted ChromaDB collection files.
- The repo has no CLI entrypoint; the notebooks and Python wrappers are the primary interfaces.

## Existing artifacts in this repository

- `data/processed/manifest.json`
- `data/processed/chunks.json`
- `vector_rag/chroma_db/` persisted vector store
- `vectorless_rag/bm25_index.pkl`
- `evaluation/results/full_results.csv`
- `evaluation/results/three_way_results.csv`
- `evaluation/results/three_way_summary.csv`
- evaluation result charts in `evaluation/results/`

## Known caveats

- The evaluation results are still tied to live Mistral API access.
- The checked-in result CSV is a snapshot and may not reflect the exact current code behavior.
- The codebase is a prototype/benchmark, not a packaged production application.

## Summary

This repository is a compact benchmark showing how semantic retrieval, BM25-based retrieval, and a hybrid Vector+BM25 fusion compare on the same financial 10-K corpus, while sharing chunking, reranking, and answer generation to keep the comparison as fair as possible.
