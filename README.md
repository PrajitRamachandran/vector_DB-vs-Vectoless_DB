# RAG Benchmark for Financial 10-K Reports

This project compares two retrieval strategies for question answering over company 10-K annual reports:

- `Vector RAG`: semantic retrieval with embeddings stored in ChromaDB
- `Vectorless RAG`: lexical retrieval with BM25

Both pipelines share the same preprocessing, company-aware query handling, reranking stage, and answer-generation model. That makes the comparison much cleaner: the main thing changing is the first-stage retriever.

The repository is best understood as a notebook-driven benchmark/prototype rather than a packaged production service. It ingests PDF reports, chunks them, builds two indexes, answers finance questions, and evaluates both approaches on the same test set.

## What this project aims to do

The goal of the project is to answer a practical question:

> For structured financial documents like 10-K reports, how much do we gain from embedding-based retrieval over a strong keyword baseline?

More specifically, the project aims to:

- benchmark `vector` vs `vectorless` retrieval on the same document set
- reduce cross-company contamination by detecting the company named in the question
- reuse the same downstream reranker and LLM so the benchmark isolates retrieval differences
- build a reusable workflow for PDF ingestion, chunking, indexing, querying, and evaluation
- measure both retrieval quality and response behavior on a small multi-company benchmark

## Benchmark scope

The current dataset contains 4 public company 10-K reports:

| Company   | PDF file                     |Pages|Chunks|
| ----------|------------------------------|-----|------|
| Amazon    | `data/raw/amazon_10k.pdf`    | 90  | 442  |
| Microsoft | `data/raw/microsoft_10k.pdf` | 156 | 671  |
| Netflix   | `data/raw/netflix_10k.pdf`   | 121 | 552  |
| NVIDIA    | `data/raw/nvidia_10k.pdf`    | 93  | 443  |

Current processed corpus totals:

- `4` PDFs
- `460` extracted pages
- `2,108` text chunks

The evaluation set contains `20` questions across:

- `financial_metrics`
- `risk_factors`
- `business_segments`
- `strategy`

## What is actually being compared

This is not a comparison between a fully neural pipeline and a fully traditional pipeline.

What stays the same in both methods:

- the same source PDFs
- the same chunking settings
- the same company-aware query preprocessing
- the same cross-encoder reranker
- the same answer-generation LLM

What changes:

- `Vector RAG` uses dense embeddings plus ChromaDB for first-stage retrieval
- `Vectorless RAG` uses BM25 keyword matching for first-stage retrieval

That means the benchmark is really comparing:

> semantic first-stage retrieval vs lexical first-stage retrieval

with everything else held mostly constant.

## End-to-end architecture

```text
Raw 10-K PDFs
    ->
PyMuPDF text extraction
    ->
text cleaning
    ->
RecursiveCharacterTextSplitter chunking
    ->
chunks.json + manifest.json
    ->
    +--> ChromaDB vector index
    |
    +--> BM25 index

User question
    ->
query preprocessing (company detection, year detection, clean query)
    ->
retrieval
    ->
fetch extra candidates
    ->
cross-encoder reranking
    ->
formatted context
    ->
Groq-hosted Llama answer generation
    ->
evaluation + CSV results
```

## Models and main components used

| Component              | Model / Tool                             | Location                               | Purpose / Notes                                                   |
| ---------------------- | ---------------------------------------- | -------------------------------------- | ----------------------------------------------------------------- |
| PDF Parsing            | PyMuPDF (`fitz`)                         | `data_loader.py`                       | Extracts text from 10-K PDF documents page by page                |
| Chunking               | `RecursiveCharacterTextSplitter`         | `data_loader.py`                       | Splits large documents into overlapping text chunks for retrieval |
| Embedding Model        | `sentence-transformers/all-MiniLM-L6-v2` | `vector_rag/indexer.py`, `config.py`   | Generates dense vector embeddings for semantic search             |
| Vector Store           | ChromaDB                                 | `vector_rag/`                          | Stores embeddings and performs cosine similarity retrieval        |
| Vectorless Retriever   | `BM25Okapi` (`rank-bm25`)                | `vectorless_rag/`                      | Performs keyword-based lexical retrieval without embeddings       |
| Reranker               | `cross-encoder/ms-marco-MiniLM-L-6-v2`   | `reranker.py`                          | Re-ranks retrieved chunks using query-passage relevance scoring   |
| Answer Generation LLM  | Llama 3.3 70B Versatile                  | `llm.py`, `config.py`                  | Generates final answers grounded in retrieved context             |
| Evaluation / Judge LLM | Gemini 2.5 Flash                         | `evaluation/evaluator.py`, `config.py` | Independently evaluates and scores generated answers              |


## Methods and techniques used

### 1. Incremental preprocessing

The preprocessing pipeline is designed so you do not have to reprocess every PDF every time.

It uses:

- file hashing with `md5`
- a manifest file that records processed PDFs
- automatic detection of new or changed source documents
- selective reprocessing only for changed files

This is handled in `data_loader.py`.

### 2. Page-level extraction and cleaning

Each PDF is read page by page. Pages with almost no text are skipped. The text is then lightly cleaned by:

- collapsing repeated newlines
- collapsing repeated spaces
- removing standalone page numbers
- removing "Table of Contents" text

### 3. Overlapping chunking

Chunks are created with:

- `chunk_size = 1000`
- `chunk_overlap = 150`

The code comments explain that these values were increased to preserve more financial context and reduce sentence cut-off problems.

Each chunk keeps metadata such as:

- `source`
- `company`
- `page`
- `chunk_id`
- `chunk_num`

### 4. Company-aware query preprocessing

Before retrieval, the project tries to detect the company named in the question using the `KNOWN_COMPANIES` dictionary in `config.py`.

This is important because it helps prevent failure cases like:

- asking about NVIDIA
- retrieving Amazon or Microsoft chunks

The query processor also detects a year if one is present, although the current retrievers do not yet use the year as a retrieval filter.

### 5. Vector retrieval path

The vector pipeline:

1. embeds chunks with `all-MiniLM-L6-v2`
2. stores them in ChromaDB
3. applies a metadata filter by company when possible
4. retrieves `top_k * 2` candidates
5. reranks those candidates down to the final `top_k`

Similarity is derived from the Chroma distance output and converted to a simple `1 / (1 + distance)` style score for display.

### 6. Vectorless retrieval path

The vectorless pipeline:

1. tokenizes text with a simple lowercase whitespace tokenizer
2. builds a BM25 index over all chunks
3. narrows the chunk list by company before scoring
4. scores chunks with BM25
5. keeps `top_k * 2` candidates
6. reranks to the final `top_k`

Even though this is called "vectorless", the pipeline still uses a neural reranker after BM25 retrieval. So the first-stage retrieval is vectorless, but the full pipeline is not purely traditional.

### 7. Cross-encoder reranking

Both pipelines use the same reranker:

- `cross-encoder/ms-marco-MiniLM-L-6-v2`

This model reads the query and chunk together and produces a new relevance score. It is slower than first-stage retrieval but usually improves which chunk ranks first.

### 8. Grounded answer generation

The final answer is generated by a Groq-hosted Llama model using a strict prompt that tells the model to:

- answer only from the retrieved context
- avoid outside knowledge
- mention when information is missing
- keep the answer concise
- mention which company the answer refers to

### 9. Evaluation strategy

The evaluator measures:

- `judge_score`: intended 1 to 5 answer quality score from a judge model
- `pass`: whether the score is at least 3
- `company_accuracy`: how many retrieved chunks match the expected company
- `avg_chunk_score`: average retrieval or rerank score
- `retrieval_time`
- `generation_time`
- `total_time`

Results are saved to `evaluation/results/full_results.csv`.

## How to run the project

### 1. Create a virtual environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Create a `.env` file

Use placeholders like this:

```env
GROQ_API_KEY=your_groq_key_here
GEMINI_KEY=your_gemini_key_here
HF_TOKEN=your_huggingface_token_here
```

Notes:

- `GROQ_API_KEY` is required for answer generation
- `GEMINI_KEY` is intended for answer judging during evaluation
- `HF_TOKEN` is configured but not strongly required for the current public models

### 4. Run the notebooks in order

Recommended notebook order:

1. `notebooks/01_data_exploration.ipynb`
2. `notebooks/02_vector_rag.ipynb`
3. `notebooks/03_vectorless_rag.ipynb`
4. `notebooks/04_comparison.ipynb`

What each notebook does:

- `01_data_exploration.ipynb`: validates libraries, runs preprocessing, inspects chunks, plots distributions, builds both indexes
- `02_vector_rag.ipynb`: interactive questions against the vector pipeline
- `03_vectorless_rag.ipynb`: interactive questions against the BM25 pipeline
- `04_comparison.ipynb`: runs the two methods on the benchmark question set

### 5. Optional: run programmatically from Python

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

## Repository structure

```text
rag-benchmark/
|-- config.py
|-- data_loader.py
|-- llm.py
|-- reranker.py
|-- requirements.txt
|-- data/
|   |-- raw/
|   |-- processed/
|-- vector_rag/
|-- vectorless_rag/
|-- evaluation/
|-- notebooks/
|-- utils/
|-- .env
|-- .gitignore
```

## File-by-file guide

The lists below focus on project-owned files and meaningful generated artifacts. The local virtual environment is grouped rather than documented package by package because it contains third-party dependencies, not project logic.

### Root files

| Path | What it does |
| --- | --- |
| `.env` | Local environment file for API keys. Loaded by `config.py`. This should contain secrets only on your machine. |
| `.gitignore` | Currently empty. In a cleaner public version, it should ignore `.env`, `venv/`, `__pycache__/`, and generated indexes/results. |
| `config.py` | Central configuration file. Defines model IDs, API-key environment variable names, chunking settings, retrieval settings, data/result paths, and the known company name map. |
| `data_loader.py` | Main ingestion and preprocessing module. Hashes PDFs, tracks processing state in a manifest, extracts text with PyMuPDF, cleans it, chunks it, and writes `data/processed/chunks.json`. |
| `llm.py` | Wraps the Groq client, builds the answer-generation prompt, formats retrieved chunks into a labeled context block, and returns the final model answer. |
| `reranker.py` | Loads the cross-encoder reranker once and uses it to re-score retrieved chunks before generation. |
| `requirements.txt` | Dependency list for parsing, indexing, retrieval, notebooks, and evaluation support. |

### Data files

| Path | What it does |
| --- | --- |
| `data/raw/amazon_10k.pdf` | Source annual report PDF for Amazon. |
| `data/raw/microsoft_10k.pdf` | Source annual report PDF for Microsoft. |
| `data/raw/netflix_10k.pdf` | Source annual report PDF for Netflix. |
| `data/raw/nvidia_10k.pdf` | Source annual report PDF for NVIDIA. |
| `data/processed/manifest.json` | Processing manifest that records each PDF hash, processing time, chunk count, and page count. Used to skip unchanged files. |
| `data/processed/chunks.json` | Full processed corpus as JSON. Each entry stores a chunk plus metadata such as company, source file, page, and chunk number. |

### `vector_rag/` files

| Path | What it does |
| --- | --- |
| `vector_rag/indexer.py` | Builds and loads the ChromaDB collection, embeds chunks with `all-MiniLM-L6-v2`, and skips sources already indexed. |
| `vector_rag/retriever.py` | Runs vector retrieval: query preprocessing, optional company metadata filter, Chroma query, candidate scoring, reranking, and timing. |
| `vector_rag/pipeline.py` | End-to-end vector pipeline wrapper with `ask()` and `show()` methods. |
| `vector_rag/chroma_db/chroma.sqlite3` | Chroma metadata database persisted to disk. Stores collection-level information and index metadata. |
| `vector_rag/chroma_db/8870c2c3-309a-493d-b865-cba19951e2e5/header.bin` | Binary header for the persisted HNSW vector index created by ChromaDB. |
| `vector_rag/chroma_db/8870c2c3-309a-493d-b865-cba19951e2e5/data_level0.bin` | Main HNSW vector graph data for approximate nearest-neighbor search. |
| `vector_rag/chroma_db/8870c2c3-309a-493d-b865-cba19951e2e5/length.bin` | Binary file used by the HNSW index to track node/edge lengths. |
| `vector_rag/chroma_db/8870c2c3-309a-493d-b865-cba19951e2e5/link_lists.bin` | HNSW graph link structure used during vector retrieval. |
| `vector_rag/chroma_db/8870c2c3-309a-493d-b865-cba19951e2e5/index_metadata.pickle` | Pickled metadata describing the persisted HNSW index. |

### `vectorless_rag/` files

| Path | What it does |
| --- | --- |
| `vectorless_rag/indexer.py` | Builds or loads the BM25 index, fingerprints the chunk list, and skips rebuilds when chunks have not changed. |
| `vectorless_rag/retriever.py` | Runs BM25 retrieval, narrows the search set by detected company, rebuilds BM25 over that subset, and reranks the candidates. |
| `vectorless_rag/pipeline.py` | End-to-end BM25 pipeline wrapper with the same interface as the vector pipeline. |

### `utils/` files

| Path | What it does |
| --- | --- |
| `utils/__init__.py` | Empty package marker so `utils` can be imported as a module. |
| `utils/query_processor.py` | Detects company names, detects years, strips those terms out of the query when needed, and returns structured query metadata. |

### `evaluation/` files

| Path | What it does |
| --- | --- |
| `evaluation/evaluator.py` | Runs the benchmark over the question set, computes retrieval metrics, calls the judge model, writes the CSV, and prints summary statistics. |
| `evaluation/test_questions.json` | Benchmark dataset with 20 questions, company labels, categories, and difficulty tags. |
| `evaluation/results/full_results.csv` | Saved benchmark results for both methods. The current checked-in file has 40 rows but the judge-scoring stage failed, so `judge_score` values are not trustworthy yet. |
| `evaluation/results/chunk_distribution.png` | Plot exported from notebook exploration showing chunk counts and chunk-length distribution. |

### `notebooks/` files

| Path | What it does |
| --- | --- |
| `notebooks/01_data_exploration.ipynb` | Setup and exploration notebook. Checks imports, runs preprocessing, inspects a sample chunk, summarizes chunk counts, draws plots, and rebuilds both indexes. |
| `notebooks/02_vector_rag.ipynb` | Demo notebook for the vector pipeline using direct question examples. |
| `notebooks/03_vectorless_rag.ipynb` | Demo notebook for the BM25 pipeline using the same style of questions for comparison. |
| `notebooks/04_comparison.ipynb` | Comparison notebook that instantiates both pipelines and runs the evaluation routine. |
| `notebooks/Test_notebook.ipynb` | Empty scratch notebook. Currently contains no workflow logic. |
| `notebooks/vectorless_rag/bm25_index.pkl` | Serialized BM25 payload saved from notebook execution. It contains the BM25 object plus the chunk list. |
| `notebooks/vectorless_rag/bm25_manifest.json` | Notebook-local BM25 manifest storing the chunk fingerprint used to decide whether the BM25 index must be rebuilt. |

### Generated Python cache files

These are interpreter-generated bytecode caches. They mirror the corresponding source modules and do not contain unique project logic.

| Path | What it does |
| --- | --- |
| `__pycache__/config.cpython-313.pyc` | Compiled bytecode cache for `config.py`. |
| `__pycache__/data_loader.cpython-313.pyc` | Compiled bytecode cache for `data_loader.py`. |
| `__pycache__/llm.cpython-313.pyc` | Compiled bytecode cache for `llm.py`. |
| `__pycache__/reranker.cpython-313.pyc` | Compiled bytecode cache for `reranker.py`. |
| `evaluation/__pycache__/evaluator.cpython-313.pyc` | Compiled bytecode cache for `evaluation/evaluator.py`. |
| `utils/__pycache__/__init__.cpython-313.pyc` | Compiled bytecode cache for `utils/__init__.py`. |
| `utils/__pycache__/query_processor.cpython-313.pyc` | Compiled bytecode cache for `utils/query_processor.py`. |
| `vector_rag/__pycache__/indexer.cpython-313.pyc` | Compiled bytecode cache for `vector_rag/indexer.py`. |
| `vector_rag/__pycache__/pipeline.cpython-313.pyc` | Compiled bytecode cache for `vector_rag/pipeline.py`. |
| `vector_rag/__pycache__/retriever.cpython-313.pyc` | Compiled bytecode cache for `vector_rag/retriever.py`. |
| `vectorless_rag/__pycache__/indexer.cpython-313.pyc` | Compiled bytecode cache for `vectorless_rag/indexer.py`. |
| `vectorless_rag/__pycache__/pipeline.cpython-313.pyc` | Compiled bytecode cache for `vectorless_rag/pipeline.py`. |
| `vectorless_rag/__pycache__/retriever.cpython-313.pyc` | Compiled bytecode cache for `vectorless_rag/retriever.py`. |

### Local environment directory

| Path | What it does |
| --- | --- |
| `venv/` | Local Python virtual environment created on this machine. It contains installed third-party packages and is not part of the project logic itself. |

## Important implementation details

### Shared settings

From `config.py`:

- embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- answer model: `llama-3.3-70b-versatile`
- judge model: `gemini-2.5-flash`
- `TOP_K = 8`
- chunk size: `1000`
- chunk overlap: `150`
- generation max tokens: `512`
- generation temperature: `0.1`

### Why the company filter matters

The biggest retrieval-quality safeguard in this repo is company detection.

Without it, financial questions like:

- "What was NVIDIA's revenue?"
- "What are Microsoft's risk factors?"

can accidentally retrieve chunks from another company that uses similar finance language. Both retrievers try to fix that early.

### Why reranking is used in both pipelines

BM25 and embedding retrieval are both imperfect as first-stage retrievers. The reranker is used to make the final context more comparable and more relevant, especially when multiple chunks mention similar financial terms.

## Current outputs in the repository

The repo already contains several generated outputs:

- processed chunks in `data/processed/chunks.json`
- a processing manifest in `data/processed/manifest.json`
- a persisted ChromaDB index in `vector_rag/chroma_db/`
- a notebook-generated BM25 index in `notebooks/vectorless_rag/`
- a results CSV in `evaluation/results/full_results.csv`
- an exploratory visualization in `evaluation/results/chunk_distribution.png`

## Known caveats and rough edges

This README describes the repo as it exists today, so these limitations are worth knowing:

1. `evaluation/evaluator.py` is configured to use `google.genai.Client`, but the saved results show the scoring call failed with `Client object has no attribute chat`. That means the checked-in `judge_score`, `pass`, and judge summary fields are currently invalid.
2. `requirements.txt` does not clearly include the Google GenAI SDK used by `evaluation/evaluator.py`, so evaluation may need an additional dependency before it runs cleanly.
3. The BM25 index path in `vectorless_rag/indexer.py` is relative, so the saved index location depends on the current working directory. That is why the checked-in BM25 artifact currently lives under `notebooks/vectorless_rag/`.
4. The repo includes a local `.env` workflow and an empty `.gitignore`; in a public/shared version, secrets and generated artifacts should be ignored more carefully.
5. There is no dedicated CLI entrypoint or automated test suite yet. The notebooks are the main user interface.

## In one sentence

This project is a compact benchmark for comparing semantic vector retrieval against BM25-style vectorless retrieval on real financial 10-K documents, using shared chunking, reranking, and LLM answer generation to keep the comparison as fair and understandable as possible.
