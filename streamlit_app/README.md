# Financial RAG Benchmark — Streamlit App

A multi-page Streamlit UI wrapping the existing Vector / Vectorless / Hybrid RAG
pipelines for interactive exploration, chat, and benchmarking.

## Quick start

```bash
# 1. Install dependencies (from the repo root)
pip install -r requirements.txt
pip install -r streamlit_app/requirements_streamlit.txt

# 2. Make sure your .env is in the repo root (MISTRAL_API_KEY etc.)

# 3. Launch
cd streamlit_app
streamlit run app.py
```

## Pages

| Page | What it does |
|---|---|
| 📊 Dashboard | Corpus stats, pipeline health, latest eval snapshot |
| 📂 Upload Documents | Upload PDFs → trigger preprocessing |
| 🗂️ Index Manager | Build/rebuild ChromaDB + BM25 indices, browse chunks |
| 💬 Chat | Interactive RAG chat with source attribution and latency |
| 🗃️ Conversations | Browse and replay saved conversations |
| 🔬 Evaluation | Run three-way benchmark; charts by method/company/category |

## Project layout

```
streamlit_app/
├── app.py                      # Entry point
├── requirements_streamlit.txt
├── .streamlit/
│   └── config.toml             # Theme + server settings
├── components/
│   └── sidebar.py              # Navigation sidebar
├── pages/
│   ├── router.py               # Page dispatcher
│   ├── dashboard.py
│   ├── upload_documents.py
│   ├── index_manager.py
│   ├── chat.py
│   ├── conversations.py
│   └── evaluation.py
└── services/
    ├── pipeline_manager.py     # Lazy singleton loader for all 3 RAG pipelines
    ├── conversation_store.py   # In-memory conversation history (session state)
    └── index_service.py        # Wrappers for data_loader + indexer calls
```

## Design principles

- **Zero changes to existing code** — all RAG modules (`vector_rag/`, `vectorless_rag/`,
  `hybrid_rag/`, `evaluation/`) are imported unchanged.
- **Lazy pipeline loading** — pipelines are loaded on first use, not at startup,
  so the UI is responsive even before heavy models finish downloading.
- **Singleton via session state** — `PipelineManager` lives in `st.session_state`
  so pipelines survive page navigations without being reloaded.
- **Modular pages** — each page is a standalone module with a single `render()`
  function; `pages/router.py` dispatches based on `st.session_state.current_page`.
