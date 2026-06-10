# vectorless_rag/pipeline.py

import sys
import time
import json
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from vectorless_rag.indexer import load_bm25_index
from vector_rag.indexer import build_parent_lookup
from vectorless_rag.retriever import retrieve
from llm import load_llm, generate_answer, format_context


class VectorlessRAGPipeline:

    def __init__(self):
        print("🔧 Initialising Vectorless RAG Pipeline...")

        self.bm25, self.children = load_bm25_index()
        self.llm                 = load_llm()

        # Resolve chunks.json from __file__ — always correct, CWD-independent (fix #1)
        chunks_path = (
            Path(__file__).resolve().parent.parent
            / "data" / "processed" / "chunks.json"
        )
        with open(chunks_path, encoding="utf-8") as f:
            data = json.load(f)
        self.parent_lookup = build_parent_lookup(data)

        print(
            f"✅ Vectorless RAG ready — "
            f"{len(self.children)} children, "
            f"{len(self.parent_lookup)} parents\n"
        )

    # ─────────────────────────────────────────────────────────────────────────

    def ask(self, question: str, top_k: int = None) -> dict:
        from config import TOP_K
        top_k = top_k or TOP_K

        ret     = retrieve(question, self.bm25, self.children, self.parent_lookup, top_k)
        retrieval_time = round(ret.get("retrieval_latency", ret.get("latency", 0.0)), 4)
        rerank_time = round(ret.get("rerank_latency", 0.0), 4)
        context = format_context(ret["chunks"])

        # Guard: don't call the LLM with empty context (fix #3)
        if not context:
            return {
                "question"       : question,
                "answer"         : "This information was not found in the retrieved sections.",
                "retrieved"      : ret["chunks"],
                "retrieval_time" : retrieval_time,
                "rerank_time"    : rerank_time,
                "generation_time": 0.0,
                "total_time"     : round(retrieval_time + rerank_time, 4),
                "method"         : "vectorless",   # consistent label (fix #2)
                "retrieval_latency": retrieval_time,
                "rerank_latency"   : rerank_time,
            }

        gen_start = time.perf_counter()
        answer    = generate_answer(self.llm, context, question)
        gen_time  = round(time.perf_counter() - gen_start, 4)

        return {
            "question"       : question,
            "answer"         : answer,
            "retrieved"      : ret["chunks"],
            "retrieval_time" : retrieval_time,
            "rerank_time"    : rerank_time,
            "generation_time": gen_time,
            "total_time"     : round(retrieval_time + rerank_time + gen_time, 4),
            "method"         : "vectorless",   # consistent label (fix #2)
            "retrieval_latency": retrieval_time,
            "rerank_latency"   : rerank_time,
        }

    # ─────────────────────────────────────────────────────────────────────────

    def show(self, result: dict):
        print(f"\n{'='*55}")
        print(f"  METHOD  : Vectorless RAG — BM25 + Financial Tokenizer")
        print(f"{'='*55}")
        print(f"  Q: {result['question']}")
        print(f"{'─'*55}")
        print(f"  A: {result['answer']}")
        print(f"{'─'*55}")
        print(f"  Retrieved chunks:")
        for c in result.get("retrieved", []):
            m = c.get("metadata", {})
            print(
                f"   • {m.get('company', 'Unknown')} | Page {m.get('page', '?')} "
                f"| Rerank {c.get('rerank_score', c.get('score', 0)):.4f}"
            )
        print(f"{'─'*55}")
        print(
            f"  Retrieval: {result['retrieval_time']:.4f}s | "
            f"Rerank: {result.get('rerank_time', 0):.4f}s | "
            f"Generation: {result['generation_time']:.4f}s | "
            f"Total: {result['total_time']:.4f}s"
        )
        print(f"{'='*55}\n")
