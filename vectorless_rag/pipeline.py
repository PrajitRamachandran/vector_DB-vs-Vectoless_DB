# vectorless_rag/pipeline.py
#
# Bug fixes applied:
#
# 1. RELATIVE PATH CRASH
#    Old: Path("../data/processed/chunks.json")
#    This path is resolved relative to the PROCESS working directory (CWD),
#    which is the notebook directory when run from Jupyter. That means the
#    path resolves to notebooks/../data/..., which works only sometimes and
#    breaks completely when the pipeline is imported from anywhere else.
#    Fix: resolve from __file__ so the path is always absolute and correct
#    regardless of where the code is called from.
#
# 2. INCONSISTENT METHOD LABEL
#    Old: "method": "bm25"
#    The evaluator registers this pipeline under the label "vectorless".
#    Returning "bm25" from ask() creates a mismatch in the results CSV and
#    in any downstream code that branches on method name.
#    Fix: return "method": "vectorless" consistently.
#
# 3. MISSING EMPTY-CONTEXT GUARD
#    If retrieval returns nothing, the old code still called generate_answer()
#    with an empty context, which either errors or returns a hallucinated answer.
#    Fix: return the standard "not found" message early when context is empty.

import sys
import time
import json
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from vectorless_rag.indexer   import load_bm25_index
from vector_rag.indexer       import build_parent_lookup
from vectorless_rag.retriever import retrieve
from llm                      import load_llm, generate_answer, format_context


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
        context = format_context(ret["chunks"])

        # Guard: don't call the LLM with empty context (fix #3)
        if not context:
            return {
                "question"       : question,
                "answer"         : "This information was not found in the retrieved sections.",
                "retrieved"      : ret["chunks"],
                "retrieval_time" : ret["latency"],
                "generation_time": 0.0,
                "total_time"     : ret["latency"],
                "method"         : "vectorless",   # consistent label (fix #2)
            }

        gen_start = time.perf_counter()
        answer    = generate_answer(self.llm, context, question)
        gen_time  = round(time.perf_counter() - gen_start, 4)

        return {
            "question"       : question,
            "answer"         : answer,
            "retrieved"      : ret["chunks"],
            "retrieval_time" : ret["latency"],
            "generation_time": gen_time,
            "total_time"     : round(ret["latency"] + gen_time, 4),
            "method"         : "vectorless",   # consistent label (fix #2)
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
            f"  Retrieval: {result['retrieval_time']}s | "
            f"Generation: {result['generation_time']}s | "
            f"Total: {result['total_time']}s"
        )
        print(f"{'='*55}\n")