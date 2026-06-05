import sys
import time
import json
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from vector_rag.indexer import load_index, build_parent_lookup
from vectorless_rag.indexer import load_bm25_index
from hybrid_rag.retriever import retrieve
from llm import load_llm, generate_answer, format_context


class HybridRAGPipeline:
    def __init__(self):
        print("🔧 Initialising Hybrid RAG Pipeline...")

        self.collection = load_index()
        self.bm25, self.bm25_children = load_bm25_index()
        self.llm = load_llm()

        self._data = self._load_chunks()
        self.parent_lookup = build_parent_lookup(self._data)

        print(
            f"✅ Hybrid RAG ready — "
            f"{self.collection.count()} vectors | "
            f"{len(self.bm25_children)} BM25 children | "
            f"{len(self.parent_lookup)} parents\n"
        )

    def _load_chunks(self) -> dict:
        path = Path(__file__).resolve().parent.parent / "data" / "processed" / "chunks.json"
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def ask(self, question: str, top_k: int = None) -> dict:
        from config import TOP_K

        top_k = top_k or TOP_K

        ret = retrieve(
            query=question,
            collection=self.collection,
            bm25_model=self.bm25,
            all_children=self.bm25_children,
            parent_lookup=self.parent_lookup,
            top_k=top_k,
        )

        context = format_context(ret["chunks"])

        if not context:
            return {
                "question": question,
                "answer": "This information was not found in the retrieved sections.",
                "retrieved": ret["chunks"],
                "retrieval_time": ret["latency"],
                "generation_time": 0.0,
                "total_time": ret["latency"],
                "method": "hybrid",
                "vector_latency": ret.get("vector_latency", 0.0),
                "bm25_latency": ret.get("bm25_latency", 0.0),
                "rerank_latency": ret.get("rerank_latency", 0.0),
                "vector_candidates": ret.get("vector_candidates", 0),
                "bm25_candidates": ret.get("bm25_candidates", 0),
                "fused_candidates": ret.get("fused_candidates", 0),
            }

        gen_start = time.perf_counter()
        answer = generate_answer(self.llm, context, question)
        gen_time = round(time.perf_counter() - gen_start, 4)

        return {
            "question": question,
            "answer": answer,
            "retrieved": ret["chunks"],
            "retrieval_time": ret["latency"],
            "generation_time": gen_time,
            "total_time": round(ret["latency"] + gen_time, 4),
            "method": "hybrid",
            "vector_latency": ret["vector_latency"],
            "bm25_latency": ret["bm25_latency"],
            "rerank_latency": ret["rerank_latency"],
            "vector_candidates": ret["vector_candidates"],
            "bm25_candidates": ret["bm25_candidates"],
            "fused_candidates": ret["fused_candidates"],
        }

    def show(self, result: dict):
        print(f"\n{'='*55}")
        print(f"  METHOD  : Hybrid RAG — RRF(Vector + BM25) + Rerank")
        print(f"{'='*55}")
        print(f"  Q: {result['question']}")
        print(f"{'─'*55}")
        print(f"  A: {result['answer']}")
        print(f"{'─'*55}")
        print(f"  Retrieved chunks:")
        for c in result.get("retrieved", []):
            m = c.get("metadata", {})
            sources = "+".join(c.get("rrf_sources", ["?"]))
            print(
                f"   • {m.get('company', 'Unknown')} | Page {m.get('page', '?')} "
                f"| Rerank {c.get('rerank_score', 0):.4f} "
                f"| RRF {c.get('rrf_score', 0):.4f} "
                f"| via [{sources}]"
            )
        print(f"{'─'*55}")
        print(
            f"  Vector: {result.get('vector_latency', 0):.4f}s  "
            f"BM25: {result.get('bm25_latency', 0):.4f}s  "
            f"Rerank: {result.get('rerank_latency', 0):.4f}s"
        )
        print(
            f"  Candidates → vector: {result.get('vector_candidates', '?')} | "
            f"bm25: {result.get('bm25_candidates', '?')} | "
            f"fused: {result.get('fused_candidates', '?')} | "
            f"final: {len(result.get('retrieved', []))}"
        )
        print(f"{'─'*55}")
        print(
            f"  Retrieval: {result.get('retrieval_time', 0)}s | "
            f"Generation: {result.get('generation_time', 0)}s | "
            f"Total: {result.get('total_time', 0)}s"
        )
        print(f"{'='*55}\n")