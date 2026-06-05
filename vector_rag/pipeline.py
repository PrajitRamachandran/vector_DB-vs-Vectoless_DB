# vector_rag/pipeline.py
import sys, time
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from vector_rag.indexer   import load_index, build_parent_lookup
from vector_rag.retriever import retrieve
from llm import load_llm, generate_answer, format_context


class VectorRAGPipeline:
    def __init__(self):
        print("🔧 Initialising Vector RAG Pipeline...")
        self.collection    = load_index()
        self.llm           = load_llm()
        # Parent lookup built once at init — O(1) lookups during retrieval
        self._data         = self._load_data()
        self.parent_lookup = build_parent_lookup(self._data)
        print(f"✅ Vector RAG ready — "
              f"{len(self.parent_lookup)} parents in lookup\n")

    def _load_data(self) -> dict:
        import json
        from pathlib import Path
        path = Path("../data/processed/chunks.json")
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def ask(self, question: str, top_k: int = None) -> dict:
        from config import TOP_K
        top_k = top_k or TOP_K

        ret          = retrieve(question, self.collection,
                                self.parent_lookup, top_k)
        context      = format_context(ret["chunks"])

        gen_start    = time.perf_counter()
        answer       = generate_answer(self.llm, context, question)
        gen_time     = round(time.perf_counter() - gen_start, 4)

        return {
            "question"       : question,
            "answer"         : answer,
            "retrieved"      : ret["chunks"],
            "retrieval_time" : ret["latency"],
            "generation_time": gen_time,
            "total_time"     : round(ret["latency"] + gen_time, 4),
            "method"         : "vector"
        }

    def show(self, result: dict):
        print(f"\n{'='*55}")
        print(f"  METHOD  : Vector RAG — BGE + HNSW + Parent-Child")
        print(f"{'='*55}")
        print(f"  Q: {result['question']}")
        print(f"{'─'*55}")
        print(f"  A: {result['answer']}")
        print(f"{'─'*55}")
        for c in result["retrieved"]:
            m = c["metadata"]
            print(f"   • {m['company']} | Page {m['page']} "
                  f"| Score {c.get('rerank_score', c.get('score', 0)):.4f}")
        print(f"{'─'*55}")
        print(f"  Retrieval: {result['retrieval_time']}s | "
              f"Generation: {result['generation_time']}s | "
              f"Total: {result['total_time']}s")
        print(f"{'='*55}\n")