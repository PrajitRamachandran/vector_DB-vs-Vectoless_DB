# vector_rag/pipeline.py
import sys
import time
from vector_rag.indexer import load_index
from vector_rag.retriever import retrieve
from llm import load_llm, generate_answer, format_context
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))


class VectorRAGPipeline:
    """
    End-to-end Vector RAG pipeline.
    Initialise once, call .ask() as many times as you want.
    """

    def __init__(self):
        print("🔧 Initialising Vector RAG Pipeline...")
        self.collection = load_index()
        self.llm        = load_llm()
        print("✅ Vector RAG Pipeline ready\n")

    def ask(self, question: str, top_k: int = None) -> dict:
        """
        Full pipeline: question → retrieve → generate → answer

        Returns a dict with everything needed for evaluation:
        {
            question      : str,
            answer        : str,
            retrieved     : list of chunks with scores,
            retrieval_time: float (seconds),
            generation_time: float (seconds),
            total_time    : float (seconds),
            method        : "vector"
        }
        """
        from config import TOP_K
        top_k = top_k or TOP_K

        # Step 1: Retrieve
        retrieval_result = retrieve(question, self.collection, top_k)
        chunks           = retrieval_result["chunks"]
        retrieval_time   = retrieval_result["latency"]

        # Step 2: Format context
        context = format_context(chunks)

        # Step 3: Generate answer
        gen_start = time.perf_counter()
        answer    = generate_answer(self.llm, context, question)
        gen_time  = round(time.perf_counter() - gen_start, 4)

        return {
            "question"       : question,
            "answer"         : answer,
            "retrieved"      : chunks,
            "retrieval_time" : retrieval_time,
            "generation_time": gen_time,
            "total_time"     : round(retrieval_time + gen_time, 4),
            "method"         : "vector"
        }

    def show(self, result: dict):
        """Pretty-prints a pipeline result."""
        print(f"\n{'='*55}")
        print(f"  METHOD  : Vector RAG (ChromaDB + Sentence Transformers)")
        print(f"{'='*55}")
        print(f"  QUESTION: {result['question']}")
        print(f"{'─'*55}")
        print(f"  ANSWER  :\n  {result['answer']}")
        print(f"{'─'*55}")
        print(f"  SOURCES :")
        for c in result["retrieved"]:
            m = c["metadata"]
            print(f"    • {m['company']} | Page {m['page']} | Score: {c['score']}")
        print(f"{'─'*55}")
        print(f"  Retrieval : {result['retrieval_time']}s")
        print(f"  Generation: {result['generation_time']}s")
        print(f"  Total     : {result['total_time']}s")
        print(f"{'='*55}\n")