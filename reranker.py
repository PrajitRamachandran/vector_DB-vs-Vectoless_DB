import time

start = time.time()
print("Loading ReRanker...")

from sentence_transformers import CrossEncoder
import config

_reranker = None

RERANKER_MODEL = "BAAI/bge-reranker-large"



def load_reranker():
    global _reranker
    if _reranker is None:
        print(f"🔁 Loading reranker: {RERANKER_MODEL}")
        _reranker = CrossEncoder(RERANKER_MODEL)
        print("✅ Reranker ready")
    return _reranker


def rerank(query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    """
    Takes the initially retrieved chunks and re-scores them
    using a cross-encoder that reads query + chunk together.

    Steps:
    1. Pair the query with each chunk text
    2. Cross-encoder scores each pair (higher = more relevant)
    3. Return top_k chunks sorted by new score
    """
    reranker = load_reranker()

    pairs  = [(query, c["text"]) for c in chunks]
    scores = reranker.predict(pairs)

    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = round(float(score), 4)

    reranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_k]


print(
    f"ReRanker loaded in "
    f"{time.time()-start:.2f}s"
)