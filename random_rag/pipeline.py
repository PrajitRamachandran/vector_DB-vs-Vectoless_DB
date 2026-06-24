import time

start = time.time()
print("Loading Random_pipeline...")

import random

from vector_rag.pipeline import VectorRAGPipeline
from vectorless_rag.pipeline import VectorlessRAGPipeline
from hybrid_rag.pipeline import HybridRAGPipeline


class RandomRAGPipeline:

    def __init__(self):

        self.vector = VectorRAGPipeline()
        self.vectorless = VectorlessRAGPipeline()
        self.hybrid = HybridRAGPipeline()

    def ask(self, question: str):

        selected = random.choice([
            "vector",
            "vectorless",
            "hybrid"
        ])

        pipeline_map = {
            "vector": self.vector,
            "vectorless": self.vectorless,
            "hybrid": self.hybrid
        }

        result = pipeline_map[selected].ask(
            question
        )

        result["random_selected_method"] = selected
        result["method"] = "random"

        return result
    

print(
    f"Random_pipeline loaded in "
    f"{time.time()-start:.2f}s"
)