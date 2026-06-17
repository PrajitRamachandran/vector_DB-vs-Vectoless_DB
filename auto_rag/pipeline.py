from auto_rag.classifier import (
    classify_question
)

from auto_rag.rules import (
    ROUTING_RULES
)

from vector_rag.pipeline import (
    VectorRAGPipeline
)

from vectorless_rag.pipeline import (
    VectorlessRAGPipeline
)

from hybrid_rag.pipeline import (
    HybridRAGPipeline
)


class AutoRAGPipeline:

    def __init__(self):

        self.vector = VectorRAGPipeline()

        self.vectorless = (
            VectorlessRAGPipeline()
        )

        self.hybrid = (
            HybridRAGPipeline()
        )

    def ask(
        self,
        question: str
    ):

        query_type = classify_question(
            question
        )

        selected_method = ROUTING_RULES.get(
            query_type,
            "vector"
        )

        pipeline_map = {
            "vector": self.vector,
            "vectorless": self.vectorless,
            "hybrid": self.hybrid
        }

        result = pipeline_map[
            selected_method
        ].ask(question)

        result["method"] = "auto"

        result[
            "auto_selected_method"
        ] = selected_method

        result[
            "query_type"
        ] = query_type

        return result