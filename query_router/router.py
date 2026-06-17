from query_router.classifier import (
    classify
)

from query_router.intents import (
    DOCUMENT_QUESTION
)

def route(question):

    intent = classify(question)

    return {
        "intent": intent,
        "use_rag": (
            intent == DOCUMENT_QUESTION
        )
    }