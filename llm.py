# llm.py
from groq import Groq
import config

# ── Singleton — client created once, reused across both pipelines ──
_client = None


def load_llm(verbose: bool = True):
    """
    Creates a Groq client.
    Lightweight — no model download, no RAM usage.
    All inference happens on Groq's servers.
    """
    global _client

    if _client is not None:
        if verbose:
            print("✅ Groq client already initialised — reusing")
        return _client

    if not config.GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY not found. "
            "Add it to your .env file: GROQ_API_KEY=gsk_..."
        )

    _client = Groq(api_key=config.GROQ_API_KEY)

    if verbose:
        print(f"✅ Groq client ready — model: {config.LLM_MODEL_ID}")

    return _client


def generate_answer(client, context: str, question: str) -> str:
    """
    Sends context + question to Llama 3.3 70B on Groq.
    Returns the answer string.
    """
    system_prompt = """You are a precise financial analyst assistant.
Your job is to answer questions about company 10-K financial reports accurately.

Rules:
1. Answer ONLY using the provided context. Do not use outside knowledge.
2. If the exact figure is in the context, state it directly.
3. If the context contains partial information, state what you found and what is missing.
4. If the answer is not in the context, say: "This information was not found in the retrieved sections."
5. Always mention which company your answer refers to.
6. Be concise — 2 to 4 sentences maximum."""

    user_message = f"""Context from financial reports:
{context}

Question: {question}

Answer based strictly on the context above:"""

    response = client.chat.completions.create(
        model       = config.LLM_MODEL_ID,
        messages    = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message}
        ],
        temperature = config.TEMPERATURE,
        max_tokens  = config.MAX_NEW_TOKENS,
    )

    return response.choices[0].message.content.strip()


def format_context(retrieved_chunks: list[dict]) -> str:
    """
    Formats retrieved chunks into a single labelled context string.
    """
    context_parts = []

    for i, chunk in enumerate(retrieved_chunks, 1):
        meta    = chunk.get("metadata", {})
        company = meta.get("company", "Unknown")
        page    = meta.get("page", "?")
        text    = chunk.get("text", "")

        context_parts.append(
            f"[Source {i} — {company}, Page {page}]\n{text}"
        )

    return "\n\n".join(context_parts)