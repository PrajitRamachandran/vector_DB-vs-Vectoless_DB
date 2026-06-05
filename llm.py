from __future__ import annotations

from openai import OpenAI

import config

_client: OpenAI | None = None


def _extract_text_content(content) -> str:
    """
    Normalises OpenAI-style message content into a plain string.
    """
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif hasattr(item, "text"):
                parts.append(getattr(item, "text", ""))
        return "".join(parts).strip()

    return ""


def load_llm(verbose: bool = True) -> OpenAI:
    """
    Creates the Mistral client.
    """
    global _client

    if _client is not None:
        if verbose:
            print("Mistral client already initialised - reusing")
        return _client

    if not config.MISTRAL_API_KEY:
        raise ValueError(
            "MISTRAL_API_KEY not found. "
            "Add it to your .env file: MISTRAL_API_KEY=..."
        )

    if not config.MISTRAL_BASE_URL:
        raise ValueError(
            "MISTRAL_BASE_URL not found. "
            "Add it to your config or .env file."
        )

    _client = OpenAI(
        api_key=config.MISTRAL_API_KEY,
        base_url=config.MISTRAL_BASE_URL,
    )

    if verbose:
        print(f"Mistral client ready - model: {config.LLM_MODEL_ID}")

    return _client


def generate_answer(client, context: str, question: str) -> str:
    """
    Sends context and question to Mistral.
    Returns the answer string.
    """
    if client is None:
        client = load_llm(verbose=False)

    context = (context or "").strip()
    question = (question or "").strip()

    if not question:
        raise ValueError("Question is empty.")

    if not context:
        return "This information was not found in the retrieved sections."

    system_prompt = """You are a precise financial analyst assistant.
Your job is to answer questions about company 10-K financial reports accurately.

Rules:
1. Answer ONLY using the provided context. Do not use outside knowledge.
2. If the exact figure is in the context, state it directly.
3. If the context contains partial information, state what you found and what is missing.
4. If the answer is not in the context, say: "This information was not found in the retrieved sections."
5. Always mention which company your answer refers to.
6. Be concise - 2 to 4 sentences maximum."""

    user_message = f"""Context from financial reports:
{context}

Question: {question}

Answer based strictly on the context above:"""

    try:
        response = client.chat.completions.create(
            model=config.LLM_MODEL_ID,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=config.TEMPERATURE,
            max_tokens=config.MAX_NEW_TOKENS,
        )
    except Exception as e:
        raise RuntimeError(f"LLM generation failed: {e}") from e

    if not getattr(response, "choices", None):
        raise ValueError("Mistral returned no choices.")

    content = response.choices[0].message.content
    answer = _extract_text_content(content)

    if answer:
        return answer

    return "This information was not found in the retrieved sections."


def format_context(retrieved_chunks: list[dict]) -> str:
    """
    Formats retrieved chunks into a single labelled context string.
    """
    if not retrieved_chunks:
        return ""

    context_parts = []

    for i, chunk in enumerate(retrieved_chunks, 1):
        if not isinstance(chunk, dict):
            continue

        meta = chunk.get("metadata", {})
        if not isinstance(meta, dict):
            meta = {}

        company = str(meta.get("company", "Unknown")).strip() or "Unknown"
        page = meta.get("page", "?")
        text = str(chunk.get("text", "")).strip()

        if not text:
            continue

        context_parts.append(f"[Source {i} - {company}, Page {page}]\n{text}")

    return "\n\n".join(context_parts)