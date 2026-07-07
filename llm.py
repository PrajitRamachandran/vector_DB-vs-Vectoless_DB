# #llm.py

# from __future__ import annotations

# from openai import OpenAI

# import config

# _client: OpenAI | None = None


# def _extract_text_content(content) -> str:
#     """
#     Normalises OpenAI-style message content into a plain string.
#     """
#     if isinstance(content, str):
#         return content.strip()

#     if isinstance(content, list):
#         parts = []
#         for item in content:
#             if isinstance(item, dict) and item.get("type") == "text":
#                 parts.append(item.get("text", ""))
#             elif hasattr(item, "text"):
#                 parts.append(getattr(item, "text", ""))
#         return "".join(parts).strip()

#     return ""


# def load_llm(verbose: bool = True) -> OpenAI:
#     """
#     Creates the Mistral client.
#     """
#     global _client

#     if _client is not None:
#         if verbose:
#             print("Mistral client already initialised - reusing")
#         return _client

#     if not config.MISTRAL_API_KEY:
#         raise ValueError(
#             "MISTRAL_API_KEY not found. "
#             "Add it to your .env file: MISTRAL_API_KEY=..."
#         )

#     if not config.MISTRAL_BASE_URL:
#         raise ValueError(
#             "MISTRAL_BASE_URL not found. "
#             "Add it to your config or .env file."
#         )

#     _client = OpenAI(
#         api_key=config.MISTRAL_API_KEY,
#         base_url=config.MISTRAL_BASE_URL,
#     )

#     if verbose:
#         print(f"Mistral client ready - model: {config.LLM_MODEL_ID}")

#     return _client


# def generate_answer(client, context: str, question: str) -> str:
#     """
#     Sends context and question to Mistral.
#     Returns the answer string.
#     """
#     if client is None:
#         client = load_llm(verbose=False)

#     context = (context or "").strip()
#     question = (question or "").strip()

#     if not question:
#         raise ValueError("Question is empty.")

#     if not context:
#         return "This information was not found in the retrieved sections."

#     system_prompt = """
# You are a financial analyst assistant.

# Answer ONLY using the provided context.

# For company overview or summary requests:

# Generate a structured summary with:

# 1. Company Overview
# 2. Business Segments
# 3. Products and Services
# 4. Revenue Drivers
# 5. Strategic Priorities
# 6. Key Risks

# If information is missing from the context, omit that section.

# Never invent facts.

# For factual questions:
# Answer directly and concisely using only the context.

# If the answer is not present, say:

# "This information was not found in the retrieved sections."
# """

#     user_message = f"""Context from financial reports. The most relevant excerpt appears first:
# {context}

# Question: {question}

# Answer based strictly on the context above:"""

#     try:
#         response = client.chat.completions.create(
#             model=config.LLM_MODEL_ID,
#             messages=[
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": user_message},
#             ],
#             temperature=config.TEMPERATURE,
#             max_tokens=config.MAX_NEW_TOKENS,
#         )
#     except Exception as e:
#         raise RuntimeError(f"LLM generation failed: {e}") from e

#     if not getattr(response, "choices", None):
#         raise ValueError("Mistral returned no choices.")

#     content = response.choices[0].message.content
#     answer = _extract_text_content(content)

#     if answer:
#         return answer

#     return "This information was not found in the retrieved sections."


# def format_context(retrieved_chunks: list[dict]) -> str:
#     """
#     Formats retrieved chunks into a single labelled context string.
#     """
#     if not retrieved_chunks:
#         return ""

#     def _trim_text(text: str, limit: int) -> str:
#         cleaned = " ".join(str(text or "").split())
#         if len(cleaned) <= limit:
#             return cleaned
#         return cleaned[:limit].rstrip() + "..."

#     context_parts = []

#     for i, chunk in enumerate(retrieved_chunks, 1):
#         if not isinstance(chunk, dict):
#             continue

#         meta = chunk.get("metadata", {})
#         if not isinstance(meta, dict):
#             meta = {}

#         company = str(meta.get("company", "Unknown")).strip() or "Unknown"
#         page = meta.get("page", "?")
#         parent_text = str(chunk.get("text", "")).strip()
#         child_text = str(chunk.get("child_text", "")).strip()

#         if not parent_text and not child_text:
#             continue

#         primary_text = child_text or parent_text

#         context_parts.append(
#             f"[Source {i} - {company}, Page {page}]\n"
#             f"Relevant excerpt: {primary_text}"
#         )

#         if child_text and parent_text and child_text != parent_text:
#             context_parts.append(
#                 f"Broader context: {parent_text}"
#             )

#     return "\n\n".join(context_parts)

# # ================================================================================
# # GENERAL CHATTING
# # ================================================================================

# def generate_chat_response(
#     client,
#     question: str
# ) -> str:

#     if client is None:
#         client = load_llm(
#             verbose=False
#         )

#     system_prompt = """
# You are a helpful AI assistant.

# You can:
# - Have normal conversations
# - Explain concepts
# - Answer general knowledge questions

# Be concise and helpful.
# """

#     response = client.chat.completions.create(
#         model=config.LLM_MODEL_ID,
#         messages=[
#             {
#                 "role": "system",
#                 "content": system_prompt
#             },
#             {
#                 "role": "user",
#                 "content": question
#             }
#         ],
#         temperature=config.TEMPERATURE,
#         max_tokens=config.MAX_NEW_TOKENS
#     )

#     content = (
#         response.choices[0]
#         .message.content
#     )

#     return _extract_text_content(
#         content
#     )

# llm.py
"""
Thin wrapper around the Mistral (OpenAI-compatible) chat completions API:
client construction, answer generation grounded in retrieved context, and
general-purpose chat.

Integration note on format_context(): vector_rag.retriever.retrieve()
returns each final chunk with "text" = the PARENT chunk's content (rich
surrounding context) and "child_text" = the CHILD chunk's content (the
small, precisely-matched excerpt that won it a place in the results — see
retriever.py / indexer.py's parent-child docstrings). format_context()
puts the parent text first as the main body the LLM should reason over,
and calls out the child text separately only as a "key matching excerpt"
pointer — not the other way around — because feeding the LLM only the
300-char child fragment as primary content would defeat the entire
purpose of doing the child->parent swap during retrieval.
"""

from __future__ import annotations

import logging
import threading
import time

from openai import OpenAI

import config

logger = logging.getLogger(__name__)

_client: OpenAI | None = None
_client_lock = threading.Lock()

NOT_FOUND_ANSWER = "This information was not found in the retrieved sections."

# Per-excerpt and total-context character budgets, so a handful of large
# parent chunks can't silently balloon the prompt past the model's context
# window. Sources are added in the order they arrive (already
# relevance-sorted by the retriever) and dropped once the total budget is
# exhausted, rather than truncated mid-block.
MAX_CHARS_PER_EXCERPT = getattr(config, "MAX_CHARS_PER_EXCERPT", 1500)
MAX_CONTEXT_CHARS = getattr(config, "MAX_CONTEXT_CHARS", 12000)

LLM_MAX_RETRIES = getattr(config, "LLM_MAX_RETRIES", 2)
LLM_RETRY_BACKOFF_SECONDS = getattr(config, "LLM_RETRY_BACKOFF_SECONDS", 1.5)


def _extract_text_content(content) -> str:
    """
    Normalizes OpenAI-style message content into a plain string.

    Multiple content blocks are joined with a space rather than
    concatenated directly — providers are not guaranteed to include
    inter-block whitespace, so naive concatenation risks merging the end
    of one block into the start of the next (e.g. "...report" + "Revenue..."
    -> "...reportRevenue...").
    """
    if content is None:
        return ""

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text", "")
                if text:
                    parts.append(str(text))
            elif hasattr(item, "text"):
                text = getattr(item, "text", "")
                if text:
                    parts.append(str(text))
        return " ".join(parts).strip()

    return str(content).strip()


def load_llm(verbose: bool = True) -> OpenAI:
    """
    Creates (or returns the cached) Mistral client. Thread-safe: concurrent
    first callers block on a lock instead of each constructing and
    discarding their own client.
    """
    global _client

    if _client is not None:
        if verbose:
            logger.info("Mistral client already initialised - reusing")
        return _client

    with _client_lock:
        if _client is not None:
            return _client

        missing = [
            name for name in ("MISTRAL_API_KEY", "MISTRAL_BASE_URL", "LLM_MODEL_ID")
            if not getattr(config, name, None)
        ]
        if missing:
            raise ValueError(
                f"Missing required config value(s) {missing}. "
                f"Add them to your .env / config before starting the LLM client."
            )

        _client = OpenAI(
            api_key=config.MISTRAL_API_KEY,
            base_url=config.MISTRAL_BASE_URL,
        )

        if verbose:
            logger.info("Mistral client ready - model: %s", config.LLM_MODEL_ID)

    return _client


def _call_chat_completion(client: OpenAI, messages: list[dict], *, context_label: str):
    """
    Calls chat.completions.create() with a short linear-backoff retry for
    transient failures (timeouts, 5xx, connection resets), so a single
    dropped request doesn't fail a user-facing answer outright.
    """
    last_exc: Exception | None = None
    for attempt in range(1, LLM_MAX_RETRIES + 2):  # initial try + retries
        try:
            return client.chat.completions.create(
                model=config.LLM_MODEL_ID,
                messages=messages,
                temperature=config.TEMPERATURE,
                max_tokens=config.MAX_NEW_TOKENS,
            )
        except Exception as exc:
            last_exc = exc
            logger.warning(
                "%s: LLM call attempt %d/%d failed: %s",
                context_label, attempt, LLM_MAX_RETRIES + 1, exc,
            )
            if attempt <= LLM_MAX_RETRIES:
                time.sleep(LLM_RETRY_BACKOFF_SECONDS * attempt)

    raise RuntimeError(f"{context_label} failed after retries: {last_exc}") from last_exc


def generate_answer(client, context: str, question: str) -> str:
    """
    Sends context and question to the LLM, grounded strictly in the
    provided context. Returns the answer string.
    """
    if client is None:
        client = load_llm(verbose=False)

    context = (context or "").strip()
    question = (question or "").strip()

    if not question:
        raise ValueError("Question is empty.")

    if not context:
        return NOT_FOUND_ANSWER

    system_prompt = """You are a financial analyst assistant.

Answer ONLY using the provided context.

For company overview or summary requests:

Generate a structured summary with:

1. Company Overview
2. Business Segments
3. Products and Services
4. Revenue Drivers
5. Strategic Priorities
6. Key Risks

If information is missing from the context, omit that section.

Never invent facts.

For factual questions:
Answer directly and concisely using only the context.

If the answer is not present, say:

"This information was not found in the retrieved sections."
"""

    user_message = f"""Context from financial reports. The most relevant excerpt appears first:
{context}

Question: {question}

Answer based strictly on the context above:"""

    try:
        response = _call_chat_completion(
            client,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            context_label="generate_answer",
        )
    except Exception as exc:
        raise RuntimeError(f"LLM generation failed: {exc}") from exc

    if not getattr(response, "choices", None):
        raise ValueError("Mistral returned no choices.")

    answer = _extract_text_content(response.choices[0].message.content)
    return answer if answer else NOT_FOUND_ANSWER


def format_context(retrieved_chunks: list[dict]) -> str:
    """
    Formats retrieved chunks into a single labelled context string for the
    LLM prompt.

    Each source's PARENT text is the primary excerpt (rich context); the
    CHILD text — the small fragment that actually matched the query — is
    called out separately as a "Key matching excerpt" only when it differs
    from the parent, so the LLM's attention is drawn to the precise hit
    without losing the surrounding context around it.

    Sources are included in the order given (assumed relevance-sorted) up
    to MAX_CONTEXT_CHARS total; once the budget is exhausted, remaining
    (lower-relevance) sources are dropped entirely rather than truncated
    mid-block, which would risk cutting off a sentence the LLM then
    misreads as complete.
    """
    if not retrieved_chunks:
        return ""

    def _trim_text(text: str, limit: int) -> str:
        cleaned = " ".join(str(text or "").split())
        if len(cleaned) <= limit:
            return cleaned
        return cleaned[:limit].rstrip() + "..."

    def _display_page(meta: dict) -> str:
        page = meta.get("page", "?")
        if page in (None, "", -1):
            return "?"
        return str(page)

    context_parts: list[str] = []
    total_chars = 0
    dropped = 0

    for i, chunk in enumerate(retrieved_chunks, 1):
        if not isinstance(chunk, dict):
            continue

        meta = chunk.get("metadata", {})
        if not isinstance(meta, dict):
            meta = {}

        company = str(meta.get("company", "Unknown")).strip() or "Unknown"
        page = _display_page(meta)

        parent_text = _trim_text(chunk.get("text", ""), MAX_CHARS_PER_EXCERPT)
        child_text = _trim_text(chunk.get("child_text", ""), MAX_CHARS_PER_EXCERPT)

        if not parent_text and not child_text:
            continue

        primary_text = parent_text or child_text

        block_lines = [f"[Source {i} - {company}, Page {page}]"]
        if child_text and parent_text and child_text != parent_text:
            block_lines.append(f"Key matching excerpt: {child_text}")
        block_lines.append(f"Context: {primary_text}")
        block = "\n".join(block_lines)

        # Always keep at least one source, even if it alone exceeds the
        # budget — an over-budget single source is still better context
        # than returning nothing.
        if context_parts and total_chars + len(block) > MAX_CONTEXT_CHARS:
            dropped += 1
            continue

        context_parts.append(block)
        total_chars += len(block)

    if dropped:
        logger.debug(
            "format_context: dropped %d lower-relevance source(s) to stay "
            "under the %d-char context budget.", dropped, MAX_CONTEXT_CHARS,
        )

    return "\n\n".join(context_parts)


# ================================================================================
# GENERAL CHATTING
# ================================================================================

def generate_chat_response(client, question: str) -> str:
    """
    Sends a free-form question to the LLM with no retrieval context — for
    general conversation rather than document-grounded Q&A.
    """
    if client is None:
        client = load_llm(verbose=False)

    question = (question or "").strip()
    if not question:
        raise ValueError("Question is empty.")

    system_prompt = """You are a helpful AI assistant.

You can:
- Have normal conversations
- Explain concepts
- Answer general knowledge questions

Be concise and helpful.
"""

    try:
        response = _call_chat_completion(
            client,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            context_label="generate_chat_response",
        )
    except Exception as exc:
        raise RuntimeError(f"LLM chat generation failed: {exc}") from exc

    if not getattr(response, "choices", None):
        raise ValueError("Mistral returned no choices.")

    answer = _extract_text_content(response.choices[0].message.content)
    return answer if answer else "I'm not sure how to respond to that — could you rephrase?"