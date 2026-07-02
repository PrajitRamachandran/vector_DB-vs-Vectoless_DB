"""
Repository Layer

All database interactions happen here.
"""

from datetime import datetime
import uuid

from streamlit_app.database.db import (
    get_connection
)

# ============================================================
# CONVERSATIONS
# ============================================================

def save_conversation(
    session_id: str,
    user_id,
    method: str,
    model_name: str,
    prompt: str,
    response: str,
    company_filter: str = None,
    retrieval_latency: float = None,
    rerank_latency: float = None,
    generation_latency: float = None,
    total_latency: float = None,
    vector_candidates: int = None,
    bm25_candidates: int = None,
    fused_candidates: int = None,
    status: str = "SUCCESS",
    error_message: str = None
):
    """
    Insert a conversation.
    """

    chat_id = str(uuid.uuid4())

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO conversations (
            chat_id,
            session_id,
            user_id,
            timestamp,

            method,
            model_name,

            prompt,
            response,

            company_filter,

            retrieval_latency,
            rerank_latency,
            generation_latency,
            total_latency,

            vector_candidates,
            bm25_candidates,
            fused_candidates,

            status,
            error_message
        )
        VALUES (
            ?, ?,?, ?,
            ?, ?,
            ?, ?,
            ?,
            ?, ?, ?, ?,
            ?, ?, ?,
            ?, ?
        )
        """,
        (
            chat_id,
            session_id,
            user_id,
            datetime.now().isoformat(),

            method,
            model_name,

            prompt,
            response,

            company_filter,

            retrieval_latency,
            rerank_latency,
            generation_latency,
            total_latency,

            vector_candidates,
            bm25_candidates,
            fused_candidates,

            status,
            error_message
        )
    )

    conn.commit()

    conn.close()

    return chat_id

# ============================================================
# RETRIEVED CHUNKS
# ============================================================

def save_retrieved_chunks(
    chat_id: str,
    chunks: list
):
    """
    Persist retrieved chunks.
    """

    conn = get_connection()

    cursor = conn.cursor()

    for rank, chunk in enumerate(
        chunks,
        start=1
    ):

        metadata = chunk.get(
            "metadata",
            {}
        )

        cursor.execute(
            """
            INSERT INTO retrieved_chunks (

                chat_id,

                rank_position,

                source,
                company,
                page,

                retrieval_score,
                rerank_score,
                rrf_score,

                chunk_text

            )
            VALUES (
                ?, ?, ?, ?, ?,
                ?, ?, ?, ?
            )
            """,
            (
                chat_id,

                rank,

                metadata.get("source"),
                metadata.get("company"),
                metadata.get("page"),

                chunk.get("score"),
                chunk.get("rerank_score"),
                chunk.get("rrf_score"),

                chunk.get("text")
            )
        )

    conn.commit()

    conn.close()

# ============================================================
# GET CONVERSATIONS
# ============================================================

def get_conversations():

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT *
        FROM conversations
        ORDER BY timestamp DESC
        """
    )

    rows = cursor.fetchall()

    conn.close()

    return [
        dict(row)
        for row in rows
    ]

# ============================================================
# GET SINGLE CHAT
# ============================================================

def get_chat(
    chat_id: str
):

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT *
        FROM conversations
        WHERE chat_id = ?
        """,
        (chat_id,)
    )

    chat = cursor.fetchone()

    cursor.execute(
        """
        SELECT *
        FROM retrieved_chunks
        WHERE chat_id = ?
        ORDER BY rank_position
        """,
        (chat_id,)
    )

    chunks = cursor.fetchall()

    conn.close()

    return {
        "conversation":
            dict(chat)
            if chat else None,

        "chunks":
            [
                dict(c)
                for c in chunks
            ]
    }

# ============================================================
# DELETE CHAT
# ============================================================

def delete_chat(
    chat_id: str
):

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute(
        """
        DELETE FROM retrieved_chunks
        WHERE chat_id = ?
        """,
        (chat_id,)
    )

    cursor.execute(
        """
        DELETE FROM conversations
        WHERE chat_id = ?
        """,
        (chat_id,)
    )

    conn.commit()

    conn.close()

# ============================================================
# DELETE ALL
# ============================================================

def clear_conversations():

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute(
        "DELETE FROM retrieved_chunks"
    )

    cursor.execute(
        "DELETE FROM conversations"
    )

    conn.commit()

    conn.close()

# ============================================================
# SYSTEM LOGS
# ============================================================

def save_log(
    level: str,
    module: str,
    message: str
):

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO system_logs (

            timestamp,
            level,
            module,
            message

        )
        VALUES (?, ?, ?, ?)
        """,
        (
            datetime.now().isoformat(),
            level,
            module,
            message
        )
    )

    conn.commit()

    conn.close()

# ============================================================
# EVALUATIONS
# ============================================================
# ============================================================
# EVALUATIONS
# ============================================================

def save_evaluation(
    evaluation_id,
    method,
    evaluator,

    total_questions,

    avg_judge_score,
    pass_rate,
    company_accuracy,

    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    contextual_relevancy,

    overall_score
):

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO evaluations (

            evaluation_id,
            timestamp,

            method,
            evaluator,

            total_questions,

            avg_judge_score,
            pass_rate,
            company_accuracy,

            faithfulness,
            answer_relevancy,

            context_precision,
            context_recall,
            contextual_relevancy,

            overall_score

        )
        VALUES (

            ?, ?,

            ?, ?,

            ?,

            ?, ?, ?,

            ?, ?,

            ?, ?, ?,

            ?
        )
        """,
        (
            evaluation_id,

            datetime.now().isoformat(),

            method,
            evaluator,

            total_questions,

            avg_judge_score,
            pass_rate,
            company_accuracy,

            faithfulness,
            answer_relevancy,

            context_precision,
            context_recall,
            contextual_relevancy,

            overall_score
        )
    )

    conn.commit()

    conn.close()

def get_chunks_for_chat(chat_id):

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT *
        FROM retrieved_chunks
        WHERE chat_id = ?
        ORDER BY rank_position
        """,
        (chat_id,)
    )

    rows = cursor.fetchall()

    conn.close()

    return [
        dict(row)
        for row in rows
    ]


def get_evaluations():

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT *
        FROM evaluations
        ORDER BY timestamp DESC
        """
    )

    rows = cursor.fetchall()

    conn.close()

    return [
        dict(row)
        for row in rows
    ]

def clear_evaluations():

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute(
        "DELETE FROM evaluations"
    )

    conn.commit()

    conn.close()

def get_logs():

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT *
        FROM system_logs
        ORDER BY timestamp DESC
        """
    )

    rows = cursor.fetchall()

    conn.close()

    return [
        dict(row)
        for row in rows
    ]

def get_best_method():

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT *
        FROM evaluations
        ORDER BY overall_score DESC
        LIMIT 1
        """
    )

    row = cursor.fetchone()

    conn.close()

    return dict(row) if row else None


def get_dashboard_stats():

    conn = get_connection()
    cursor = conn.cursor()

    stats = {}

    cursor.execute(
        "SELECT COUNT(*) FROM conversations"
    )
    stats["total_chats"] = cursor.fetchone()[0]

    cursor.execute(
        """
        SELECT COUNT(DISTINCT method)
        FROM conversations
        """
    )
    stats["methods_used"] = cursor.fetchone()[0]

    cursor.execute(
        """
        SELECT AVG(total_latency)
        FROM conversations
        """
    )
    stats["avg_latency"] = cursor.fetchone()[0]

    cursor.execute(
        """
        SELECT COUNT(*)
        FROM evaluations
        """
    )
    stats["evaluation_runs"] = cursor.fetchone()[0]

    conn.close()

    return stats

def get_dashboard_stats_by_user(
    user_id
):

    conn = get_connection()

    cursor = conn.cursor()

    stats = {}

    cursor.execute(
        """
        SELECT COUNT(*)
        FROM conversations
        WHERE user_id = ?
        """,
        (user_id,)
    )

    stats["total_chats"] = (
        cursor.fetchone()[0]
    )

    cursor.execute(
        """
        SELECT COUNT(
            DISTINCT method
        )
        FROM conversations
        WHERE user_id = ?
        """,
        (user_id,)
    )

    stats["methods_used"] = (
        cursor.fetchone()[0]
    )

    cursor.execute(
        """
        SELECT AVG(total_latency)
        FROM conversations
        WHERE user_id = ?
        """,
        (user_id,)
    )

    stats["avg_latency"] = (
        cursor.fetchone()[0]
    )

    cursor.execute(
        """
        SELECT COUNT(*)
        FROM evaluations
        """
    )

    stats["evaluation_runs"] = (
        cursor.fetchone()[0]
    )

    conn.close()

    return stats

def get_recent_conversations_by_user(
    user_id,
    limit=10
):

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT *
        FROM conversations
        WHERE user_id = ?
        ORDER BY timestamp DESC
        LIMIT ?
        """,
        (
            user_id,
            limit
        )
    )

    rows = cursor.fetchall()

    conn.close()

    return [
        dict(row)
        for row in rows
    ]