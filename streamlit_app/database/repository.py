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
    error_message: str = None,
    title: str = None,
    query_type: str = None,
    confidence_score: float = None,
    hallucination_risk: str = None,
    tokens_prompt: int = None,
    tokens_completion: int = None,
    estimated_cost_usd: float = None,
    is_benchmark: bool = False,
    benchmark_tag: str = None
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
            error_message,

            title,
            query_type,
            confidence_score,
            hallucination_risk,

            tokens_prompt,
            tokens_completion,
            estimated_cost_usd,

            is_benchmark,
            benchmark_tag
        )
        VALUES (
            ?, ?, ?, ?,
            ?, ?,
            ?, ?,
            ?,
            ?, ?, ?, ?,
            ?, ?, ?,
            ?, ?,
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
            error_message,

            title or (prompt[:60] if prompt else None),
            query_type,
            confidence_score,
            hallucination_risk,

            tokens_prompt,
            tokens_completion,
            estimated_cost_usd,

            1 if is_benchmark else 0,
            benchmark_tag
        )
    )

    conn.commit()

    conn.close()

    return chat_id


def update_conversation_quality(
    chat_id: str,
    query_type: str = None,
    confidence_score: float = None,
    hallucination_risk: str = None,
    tokens_prompt: int = None,
    tokens_completion: int = None,
    estimated_cost_usd: float = None
):
    """
    Patch quality / usage metadata onto an already-saved
    conversation row (used once post-hoc analytics are computed).
    """

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute(
        """
        UPDATE conversations
        SET
            query_type = COALESCE(?, query_type),
            confidence_score = COALESCE(?, confidence_score),
            hallucination_risk = COALESCE(?, hallucination_risk),
            tokens_prompt = COALESCE(?, tokens_prompt),
            tokens_completion = COALESCE(?, tokens_completion),
            estimated_cost_usd = COALESCE(?, estimated_cost_usd)
        WHERE chat_id = ?
        """,
        (
            query_type,
            confidence_score,
            hallucination_risk,
            tokens_prompt,
            tokens_completion,
            estimated_cost_usd,
            chat_id
        )
    )

    conn.commit()
    conn.close()


def update_conversation_title(chat_id: str, title: str):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "UPDATE conversations SET title = ? WHERE chat_id = ?",
        (title, chat_id)
    )

    conn.commit()
    conn.close()


def toggle_favorite(chat_id: str) -> bool:
    """
    Flips is_favorite for a chat. Returns the new state.
    """

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT is_favorite FROM conversations WHERE chat_id = ?",
        (chat_id,)
    )
    row = cursor.fetchone()
    current = bool(row[0]) if row else False
    new_state = not current

    cursor.execute(
        "UPDATE conversations SET is_favorite = ? WHERE chat_id = ?",
        (1 if new_state else 0, chat_id)
    )

    conn.commit()
    conn.close()

    return new_state


def toggle_bookmark(chat_id: str) -> bool:
    """
    Flips is_bookmarked for a chat. Returns the new state.
    """

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT is_bookmarked FROM conversations WHERE chat_id = ?",
        (chat_id,)
    )
    row = cursor.fetchone()
    current = bool(row[0]) if row else False
    new_state = not current

    cursor.execute(
        "UPDATE conversations SET is_bookmarked = ? WHERE chat_id = ?",
        (1 if new_state else 0, chat_id)
    )

    conn.commit()
    conn.close()

    return new_state


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
        DELETE FROM feedback
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


def clear_conversations():

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute(
        "DELETE FROM retrieved_chunks"
    )

    cursor.execute(
        "DELETE FROM feedback"
    )

    cursor.execute(
        "DELETE FROM conversations"
    )

    conn.commit()

    conn.close()


def clear_session(session_id: str):
    """
    Deletes only the conversations belonging to one
    Streamlit session (used by 'Clear Chat').
    """

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        DELETE FROM retrieved_chunks
        WHERE chat_id IN (
            SELECT chat_id FROM conversations
            WHERE session_id = ?
        )
        """,
        (session_id,)
    )

    cursor.execute(
        """
        DELETE FROM feedback
        WHERE chat_id IN (
            SELECT chat_id FROM conversations
            WHERE session_id = ?
        )
        """,
        (session_id,)
    )

    cursor.execute(
        "DELETE FROM conversations WHERE session_id = ?",
        (session_id,)
    )

    conn.commit()
    conn.close()


# ============================================================
# SEARCH / FILTER / RECENT
# ============================================================

def search_conversations(
    user_id=None,
    keyword: str = None,
    method_filter: str = None,
    date_from: str = None,
    date_to: str = None,
    favorites_only: bool = False,
    bookmarked_only: bool = False,
    limit: int = 50
):
    """
    Flexible conversation search used by the sidebar
    history browser (search box + filters).
    """

    conn = get_connection()
    cursor = conn.cursor()

    clauses = []
    params = []

    if user_id is not None:
        clauses.append("user_id = ?")
        params.append(user_id)

    if keyword:
        clauses.append("(prompt LIKE ? OR response LIKE ? OR title LIKE ?)")
        like = f"%{keyword}%"
        params.extend([like, like, like])

    if method_filter and method_filter != "All":
        clauses.append("method LIKE ?")
        params.append(f"%{method_filter}%")

    if date_from:
        clauses.append("timestamp >= ?")
        params.append(date_from)

    if date_to:
        clauses.append("timestamp <= ?")
        params.append(date_to)

    if favorites_only:
        clauses.append("is_favorite = 1")

    if bookmarked_only:
        clauses.append("is_bookmarked = 1")

    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""

    cursor.execute(
        f"""
        SELECT *
        FROM conversations
        {where_sql}
        ORDER BY timestamp DESC
        LIMIT ?
        """,
        (*params, limit)
    )

    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]


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


def get_session_messages(session_id: str):
    """
    All Q&A turns for the current browser session, in order —
    used to rebuild chat history after a page refresh.
    """

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT *
        FROM conversations
        WHERE session_id = ?
        ORDER BY timestamp ASC
        """,
        (session_id,)
    )

    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]


def get_session_stats(session_id: str):
    """
    Powers the session summary card: message count,
    duration, and average latency for the active session.
    """

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT
            COUNT(*) as total_messages,
            AVG(total_latency) as avg_latency,
            MIN(timestamp) as first_ts,
            MAX(timestamp) as last_ts,
            SUM(COALESCE(tokens_prompt, 0) + COALESCE(tokens_completion, 0)) as total_tokens,
            SUM(COALESCE(estimated_cost_usd, 0)) as total_cost
        FROM conversations
        WHERE session_id = ? AND status = 'SUCCESS'
        """,
        (session_id,)
    )

    row = cursor.fetchone()
    conn.close()

    if not row or row["total_messages"] == 0:
        return {
            "total_messages": 0,
            "avg_latency": 0,
            "duration_seconds": 0,
            "total_tokens": 0,
            "total_cost": 0.0
        }

    duration = 0
    if row["first_ts"] and row["last_ts"]:
        try:
            first = datetime.fromisoformat(row["first_ts"])
            last = datetime.fromisoformat(row["last_ts"])
            duration = (last - first).total_seconds()
        except ValueError:
            duration = 0

    return {
        "total_messages": row["total_messages"],
        "avg_latency": row["avg_latency"] or 0,
        "duration_seconds": duration,
        "total_tokens": row["total_tokens"] or 0,
        "total_cost": row["total_cost"] or 0.0
    }


# ============================================================
# FEEDBACK
# ============================================================

def save_feedback(
    chat_id: str,
    user_id,
    rating: str,
    comment: str = None
):
    """
    rating is expected to be 'up' or 'down'.
    """

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO feedback (
            chat_id, user_id, rating, comment, timestamp
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            chat_id,
            user_id,
            rating,
            comment,
            datetime.now().isoformat()
        )
    )

    conn.commit()
    conn.close()


def get_feedback_for_chat(chat_id: str):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT * FROM feedback
        WHERE chat_id = ?
        ORDER BY timestamp DESC
        """,
        (chat_id,)
    )

    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]


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


def get_logs(limit: int = 200):

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT *
        FROM system_logs
        ORDER BY timestamp DESC
        LIMIT ?
        """,
        (limit,)
    )

    rows = cursor.fetchall()

    conn.close()

    return [
        dict(row)
        for row in rows
    ]


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


# ============================================================
# DASHBOARD / ANALYTICS
# ============================================================

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


def get_latency_breakdown():

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT
            AVG(retrieval_latency),
            AVG(rerank_latency),
            AVG(generation_latency),
            AVG(total_latency)
        FROM conversations
        """
    )

    row = cursor.fetchone()
    conn.close()

    return {
        "avg_retrieval_latency": row[0] or 0,
        "avg_rerank_latency": row[1] or 0,
        "avg_generation_latency": row[2] or 0,
        "avg_total_latency": row[3] or 0
    }


def get_top_queries(limit=10):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT
            prompt,
            COUNT(*) as frequency
        FROM conversations
        GROUP BY prompt
        ORDER BY frequency DESC
        LIMIT ?
        """,
        (limit,)
    )

    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]


def get_top_companies(limit=10):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT
            company_filter,
            COUNT(*) as frequency
        FROM conversations
        WHERE company_filter IS NOT NULL
        GROUP BY company_filter
        ORDER BY frequency DESC
        LIMIT ?
        """,
        (limit,)
    )

    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]


def get_top_methods(limit=10):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT
            method,
            COUNT(*) as frequency
        FROM conversations
        GROUP BY method
        ORDER BY frequency DESC
        LIMIT ?
        """,
        (limit,)
    )

    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]


def get_user_analytics():

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT
            user_id,
            COUNT(*) as total_chats,
            AVG(total_latency) as avg_latency,
            MAX(timestamp) as last_active
        FROM conversations
        GROUP BY user_id
        ORDER BY total_chats DESC
        """
    )

    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]


def get_evaluation_trend(limit=10):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT
            method,
            overall_score,
            timestamp
        FROM evaluations
        ORDER BY timestamp ASC
        """
    )

    rows = cursor.fetchall()
    conn.close()

    data = [dict(row) for row in rows]
    return data[-limit:] if limit else data


def get_recent_activity(limit=15):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT
            timestamp,
            'conversation' as activity_type,
            method as detail
        FROM conversations

        UNION ALL

        SELECT
            timestamp,
            'evaluation' as activity_type,
            method as detail
        FROM evaluations

        ORDER BY timestamp DESC
        LIMIT ?
        """,
        (limit,)
    )

    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]


def get_system_health():
    """
    Lightweight, dependency-free health signal used by the
    chat sidebar's system health panel: can we open a
    connection and read from the core tables.
    """

    health = {
        "database": False,
        "conversations_table": False,
        "evaluations_table": False
    }

    try:
        conn = get_connection()
        cursor = conn.cursor()
        health["database"] = True

        cursor.execute("SELECT 1 FROM conversations LIMIT 1")
        health["conversations_table"] = True

        cursor.execute("SELECT 1 FROM evaluations LIMIT 1")
        health["evaluations_table"] = True

        conn.close()
    except Exception:
        pass

    return health