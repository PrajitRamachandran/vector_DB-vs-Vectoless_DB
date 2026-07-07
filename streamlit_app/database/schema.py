"""
Database Schema

Creates and migrates all benchmark tables.

This module is intentionally idempotent: re-running
`initialize_schema()` on an existing database is safe.
New columns are added via best-effort ALTER TABLE
statements so existing installations upgrade in place
without losing data.
"""

import sqlite3

from streamlit_app.database.db import (
    get_connection
)

# ============================================================
# COLUMN MIGRATIONS
# ============================================================
# (table, column, column_ddl) — added only if missing.

_CONVERSATION_COLUMNS = [

    # Ownership (repository.save_conversation already writes
    # this column; it must exist on the table).
    ("user_id", "TEXT"),

    # Conversation organization
    ("title", "TEXT"),
    ("is_favorite", "INTEGER DEFAULT 0"),
    ("is_bookmarked", "INTEGER DEFAULT 0"),

    # Retrieval / answer quality signals
    ("query_type", "TEXT"),
    ("confidence_score", "REAL"),
    ("hallucination_risk", "TEXT"),

    # Usage / cost tracking
    ("tokens_prompt", "INTEGER"),
    ("tokens_completion", "INTEGER"),
    ("estimated_cost_usd", "REAL"),

    # Benchmark / evaluation tagging
    ("is_benchmark", "INTEGER DEFAULT 0"),
    ("benchmark_tag", "TEXT"),
]


def _add_column_if_missing(cursor, table, column, ddl):

    try:
        cursor.execute(
            f"ALTER TABLE {table} ADD COLUMN {column} {ddl}"
        )
    except sqlite3.OperationalError as e:
        # Column already exists — safe to ignore.
        if "duplicate column name" not in str(e).lower():
            raise


# ============================================================
# CREATE TABLES
# ============================================================

def create_tables():

    conn = get_connection()

    cursor = conn.cursor()

    # ========================================================
    # CONVERSATIONS
    # ========================================================

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS conversations (

        chat_id TEXT PRIMARY KEY,

        session_id TEXT,

        timestamp DATETIME,

        method TEXT,

        model_name TEXT,

        prompt TEXT,

        response TEXT,

        company_filter TEXT,

        retrieval_latency REAL,

        rerank_latency REAL,

        generation_latency REAL,

        total_latency REAL,

        vector_candidates INTEGER,

        bm25_candidates INTEGER,

        fused_candidates INTEGER,

        status TEXT,

        error_message TEXT
    )
    """)

    for column, ddl in _CONVERSATION_COLUMNS:
        _add_column_if_missing(
            cursor, "conversations", column, ddl
        )

    # ========================================================
    # RETRIEVED CHUNKS
    # ========================================================

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS retrieved_chunks (

        id INTEGER PRIMARY KEY AUTOINCREMENT,

        chat_id TEXT,

        rank_position INTEGER,

        source TEXT,

        company TEXT,

        page INTEGER,

        retrieval_score REAL,

        rerank_score REAL,

        rrf_score REAL,

        chunk_text TEXT,

        FOREIGN KEY(chat_id)
        REFERENCES conversations(chat_id)
    )
    """)

    # ========================================================
    # FEEDBACK (thumbs up / down + optional comment)
    # ========================================================

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS feedback (

        id INTEGER PRIMARY KEY AUTOINCREMENT,

        chat_id TEXT,

        user_id TEXT,

        rating TEXT,

        comment TEXT,

        timestamp DATETIME,

        FOREIGN KEY(chat_id)
        REFERENCES conversations(chat_id)
    )
    """)

    # ========================================================
    # EVALUATIONS
    # ========================================================

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS evaluations (

        evaluation_id TEXT PRIMARY KEY,

        timestamp DATETIME,

        method TEXT,

        evaluator TEXT,

        total_questions INTEGER,

        avg_judge_score REAL,

        pass_rate REAL,

        company_accuracy REAL,

        faithfulness REAL,

        answer_relevancy REAL,

        context_precision REAL,

        context_recall REAL,

        contextual_relevancy REAL,

        overall_score REAL
    )
    """)

    # ========================================================
    # SYSTEM LOGS
    # ========================================================

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS system_logs (

        id INTEGER PRIMARY KEY AUTOINCREMENT,

        timestamp DATETIME,

        level TEXT,

        module TEXT,

        message TEXT
    )
    """)

    # ========================================================
    # INDEXES
    # ========================================================

    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_conv_session "
        "ON conversations(session_id)"
    )

    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_conv_user "
        "ON conversations(user_id)"
    )

    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_conv_timestamp "
        "ON conversations(timestamp)"
    )

    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_chunks_chat "
        "ON retrieved_chunks(chat_id)"
    )

    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_feedback_chat "
        "ON feedback(chat_id)"
    )

    conn.commit()

    conn.close()


# ============================================================
# INITIALIZE
# ============================================================

def initialize_schema():

    create_tables()

    print(
        "Database schema initialized."
    )