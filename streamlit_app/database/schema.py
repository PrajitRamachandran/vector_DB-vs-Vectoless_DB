"""
Database Schema

Creates all benchmark tables.
"""

from streamlit_app.database.db import (
    get_connection
)

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