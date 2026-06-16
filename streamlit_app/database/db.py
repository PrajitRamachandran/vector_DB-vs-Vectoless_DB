"""
SQLite Database Manager

Centralized database connection management
for the Financial RAG Benchmark.
"""

from pathlib import Path
import sqlite3

# ============================================================
# PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

STORAGE_DIR = ROOT_DIR / "storage"

DATABASE_PATH = STORAGE_DIR / "benchmark.db"

# ============================================================
# INITIALIZATION
# ============================================================

def initialize_database():
    """
    Creates storage directory if missing.
    Creates SQLite database file if missing.
    """

    STORAGE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    conn = sqlite3.connect(
        DATABASE_PATH
    )

    conn.close()

# ============================================================
# CONNECTION
# ============================================================

def get_connection():
    """
    Returns SQLite connection.

    Row factory enabled for
    dict-like access.
    """

    initialize_database()

    conn = sqlite3.connect(
        DATABASE_PATH
    )

    conn.row_factory = sqlite3.Row

    return conn

# ============================================================
# TEST
# ============================================================

def health_check():
    """
    Verify database accessibility.
    """

    try:

        conn = get_connection()

        cursor = conn.cursor()

        cursor.execute(
            "SELECT 1"
        )

        cursor.fetchone()

        conn.close()

        return {
            "success": True
        }

    except Exception as e:

        return {
            "success": False,
            "error": str(e)
        }

# ============================================================
# CLOSE
# ============================================================

def close_connection(conn):

    if conn:

        conn.close()