from pathlib import Path
import sqlite3

ROOT_DIR = Path(__file__).resolve().parents[2]

STORAGE_DIR = ROOT_DIR / "storage"

USERS_DB = STORAGE_DIR / "users.db"


def get_auth_connection():

    STORAGE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    conn = sqlite3.connect(
        USERS_DB
    )

    conn.row_factory = sqlite3.Row

    return conn