from streamlit_app.auth.auth_db import (
    get_auth_connection
)


def initialize_auth_schema():

    conn = get_auth_connection()

    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (

        user_id INTEGER PRIMARY KEY AUTOINCREMENT,

        username TEXT UNIQUE NOT NULL,

        email TEXT UNIQUE NOT NULL,

        password_hash TEXT NOT NULL,

        role TEXT DEFAULT 'user',

        created_at DATETIME
            DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()

    conn.close()