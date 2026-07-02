from streamlit_app.auth.auth_db import (
    get_auth_connection
)


def create_user(
    username,
    email,
    password_hash,
    role="user"
):

    conn = get_auth_connection()

    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO users
        (
            username,
            email,
            password_hash,
            role
        )
        VALUES
        (
            ?, ?, ?, ?
        )
        """,
        (
            username,
            email,
            password_hash,
            role
        )
    )

    conn.commit()

    conn.close()


def get_user_by_username(
    username
):

    conn = get_auth_connection()

    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT *
        FROM users
        WHERE username = ?
        """,
        (username,)
    )

    row = cursor.fetchone()

    conn.close()

    return dict(row) if row else None


def get_all_users():

    conn = get_auth_connection()

    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT *
        FROM users
        ORDER BY created_at DESC
        """
    )

    rows = cursor.fetchall()

    conn.close()

    return [
        dict(row)
        for row in rows
    ]

def user_exists(
    username,
    email
):

    conn = get_auth_connection()

    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT *
        FROM users
        WHERE username = ?
        OR email = ?
        """,
        (
            username,
            email
        )
    )

    row = cursor.fetchone()

    conn.close()

    return row is not None