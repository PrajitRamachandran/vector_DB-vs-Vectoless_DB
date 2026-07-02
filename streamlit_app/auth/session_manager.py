import streamlit as st


def login_user(user):
    """
    Create authenticated session.
    """

    st.session_state.logged_in = True

    st.session_state.user_id = (
        user["user_id"]
    )

    st.session_state.username = (
        user["username"]
    )

    st.session_state.role = (
        user["role"]
    )


def logout_user():
    """
    Clear all authentication-related
    session variables.
    """

    keys = [

        "logged_in",

        "user_id",

        "username",

        "role",

        "chat_history",

        "session_id"
    ]

    for key in keys:

        if key in st.session_state:

            del st.session_state[key]


def is_authenticated():
    """
    Returns True if user is logged in.
    """

    return st.session_state.get(
        "logged_in",
        False
    )


def is_admin():
    """
    Returns True if current user
    is an admin.
    """

    return (
        st.session_state.get(
            "role"
        )
        ==
        "admin"
    )


def get_current_user_id():
    """
    Returns current user id.
    """

    return st.session_state.get(
        "user_id"
    )


def get_current_username():
    """
    Returns current username.
    """

    return st.session_state.get(
        "username"
    )


def get_current_role():
    """
    Returns current role.
    """

    return st.session_state.get(
        "role"
    )