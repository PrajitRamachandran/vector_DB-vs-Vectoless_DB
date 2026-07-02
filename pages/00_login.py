import streamlit as st

from streamlit_app.auth.auth_service import (
    authenticate
)

from streamlit_app.auth.session_manager import (
    login_user
)

st.set_page_config(
    page_title="Login"
)

st.title("Financial RAG Login")

username = st.text_input(
    "Username"
)

password = st.text_input(
    "Password",
    type="password"
)

if st.button("Login"):

    user = authenticate(
        username,
        password
    )

    if user:

        login_user(user)

        st.success(
            "Login successful"
        )

        st.switch_page(
            "pages/01_dashboard.py"
        )

    else:

        st.error(
            "Invalid username or password"
        )

st.divider()

st.write(
    "Don't have an account?"
)

if st.button(
    "Register"
):

    st.switch_page(
        "pages/00_register.py"
    )