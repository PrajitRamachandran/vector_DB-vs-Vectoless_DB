import streamlit as st

from streamlit_app.auth.auth_repository import (
    create_user,
    user_exists
)

from streamlit_app.auth.password_utils import (
    hash_password
)

st.set_page_config(
    page_title="Register"
)

st.title("Create Account")

username = st.text_input(
    "Username"
)

email = st.text_input(
    "Email"
)

password = st.text_input(
    "Password",
    type="password"
)

confirm_password = st.text_input(
    "Confirm Password",
    type="password"
)

if st.button(
    "Register"
):

    if not username:

        st.error(
            "Username required"
        )

    elif not email:

        st.error(
            "Email required"
        )

    elif password != confirm_password:

        st.error(
            "Passwords do not match"
        )

    elif user_exists(
        username,
        email
    ):

        st.error(
            "User already exists"
        )

    else:

        create_user(
            username=username,
            email=email,
            password_hash=hash_password(
                password
            )
        )

        st.success(
            "Account created"
        )

        st.switch_page(
            "pages/00_login.py"
        )

st.divider()

st.write(
    "Already have an account?"
)

if st.button(
    "Login"
):

    st.switch_page(
        "pages/00_login.py"
    )