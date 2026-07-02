import streamlit as st

from streamlit_app.auth.session_manager import (
    is_authenticated
)


def require_login():

    if not is_authenticated():

        st.switch_page(
            "pages/00_login.py"
        )

        st.stop()