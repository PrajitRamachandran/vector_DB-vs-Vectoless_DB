from streamlit_app.auth.auth_repository import (
    get_user_by_username
)

from streamlit_app.auth.password_utils import (
    verify_password
)


def authenticate(
    username,
    password
):

    user = get_user_by_username(
        username
    )

    if not user:
        return None

    if not verify_password(
        password,
        user["password_hash"]
    ):
        return None

    return user