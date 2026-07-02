from streamlit_app.auth.auth_schema import (
    initialize_auth_schema
)

from streamlit_app.auth.auth_repository import (
    create_user
)

from streamlit_app.auth.password_utils import (
    hash_password
)

initialize_auth_schema()

create_user(
    username="Thalaivar_Thimingalam",
    email="ramachandranprajit@gmail.com",
    password_hash=hash_password(
        "Prajit@2006"
    ),
    role="admin"
)

print(
    "Admin account created"
)