import bcrypt


def hash_password(password):

    return bcrypt.hashpw(
        password.encode(),
        bcrypt.gensalt()
    ).decode()


def verify_password(
    password,
    password_hash
):

    return bcrypt.checkpw(
        password.encode(),
        password_hash.encode()
    )