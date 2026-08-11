import bcrypt


def hash_password(plain: str) -> str:
    """Hash a plain-text password with bcrypt and return the decoded string."""
    return bcrypt.hashpw(
        plain.encode("utf-8"),
        bcrypt.gensalt()
    ).decode("utf-8")
