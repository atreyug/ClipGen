import secrets
import bcrypt

from datetime import datetime, timedelta, timezone


def generate_otp() -> str:
    return f"{secrets.randbelow(1_000_000):06d}"


def hash_otp(otp: str) -> str:
    return bcrypt.hashpw(
        otp.encode("utf-8"),
        bcrypt.gensalt()
    ).decode("utf-8")


def verify_otp(otp: str, otp_hash: str) -> bool:
    return bcrypt.checkpw(
        otp.encode("utf-8"),
        otp_hash.encode("utf-8")
    )


def get_otp_expiry():
    return datetime.now(timezone.utc) + timedelta(minutes=10)


