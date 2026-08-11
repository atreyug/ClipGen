import jwt
from datetime import datetime, timedelta, timezone

from config.config import settings


def create_access_token(data: dict) -> str:
    payload = data.copy()

    expire_time = (
        datetime.now(timezone.utc)
        + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRES_IN)
    )
    payload["exp"] = expire_time

    token = jwt.encode(
        payload,
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM,
    )

    return token


def decode_access_token(token: str) -> dict:
    payload = jwt.decode(
        token,
        settings.JWT_SECRET_KEY,
        algorithms=[settings.JWT_ALGORITHM],
    )

    return payload