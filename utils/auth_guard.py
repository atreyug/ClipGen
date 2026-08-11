from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt

from utils.jwt import decode_access_token

_bearer = HTTPBearer()


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(_bearer),) -> dict:
    token = credentials.credentials

    try:
        user_data = decode_access_token(token)
        return user_data  # e.g. {"user_id": "abc", "email": "x@y.com", "role": "admin"}

    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=401,
            detail="Your token has expired. Please login again.",
        )

    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=401,
            detail="Invalid token. Please login again.",
        )

def require_role(*roles: str):

    def role_checker(current_user: dict = Depends(get_current_user)) -> dict:
        if current_user.get("role") not in roles:
            raise HTTPException(
                status_code=403,
                detail=f"Access denied. Required role(s): {', '.join(roles)}.",
            )
        return current_user

    return role_checker

