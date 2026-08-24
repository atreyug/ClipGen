from datetime import datetime, timezone

import bcrypt
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from models.auth import (
    Login,
    Signup,
    OTPVerificationRequest,
    ForgotPasswordRequest,
)
from models import User, OTPVerification
from database.connection import get_db
from utils.jwt import create_access_token
from utils.otp import verify_otp
from utils.password import hash_password
from utils.auth_helpers import build_token_data, create_and_send_otp
from services.gmail_otp import (
    send_signup_otp,
    send_password_reset_otp,
)

router = APIRouter(
    prefix="/auth",
    tags=["Authentication"]
)


@router.post("")
def login(
    details: Login,
    db: Session = Depends(get_db),
):
    detail = details.model_dump()

    user = (
        db.query(User)
        .filter(User.email == detail["email"])
        .first()
    )

    if not user:
        return JSONResponse(
            status_code=404,
            content={"success": False, "message": "No user found", "data": {}},
        )

    if not bcrypt.checkpw(
        detail["password"].encode(),
        user.password_hash.encode(),
    ):
        return JSONResponse(
            status_code=401,
            content={"success": False, "message": "Wrong Password", "data": {}},
        )

    if not user.verified:
        return JSONResponse(
            status_code=403,
            content={
                "success": False,
                "message": "Please verify your email before logging in",
                "data": {},
            },
        )

    token = create_access_token(build_token_data(user))

    return JSONResponse(
        status_code=200,
        content={"success": True, "message": "Login Successful", "data": {"token": token}},
    )


@router.post("/signup")
def signup(
    details: Signup,
    db: Session = Depends(get_db),
):
    existing_user = (
        db.query(User)
        .filter(User.email == details.email)
        .first()
    )

    if existing_user:
        if not existing_user.verified:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "message": "Email already registered but not verified. Please verify your OTP.",
                    "data": {},
                },
            )
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "message": "User with this email already exists",
                "data": {},
            },
        )

    existing_username = (
        db.query(User)
        .filter(User.username == details.username)
        .first()
    )

    if existing_username:
        return JSONResponse(
            status_code=400,
            content={"success": False, "message": "Username is already taken", "data": {}},
        )

    new_user = User(
        name=details.name,
        username=details.username,
        email=details.email,
        password_hash=hash_password(details.password),
        role="user",
        verified=False,
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    try:
        create_and_send_otp(db, details.email, "signup", send_signup_otp)
    except RuntimeError as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": str(e), "data": {}},
        )

    return JSONResponse(
        status_code=201,
        content={
            "success": True,
            "message": "Signup successful. Please verify your email using the OTP.",
            "data": {},
        },
    )


@router.post("/verify-signup")
def verify_signup(
    details: OTPVerificationRequest,
    db: Session = Depends(get_db),
):
    user = (
        db.query(User)
        .filter(User.email == details.email)
        .first()
    )

    if not user:
        return JSONResponse(
            status_code=404,
            content={"success": False, "message": "No user found", "data": {}},
        )

    if user.verified:
        return JSONResponse(
            status_code=400,
            content={"success": False, "message": "Email is already verified", "data": {}},
        )

    error = _check_otp(db, details.email, "signup", details.otp)
    if error:
        return error

    user.verified = True
    db.commit()
    db.refresh(user)

    token = create_access_token(build_token_data(user))

    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "message": "Email verified successfully",
            "data": {"token": token},
        },
    )


@router.post("/resend-otp")
def resend_otp(
    details: ForgotPasswordRequest,
    db: Session = Depends(get_db),
):
    user = (
        db.query(User)
        .filter(User.email == details.email)
        .first()
    )

    if not user:
        return JSONResponse(
            status_code=404,
            content={"success": False, "message": "No user found", "data": {}},
        )

    if user.verified:
        return JSONResponse(
            status_code=400,
            content={"success": False, "message": "Email is already verified", "data": {}},
        )

    try:
        create_and_send_otp(db, details.email, "signup", send_signup_otp)
    except RuntimeError as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": str(e), "data": {}},
        )

    return JSONResponse(
        status_code=200,
        content={"success": True, "message": "New OTP sent successfully", "data": {}},
    )


@router.post("/forgot-password")
def forgot_password(
    details: ForgotPasswordRequest,
    db: Session = Depends(get_db),
):
    user = (
        db.query(User)
        .filter(User.email == details.email)
        .first()
    )

    if not user:
        return JSONResponse(
            status_code=404,
            content={"success": False, "message": "No user found", "data": {}},
        )

    if not user.verified:
        return JSONResponse(
            status_code=403,
            content={"success": False, "message": "Please verify your email first", "data": {}},
        )

    try:
        create_and_send_otp(db, details.email, "forgot_password", send_password_reset_otp)
    except RuntimeError as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": str(e), "data": {}},
        )

    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "message": "Password reset OTP sent successfully",
            "data": {},
        },
    )


@router.post("/verify-reset-otp")
def verify_reset_otp(
    details: OTPVerificationRequest,
    db: Session = Depends(get_db),
):
    user = (
        db.query(User)
        .filter(User.email == details.email)
        .first()
    )

    if not user:
        return JSONResponse(
            status_code=404,
            content={"success": False, "message": "No user found", "data": {}},
        )

    error = _check_otp(db, details.email, "forgot_password", details.otp)
    if error:
        return error

    reset_token_data = {
        **build_token_data(user),
        "purpose": "password_reset",
    }
    reset_token = create_access_token(reset_token_data)

    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "message": "OTP verified successfully",
            "data": {
                "user_id": str(user.user_id),
                "reset_token": reset_token,
            },
        },
    )


def _check_otp(
    db: Session,
    email: str,
    purpose: str,
    provided_otp: str,
) -> JSONResponse | None:
    
    otp_record = (
        db.query(OTPVerification)
        .filter(
            OTPVerification.email == email,
            OTPVerification.purpose == purpose,
        )
        .order_by(OTPVerification.created_at.desc())
        .first()
    )

    if not otp_record:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "message": "No OTP found. Please request a new OTP.",
                "data": {},
            },
        )

    if otp_record.expires_at < datetime.now(timezone.utc):
        db.delete(otp_record)
        db.commit()
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "message": "OTP expired. Please request a new OTP.",
                "data": {},
            },
        )

    if not verify_otp(provided_otp, otp_record.otp_hash):
        otp_record.attempts += 1
        db.commit()
        return JSONResponse(
            status_code=400,
            content={"success": False, "message": "Invalid OTP", "data": {}},
        )

    db.delete(otp_record)
    db.commit()
    return None