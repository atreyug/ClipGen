from sqlalchemy.orm import Session

from models.otp import OTPVerification
from utils.otp import generate_otp, hash_otp, get_otp_expiry


def build_token_data(user) -> dict:
    return {
        "user_id": str(user.user_id),
        "email": user.email,
        "role": user.role,
    }


def create_and_send_otp(
    db: Session,
    email: str,
    purpose: str,
    send_fn,
) -> None:
    db.query(OTPVerification).filter(
        OTPVerification.email == email,
        OTPVerification.purpose == purpose,
    ).delete()

    otp = generate_otp()

    otp_record = OTPVerification(
        email=email,
        otp_hash=hash_otp(otp),
        purpose=purpose,
        expires_at=get_otp_expiry(),
        attempts=0,
    )

    db.add(otp_record)
    db.commit()

    try:
        send_fn(email=email, otp=otp)
    except Exception:
        db.delete(otp_record)
        db.commit()
        raise RuntimeError("Failed to send OTP email")
