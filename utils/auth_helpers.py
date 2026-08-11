from sqlalchemy.orm import Session

from models.otp import OTPVerification
from utils.otp import generate_otp, hash_otp, get_otp_expiry


def build_token_data(user) -> dict:
    """Build the standard JWT payload dict from a User ORM instance."""
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
    """
    Full OTP lifecycle:
      1. Delete any existing OTP record for (email, purpose).
      2. Generate a new OTP, hash it, and persist the record.
      3. Call ``send_fn(email=email, otp=otp)`` to deliver the code.

    Raises ``RuntimeError`` with a user-friendly message if the email
    could not be sent. In that case the OTP record has already been
    removed from the database before the error is raised, so no manual
    cleanup is required by the caller.
    """
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
