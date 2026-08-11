from utils.email import send_email


def send_signup_otp(email: str, otp: str):
    subject = "Verify your ClipGen email"

    body = f"""
Hello,

Your ClipGen verification OTP is:

{otp}

This OTP will expire in 10 minutes.

If you did not request this, please ignore this email.

Regards,
ClipGen Team
"""

    send_email(
        recipient=email,
        subject=subject,
        body=body
    )


def send_password_reset_otp(email: str, otp: str):
    subject = "ClipGen password reset OTP"

    body = f"""
Hello,

Your ClipGen password reset OTP is:

{otp}

This OTP will expire in 10 minutes.

If you did not request a password reset, please ignore this email.

Regards,
ClipGen Team
"""

    send_email(
        recipient=email,
        subject=subject,
        body=body
    )