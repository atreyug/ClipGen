from pydantic import BaseModel, EmailStr


class Login(BaseModel):
    email: str
    password: str


class Signup(BaseModel):
    name: str
    username: str
    email: EmailStr
    password: str


class ForgotPasswordRequest(BaseModel):
    """Used by /resend-otp and /forgot-password — only an email is needed."""
    email: EmailStr


class OTPVerificationRequest(BaseModel):
    """Used by /verify-signup and /verify-reset-otp — email + OTP code."""
    email: EmailStr
    otp: str