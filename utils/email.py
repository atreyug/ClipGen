import os
import smtplib

from email.message import EmailMessage
from dotenv import load_dotenv

load_dotenv()


SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USERNAME = os.getenv("SMTP_USERNAME")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
SMTP_FROM = os.getenv("SMTP_FROM")


def send_email(
    recipient: str,
    subject: str,
    body: str
):
    if not SMTP_HOST:
        raise ValueError("SMTP_HOST is not configured")

    if not SMTP_USERNAME:
        raise ValueError("SMTP_USERNAME is not configured")

    if not SMTP_PASSWORD:
        raise ValueError("SMTP_PASSWORD is not configured")

    if not SMTP_FROM:
        raise ValueError("SMTP_FROM is not configured")

    message = EmailMessage()

    message["Subject"] = subject
    message["From"] = SMTP_FROM
    message["To"] = recipient

    message.set_content(body)

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_USERNAME, SMTP_PASSWORD)
        server.send_message(message)