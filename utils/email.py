import smtplib
import ssl
from email.message import EmailMessage

from config.config import settings


def send_email(
    recipient: str,
    subject: str,
    body: str,
):
    if not recipient:
        raise ValueError("Recipient email is required")

    if not settings.SMTP_FROM:
        raise ValueError("SMTP_FROM is not configured")

    if not settings.SMTP_HOST:
        raise ValueError("SMTP_HOST is not configured")

    message = EmailMessage()
    message["To"] = recipient
    message["From"] = settings.SMTP_FROM
    message["Subject"] = subject
    message.set_content(body)

    context = ssl.create_default_context()

    try:
        if settings.SMTP_PORT == 465:
            # Implicit TLS
            with smtplib.SMTP_SSL(
                settings.SMTP_HOST,
                settings.SMTP_PORT,
                context=context,
                timeout=30,
            ) as server:
                if settings.SMTP_USERNAME and settings.SMTP_PASSWORD:
                    server.login(settings.SMTP_USERNAME, settings.SMTP_PASSWORD)
                server.send_message(message)
        else:
            # STARTTLS (587 and most other ports)
            with smtplib.SMTP(
                settings.SMTP_HOST,
                settings.SMTP_PORT,
                timeout=30,
            ) as server:
                server.ehlo()
                server.starttls(context=context)
                server.ehlo()
                if settings.SMTP_USERNAME and settings.SMTP_PASSWORD:
                    server.login(settings.SMTP_USERNAME, settings.SMTP_PASSWORD)
                server.send_message(message)
    except smtplib.SMTPAuthenticationError as exc:
        raise RuntimeError(f"SMTP authentication failed: {exc}") from exc
    except smtplib.SMTPException as exc:
        raise RuntimeError(f"Failed to send email via SMTP: {exc}") from exc

    return {"status": "sent", "to": recipient, "subject": subject}