import os
import base64
from email.message import EmailMessage
from pathlib import Path

os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
os.environ["OAUTHLIB_RELAX_TOKEN_SCOPE"] = "1"

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build

from config.config import settings


SCOPES = [
    "https://www.googleapis.com/auth/gmail.send"
]

BASE_DIR = Path(__file__).resolve().parent.parent

CREDENTIALS_FILE = Path(settings.GMAIL_CREDENTIALS_FILE)
TOKEN_FILE = Path(settings.GMAIL_TOKEN_FILE)

REDIRECT_URI = settings.GMAIL_REDIRECT_URI


def get_google_flow() -> Flow:
    flow = Flow.from_client_secrets_file(
        str(CREDENTIALS_FILE),
        scopes=SCOPES,
    )

    flow.redirect_uri = REDIRECT_URI

    return flow


def get_gmail_service():
    if not TOKEN_FILE.exists():
        raise RuntimeError(
            "Gmail API is not authorized. "
            "Visit the Google OAuth authorization endpoint first."
        )

    creds = Credentials.from_authorized_user_file(
        str(TOKEN_FILE),
        SCOPES,
    )

    if creds.expired and creds.refresh_token:
        creds.refresh(Request())

        TOKEN_FILE.write_text(
            creds.to_json(),
            encoding="utf-8",
        )

    if not creds.valid:
        raise RuntimeError(
            "Gmail OAuth credentials are invalid. "
            "Authorize the Gmail API again."
        )

    return build(
        "gmail",
        "v1",
        credentials=creds,
    )


def send_email(
    recipient: str,
    subject: str,
    body: str,
):
    if not recipient:
        raise ValueError("Recipient email is required")

    if not settings.GMAIL_SENDER:
        raise ValueError("GMAIL_SENDER is not configured")

    service = get_gmail_service()

    message = EmailMessage()

    message["To"] = recipient
    message["From"] = settings.GMAIL_SENDER
    message["Subject"] = subject

    message.set_content(body)

    encoded_message = base64.urlsafe_b64encode(
        message.as_bytes()
    ).decode()

    response = (
        service.users()
        .messages()
        .send(
            userId="me",
            body={
                "raw": encoded_message
            },
        )
        .execute()
    )

    return response