import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY")
    ACCESS_TOKEN_EXPIRES_IN: int = int(
        os.getenv("ACCESS_TOKEN_EXPIRES_IN")
    )
    JWT_ALGORITHM: str = os.getenv("JWT_ALGORITHM", "HS256")

    DATABASE_URL: str = os.getenv("DATABASE_URL")

    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY")

    GMAIL_CREDENTIALS_FILE: str = "crd.json"
    GMAIL_TOKEN_FILE: str = "token.json"
    GMAIL_SENDER: str = "yugatre47@gmail.com"
    GMAIL_REDIRECT_URI: str = "http://127.0.0.1:8000/google/callback"

    # GMAIL_CREDENTIALS_FILE: str = os.getenv("GMAIL_CREDENTIALS_FILE")
    # GMAIL_TOKEN_FILE: str = os.getenv("GMAIL_TOKEN_FILE")
    # GMAIL_SENDER: str = os.getenv("GMAIL_SENDER")
    # GMAIL_REDIRECT_URI: str = os.getenv("GMAIL_REDIRECT_URI")

    SMTP_HOST: str = os.getenv("SMTP_HOST")
    SMTP_PORT: int = os.getenv("SMTP_PORT")
    SMTP_USERNAME: str = os.getenv("SMTP_USERNAME")
    SMTP_PASSWORD: str = os.getenv("SMTP_PASSWORD")
    SMTP_FROM: str = os.getenv("SMTP_FROM")

    CLOUDINARY_CLOUD_NAME: str = os.getenv("CLOUDINARY_CLOUD_NAME")
    CLOUDINARY_API_KEY: str = os.getenv("CLOUDINARY_API_KEY")
    CLOUDINARY_API_SECRET: str = os.getenv("CLOUDINARY_API_SECRET")




settings = Config()