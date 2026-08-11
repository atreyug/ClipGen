import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY")
    ACCESS_TOKEN_EXPIRES_IN: int = int(
        os.getenv("ACCESS_TOKEN_EXPIRES_IN", 10080)
    )
    JWT_ALGORITHM: str = os.getenv("JWT_ALGORITHM", "HS256")

    DATABASE_URL: str = os.getenv("DATABASE_URL")


settings = Config()