from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, EmailStr


class UserCreate(BaseModel):
    name: str
    username: str
    email: EmailStr
    password: str


class UserUpdate(BaseModel):
    name: str | None = None
    username: str | None = None

class PasswordUpdate(BaseModel):
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    user_id: UUID
    name: str
    username: str
    email: EmailStr
    role: str
    verified: bool
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)