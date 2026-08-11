from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from database.connection import get_db
from models import User
from schema.user import UserCreate, UserUpdate, UserResponse, PasswordUpdate
from utils.auth_guard import get_current_user, require_role
from utils.password import hash_password

router = APIRouter(
    prefix="/admin",
    tags=["Admin"]
)

@router.get("/", response_model=list[UserResponse])
def get_all_users(
    db: Session = Depends(get_db),
    current_user: dict = Depends(require_role("admin"))
):
    return db.query(User).all()


@router.get("/{user_id}", response_model=UserResponse)
def get_user(
    user_id: UUID,
    db: Session = Depends(get_db),
    current_user: dict = Depends(require_role("admin"))
):
    user = (
        db.query(User)
        .filter(User.user_id == user_id)
        .first()
    )

    if not user:
        raise HTTPException(
            status_code=404,
            detail="User not found"
        )

    return user


@router.put("/{user_id}",
    response_model=UserResponse
)
def update_user(
    user_id: UUID,
    details: UserUpdate,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    user = (
        db.query(User)
        .filter(User.user_id == user_id)
        .first()
    )

    if not user:
        raise HTTPException(
            status_code=404,
            detail="User not found"
        )

    if details.name is not None:
        user.name = details.name

    if details.username is not None:
        existing_username = (
            db.query(User)
            .filter(
                User.username == details.username,
                User.user_id != user_id
            )
            .first()
        )

        if existing_username:
            raise HTTPException(
                status_code=400,
                detail="Username is already taken"
            )

        user.username = details.username

    db.commit()
    db.refresh(user)

    return user




