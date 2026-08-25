from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from database.connection import get_db
from models import User
from schema.user import UserCreate, UserUpdate, UserResponse, PasswordUpdate
from utils.auth_guard import get_current_user, require_role
from utils.password import hash_password

router = APIRouter(
    prefix="/users",
    tags=["Users"]
)



@router.get("/me", response_model=UserResponse)
def get_me(
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    user = (
        db.query(User)
        .filter(User.user_id == UUID(current_user["user_id"]))
        .first()
    )

    if not user:
        raise HTTPException(
            status_code=404,
            detail="User not found"
        )

    return user



@router.post("/",
    response_model=UserResponse,
    status_code=201
)
def create_user(
    details: UserCreate,
    db: Session = Depends(get_db)
):
    if db.query(User).filter(User.email == details.email).first():
        raise HTTPException(
            status_code=400,
            detail="Email is already registered"
        )

    if db.query(User).filter(User.username == details.username).first():
        raise HTTPException(
            status_code=400,
            detail="Username is already taken"
        )

    hashed_password = hash_password(details.password)

    new_user = User(
        name=details.name,
        username=details.username,
        email=details.email,
        password_hash=hashed_password,
        role="user",
        verified=False
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return new_user


@router.put("/password")
def update_password(
    details: PasswordUpdate,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    user = (
        db.query(User)
        .filter(User.email == details.email)
        .first()
    )

    if not user:
        raise HTTPException(
            status_code=404,
            detail="User not found"
        )

    if not user.verified:
        raise HTTPException(
            status_code=403,
            detail="Email is not verified"
        )

    user.password_hash = hash_password(details.password)

    db.commit()

    return {
        "success": True,
        "message": "Password updated successfully",
        "data": {}
    }


@router.put("/",
    response_model=UserResponse
)
def update_user(
    details: UserUpdate,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    user = (
        db.query(User)
        .filter(User.user_id == current_user["user_id"])
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


@router.delete("/{user_id}")
def delete_user(
    user_id: UUID,
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

    db.delete(user)
    db.commit()

    return {
        "success": True,
        "message": "User deleted successfully",
        "data": {}
    }


