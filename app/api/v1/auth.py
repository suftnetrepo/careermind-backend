from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel, EmailStr
from app.db.engine import get_db
from app.db.models import User
from app.core.auth import (
    hash_password, verify_password,
    create_access_token, create_refresh_token, decode_token,
)
from app.core.deps import get_current_user
import uuid

router = APIRouter(prefix="/auth", tags=["Auth"])


class RegisterRequest(BaseModel):
    name: str
    email: EmailStr
    password: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class RefreshRequest(BaseModel):
    refresh_token: str


@router.post("/register", response_model=TokenResponse, status_code=201)
async def register(req: RegisterRequest, db: AsyncSession = Depends(get_db)):
    if len(req.password) < 8:
        raise HTTPException(400, "Password must be at least 8 characters")

    existing = await db.execute(select(User).where(User.email == req.email.lower()))
    if existing.scalar_one_or_none():
        raise HTTPException(400, "An account with this email already exists")

    user = User(
        id=uuid.uuid4(),
        email=req.email.lower(),
        name=req.name.strip(),
        hashed_password=hash_password(req.password),
        free_sessions=1,
    )
    db.add(user)
    await db.commit()

    return TokenResponse(
        access_token=create_access_token(str(user.id), user.email),
        refresh_token=create_refresh_token(str(user.id)),
    )


@router.post("/login", response_model=TokenResponse)
async def login(req: LoginRequest, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.email == req.email.lower()))
    user = result.scalar_one_or_none()

    if not user or not verify_password(req.password, user.hashed_password):
        raise HTTPException(401, "Incorrect email or password")

    return TokenResponse(
        access_token=create_access_token(str(user.id), user.email),
        refresh_token=create_refresh_token(str(user.id)),
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh(req: RefreshRequest, db: AsyncSession = Depends(get_db)):
    payload = decode_token(req.refresh_token)
    if not payload or payload.get("type") != "refresh":
        raise HTTPException(401, "Invalid refresh token")

    result = await db.execute(
        select(User).where(User.id == uuid.UUID(payload["sub"]))
    )
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(401, "User not found")

    return TokenResponse(
        access_token=create_access_token(str(user.id), user.email),
        refresh_token=create_refresh_token(str(user.id)),
    )


@router.get("/me")
async def me(user: User = Depends(get_current_user)):
    return {
        "id":                 str(user.id),
        "name":               user.name,
        "email":              user.email,
        "sessions_remaining": user.sessions_remaining,
        "free_sessions":      user.free_sessions,
        "paid_sessions":      user.paid_sessions,
        "created_at":         user.created_at.isoformat() if user.created_at else None,
    }
