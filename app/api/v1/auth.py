from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, status
from app.core.rate_limit import (
    limiter, client_ip, REGISTER_LIMIT, LOGIN_LIMIT, VERIFY_EMAIL_LIMIT, RESEND_VERIFY_LIMIT,
)
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete
from pydantic import BaseModel, EmailStr
from app.db.engine import get_db
from app.db.models import User, InterviewSession
from app.core.auth import (
    hash_password, verify_password,
    create_access_token, create_refresh_token, decode_token,
    create_email_verification_token,
)
from app.services.email import send_verification_email
from app.core.deps import get_current_user
from app.core.pricing import FREE_INTERVIEW_MINUTES
import logging
import uuid

logger = logging.getLogger(__name__)

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


class VerifyEmailRequest(BaseModel):
    token: str


@router.post("/register", response_model=TokenResponse, status_code=201)
@limiter.limit(REGISTER_LIMIT, key_func=client_ip)
async def register(
    request: Request,
    req: RegisterRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
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
        free_minutes=FREE_INTERVIEW_MINUTES,
    )
    db.add(user)
    await db.commit()

    # Sent after the response so a slow or failing email never blocks sign-up
    background_tasks.add_task(
        send_verification_email,
        user.email, user.name, create_email_verification_token(str(user.id), user.email),
    )

    return TokenResponse(
        access_token=create_access_token(str(user.id), user.email),
        refresh_token=create_refresh_token(str(user.id)),
    )


@router.post("/login", response_model=TokenResponse)
@limiter.limit(LOGIN_LIMIT, key_func=client_ip)
async def login(request: Request, req: LoginRequest, db: AsyncSession = Depends(get_db)):
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
        "has_free_interview": user.has_free_interview,
        "free_minutes":       user.free_minutes,
        "email_verified":     bool(user.email_verified),
        "is_admin":           bool(user.is_admin),
        "created_at":         user.created_at.isoformat() if user.created_at else None,
    }


@router.post("/verify-email")
@limiter.limit(VERIFY_EMAIL_LIMIT, key_func=client_ip)
async def verify_email(request: Request, req: VerifyEmailRequest, db: AsyncSession = Depends(get_db)):
    """Confirm an email address from the link in the verification email. No login needed —
    the link may be opened on a different device."""
    payload = decode_token(req.token)
    if not payload or payload.get("type") != "email_verify":
        raise HTTPException(400, "This verification link is invalid or has expired")
    try:
        user_id = uuid.UUID(payload["sub"])
    except (KeyError, ValueError):
        raise HTTPException(400, "This verification link is invalid or has expired")

    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user or user.email != payload.get("email"):
        raise HTTPException(400, "This verification link is invalid or has expired")

    if not user.email_verified:
        user.email_verified = True
        await db.commit()
    return {"verified": True, "email": user.email}


@router.post("/resend-verification")
@limiter.limit(RESEND_VERIFY_LIMIT)
async def resend_verification(
    request: Request,
    background_tasks: BackgroundTasks,
    user: User = Depends(get_current_user),
):
    if user.email_verified:
        return {"sent": False, "already_verified": True}
    background_tasks.add_task(
        send_verification_email,
        user.email, user.name, create_email_verification_token(str(user.id), user.email),
    )
    return {"sent": True, "already_verified": False}


@router.delete("/account")
async def delete_account(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Hard delete the user account and all
    associated data — interviews, transcripts,
    CV text, feedback, quiz, flashcards.
    """
    interviews = await db.execute(
        delete(InterviewSession).where(InterviewSession.user_id == user.id)
    )
    await db.execute(delete(User).where(User.id == user.id))
    await db.commit()
    logger.info("Account deleted: %s (%d interviews)", user.id, interviews.rowcount)
    return {"deleted": True}
