"""Self-service password reset for Tranquis (and CareerMind)."""
import hashlib
import secrets
import uuid
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from pydantic import BaseModel, EmailStr
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import hash_password
from app.core.rate_limit import limiter, client_ip
from app.db.engine import get_db
from app.db.models import User, PasswordResetToken
from app.services.email import send_email, base_template, h1, p, btn, escape_html
from app.config import get_settings

settings = get_settings()
router = APIRouter(prefix="/auth", tags=["Auth"])

RESET_TTL_HOURS = 1
RESET_RATE_LIMIT = "5/hour"


def _hash_token(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


async def send_password_reset_email(to: str, name: str, raw_token: str, app_source: str = "careermind") -> bool:
    first = escape_html(name.split()[0] if name.strip() else "there")

    if app_source == "tranquis":
        brand = "Tranquis"
        brand_color = "#14B8A6"
        tagline = "Translate the world."
        reset_url = f"tranquis://reset-password?token={raw_token}"
        support_email = "support@tranquis.com"
    else:
        brand = "Interquis"
        brand_color = "#6366f1"
        tagline = "Practice interviews. Land the job."
        reset_url = f"{settings.frontend_url}/reset-password?token={raw_token}"
        support_email = settings.brevo_from_email

    # Build a minimal branded template for the reset email
    content = (
        h1("Reset your password")
        + p(f"Hi {first}, we received a request to reset your {brand} password.")
        + p("Tap the button below to choose a new password. This link expires in 1 hour.")
        + btn("Reset password", reset_url)
        + p(
            f'<span style="font-size:13px;color:#94a3b8">'
            f"If you didn't request a password reset, you can ignore this email — your account is safe.<br/>"
            f"Need help? Contact <a href='mailto:{support_email}' style='color:#64748b'>{support_email}</a></span>"
        )
    )

    # Swap out the Interquis branding in the base template for Tranquis
    html = base_template(content, preheader=f"Reset your {brand} password")
    if app_source == "tranquis":
        html = html.replace(
            '<div style="font-size:20px;line-height:24px;font-weight:800;letter-spacing:-0.02em;color:#111827">Inter<span style="color:#6366f1">quis</span></div>',
            f'<div style="font-size:20px;line-height:24px;font-weight:800;letter-spacing:-0.02em;color:#111827">Tran<span style="color:{brand_color}">quis</span></div>',
        ).replace(
            '<div style="font-size:11px;line-height:16px;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;color:#94a3b8">Practice interviews. Land the job.</div>',
            f'<div style="font-size:11px;line-height:16px;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;color:#94a3b8">{tagline}</div>',
        ).replace(
            '<tr><td height="5" bgcolor="#6366f1"',
            f'<tr><td height="5" bgcolor="{brand_color}"',
        )

    return await send_email(to, f"Reset your {brand} password", html)


class ForgotPasswordRequest(BaseModel):
    email: EmailStr
    app_source: str = "careermind"  # 'careermind' | 'tranquis'


class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str


@router.post("/forgot-password")
@limiter.limit(RESET_RATE_LIMIT, key_func=client_ip)
async def forgot_password(
    request: Request,
    req: ForgotPasswordRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """
    Request a password reset link.
    Always returns 200 so attackers can't enumerate accounts.
    """
    result = await db.execute(select(User).where(User.email == req.email.lower()))
    user = result.scalar_one_or_none()

    if user:
        raw_token = secrets.token_urlsafe(32)
        reset_token = PasswordResetToken(
            id=uuid.uuid4(),
            user_id=user.id,
            token_hash=_hash_token(raw_token),
            used=False,
            expires_at=datetime.now(timezone.utc) + timedelta(hours=RESET_TTL_HOURS),
        )
        db.add(reset_token)
        await db.commit()

        app_source = req.app_source if req.app_source in ("careermind", "tranquis") else "careermind"
        background_tasks.add_task(
            send_password_reset_email,
            user.email, user.name, raw_token, app_source,
        )

    return {"sent": True}


@router.post("/reset-password")
async def reset_password(
    req: ResetPasswordRequest,
    db: AsyncSession = Depends(get_db),
):
    """Consume a reset token and change the user's password."""
    if len(req.new_password) < 8:
        raise HTTPException(400, "Password must be at least 8 characters")

    token_hash = _hash_token(req.token)
    result = await db.execute(
        select(PasswordResetToken).where(
            PasswordResetToken.token_hash == token_hash,
            PasswordResetToken.used == False,  # noqa: E712
        )
    )
    reset_token = result.scalar_one_or_none()

    if not reset_token:
        raise HTTPException(400, "This reset link is invalid or has already been used")

    if reset_token.expires_at < datetime.now(timezone.utc):
        raise HTTPException(400, "This reset link has expired — please request a new one")

    # Mark token as used and update password
    reset_token.used = True
    await db.execute(
        update(User)
        .where(User.id == reset_token.user_id)
        .values(hashed_password=hash_password(req.new_password))
    )
    await db.commit()

    return {"reset": True}
