"""Per-user daily translation quota for Tranquis free tier.

Free users: FREE_DAILY_TRANSLATIONS text translations per UTC day.
Pro users (is_pro=True on User, set by RevenueCat webhook): unlimited.

Usage:
    from app.core.quota import check_and_increment_quota
    await check_and_increment_quota(db, user)   # raises HTTP 429 when exceeded
"""
from datetime import datetime, timezone, date
from fastapi import HTTPException
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.postgresql import insert as pg_insert
import uuid

from app.db.models import TranslationQuota, FREE_DAILY_TRANSLATIONS


async def check_and_increment_quota(db: AsyncSession, user) -> dict:
    """
    Check and atomically increment the user's translation count for today.
    Returns {"used": n, "limit": FREE_DAILY_TRANSLATIONS, "unlimited": bool}.
    Raises HTTP 429 if the free limit is exceeded.
    """
    # Pro users bypass quota entirely
    is_pro: bool = getattr(user, "is_pro", False)
    if is_pro or getattr(user, "is_admin", False):
        return {"used": 0, "limit": FREE_DAILY_TRANSLATIONS, "unlimited": True}

    today: date = datetime.now(timezone.utc).date()

    # Upsert: insert (count=1) or increment by 1, then return the new count
    stmt = (
        pg_insert(TranslationQuota)
        .values(
            id=uuid.uuid4(),
            user_id=user.id,
            quota_date=today,
            count=1,
        )
        .on_conflict_do_update(
            constraint="uq_translation_quota_user_date",
            set_={"count": TranslationQuota.count + 1},
        )
        .returning(TranslationQuota.count)
    )
    result = await db.execute(stmt)
    await db.commit()
    new_count: int = result.scalar_one()

    if new_count > FREE_DAILY_TRANSLATIONS:
        # Decrement back so we don't keep counting past the limit
        await db.execute(
            update(TranslationQuota)
            .where(
                TranslationQuota.user_id == user.id,
                TranslationQuota.quota_date == today,
            )
            .values(count=TranslationQuota.count - 1)
        )
        await db.commit()
        raise HTTPException(
            status_code=429,
            detail={
                "code": "quota_exceeded",
                "message": f"Free plan: {FREE_DAILY_TRANSLATIONS} translations per day. Upgrade to Pro for unlimited.",
                "limit": FREE_DAILY_TRANSLATIONS,
                "resets": "midnight UTC",
            },
        )

    return {
        "used": new_count,
        "limit": FREE_DAILY_TRANSLATIONS,
        "unlimited": False,
    }


async def get_quota_status(db: AsyncSession, user) -> dict:
    """Return quota status without incrementing (for the /me or /quota endpoint)."""
    is_pro: bool = getattr(user, "is_pro", False)
    if is_pro or getattr(user, "is_admin", False):
        return {"used": 0, "limit": FREE_DAILY_TRANSLATIONS, "unlimited": True}

    today: date = datetime.now(timezone.utc).date()
    result = await db.execute(
        select(TranslationQuota.count).where(
            TranslationQuota.user_id == user.id,
            TranslationQuota.quota_date == today,
        )
    )
    count = result.scalar_one_or_none() or 0
    return {
        "used": count,
        "limit": FREE_DAILY_TRANSLATIONS,
        "unlimited": False,
        "remaining": max(0, FREE_DAILY_TRANSLATIONS - count),
    }
