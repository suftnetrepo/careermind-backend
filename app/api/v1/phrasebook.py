"""Tranquis phrasebook and translation history endpoints."""
import uuid
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel

from app.core.deps import get_current_user
from app.db.engine import get_db
from app.db.models import User, TranslationHistory, PhrasebookEntry

router = APIRouter(tags=["phrasebook"])


# ── History ───────────────────────────────────────────────────────────────────

@router.get("/history")
async def get_history(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    limit: int = 50,
    offset: int = 0,
):
    """Return the user's recent translations, newest first."""
    result = await db.execute(
        select(TranslationHistory)
        .where(TranslationHistory.user_id == user.id)
        .order_by(TranslationHistory.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    rows = result.scalars().all()

    # Which ones are saved to phrasebook?
    ids = [r.id for r in rows]
    saved_result = await db.execute(
        select(PhrasebookEntry.translation_id)
        .where(PhrasebookEntry.user_id == user.id, PhrasebookEntry.translation_id.in_(ids))
    )
    saved_ids = {r for r in saved_result.scalars().all()}

    return [
        {
            "id": str(r.id),
            "source_text": r.source_text,
            "translated_text": r.translated_text,
            "source_lang": r.source_lang,
            "target_lang": r.target_lang,
            "mode": r.mode,
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "saved": r.id in saved_ids,
        }
        for r in rows
    ]


@router.delete("/history")
async def clear_history(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete all translation history for the current user."""
    await db.execute(
        delete(TranslationHistory).where(TranslationHistory.user_id == user.id)
    )
    await db.commit()
    return {"deleted": True}


# ── Phrasebook ────────────────────────────────────────────────────────────────

class SaveRequest(BaseModel):
    translation_id: str


@router.post("/phrasebook/save")
async def save_to_phrasebook(
    req: SaveRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Star a translation and add it to the phrasebook."""
    try:
        tid = uuid.UUID(req.translation_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid translation_id")

    # Verify it belongs to this user
    result = await db.execute(
        select(TranslationHistory).where(
            TranslationHistory.id == tid,
            TranslationHistory.user_id == user.id,
        )
    )
    entry = result.scalar_one_or_none()
    if not entry:
        raise HTTPException(status_code=404, detail="Translation not found")

    # Idempotent — don't add duplicates
    existing = await db.execute(
        select(PhrasebookEntry).where(
            PhrasebookEntry.user_id == user.id,
            PhrasebookEntry.translation_id == tid,
        )
    )
    if existing.scalar_one_or_none():
        return {"saved": True, "already_exists": True}

    db.add(PhrasebookEntry(id=uuid.uuid4(), user_id=user.id, translation_id=tid))
    await db.commit()
    return {"saved": True, "already_exists": False}


@router.get("/phrasebook")
async def get_phrasebook(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return all phrasebook entries with their full translation data."""
    result = await db.execute(
        select(PhrasebookEntry, TranslationHistory)
        .join(TranslationHistory, PhrasebookEntry.translation_id == TranslationHistory.id)
        .where(PhrasebookEntry.user_id == user.id)
        .order_by(PhrasebookEntry.created_at.desc())
    )
    rows = result.all()
    return [
        {
            "id": str(pb.id),
            "translation_id": str(th.id),
            "source_text": th.source_text,
            "translated_text": th.translated_text,
            "source_lang": th.source_lang,
            "target_lang": th.target_lang,
            "mode": th.mode,
            "saved_at": pb.created_at.isoformat() if pb.created_at else None,
        }
        for pb, th in rows
    ]


@router.delete("/phrasebook/{entry_id}")
async def remove_from_phrasebook(
    entry_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Remove a translation from the phrasebook."""
    try:
        eid = uuid.UUID(entry_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid entry_id")

    result = await db.execute(
        delete(PhrasebookEntry).where(
            PhrasebookEntry.id == eid,
            PhrasebookEntry.user_id == user.id,
        ).returning(PhrasebookEntry.id)
    )
    if not result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Entry not found")
    await db.commit()
    return {"deleted": True}
