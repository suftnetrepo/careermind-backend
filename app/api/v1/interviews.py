from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, update
from pydantic import BaseModel
from typing import Optional, List
from app.db.engine import get_db
from app.db.models import User, InterviewSession, InterviewStatus
from app.core.deps import get_current_user
from app.services.question_generator import generate_questions
from app.api.v1.sessions import RATE_PENCE_PER_MINUTE, MIN_MINUTES, MAX_MINUTES
import uuid
import json

router = APIRouter(prefix="/interviews", tags=["Interviews"])

FREE_INTERVIEW_MINUTES = 15


class SetupRequest(BaseModel):
    role:             str
    level:            str
    focus:            str
    duration_minutes: int = 15
    job_description:  Optional[str] = None


class StartRequest(BaseModel):
    interview_id: str


class EndRequest(BaseModel):
    interview_id:     str
    transcript_json:  Optional[str] = None
    duration_seconds: Optional[int] = None


@router.post("/setup")
async def setup_interview(
    req: SetupRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Generate questions and create a pending interview session.

    A user's first interview is free (15 minutes) and pre-paid; every later
    interview stays unpaid until Stripe checkout completes.
    """
    if not user.has_free_interview and not MIN_MINUTES <= req.duration_minutes <= MAX_MINUTES:
        raise HTTPException(400, f"Duration must be between {MIN_MINUTES} and {MAX_MINUTES} minutes")

    questions = await generate_questions(
        role=req.role,
        level=req.level,
        focus=req.focus,
        job_description=req.job_description,
    )

    session = InterviewSession(
        id=uuid.uuid4(),
        user_id=user.id,
        role=req.role,
        level=req.level,
        focus=req.focus,
        duration_minutes=req.duration_minutes,
        job_description=req.job_description,
        questions_json=json.dumps(questions),
        status=InterviewStatus.setup,
        paid=False,
        is_free=False,
    )

    # Claim the free interview atomically so concurrent setups can't both use it
    claimed = await db.execute(
        update(User)
        .where(User.id == user.id, User.free_minutes > 0)
        .values(free_minutes=0)
        .returning(User.id)
    )
    if claimed.scalar_one_or_none():
        session.is_free          = True
        session.paid             = True
        session.duration_minutes = FREE_INTERVIEW_MINUTES
    else:
        session.amount_pence = session.duration_minutes * RATE_PENCE_PER_MINUTE

    db.add(session)
    await db.commit()

    return {
        "interview_id": str(session.id),
        "questions":    questions,
        "role":         req.role,
        "level":        req.level,
        "focus":        req.focus,
        "duration_minutes": session.duration_minutes,
        "is_free":          session.is_free,
        "paid":             session.paid,
        "amount_pence":     session.amount_pence,
    }


@router.post("/start")
async def start_interview(
    req: StartRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Mark interview as active. Requires the interview to be paid (or free)."""
    result = await db.execute(
        select(InterviewSession).where(
            InterviewSession.id == uuid.UUID(req.interview_id),
            InterviewSession.user_id == user.id,
        )
    )
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(404, "Interview session not found")
    if session.status != InterviewStatus.setup:
        raise HTTPException(400, "Interview already started or completed")

    if not session.paid:
        raise HTTPException(402, "Payment required before starting")

    from datetime import datetime, timezone
    session.status = InterviewStatus.active
    session.started_at = datetime.now(timezone.utc)
    await db.commit()

    return {"status": "active", "interview_id": str(session.id)}


@router.post("/end")
async def end_interview(
    req: EndRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Mark interview as completed and save transcript."""
    result = await db.execute(
        select(InterviewSession).where(
            InterviewSession.id == uuid.UUID(req.interview_id),
            InterviewSession.user_id == user.id,
        )
    )
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(404, "Interview session not found")

    from datetime import datetime, timezone
    session.status = InterviewStatus.completed
    session.ended_at = datetime.now(timezone.utc)
    session.duration_seconds = req.duration_seconds
    session.transcript_json = req.transcript_json
    await db.commit()

    return {"status": "completed", "interview_id": str(session.id)}


@router.get("/history")
async def get_history(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get user's past interview sessions."""
    result = await db.execute(
        select(InterviewSession)
        .where(InterviewSession.user_id == user.id)
        .order_by(desc(InterviewSession.created_at))
        .limit(20)
    )
    sessions = result.scalars().all()

    return [
        {
            "id":               str(s.id),
            "role":             s.role,
            "level":            s.level,
            "focus":            s.focus,
            "status":           s.status,
            "overall_score":    s.overall_score,
            "duration_seconds": s.duration_seconds,
            "created_at":       s.created_at.isoformat() if s.created_at else None,
        }
        for s in sessions
    ]


@router.get("/{interview_id}")
async def get_interview(
    interview_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get a single interview session with full details."""
    result = await db.execute(
        select(InterviewSession).where(
            InterviewSession.id == uuid.UUID(interview_id),
            InterviewSession.user_id == user.id,
        )
    )
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(404, "Interview not found")

    return {
        "id":               str(session.id),
        "role":             session.role,
        "level":            session.level,
        "focus":            session.focus,
        "duration_minutes": session.duration_minutes,
        "job_description":  session.job_description,
        "questions":        json.loads(session.questions_json) if session.questions_json else [],
        "status":           session.status,
        "paid":             session.paid,
        "is_free":          session.is_free,
        "amount_pence":     session.amount_pence,
        "stripe_payment_intent": session.stripe_payment_intent,
        "overall_score":    session.overall_score,
        "feedback":         json.loads(session.feedback_json) if session.feedback_json else None,
        "transcript":       json.loads(session.transcript_json) if session.transcript_json else None,
        "duration_seconds": session.duration_seconds,
        "started_at":       session.started_at.isoformat() if session.started_at else None,
        "ended_at":         session.ended_at.isoformat() if session.ended_at else None,
        "created_at":       session.created_at.isoformat() if session.created_at else None,
    }
