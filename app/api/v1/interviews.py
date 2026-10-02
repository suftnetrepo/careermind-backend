from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, update
from pydantic import BaseModel
from typing import Optional, List
from openai import AsyncOpenAI
from app.db.engine import get_db
from app.db.models import User, InterviewSession, InterviewStatus
from app.core.deps import get_current_user
from app.config import get_settings
from app.services.question_generator import generate_questions
from app.api.v1.sessions import RATE_PENCE_PER_MINUTE, MIN_MINUTES, MAX_MINUTES
import httpx
import logging
import uuid
import json

logger = logging.getLogger(__name__)
settings = get_settings()

REALTIME_TOKEN_TTL_SECONDS = 60
COACHING_TAGS = ("positive", "tip", "pitfall")


def build_interviewer_prompt(
    role: str,
    level: str,
    duration_minutes: int,
    questions: list[dict],
) -> str:
    questions_text = "\n".join([
        f"- {q['question']} (follow-up: {q.get('follow_up', '')})"
        for q in questions
    ])

    return f"""You are Alex, a senior hiring manager at a leading technology company. You are conducting a real job interview for a {level}-level {role} position.

INTERVIEW DURATION: {duration_minutes} minutes.
You must wrap up naturally with 2 minutes remaining.

YOUR PERSONALITY:
- Professional, warm and genuinely curious
- You acknowledge good answers before moving on
- You ask natural follow-up questions
- You are encouraging but direct
- You sound completely human — never robotic
- Concise responses — this is their interview, not yours

INTERVIEW STRUCTURE:
1. Open with a warm professional greeting (15 seconds max)
   Example: "Hi, thanks for joining me today. I'm Alex and I'll be conducting your {role} interview. We have {duration_minutes} minutes together. Shall we get started?"
2. Work through your questions naturally
3. Ask at least one follow-up per answer
4. Acknowledge their points: "That's interesting...", "Good point...", "I like that approach..."
5. With 2 minutes remaining, say:
   "We're coming up on time — one final question for you..."
6. Close warmly:
   "That's everything from my side. Really enjoyed our conversation today. Do you have any questions for me?"

YOUR QUESTIONS (work through these naturally):
{questions_text}

CRITICAL RULES:
- NEVER say "Question 1", "Question 2" or number questions
- NEVER ignore what the candidate just said
- NEVER read questions robotically — weave them naturally
- ALWAYS react to their answer before asking the next one
- If they seem nervous, be warm and encouraging
- If an answer is vague, ask for a specific example
- Keep your speaking turns SHORT — under 30 seconds
- This must feel like a real human interview
"""


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


class CoachingRequest(BaseModel):
    question: str
    answer:   str
    role:     str


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


@router.post("/realtime-token")
async def get_realtime_token(
    interview_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Create a short-lived Realtime API client secret for the browser's WebRTC call."""
    result = await db.execute(
        select(InterviewSession).where(
            InterviewSession.id == uuid.UUID(interview_id),
            InterviewSession.user_id == user.id,
        )
    )
    interview = result.scalar_one_or_none()
    if not interview:
        raise HTTPException(404, "Interview not found")
    if interview.status != InterviewStatus.active:
        raise HTTPException(400, "Interview is not active")
    if not interview.paid and not interview.is_free:
        raise HTTPException(402, "Payment required")

    questions = json.loads(interview.questions_json or "[]")

    system_prompt = build_interviewer_prompt(
        role=interview.role,
        level=interview.level,
        duration_minutes=interview.duration_minutes,
        questions=questions,
    )

    async with httpx.AsyncClient(timeout=15) as client:
        response = await client.post(
            "https://api.openai.com/v1/realtime/client_secrets",
            headers={
                "Authorization": f"Bearer {settings.openai_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "expires_after": {"anchor": "created_at", "seconds": REALTIME_TOKEN_TTL_SECONDS},
                "session": {
                    "type":              "realtime",
                    "model":             settings.openai_realtime_model,
                    "instructions":      system_prompt,
                    "output_modalities": ["audio"],   # audio output still streams its transcript
                    "max_output_tokens": 500,
                    "audio": {
                        "input": {
                            "format":        {"type": "audio/pcm", "rate": 24000},
                            "transcription": {"model": "whisper-1"},
                            "turn_detection": {
                                "type":                "server_vad",
                                "threshold":           0.5,
                                "prefix_padding_ms":   300,
                                "silence_duration_ms": 1200,
                            },
                        },
                        "output": {
                            "format": {"type": "audio/pcm", "rate": 24000},
                            "voice":  "alloy",
                        },
                    },
                },
            },
        )
    if response.status_code != 200:
        logger.error("Realtime client secret failed: %s %s", response.status_code, response.text)
        raise HTTPException(502, "Failed to create Realtime session")
    data = response.json()

    return {
        "client_secret": data["value"],
        "session_id":    data["session"]["id"],
    }


@router.post("/coaching")
async def get_coaching(
    interview_id: str,
    req: CoachingRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Analyse a candidate answer and return coaching feedback."""
    result = await db.execute(
        select(InterviewSession.id).where(
            InterviewSession.id == uuid.UUID(interview_id),
            InterviewSession.user_id == user.id,
        )
    )
    if not result.scalar_one_or_none():
        raise HTTPException(404, "Interview not found")

    client = AsyncOpenAI(api_key=settings.openai_api_key)

    prompt = f"""You are an expert interview coach watching a {req.role} interview live. Analyse this exchange and give brief, actionable coaching feedback.

Question asked: {req.question}
Candidate answered: {req.answer}

Return JSON only, no other text:
{{
  "tag": "positive" | "tip" | "pitfall",
  "observation": "one sentence about what just happened",
  "coaching": "one sentence of coaching advice",
  "try_instead": "optional — a better way to phrase it (only for pitfall)"
}}

tag meanings:
- positive: they did something well — reinforce it
- tip: good answer but could be stronger — one improvement
- pitfall: something to avoid — potentially damaging

Be specific to what they actually said.
Keep each field under 20 words.
"""

    try:
        response = await client.chat.completions.create(
            model=settings.openai_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=300,
            response_format={"type": "json_object"},
        )
        note = json.loads(response.choices[0].message.content)
    except Exception:
        logger.exception("Coaching generation failed")
        raise HTTPException(502, "Coaching unavailable")

    if note.get("tag") not in COACHING_TAGS or not note.get("coaching"):
        raise HTTPException(502, "Coaching unavailable")
    if note["tag"] != "pitfall" or not note.get("try_instead"):
        note.pop("try_instead", None)
    return note


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
