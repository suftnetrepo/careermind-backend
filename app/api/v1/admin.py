from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc, and_, cast, Date
from app.db.engine import get_db
from app.db.models import User, InterviewSession, InterviewStatus
from app.core.deps import get_admin_user
from app.api.v1.interviews import parse_uuid
from datetime import datetime, timezone, timedelta
import json

router = APIRouter(prefix="/admin", tags=["Admin"])

PAGE_LIMIT_MAX = 100
REVENUE_DAYS = 30
DURATION_TIERS = [(15, "15 min — £3"), (30, "30 min — £6"), (45, "45 min — £9"), (60, "60 min — £12")]

# A paid, non-free interview is one sale
SOLD = and_(InterviewSession.paid == True, InterviewSession.is_free == False)  # noqa: E712
COMPLETED = InterviewSession.status == InterviewStatus.completed


def _pages(total: int, limit: int) -> int:
    return max(1, -(-total // limit))


@router.get("/overview")
async def get_overview(
    admin: User = Depends(get_admin_user),
    db: AsyncSession = Depends(get_db),
):
    now = datetime.now(timezone.utc)
    month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    last_month_start = (month_start - timedelta(days=1)).replace(day=1)

    total_users = await db.scalar(select(func.count(User.id)))
    users_this_month = await db.scalar(select(func.count(User.id)).where(User.created_at >= month_start))
    users_last_month = await db.scalar(
        select(func.count(User.id)).where(User.created_at >= last_month_start, User.created_at < month_start)
    )

    total_interviews = await db.scalar(select(func.count(InterviewSession.id)).where(COMPLETED))
    interviews_this_month = await db.scalar(
        select(func.count(InterviewSession.id)).where(COMPLETED, InterviewSession.created_at >= month_start)
    )

    revenue_this_month = (await db.scalar(
        select(func.sum(InterviewSession.amount_pence)).where(SOLD, InterviewSession.created_at >= month_start)
    ) or 0) / 100
    revenue_total = (await db.scalar(select(func.sum(InterviewSession.amount_pence)).where(SOLD)) or 0) / 100

    avg_score = await db.scalar(
        select(func.avg(InterviewSession.overall_score)).where(InterviewSession.overall_score.isnot(None))
    )

    free_count = await db.scalar(select(func.count(InterviewSession.id)).where(InterviewSession.is_free == True))  # noqa: E712
    paid_count = await db.scalar(select(func.count(InterviewSession.id)).where(SOLD))

    roles = await db.execute(
        select(InterviewSession.role, func.count(InterviewSession.id).label("count"))
        .where(COMPLETED)
        .group_by(InterviewSession.role)
        .order_by(desc("count"))
        .limit(5)
    )

    return {
        "users": {
            "total":      total_users,
            "this_month": users_this_month,
            "last_month": users_last_month,
        },
        "interviews": {
            "total":      total_interviews,
            "this_month": interviews_this_month,
        },
        "revenue": {
            "this_month": round(revenue_this_month, 2),
            "total":      round(revenue_total, 2),
        },
        "avg_score":     round(float(avg_score or 0), 1),
        "free_count":    free_count,
        "paid_count":    paid_count,
        "popular_roles": [{"role": r, "count": c} for r, c in roles.fetchall()],
    }


@router.get("/users")
async def get_users(
    page:  int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=PAGE_LIMIT_MAX),
    admin: User = Depends(get_admin_user),
    db:    AsyncSession = Depends(get_db),
):
    # Per-user interview stats in one grouped subquery, not three queries per user
    stats = (
        select(
            InterviewSession.user_id.label("user_id"),
            func.count(InterviewSession.id).filter(COMPLETED).label("interviews"),
            func.avg(InterviewSession.overall_score).label("avg_score"),
            func.max(InterviewSession.created_at).label("last_active"),
        )
        .group_by(InterviewSession.user_id)
        .subquery()
    )
    result = await db.execute(
        select(User, stats.c.interviews, stats.c.avg_score, stats.c.last_active)
        .outerjoin(stats, stats.c.user_id == User.id)
        .order_by(desc(User.created_at))
        .offset((page - 1) * limit)
        .limit(limit)
    )
    users = [
        {
            "id":          str(u.id),
            "name":        u.name,
            "email":       u.email,
            "is_admin":    bool(u.is_admin),
            "created_at":  u.created_at.isoformat() if u.created_at else None,
            "interviews":  interviews or 0,
            "avg_score":   round(float(avg or 0), 1),
            "last_active": last.isoformat() if last else None,
        }
        for u, interviews, avg, last in result.all()
    ]
    total = await db.scalar(select(func.count(User.id)))
    return {"users": users, "total": total, "page": page, "pages": _pages(total, limit)}


@router.get("/interviews")
async def get_interviews(
    page:  int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=PAGE_LIMIT_MAX),
    admin: User = Depends(get_admin_user),
    db:    AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(InterviewSession, User.name, User.email)
        .join(User, InterviewSession.user_id == User.id)
        .order_by(desc(InterviewSession.created_at))
        .offset((page - 1) * limit)
        .limit(limit)
    )
    interviews = [
        {
            "id":               str(s.id),
            "user_name":        name,
            "user_email":       email,
            "role":             s.role,
            "level":            s.level,
            "focus":            s.focus,
            "status":           s.status,
            "is_free":          s.is_free,
            "paid":             s.paid,
            "amount_pence":     s.amount_pence,
            "overall_score":    s.overall_score,
            "duration_seconds": s.duration_seconds,
            "voice":            s.voice,
            "created_at":       s.created_at.isoformat() if s.created_at else None,
        }
        for s, name, email in result.all()
    ]
    total = await db.scalar(select(func.count(InterviewSession.id)))
    return {"interviews": interviews, "total": total, "page": page, "pages": _pages(total, limit)}


def _admin_safe_feedback(feedback: dict | None) -> dict | None:
    """Scores and written feedback only. Each question's "evidence" quotes the
    candidate's answer verbatim — transcript content — so it is removed."""
    if not feedback:
        return None
    return {
        **feedback,
        "questions": [
            {k: v for k, v in q.items() if k != "evidence"}
            for q in feedback.get("questions", [])
        ],
    }


@router.get("/interviews/{interview_id}")
async def get_interview_detail(
    interview_id: str,
    admin: User = Depends(get_admin_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Admin view of a single interview.
    Returns scores, feedback and question
    breakdown. Never returns transcript,
    CV text or raw audio.
    """
    result = await db.execute(
        select(InterviewSession, User.name, User.email)
        .join(User, InterviewSession.user_id == User.id)
        .where(InterviewSession.id == parse_uuid(interview_id))
    )
    row = result.first()
    if not row:
        raise HTTPException(404, "Interview not found")

    session, user_name, user_email = row
    questions = json.loads(session.questions_json or "[]")

    return {
        "id":               str(session.id),
        "user_name":        user_name,
        "user_email":       user_email,
        "role":             session.role,
        "level":            session.level,
        "focus":            session.focus,
        "voice":            session.voice,
        "duration_minutes": session.duration_minutes,
        "duration_seconds": session.duration_seconds,
        "status":           session.status,
        "is_free":          session.is_free,
        "paid":             session.paid,
        "amount_pence":     session.amount_pence,
        "overall_score":    session.overall_score,
        "feedback":         _admin_safe_feedback(json.loads(session.feedback_json) if session.feedback_json else None),
        # Only the question text and labels — no follow-ups or ideal-answer keywords
        "questions": [
            {
                "question":   q.get("question", ""),
                "topic":      q.get("topic", ""),
                "type":       q.get("type", ""),
                "difficulty": q.get("difficulty", ""),
            }
            for q in questions
        ],
        "created_at": session.created_at.isoformat() if session.created_at else None,
        "started_at": session.started_at.isoformat() if session.started_at else None,
        "ended_at":   session.ended_at.isoformat() if session.ended_at else None,
    }


@router.get("/revenue")
async def get_revenue(
    admin: User = Depends(get_admin_user),
    db:    AsyncSession = Depends(get_db),
):
    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    first_day = today - timedelta(days=REVENUE_DAYS - 1)

    # One grouped query for the 30 days (UTC), then fill in the empty days
    day = cast(func.timezone("UTC", InterviewSession.created_at), Date)
    rows = await db.execute(
        select(day.label("day"), func.sum(InterviewSession.amount_pence), func.count(InterviewSession.id))
        .where(SOLD, InterviewSession.created_at >= first_day)
        .group_by("day")
    )
    by_day = {d: (pence or 0, count) for d, pence, count in rows.all()}
    daily = []
    for i in range(REVENUE_DAYS):
        d = (first_day + timedelta(days=i)).date()
        pence, count = by_day.get(d, (0, 0))
        daily.append({"date": d.strftime("%d %b"), "revenue": round(pence / 100, 2), "payments": count})

    tier_rows = await db.execute(
        select(InterviewSession.duration_minutes, func.count(InterviewSession.id))
        .where(SOLD)
        .group_by(InterviewSession.duration_minutes)
    )
    tier_counts = dict(tier_rows.all())
    tiers = [{"label": label, "count": tier_counts.get(mins, 0)} for mins, label in DURATION_TIERS]

    return {"daily": daily, "tiers": tiers}
