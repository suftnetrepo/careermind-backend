from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel
from app.db.engine import get_db
from app.db.models import User, InterviewSession, InterviewStatus
from app.core.deps import get_current_user
from app.config import get_settings
import stripe
import uuid

router = APIRouter(prefix="/sessions", tags=["Sessions"])
settings = get_settings()

RATE_PENCE_PER_MINUTE = 20   # £0.20 per minute
MIN_MINUTES = 10
MAX_MINUTES = 60


class CreateCheckoutRequest(BaseModel):
    interview_id:     str
    duration_minutes: int


@router.post("/checkout")
async def create_checkout(
    req: CreateCheckoutRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if not MIN_MINUTES <= req.duration_minutes <= MAX_MINUTES:
        raise HTTPException(400, f"Duration must be between {MIN_MINUTES} and {MAX_MINUTES} minutes")

    # Verify interview belongs to user and is in setup state
    result = await db.execute(
        select(InterviewSession).where(
            InterviewSession.id == uuid.UUID(req.interview_id),
            InterviewSession.user_id == user.id,
        )
    )
    interview = result.scalar_one_or_none()
    if not interview:
        raise HTTPException(404, "Interview not found")
    if interview.status != InterviewStatus.setup:
        raise HTTPException(400, "Interview already started or completed")
    if interview.paid:
        raise HTTPException(400, "Interview already paid")

    amount_pence = req.duration_minutes * RATE_PENCE_PER_MINUTE
    amount_pounds = f"£{amount_pence / 100:.2f}"

    if not settings.stripe_secret_key:
        raise HTTPException(503, "Payment system not configured")

    stripe.api_key = settings.stripe_secret_key

    checkout = stripe.checkout.Session.create(
        payment_method_types=["card"],
        line_items=[{
            "price_data": {
                "currency":     "gbp",
                "unit_amount":  amount_pence,
                "product_data": {
                    "name":        f"CareerMind — {req.duration_minutes} minute interview",
                    "description": f"{req.duration_minutes} minutes · {interview.role} · {interview.level}",
                },
            },
            "quantity": 1,
        }],
        mode="payment",
        success_url=f"{settings.frontend_url}/interview?id={req.interview_id}&payment=success",
        cancel_url=f"{settings.frontend_url}/preview?id={req.interview_id}&payment=cancelled",
        metadata={
            "interview_id":     req.interview_id,
            "user_id":          str(user.id),
            "duration_minutes": str(req.duration_minutes),
            "amount_pence":     str(amount_pence),
        },
        customer_email=user.email,
    )

    # Save stripe session id on interview
    interview.stripe_session_id = checkout.id
    interview.amount_pence      = amount_pence
    await db.commit()

    return {
        "checkout_url":     checkout.url,
        "amount_pence":     amount_pence,
        "amount_display":   amount_pounds,
        "duration_minutes": req.duration_minutes,
    }


@router.post("/webhook")
async def stripe_webhook(request: Request, db: AsyncSession = Depends(get_db)):
    payload    = await request.body()
    sig_header = request.headers.get("stripe-signature", "")

    if not settings.stripe_webhook_secret:
        raise HTTPException(503, "Webhook not configured")

    stripe.api_key = settings.stripe_secret_key

    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, settings.stripe_webhook_secret
        )
    except stripe.error.SignatureVerificationError:
        raise HTTPException(400, "Invalid signature")

    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]
        meta    = session.get("metadata", {})
        interview_id     = meta.get("interview_id")
        duration_minutes = int(meta.get("duration_minutes", 0))
        amount_pence     = int(meta.get("amount_pence", 0))

        if interview_id and session.get("payment_status") == "paid":
            result = await db.execute(
                select(InterviewSession).where(
                    InterviewSession.id == uuid.UUID(interview_id)
                )
            )
            interview = result.scalar_one_or_none()
            if interview and not interview.paid:
                interview.paid                  = True
                interview.duration_minutes      = duration_minutes
                interview.amount_pence          = amount_pence
                interview.stripe_payment_intent = session.get("payment_intent")
                await db.commit()

    return {"received": True}


@router.get("/pricing")
async def get_pricing():
    """Return pricing info for the frontend slider."""
    return {
        "rate_pence_per_minute": RATE_PENCE_PER_MINUTE,
        "min_minutes":           MIN_MINUTES,
        "max_minutes":           MAX_MINUTES,
        "examples": [
            {"minutes": 10, "pence": 200,  "display": "£2.00"},
            {"minutes": 15, "pence": 300,  "display": "£3.00"},
            {"minutes": 20, "pence": 400,  "display": "£4.00"},
            {"minutes": 30, "pence": 600,  "display": "£6.00"},
            {"minutes": 60, "pence": 1200, "display": "£12.00"},
        ],
    }
