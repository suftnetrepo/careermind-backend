from fastapi import APIRouter, Depends, HTTPException, Request
from app.core.rate_limit import limiter, CHECKOUT_LIMIT
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel
from app.db.engine import get_db
from app.db.models import User, InterviewSession, InterviewStatus
from app.core.deps import get_current_user
from app.core.pricing import PRICES, FREE_INTERVIEW_MINUTES, INVALID_DURATION, price_display
from app.config import get_settings
from datetime import datetime, timezone
import asyncio
import logging
import stripe
import uuid

router = APIRouter(prefix="/sessions", tags=["Sessions"])
logger = logging.getLogger(__name__)
settings = get_settings()


def parse_uuid(value: str) -> uuid.UUID:
    """A malformed ID in a URL or body is a missing resource, not a server error."""
    try:
        return uuid.UUID(value)
    except (ValueError, AttributeError, TypeError):
        raise HTTPException(status_code=404, detail="Not found")


class CreateCheckoutRequest(BaseModel):
    interview_id:     str
    duration_minutes: int
    # Consumer Contracts Regulations 2013: the buyer must agree the service
    # starts immediately and that they lose the 14-day cancellation right
    consent:          bool = False


@router.post("/checkout")
@limiter.limit(CHECKOUT_LIMIT)
async def create_checkout(
    request: Request,
    req: CreateCheckoutRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if not req.consent:
        raise HTTPException(400, "Please confirm the interview starts immediately and payments are non-refundable once it has started")
    amount_pence = PRICES.get(req.duration_minutes)
    if not amount_pence:
        raise HTTPException(400, INVALID_DURATION)

    # Verify interview belongs to user and is in setup state
    result = await db.execute(
        select(InterviewSession).where(
            InterviewSession.id == parse_uuid(req.interview_id),
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

    amount_pounds = price_display(amount_pence)

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
                    "name":        f"Interquis — {req.duration_minutes} minute interview",
                    "description": f"{req.duration_minutes} minutes · {interview.role} · {interview.level}",
                },
            },
            "quantity": 1,
        }],
        mode="payment",
        success_url=f"{settings.frontend_url}/payment-success?id={req.interview_id}",
        cancel_url=f"{settings.frontend_url}/preview?id={req.interview_id}&payment=cancelled",
        metadata={
            "interview_id":     req.interview_id,
            "user_id":          str(user.id),
            "duration_minutes": str(req.duration_minutes),
            "amount_pence":     str(amount_pence),
            # Evidence of the cancellation-right waiver, kept on the Stripe session
            "consent_immediate_start_at": datetime.now(timezone.utc).isoformat(),
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


def apply_paid_checkout(interview: InterviewSession, checkout) -> bool:
    """Mark the interview paid from a completed Stripe checkout session.
    Returns True if anything changed. Shared by the webhook and the
    check-ready fallback, so whichever sees the payment first applies it."""
    if interview.paid or checkout.get("payment_status") != "paid":
        return False
    meta = checkout.get("metadata") or {}
    if meta.get("interview_id") != str(interview.id):
        return False
    interview.paid                  = True
    interview.duration_minutes      = int(meta.get("duration_minutes", 0)) or interview.duration_minutes
    interview.amount_pence          = int(meta.get("amount_pence", 0)) or interview.amount_pence
    interview.stripe_payment_intent = checkout.get("payment_intent")
    return True


async def sync_checkout_payment(interview: InterviewSession) -> bool:
    """Ask Stripe directly whether this interview's checkout was paid — the
    fallback for a webhook that is late or never arrives. Never raises."""
    if interview.paid or not interview.stripe_session_id or not settings.stripe_secret_key:
        return False
    try:
        stripe.api_key = settings.stripe_secret_key
        checkout = await asyncio.to_thread(stripe.checkout.Session.retrieve, interview.stripe_session_id)
    except Exception:
        logger.warning("Could not retrieve Stripe session %s", interview.stripe_session_id, exc_info=True)
        return False
    return apply_paid_checkout(interview, checkout)


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
        interview_id = session.get("metadata", {}).get("interview_id")

        if interview_id and session.get("payment_status") == "paid":
            try:
                interview_uuid = uuid.UUID(interview_id)
            except ValueError:
                # Acknowledge anyway: a non-2xx makes Stripe retry the event for days
                logger.error("Stripe webhook with malformed interview_id %r (session %s)", interview_id, session.get("id"))
                return {"received": True}
            result = await db.execute(
                select(InterviewSession).where(
                    InterviewSession.id == interview_uuid
                )
            )
            interview = result.scalar_one_or_none()
            if interview and apply_paid_checkout(interview, session):
                await db.commit()

    return {"received": True}


@router.get("/pricing")
async def get_pricing():
    """Session prices and the free interview length."""
    return {
        "prices": [
            {"minutes": minutes, "pence": pence, "display": price_display(pence)}
            for minutes, pence in PRICES.items()
        ],
        "free_minutes": FREE_INTERVIEW_MINUTES,
    }
