from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel
from app.db.engine import get_db
from app.db.models import User, SessionPack, PackStatus
from app.core.deps import get_current_user
from app.config import get_settings
import stripe
import uuid
from datetime import datetime, timezone

router = APIRouter(prefix="/sessions", tags=["Sessions"])
settings = get_settings()

PACKS = {
    "1":  {"sessions": 1,  "amount_pence": 299,  "label": "1 session"},
    "5":  {"sessions": 5,  "amount_pence": 1199, "label": "5 sessions"},
    "10": {"sessions": 10, "amount_pence": 1999, "label": "10 sessions"},
}


class CreateCheckoutRequest(BaseModel):
    pack: str   # "1", "5", or "10"


@router.get("/balance")
async def get_balance(user: User = Depends(get_current_user)):
    return {
        "sessions_remaining": user.sessions_remaining,
        "free_sessions":      user.free_sessions,
        "paid_sessions":      user.paid_sessions,
    }


@router.post("/checkout")
async def create_checkout(
    req: CreateCheckoutRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if req.pack not in PACKS:
        raise HTTPException(400, "Invalid pack. Choose 1, 5, or 10.")

    pack_info = PACKS[req.pack]

    if not settings.stripe_secret_key:
        raise HTTPException(503, "Payment system not configured")

    stripe.api_key = settings.stripe_secret_key

    # Create pending pack record
    pack = SessionPack(
        id=uuid.uuid4(),
        user_id=user.id,
        sessions_count=pack_info["sessions"],
        amount_pence=pack_info["amount_pence"],
        currency="gbp",
        status=PackStatus.pending,
    )
    db.add(pack)
    await db.commit()

    # Create Stripe checkout session
    checkout = stripe.checkout.Session.create(
        payment_method_types=["card"],
        line_items=[{
            "price_data": {
                "currency":     "gbp",
                "unit_amount":  pack_info["amount_pence"],
                "product_data": {"name": f"CareerMind — {pack_info['label']}"},
            },
            "quantity": 1,
        }],
        mode="payment",
        success_url=f"{settings.frontend_url}/dashboard?payment=success",
        cancel_url=f"{settings.frontend_url}/buy-sessions?payment=cancelled",
        metadata={
            "pack_id":  str(pack.id),
            "user_id":  str(user.id),
            "sessions": str(pack_info["sessions"]),
        },
        customer_email=user.email,
    )

    # Save stripe session id
    pack.stripe_session_id = checkout.id
    await db.commit()

    return {"checkout_url": checkout.url, "pack_id": str(pack.id)}


@router.post("/webhook")
async def stripe_webhook(request: Request, db: AsyncSession = Depends(get_db)):
    payload = await request.body()
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
        meta = session.get("metadata", {})
        pack_id = meta.get("pack_id")
        sessions_count = int(meta.get("sessions", 0))
        user_id = meta.get("user_id")

        if pack_id and user_id and sessions_count:
            # Update pack status
            result = await db.execute(
                select(SessionPack).where(
                    SessionPack.id == uuid.UUID(pack_id)
                )
            )
            pack = result.scalar_one_or_none()
            if pack and pack.status == PackStatus.pending:
                pack.status = PackStatus.completed
                pack.stripe_payment_intent = session.get("payment_intent")
                pack.completed_at = datetime.now(timezone.utc)

                # Credit the user
                user_result = await db.execute(
                    select(User).where(User.id == uuid.UUID(user_id))
                )
                user = user_result.scalar_one_or_none()
                if user:
                    user.paid_sessions += sessions_count

                await db.commit()

    return {"received": True}
