from fastapi import Request
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from app.core.auth import decode_token

# Per-instance, in-memory limits. They cap OpenAI spend and brute force on a
# single instance; move storage to Redis if the API is scaled out.
REGISTER_LIMIT       = "5/hour"
LOGIN_LIMIT          = "10/minute"
SETUP_LIMIT          = "15/hour"      # each setup is a question-generation call
UPLOAD_CV_LIMIT      = "10/hour"
REALTIME_TOKEN_LIMIT = "10/hour"      # each token opens a voice session
COACHING_LIMIT       = "120/hour"     # one per answer; a 60-minute interview stays well under
STUDY_LIMIT          = "10/hour"
CHECKOUT_LIMIT       = "20/hour"
VERIFY_EMAIL_LIMIT   = "20/hour"      # per IP — the link may be opened on another device
RESEND_VERIFY_LIMIT  = "3/hour"       # each one sends an email
TRANSLATE_LIMIT      = "200/hour"     # each one is a GPT-4o call
TRANSLATE_IMAGE_LIMIT = "40/hour"     # vision calls cost more
TRANSCRIBE_LIMIT     = "10/minute"    # audio is expensive — per user
TTS_LIMIT            = "10/minute"


def client_ip(request: Request) -> str:
    # Behind Render's proxy request.client is the proxy. The proxy appends the
    # address it saw to X-Forwarded-For; earlier entries are client-supplied
    # and could be spoofed, so use the last one.
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[-1].strip()
    return request.client.host if request.client else "unknown"


def user_or_ip(request: Request) -> str:
    """Limit signed-in endpoints per user, so people sharing an IP don't block each other."""
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        payload = decode_token(auth[7:])
        if payload and payload.get("sub"):
            return f"user:{payload['sub']}"
    return f"ip:{client_ip(request)}"


limiter = Limiter(key_func=user_or_ip)


async def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    # Shape matches FastAPI errors so the frontend shows the message
    return JSONResponse(
        status_code=429,
        content={"detail": "Too many requests — please wait a few minutes and try again."},
    )
