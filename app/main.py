import logging
import os
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration
from sentry_sdk.scrubber import DEFAULT_DENYLIST, EventScrubber
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.config import get_settings
from app.db.migrations import run_migrations
from app.api.v1 import auth, sessions, interviews, admin, translate
from app.core.rate_limit import limiter, rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

settings = get_settings()

# Error tracking — set up like Edquis: off until SENTRY_DSN is set, 10% of
# requests traced. Request bodies here carry CVs, transcripts and answers as
# well as passwords and tokens, so those fields are scrubbed before sending.
SENTRY_SCRUBBED_FIELDS = [
    "refresh_token", "access_token", "client_secret",
    "cv_text", "transcript_json", "transcript", "answer", "question",
    "job_description", "custom_prompt", "stripe-signature",
    # Tranquis — what users translate, say or photograph
    "text", "source_text", "translated_text", "audio_base64", "image_base64",
    "transcription", "translation",
]
sentry_dsn = os.getenv("SENTRY_DSN", "")
if sentry_dsn:
    sentry_sdk.init(
        dsn=sentry_dsn,
        integrations=[FastApiIntegration(), SqlalchemyIntegration()],
        traces_sample_rate=0.1,
        environment=os.getenv("SENTRY_ENVIRONMENT") or os.getenv("ENVIRONMENT", "development"),
        send_default_pii=False,
        # Frame variables hold raw request headers, CVs and transcripts that the
        # scrubber can't recognise — keep stack traces, drop the variables
        include_local_variables=False,
        event_scrubber=EventScrubber(denylist=DEFAULT_DENYLIST + SENTRY_SCRUBBED_FIELDS, recursive=True),
    )

# App loggers (email delivery, feedback jobs, rate limits) go to stdout for Render's log view
logging.basicConfig(level=logging.INFO, format="%(levelname)s:     %(name)s - %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Migrations own the schema: create_all never adds columns to existing tables
    await run_migrations()
    yield


app = FastAPI(
    title="CareerMind API",
    version="1.0.0",
    description="AI-powered voice interview coaching",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)

# Local dev servers are only allowed when the configured frontend is itself local
_dev_origins = (
    ["http://localhost:3000", "http://localhost:3001"]
    if settings.frontend_url.startswith("http://localhost") else []
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_url, *_dev_origins],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router,      prefix="/api/v1")
app.include_router(sessions.router,  prefix="/api/v1")
app.include_router(interviews.router, prefix="/api/v1")
app.include_router(admin.router,      prefix="/api/v1")
app.include_router(translate.router,  prefix="/api/v1")


@app.get("/health")
async def health():
    return {"status": "ok", "service": "careermind-api"}


@app.get("/")
async def root():
    return {"message": "CareerMind API", "docs": "/docs"}
