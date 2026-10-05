import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.config import get_settings
from app.db.engine import create_tables
from app.api.v1 import auth, sessions, interviews
from app.core.rate_limit import limiter, rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

settings = get_settings()

# App loggers (email delivery, feedback jobs, rate limits) go to stdout for Render's log view
logging.basicConfig(level=logging.INFO, format="%(levelname)s:     %(name)s - %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await create_tables()
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


@app.get("/health")
async def health():
    return {"status": "ok", "service": "careermind-api"}


@app.get("/")
async def root():
    return {"message": "CareerMind API", "docs": "/docs"}
