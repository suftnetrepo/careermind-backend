from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.config import get_settings
from app.db.engine import create_tables
from app.api.v1 import auth, sessions, interviews

settings = get_settings()


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

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_url, "http://localhost:3000"],
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
