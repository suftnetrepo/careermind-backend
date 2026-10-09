import uuid
import enum
from sqlalchemy import (
    Column, String, Boolean, Integer, DateTime,
    Text, ForeignKey, Enum as SAEnum, Date, UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import false, func
from app.db.engine import Base


class UserStatus(str, enum.Enum):
    active    = "active"
    suspended = "suspended"


class InterviewStatus(str, enum.Enum):
    setup     = "setup"
    active    = "active"
    completed = "completed"
    abandoned = "abandoned"


class User(Base):
    __tablename__ = "users"

    id              = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email           = Column(String, unique=True, nullable=False, index=True)
    name            = Column(String, nullable=False)
    hashed_password = Column(String, nullable=False)
    status          = Column(SAEnum(UserStatus), default=UserStatus.active)
    free_minutes    = Column(Integer, default=10)  # one free interview on signup
    email_verified  = Column(Boolean, default=False)
    is_admin        = Column(Boolean, nullable=False, default=False, server_default=false())
    # Which app registered this user: 'careermind' | 'tranquis'
    app_source      = Column(String, nullable=False, default="careermind", server_default="careermind")
    created_at      = Column(DateTime(timezone=True), server_default=func.now())
    updated_at      = Column(DateTime(timezone=True), onupdate=func.now())

    @property
    def has_free_interview(self) -> bool:
        return self.free_minutes > 0


class InterviewSession(Base):
    """A single voice interview session."""
    __tablename__ = "interview_sessions"

    id               = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id          = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)

    # Setup
    role             = Column(String, nullable=False)
    level            = Column(String, nullable=False)
    focus            = Column(String, nullable=False)
    duration_minutes = Column(Integer, default=15)
    voice            = Column(String, default="alloy")   # Realtime API voice for Alex
    cv_text          = Column(Text, nullable=True)
    custom_prompt    = Column(Text, nullable=True)
    preset_prompts   = Column(Text, nullable=True)        # JSON list of preset strings
    job_description  = Column(Text, nullable=True)

    # Generated content
    questions_json   = Column(Text, nullable=True)

    # State
    status           = Column(SAEnum(InterviewStatus), default=InterviewStatus.setup)
    started_at       = Column(DateTime(timezone=True), nullable=True)
    ended_at         = Column(DateTime(timezone=True), nullable=True)
    duration_seconds = Column(Integer, nullable=True)

    # Billing
    paid                  = Column(Boolean, default=False)
    is_free               = Column(Boolean, default=False)
    amount_pence          = Column(Integer, nullable=True)
    stripe_session_id     = Column(String, nullable=True)
    stripe_payment_intent = Column(String, nullable=True)

    # Results
    transcript_json  = Column(Text, nullable=True)
    feedback_json    = Column(Text, nullable=True)
    overall_score    = Column(Integer, nullable=True)

    # Study materials (generated once after the interview, then cached)
    quiz_json          = Column(Text, nullable=True)
    flashcards_json    = Column(Text, nullable=True)
    study_generated_at = Column(DateTime(timezone=True), nullable=True)
    study_model        = Column(String, nullable=True)   # which model wrote them

    created_at       = Column(DateTime(timezone=True), server_default=func.now())


# ── Tranquis: per-user daily translation quota ────────────────────────────────

FREE_DAILY_TRANSLATIONS = 10


class TranslationQuota(Base):
    """Tracks how many text translations a free user has used today (UTC date)."""
    __tablename__ = "translation_quotas"
    __table_args__ = (UniqueConstraint("user_id", "quota_date", name="uq_translation_quota_user_date"),)

    id         = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id    = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    quota_date = Column(Date, nullable=False)           # UTC date the count belongs to
    count      = Column(Integer, nullable=False, default=0)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


# ── Password reset tokens ─────────────────────────────────────────────────────

class PasswordResetToken(Base):
    """Single-use, short-lived tokens for self-service password reset."""
    __tablename__ = "password_reset_tokens"

    id         = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id    = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    token_hash = Column(String, nullable=False, unique=True)  # SHA-256 of the raw token
    used       = Column(Boolean, nullable=False, default=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


# ── Tranquis: translation history & phrasebook ───────────────────────────────

class TranslationHistory(Base):
    """Every translation a user makes (text, voice, or camera)."""
    __tablename__ = "translation_history"

    id              = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id         = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    source_text     = Column(Text, nullable=False)
    translated_text = Column(Text, nullable=False)
    source_lang     = Column(String, nullable=False, default="auto")
    target_lang     = Column(String, nullable=False)
    mode            = Column(String, nullable=False, default="text")  # text|voice|camera
    created_at      = Column(DateTime(timezone=True), server_default=func.now())


class PhrasebookEntry(Base):
    """User-saved translations (starred from history)."""
    __tablename__ = "phrasebook_entries"

    id             = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id        = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    translation_id = Column(UUID(as_uuid=True), ForeignKey("translation_history.id", ondelete="CASCADE"), nullable=False)
    created_at     = Column(DateTime(timezone=True), server_default=func.now())
