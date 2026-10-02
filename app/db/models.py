import uuid
import enum
from sqlalchemy import (
    Column, String, Boolean, Integer, DateTime,
    Text, ForeignKey, Enum as SAEnum,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
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
    free_minutes    = Column(Integer, default=15)  # one free interview on signup
    email_verified  = Column(Boolean, default=False)
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

    created_at       = Column(DateTime(timezone=True), server_default=func.now())
