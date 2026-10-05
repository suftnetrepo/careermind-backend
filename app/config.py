from pydantic_settings import BaseSettings
from pydantic import Field, field_validator
from functools import lru_cache


class Settings(BaseSettings):
    app_name: str = "CareerMind API"
    debug: bool = False

    database_url: str = Field(default="", env="DATABASE_URL")

    jwt_secret_key: str = Field(default="change-me-in-production", env="JWT_SECRET_KEY")
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60
    refresh_token_expire_days: int = 30

    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
    openai_model: str = "gpt-4o"
    openai_realtime_model: str = Field(default="gpt-realtime", env="OPENAI_REALTIME_MODEL")

    stripe_secret_key: str = Field(default="", env="STRIPE_SECRET_KEY")
    stripe_webhook_secret: str = Field(default="", env="STRIPE_WEBHOOK_SECRET")

    frontend_url: str = Field(default="http://localhost:3000", env="FRONTEND_URL")

    # Transactional email (Brevo) — same provider as Learnify
    brevo_api_key: str = Field(default="", env="BREVO_API_KEY")
    brevo_from_email: str = Field(default="", env="BREVO_FROM_EMAIL")
    brevo_from_name: str = Field(default="CareerMind", env="BREVO_FROM_NAME")
    email_verify_ttl_hours: int = 24

    # Shared secret for the scheduled /interviews/cleanup-old-data job
    cleanup_secret: str = Field(default="", env="CLEANUP_SECRET")
    api_url: str = Field(default="http://localhost:8000", env="API_URL")

    @field_validator("jwt_secret_key")
    @classmethod
    def _require_strong_jwt_secret(cls, v: str) -> str:
        # A missing or default secret would let anyone forge login tokens — refuse to start
        if v in ("", "change-me-in-production") or len(v) < 32:
            raise ValueError("JWT_SECRET_KEY must be set to a random secret of at least 32 characters")
        return v

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
