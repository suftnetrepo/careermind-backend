from pydantic_settings import BaseSettings
from pydantic import Field
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

    stripe_secret_key: str = Field(default="", env="STRIPE_SECRET_KEY")
    stripe_webhook_secret: str = Field(default="", env="STRIPE_WEBHOOK_SECRET")
    stripe_price_1_session: str = Field(default="", env="STRIPE_PRICE_1_SESSION")
    stripe_price_5_sessions: str = Field(default="", env="STRIPE_PRICE_5_SESSIONS")
    stripe_price_10_sessions: str = Field(default="", env="STRIPE_PRICE_10_SESSIONS")

    frontend_url: str = Field(default="http://localhost:3000", env="FRONTEND_URL")
    api_url: str = Field(default="http://localhost:8000", env="API_URL")

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
