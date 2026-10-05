from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from app.config import get_settings

settings = get_settings()

_url = settings.database_url or "postgresql+asyncpg://localhost/careermind"
if _url.startswith("postgresql://"):
    _url = _url.replace("postgresql://", "postgresql+asyncpg://", 1)

# Neon suspends idle computes and drops their connections: test each pooled
# connection before use and recycle long-lived ones
engine = create_async_engine(
    _url,
    echo=settings.debug,
    pool_pre_ping=True,
    pool_recycle=300,
)

AsyncSessionLocal = async_sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)


class Base(DeclarativeBase):
    pass


async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


async def create_tables():
    async with engine.begin() as conn:
        from app.db import models  # noqa
        await conn.run_sync(Base.metadata.create_all)
