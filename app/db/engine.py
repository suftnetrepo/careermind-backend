from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from app.config import get_settings

settings = get_settings()


def _asyncpg_url(raw: str) -> tuple[str, dict]:
    """Make a standard Postgres URL (e.g. Neon's '?sslmode=require&channel_binding=require')
    usable with asyncpg, which rejects libpq-only query options."""
    url = make_url(raw.replace("postgresql://", "postgresql+asyncpg://", 1) if raw.startswith("postgresql://") else raw)
    query = dict(url.query)
    connect_args = {}
    sslmode = query.pop("sslmode", None)
    query.pop("channel_binding", None)   # libpq-only; asyncpg has no equivalent
    if sslmode and "ssl" not in query and sslmode != "disable":
        connect_args["ssl"] = sslmode    # asyncpg accepts the same mode names
    return url.set(query=query).render_as_string(hide_password=False), connect_args


_url, _connect_args = _asyncpg_url(settings.database_url or "postgresql+asyncpg://localhost/careermind")

# Neon suspends idle computes and drops their connections: test each pooled
# connection before use and recycle long-lived ones
engine = create_async_engine(
    _url,
    connect_args=_connect_args,
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
