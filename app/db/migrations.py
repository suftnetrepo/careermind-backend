"""Apply Alembic migrations at startup, so every deploy brings the schema up to date."""
import logging
from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from sqlalchemy import inspect

from app.db.engine import engine

logger = logging.getLogger(__name__)

BACKEND_DIR = Path(__file__).resolve().parents[2]
# Databases created before Alembic (by create_all) match this revision
BASELINE_REVISION = "1b96b0538a36"


def _alembic_config(connection) -> Config:
    cfg = Config(str(BACKEND_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(BACKEND_DIR / "alembic"))
    cfg.attributes["connection"] = connection
    cfg.attributes["configure_logger"] = False   # keep the app's logging setup
    return cfg


def migrate(connection) -> None:
    """Bring the database on `connection` to the latest revision."""
    cfg = _alembic_config(connection)
    tables = inspect(connection).get_table_names()
    current = MigrationContext.configure(connection).get_current_revision()
    if "users" in tables and current is None:
        # Tables exist but Alembic has never run here — record the baseline
        # instead of trying to create tables that already exist
        logger.warning("Database has tables but no Alembic version; stamping baseline %s", BASELINE_REVISION)
        command.stamp(cfg, BASELINE_REVISION)
    command.upgrade(cfg, "head")
    logger.info("Database schema at %s", MigrationContext.configure(connection).get_current_revision())


async def run_migrations() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(migrate)
