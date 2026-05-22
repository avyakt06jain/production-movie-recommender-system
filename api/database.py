"""
Database module — provides both async and sync database connections.

For local development:
  - Uses SQLite (data/movierec.db) when DATABASE_URL is not set or set to 'sqlite'
  - Automatically falls back to SQLite if PostgreSQL is unavailable

For production:
  - Uses PostgreSQL via DATABASE_URL (Supabase)
  - Async engine via asyncpg

Environment variables:
  DATABASE_URL       — async PostgreSQL URL (e.g. postgresql+asyncpg://...)
  DATABASE_URL_SYNC  — sync PostgreSQL URL  (e.g. postgresql://...)
"""

import os
import sqlite3
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load .env from project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

DATABASE_URL = os.environ.get("DATABASE_URL", "")
DATABASE_URL_SYNC = os.environ.get("DATABASE_URL_SYNC", "")

# Default SQLite path for local development
SQLITE_PATH = str(_PROJECT_ROOT / "data" / "movierec.db")


# ─── Detect database backend ─────────────────────────────────────────────────
def _is_postgres() -> bool:
    """Check if we're configured for PostgreSQL."""
    return bool(DATABASE_URL) and "postgresql" in DATABASE_URL


# ─── Sync connection (for scripts like load_data.py) ─────────────────────────
def get_sync_connection():
    """Return a synchronous database connection.

    Returns a psycopg2 connection for PostgreSQL, or sqlite3 connection for SQLite.
    """
    if _is_postgres():
        try:
            import psycopg2
            # Parse the sync URL for psycopg2
            url = DATABASE_URL_SYNC or DATABASE_URL.replace("+asyncpg", "")
            conn = psycopg2.connect(url)
            conn.autocommit = False
            logger.info(f"Connected to PostgreSQL: {url.split('@')[-1] if '@' in url else 'local'}")
            return conn
        except Exception as e:
            logger.warning(f"PostgreSQL connection failed: {e}. Falling back to SQLite.")

    # SQLite fallback
    os.makedirs(os.path.dirname(SQLITE_PATH), exist_ok=True)
    conn = sqlite3.connect(SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    logger.info(f"Connected to SQLite: {SQLITE_PATH}")
    return conn


def is_sqlite(conn) -> bool:
    """Check if the connection is SQLite."""
    return isinstance(conn, sqlite3.Connection)


# ─── Async engine (for FastAPI) ──────────────────────────────────────────────
_async_engine = None
_AsyncSessionLocal = None


def get_async_engine():
    """Get or create the async SQLAlchemy engine."""
    global _async_engine
    if _async_engine is not None:
        return _async_engine

    if _is_postgres():
        from sqlalchemy.ext.asyncio import create_async_engine
        _async_engine = create_async_engine(
            DATABASE_URL,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
            echo=False,
        )
        logger.info("Created async PostgreSQL engine")
    else:
        from sqlalchemy.ext.asyncio import create_async_engine
        # Use aiosqlite for async SQLite
        sqlite_url = f"sqlite+aiosqlite:///{SQLITE_PATH}"
        _async_engine = create_async_engine(sqlite_url, echo=False)
        logger.info(f"Created async SQLite engine: {SQLITE_PATH}")

    return _async_engine


def get_async_session_factory():
    """Get or create the async session factory."""
    global _AsyncSessionLocal
    if _AsyncSessionLocal is not None:
        return _AsyncSessionLocal

    from sqlalchemy.ext.asyncio import async_sessionmaker
    engine = get_async_engine()
    _AsyncSessionLocal = async_sessionmaker(
        engine, expire_on_commit=False, autoflush=False
    )
    return _AsyncSessionLocal


async def get_db():
    """FastAPI dependency: yield an async database session."""
    SessionLocal = get_async_session_factory()
    async with SessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
