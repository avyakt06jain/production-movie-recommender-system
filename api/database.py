"""
Database module for the MovieRec API.

Provides:
- Async SQLAlchemy engine + session factory for FastAPI request handling
- Sync SQLAlchemy engine for scripts that need synchronous DB access
- get_db() async generator for FastAPI dependency injection
- get_sync_connection() context manager using psycopg2 for bulk loading scripts
"""

import os
from urllib.parse import urlparse

from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine

load_dotenv()

# ---------------------------------------------------------------------------
# Async engine (for FastAPI request handling via asyncpg)
# ---------------------------------------------------------------------------
DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://postgres:postgres@localhost:5432/movierec",
)

async_engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
)

AsyncSessionLocal = sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

# ---------------------------------------------------------------------------
# Sync engine (for scripts / migrations that need synchronous access)
# ---------------------------------------------------------------------------
DATABASE_URL_SYNC: str = os.getenv(
    "DATABASE_URL_SYNC",
    "postgresql://postgres:postgres@localhost:5432/movierec",
)

sync_engine = create_engine(
    DATABASE_URL_SYNC,
    echo=False,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
)


# ---------------------------------------------------------------------------
# FastAPI dependency injection
# ---------------------------------------------------------------------------
async def get_db() -> AsyncSession:
    """Yield an async SQLAlchemy session for use as a FastAPI dependency.

    Usage::

        @router.get("/items")
        async def list_items(db: AsyncSession = Depends(get_db)):
            result = await db.execute(select(Item))
            return result.scalars().all()
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# ---------------------------------------------------------------------------
# psycopg2 sync connection for bulk loading scripts
# ---------------------------------------------------------------------------
def get_sync_connection():
    """Return a raw psycopg2 connection for bulk-insert scripts.

    Parses ``DATABASE_URL_SYNC`` to extract host/port/dbname/user/password
    and opens a plain psycopg2 connection (no SQLAlchemy overhead).

    Usage::

        conn = get_sync_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
            conn.commit()
        finally:
            conn.close()
    """
    import psycopg2

    url = DATABASE_URL_SYNC
    parsed = urlparse(url)

    return psycopg2.connect(
        host=parsed.hostname or "localhost",
        port=parsed.port or 5432,
        dbname=parsed.path.lstrip("/") if parsed.path else "movierec",
        user=parsed.username or "postgres",
        password=parsed.password or "postgres",
    )
