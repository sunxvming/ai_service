"""
Database connection management for camera storage.
"""

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base

Base = declarative_base()

# Global engine and session factory
_engine = None
_async_session_factory = None


def init_database(db_path: str):
    """
    Initialize database engine and session factory.

    Args:
        db_path: Path to SQLite database file
    """
    global _engine, _async_session_factory

    database_url = f"sqlite+aiosqlite:///{db_path}"

    _engine = create_async_engine(
        database_url,
        echo=False,
        future=True,
    )

    _async_session_factory = async_sessionmaker(
        _engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )



async def create_tables():
    """Create all database tables."""
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


def get_session() -> AsyncSession:
    """Get database session."""
    if _async_session_factory is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")
    return _async_session_factory()


async def close_database():
    """Close database engine."""
    if _engine:
        await _engine.dispose()
