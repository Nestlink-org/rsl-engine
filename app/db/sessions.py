"""Database session management"""

# import ssl
from typing import Generator, AsyncGenerator
from sqlmodel import Session, create_engine
from sqlmodel.ext.asyncio.session import AsyncSession 
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool
from ..core.config import settings
# ssl_context = ssl.create_default_context()

# Synchronous engine
engine = create_engine(
    settings.PSQL_URI,
    # connect_args={"ssl": ssl_context},
    echo=True,
    future=True
)

# Asynchronous engine
async_engine = create_async_engine(
    settings.PSQL_URI_ASYNC,
    # connect_args={"ssl": ssl_context},
    echo=True,
    future=True,
    poolclass=NullPool,  # This prevents connection pool conflicts
    pool_pre_ping=True   # This checks if connection is alive before using
)

# Session factories
sync_session = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

async_session = sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False
)

def get_db() -> Generator[Session, None, None]:
    """Synchronous session generator"""
    db = sync_session()
    try:
        yield db
    finally:
        db.close()

async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """Asynchronous session generator"""
    async with async_session() as session:
        try:
            yield session
        finally:
            await session.close()