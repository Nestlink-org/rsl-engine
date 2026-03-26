import asyncio
import sys
from logging.config import fileConfig
from pathlib import Path
# import ssl

import sqlmodel
from alembic import context
from alembic.autogenerate import renderers
from sqlalchemy import pool
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel

# Ensure project root is in sys.path
root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir))

from app.core.config import settings
from app.models import *


# Alembic Config object
config = context.config

# Setup logging from alembic.ini
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Ensure AutoString migrations include import sqlmodel
@renderers.dispatch_for(sqlmodel.sql.sqltypes.AutoString)
def _render_autostring(type_, autogen_context):
    autogen_context.imports.add("import sqlmodel")
    return "sqlmodel.sql.sqltypes.AutoString()"

# ✅ Metadata for autogeneration
target_metadata = SQLModel.metadata

# --- SSL context for asyncpg ---
# ssl_context = ssl.create_default_context()

def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    context.configure(
        url=f"{settings.PSQL_URI}",
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()

async def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    connectable = create_async_engine(
        settings.PSQL_URI_ASYNC,
        # connect_args={"ssl": ssl_context},
        poolclass=pool.NullPool
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

def do_run_migrations(connection) -> None:
    """Run migrations using the given connection."""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
        include_schemas=True,
    )

    with context.begin_transaction():
        context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())

print("Loaded tables:", SQLModel.metadata.tables.keys())