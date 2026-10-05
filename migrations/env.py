"""Runs migrations: from the app at startup (app.db.migrate) or from the alembic command."""

from alembic import context
from sqlalchemy import Connection, create_engine
from sqlmodel import SQLModel

import app.models  # noqa: F401  (registers the tables on SQLModel.metadata)
from app.config import DATABASE_URL


def run(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=SQLModel.metadata,
        render_as_batch=connection.dialect.name == "sqlite",  # SQLite can't ALTER most things
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


shared = context.config.attributes.get("connection")
if shared is not None:
    run(shared)
else:
    with create_engine(DATABASE_URL).connect() as connection:
        run(connection)
