from collections.abc import Iterator
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, inspect
from sqlmodel import Session, create_engine, select

from app.config import DATABASE_URL, ROOT
from app.models import GameType
from app.rating.margin import FOOSBALL

FIRST_MIGRATION = "2fe655bff3db"  # migrations/versions/2fe655bff3db_initial_schema.py


def make_engine(url: str) -> Engine:
    if url.startswith("sqlite"):
        return create_engine(url, connect_args={"check_same_thread": False})
    # Neon closes idle connections (it sleeps after a few minutes without queries), so check
    # each pooled connection before use and replace old ones.
    return create_engine(url, pool_pre_ping=True, pool_recycle=300)


engine = make_engine(DATABASE_URL)


def migrate(engine: Engine) -> None:
    """Bring the database tables up to date, creating them on the very first start."""
    config = Config(str(ROOT / "alembic.ini"))
    with engine.begin() as connection:
        config.attributes["connection"] = connection
        tables = set(inspect(connection).get_table_names())
        if "gametype" in tables and "alembic_version" not in tables:
            # Made before migrations existed: by the first version with logins, it already
            # matches the first migration; anything older only ever held test data.
            if "users" not in tables:
                raise RuntimeError(
                    "This database comes from an early test version of Bayesball. "
                    "Delete it (data/bayesball.db) and start again."
                )
            command.stamp(config, FIRST_MIGRATION)
        command.upgrade(config, "head")


def init_db(engine: Engine) -> None:
    """Update the tables and add the Foosball game type if there isn't one yet."""
    if engine.url.get_backend_name() == "sqlite" and engine.url.database not in (None, ":memory:"):
        Path(engine.url.database).parent.mkdir(parents=True, exist_ok=True)
    migrate(engine)
    with Session(engine) as session:
        if session.exec(select(GameType)).first() is None:
            session.add(GameType.from_settings("Foosball", FOOSBALL))
            session.commit()


def get_session() -> Iterator[Session]:
    with Session(engine) as session:
        yield session
