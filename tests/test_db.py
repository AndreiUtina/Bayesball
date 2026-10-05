import pytest
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import inspect, text
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine, select

from app.config import database_url
from app.db import init_db
from app.models import GameType, Player


def memory_engine():
    return create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )


def test_migrations_match_the_models(engine):
    """Fails if app/models.py changed without a new migration (alembic revision --autogenerate)."""
    with engine.connect() as connection:
        context = MigrationContext.configure(connection, opts={"compare_type": True})
        assert compare_metadata(context, SQLModel.metadata) == []


def test_init_db_is_repeatable(engine):
    init_db(engine)  # a restart: nothing to migrate, Foosball not added twice
    with Session(engine) as session:
        assert len(session.exec(select(GameType)).all()) == 1


def test_database_from_before_migrations_is_adopted():
    engine = memory_engine()
    SQLModel.metadata.create_all(engine)  # how the first version with logins made its tables
    with Session(engine) as session:
        session.add(Player(name="ann"))
        session.commit()
    init_db(engine)
    with engine.connect() as connection:
        assert "alembic_version" in inspect(connection).get_table_names()
        assert connection.execute(text("select name from player")).scalar() == "ann"


def test_older_test_database_is_refused():
    engine = memory_engine()
    with engine.begin() as connection:
        connection.execute(text("create table gametype (id integer primary key)"))
    with pytest.raises(RuntimeError, match="Delete it"):
        init_db(engine)


@pytest.mark.parametrize(
    ("given", "used"),
    [
        (
            "postgres://u:p@host/db?sslmode=require",
            "postgresql+psycopg://u:p@host/db?sslmode=require",
        ),
        ("postgresql://u:p@host/db", "postgresql+psycopg://u:p@host/db"),
        ("postgresql+psycopg://u:p@host/db", "postgresql+psycopg://u:p@host/db"),
        ("sqlite:///data/bayesball.db", "sqlite:///data/bayesball.db"),
    ],
)
def test_database_url(given, used):
    assert database_url(given) == used
