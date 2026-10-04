from collections.abc import Iterator
from pathlib import Path

from sqlalchemy import Engine
from sqlmodel import Session, SQLModel, create_engine, select

from app.config import DATABASE_URL
from app.models import GameType
from app.rating.margin import FOOSBALL


def make_engine(url: str) -> Engine:
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    return create_engine(url, connect_args=connect_args)


engine = make_engine(DATABASE_URL)


def init_db(engine: Engine) -> None:
    """Create the tables and the Foosball game type if they don't exist yet."""
    if engine.url.get_backend_name() == "sqlite" and engine.url.database not in (None, ":memory:"):
        Path(engine.url.database).parent.mkdir(parents=True, exist_ok=True)
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        if session.exec(select(GameType)).first() is None:
            session.add(GameType.from_settings("Foosball", FOOSBALL))
            session.commit()


def get_session() -> Iterator[Session]:
    with Session(engine) as session:
        yield session
