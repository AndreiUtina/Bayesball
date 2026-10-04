import pytest
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, create_engine

from app.db import get_session, init_db
from app.main import app


@pytest.fixture
def client():
    """A test client backed by a fresh in-memory database with the Foosball game type."""
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    init_db(engine)

    def session_override():
        with Session(engine) as session:
            yield session

    app.dependency_overrides[get_session] = session_override
    yield TestClient(app)
    app.dependency_overrides.clear()
