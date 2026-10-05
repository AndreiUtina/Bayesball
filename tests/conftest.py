import os

import pytest
from argon2 import PasswordHasher
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, create_engine

from app import auth
from app.config import database_url
from app.db import get_session, init_db
from app.main import app
from app.models import User

PASSWORD = "correct horse battery"

# Set this to a Postgres database to run the tests against it (CI does); it gets wiped.
TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL")


@pytest.fixture(autouse=True)
def fast_logins(monkeypatch):
    """Cheap password hashing (the real settings are deliberately slow) and no lockouts."""
    monkeypatch.setattr(auth, "hasher", PasswordHasher(time_cost=1, memory_cost=8, parallelism=1))
    auth._dummy_hash.cache_clear()
    auth._failures.clear()


@pytest.fixture
def engine():
    """A fresh database with the Foosball game type: in memory, or TEST_DATABASE_URL."""
    if TEST_DATABASE_URL:
        engine = create_engine(database_url(TEST_DATABASE_URL))
        with engine.begin() as connection:
            connection.execute(text("drop schema public cascade"))
            connection.execute(text("create schema public"))
    else:
        engine = create_engine(
            "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
        )
    init_db(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def new_client(engine):
    """Makes logged-out test clients; they all share the test database."""

    def session_override():
        with Session(engine) as session:
            yield session

    app.dependency_overrides[get_session] = session_override
    yield lambda: TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def add_user(engine):
    def add(username: str, role: str = "admin", password: str = PASSWORD) -> int:
        with Session(engine) as session:
            user = User(username=username, password_hash=auth.hasher.hash(password), role=role)
            session.add(user)
            session.commit()
            return user.id

    return add


def log_in(client: TestClient, username: str, password: str = PASSWORD) -> None:
    response = client.post(
        "/login", data={"username": username, "password": password}, follow_redirects=False
    )
    assert response.status_code == 303, response.text


@pytest.fixture
def visitor(new_client):
    """A test client that isn't logged in."""
    return new_client()


@pytest.fixture
def client(new_client, add_user):
    """A test client logged in as the owner."""
    add_user("owner", role="owner")
    client = new_client()
    log_in(client, "owner")
    return client


@pytest.fixture
def login():
    """log_in(client, username, password=PASSWORD), for tests with several users."""
    return log_in
