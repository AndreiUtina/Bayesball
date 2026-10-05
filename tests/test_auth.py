import re
from datetime import timedelta

import pytest
from sqlmodel import Session, select

from app.auth import owner_setup_link
from app.models import Game, Invite, Player, utcnow

TWO_V_TWO = {
    "game_type_id": 1,
    "format": "2v2",
    "a": ["1", "2"],
    "b": ["3", "4"],
    "row": ["0"],
    "score": ["10-4"],
}


@pytest.fixture
def four(client):
    for name in ("ann", "bob", "cat", "dan"):
        client.post("/players/new", data={"game_type_id": 1, "name": name})


def invite_token(client) -> str:
    page = client.post("/admin/users/invite").text
    return re.search(r"/invite/([\w-]+)", page)[1]


# --- Visitors ----------------------------------------------------------------------------------


def test_visitors_can_view_everything(visitor, client, four):
    client.post("/games/new", data=TWO_V_TWO)
    for path in ("/", "/predict", "/balance", "/games", "/players/1", "/api/players", "/docs"):
        assert visitor.get(path).status_code == 200, path
    nav = visitor.get("/").text
    assert "Log in" in nav and "Record game" not in nav and "Edit" not in visitor.get("/games").text


@pytest.mark.parametrize("path", ["/players/new", "/games/new", "/games/1/edit", "/players/1/edit"])
def test_visitors_are_sent_to_log_in(visitor, path):
    response = visitor.get(path, follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"] == f"/login?next={path}"


def test_login_redirect_keeps_the_query(visitor):
    response = visitor.get("/games/new?format=1v1&a=2", follow_redirects=False)
    assert response.headers["location"] == "/login?next=/games/new%3Fformat%3D1v1%26a%3D2"


def test_visitors_cannot_write(visitor, client, four):
    response = visitor.post("/games/new", data=TWO_V_TWO, follow_redirects=False)
    assert response.headers["location"].startswith("/login")
    assert visitor.post("/api/players", json={"name": "eve"}).status_code == 401
    assert visitor.delete("/api/games/1").status_code == 401
    assert client.get("/api/games").json() == []


def test_admins_cannot_manage_admins(new_client, add_user, login):
    add_user("ada")
    admin = new_client()
    login(admin, "ada")
    assert admin.get("/admin/users").status_code == 403
    assert admin.post("/admin/users/invite").status_code == 403
    assert admin.patch("/api/game-types/1", json={"beta": 1}).status_code == 403
    assert admin.post("/api/players", json={"name": "eve"}).status_code == 201
    assert "Admins" not in admin.get("/").text


# --- Logging in --------------------------------------------------------------------------------


def test_login_and_logout(visitor, add_user):
    add_user("ada")
    wrong = visitor.post("/login", data={"username": "ada", "password": "nope"})
    assert wrong.status_code == 401 and "wrong username or password" in wrong.text

    response = visitor.post(
        "/login",
        data={"username": "ADA", "password": "correct horse battery", "next": "/games/new"},
        follow_redirects=False,
    )
    assert response.headers["location"] == "/games/new"  # usernames ignore case
    assert visitor.get("/games/new").status_code == 200

    visitor.post("/logout")
    assert visitor.get("/games/new", follow_redirects=False).status_code == 303


def test_login_only_redirects_within_the_site(visitor, add_user):
    add_user("ada")
    data = {"username": "ada", "password": "correct horse battery", "next": "//evil.example"}
    response = visitor.post("/login", data=data, follow_redirects=False)
    assert response.headers["location"] == "/"


def test_too_many_failed_logins(visitor, add_user):
    add_user("ada")
    for _ in range(5):
        visitor.post("/login", data={"username": "ada", "password": "guess"})
    right = visitor.post("/login", data={"username": "ada", "password": "correct horse battery"})
    assert right.status_code == 401 and "too many failed logins" in right.text


# --- Invites and removing admins ---------------------------------------------------------------


def test_invite_flow(client, new_client):
    token = invite_token(client)
    assert "Unused invite links" in client.get("/admin/users").text

    newcomer = new_client()
    assert "Join as an admin" in newcomer.get(f"/invite/{token}").text
    data = {"username": "ada", "password": "a long password", "password2": "a long password"}
    response = newcomer.post(f"/invite/{token}", data=data, follow_redirects=False)
    assert response.status_code == 303
    assert newcomer.get("/games/new").status_code == 200  # logged in straight away

    again = new_client().post(f"/invite/{token}", data={**data, "username": "eve"})
    assert again.status_code == 404 and "not valid any more" in again.text
    assert "Unused invite links" not in client.get("/admin/users").text


@pytest.mark.parametrize(
    ("username", "password", "password2", "message"),
    [
        ("ada", "a long password", "a long passwort", "don&#39;t match"),
        ("ada", "short", "short", "at least 10 characters"),
        ("owner", "a long password", "a long password", "is taken"),
        ("a d", "a long password", "a long password", "usernames are"),
    ],
)
def test_invite_rules(client, new_client, username, password, password2, message):
    token = invite_token(client)
    data = {"username": username, "password": password, "password2": password2}
    response = new_client().post(f"/invite/{token}", data=data)
    assert response.status_code == 422 and message in response.text


def test_expired_and_cancelled_invites(client, new_client, engine):
    token = invite_token(client)
    with Session(engine) as session:
        invite = session.exec(select(Invite)).one()
        invite.expires_at = utcnow() - timedelta(minutes=1)
        session.add(invite)
        session.commit()
    assert new_client().get(f"/invite/{token}").status_code == 404

    token = invite_token(client)
    with Session(engine) as session:
        pending = session.exec(select(Invite).where(Invite.used_at.is_(None))).all()
    client.post(f"/admin/invites/{pending[-1].id}/cancel")
    assert new_client().get(f"/invite/{token}").status_code == 404


def test_removing_an_admin_logs_them_out(client, new_client, add_user, login):
    ada_id = add_user("ada")
    admin = new_client()
    login(admin, "ada")
    assert admin.get("/games/new").status_code == 200

    client.post(f"/admin/users/{ada_id}/remove")
    assert admin.get("/games/new", follow_redirects=False).status_code == 303
    again = new_client().post(
        "/login", data={"username": "ada", "password": "correct horse battery"}
    )
    assert again.status_code == 401
    assert "removed" in client.get("/admin/users").text


def test_owner_setup_link(engine):
    with Session(engine) as session:
        first = owner_setup_link(session)
        second = owner_setup_link(session)  # e.g. after a restart
        assert first and second and first != second
        assert len(session.exec(select(Invite)).all()) == 1  # the old link stopped working


def test_owner_setup_flow(engine, new_client):
    with Session(engine) as session:
        link = owner_setup_link(session)
    token = link.rsplit("/", 1)[1]
    owner = new_client()
    assert "Create the owner account" in owner.get(f"/invite/{token}").text
    data = {"username": "andrei", "password": "a long password", "password2": "a long password"}
    owner.post(f"/invite/{token}", data=data)
    assert owner.get("/admin/users").status_code == 200
    with Session(engine) as session:
        assert owner_setup_link(session) is None


# --- Audit trail, CSRF and headers -------------------------------------------------------------


def test_audit_trail(client, new_client, add_user, login, engine, four):
    add_user("ada")
    admin = new_client()
    login(admin, "ada")
    admin.post("/games/new", data=TWO_V_TWO)
    assert "last edited" not in client.get("/games/1/edit").text
    client.post("/games/1/edit", data={"a": ["1", "2"], "b": ["3", "4"], "score": "10-5"})
    with Session(engine) as session:
        game = session.get(Game, 1)
        player = session.get(Player, 1)
        assert (game.created_by, game.updated_by) == (2, 1)  # ada recorded it, the owner edited
        assert player.created_by == 1
    page = client.get("/games/1/edit").text
    assert "recorded by ada" in page and "last edited by owner" in page


def test_cross_site_requests_are_blocked(client, four):
    evil = client.post("/games/new", data=TWO_V_TWO, headers={"Origin": "https://evil.example"})
    assert evil.status_code == 403
    fetch = client.post(
        "/api/players", json={"name": "x"}, headers={"Sec-Fetch-Site": "cross-site"}
    )
    assert fetch.status_code == 403
    ours = client.post("/games/new", data=TWO_V_TWO, headers={"Origin": "http://testserver"})
    assert ours.status_code == 200 and "beat" in ours.text


def test_security_headers_and_robots(visitor):
    response = visitor.get("/")
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert '<meta name="robots" content="noindex">' in response.text
    assert "integrity=" in response.text
    assert visitor.get("/robots.txt").text == "User-agent: *\nDisallow: /\n"
