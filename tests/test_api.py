import pytest
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, create_engine

from app.db import get_session, init_db
from app.main import app


@pytest.fixture
def client():
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


@pytest.fixture
def gt(client):
    """The id of the seeded Foosball game type."""
    return client.get("/api/game-types").json()[0]["id"]


@pytest.fixture
def players(client):
    """Four average players: name -> id."""
    return {name: add_player(client, name)["id"] for name in ("ann", "bob", "cat", "dan")}


def add_player(client, name, **body):
    response = client.post("/api/players", json={"name": name, **body})
    assert response.status_code == 201, response.text
    return response.json()


def two_v_two(gt, ids, score_a, score_b, **extra):
    """ids = (A attack, A defence, B attack, B defence)."""
    a_att, a_def, b_att, b_def = ids
    return {
        "game_type_id": gt,
        "side_a": [{"player_id": a_att, "role": "attack"}, {"player_id": a_def, "role": "defence"}],
        "side_b": [{"player_id": b_att, "role": "attack"}, {"player_id": b_def, "role": "defence"}],
        "score_a": score_a,
        "score_b": score_b,
        **extra,
    }


def rating(client, player_id):
    return client.get(f"/api/players/{player_id}").json()["ratings"][0]


def test_foosball_is_seeded(client):
    [foosball] = client.get("/api/game-types").json()
    assert foosball["name"] == "Foosball"
    assert foosball["roles"] == ["attack", "defence"]
    assert foosball["points_to_win"] == 10


def test_create_player_with_prior(client, gt):
    player = add_player(client, "ace", game_type_id=gt, prior={"level": 2, "stronger": "attack"})
    r = player["ratings"][0]
    assert r["overall"]["mu"] == pytest.approx(3.0)
    assert r["roles"]["attack"]["mu"] > r["roles"]["defence"]["mu"]
    assert r["provisional"] and r["games_played"] == 0

    plain = add_player(client, "joe")["ratings"][0]
    assert plain["overall"]["mu"] == 0


def test_player_errors(client, gt):
    add_player(client, "ann")
    assert client.post("/api/players", json={"name": "ann"}).status_code == 409
    bad_role = {"name": "x", "game_type_id": gt, "prior": {"stronger": "goalie"}}
    assert client.post("/api/players", json=bad_role).status_code == 422
    no_game_type = {"name": "y", "prior": {"level": 1}}
    assert client.post("/api/players", json=no_game_type).status_code == 422
    assert client.get("/api/players/999").status_code == 404


def test_record_game_updates_ratings(client, gt, players):
    ids = list(players.values())
    response = client.post("/api/games", json=two_v_two(gt, ids, 10, 4))
    assert response.status_code == 201, response.text
    game = response.json()
    assert game["outcome"] == "A"
    assert game["side_a"][0]["name"] == "ann"
    assert game["side_a"][0]["mu_after"]["attack"] > game["side_a"][0]["mu_before"]["attack"]

    ann, cat = rating(client, players["ann"]), rating(client, players["cat"])
    assert ann["overall"]["mu"] > 0 > cat["overall"]["mu"]
    assert (ann["wins"], ann["losses"], cat["wins"], cat["losses"]) == (1, 0, 0, 1)

    board = client.get("/api/players", params={"game_type_id": gt}).json()
    scores = [p["ratings"][0]["ranking_score"] for p in board]
    assert scores == sorted(scores, reverse=True)


def test_one_v_one(client, gt, players):
    body = {
        "game_type_id": gt,
        "side_a": [{"player_id": players["ann"]}],
        "side_b": [{"player_id": players["bob"]}],
        "score_a": 7,
        "score_b": 10,
    }
    assert client.post("/api/games", json=body).json()["outcome"] == "B"
    bob = rating(client, players["bob"])
    assert bob["roles"]["attack"]["mu"] == pytest.approx(bob["roles"]["defence"]["mu"])
    assert bob["overall"]["mu"] > 0


@pytest.mark.parametrize(
    "change",
    [
        {"score_a": 9, "score_b": 4},  # nobody reached 10
        {"score_a": 10, "score_b": 10},  # draw
        {"score_a": 11, "score_b": 3},
        {"score_a": 10, "score_b": -1},
        {"side_b": [{"player_id": 1, "role": "attack"}, {"player_id": 4, "role": "defence"}]},
        {"side_b": [{"player_id": 3, "role": "attack"}, {"player_id": 4, "role": "attack"}]},
        {"side_b": [{"player_id": 3, "role": "attack"}, {"player_id": 4}]},
        {"side_b": [{"player_id": 3, "role": "attack"}, {"player_id": 99, "role": "defence"}]},
        {"side_b": [{"player_id": 3}]},  # 2v1
        {"side_a": [{"player_id": 1, "role": "attack"}], "side_b": [{"player_id": 3}]},  # 1v1 role
    ],
)
def test_game_rules(client, gt, players, change):
    body = {**two_v_two(gt, list(players.values()), 10, 4), **change}
    response = client.post("/api/games", json=body)
    assert response.status_code == 422, response.text


def test_edit_and_delete_game_recompute(client, gt, players):
    ids = list(players.values())
    game = client.post("/api/games", json=two_v_two(gt, ids, 10, 4)).json()

    flipped = client.patch(f"/api/games/{game['id']}", json={"score_a": 4, "score_b": 10})
    assert flipped.status_code == 200
    ann = rating(client, players["ann"])
    assert ann["overall"]["mu"] < 0
    assert (ann["wins"], ann["losses"]) == (0, 1)

    # Swapping the teams' players is the same as the original result.
    swapped = client.patch(f"/api/games/{game['id']}", json=two_v_two(gt, ids[2:] + ids[:2], 10, 4))
    assert swapped.json()["side_a"][0]["name"] == "cat"
    assert rating(client, players["cat"])["overall"]["mu"] > 0

    assert client.delete(f"/api/games/{game['id']}").status_code == 204
    assert client.get(f"/api/games/{game['id']}").status_code == 404
    for player_id in ids:
        r = rating(client, player_id)
        assert r["overall"]["mu"] == 0 and r["games_played"] == 0


def test_games_are_replayed_in_date_order(client, gt, players):
    ids = list(players.values())
    later = two_v_two(gt, ids, 10, 2, played_at="2026-05-02T20:00:00Z")
    earlier = two_v_two(gt, ids, 3, 10, played_at="2026-05-01T20:00:00Z")
    client.post("/api/games", json=later)
    client.post("/api/games", json=earlier)

    history = client.get("/api/games", params={"player_id": players["ann"]}).json()
    assert [g["score_a"] for g in history] == [10, 3]  # newest first
    first_game = history[1]
    assert first_game["side_a"][0]["mu_before"] == {"attack": 0, "defence": 0}


def test_changing_a_prior_recomputes(client, gt, players):
    ids = list(players.values())
    client.post("/api/games", json=two_v_two(gt, ids, 10, 4))
    before = rating(client, players["ann"])["overall"]["mu"]
    body = {"game_type_id": gt, "prior": {"level": 2}}
    response = client.patch(f"/api/players/{players['ann']}", json=body)
    assert response.status_code == 200
    assert rating(client, players["ann"])["overall"]["mu"] > before + 1


def test_rename_and_deactivate_player(client, players):
    response = client.patch(
        f"/api/players/{players['ann']}", json={"name": "anna", "active": False}
    )
    assert response.json()["name"] == "anna" and response.json()["active"] is False
    taken = client.patch(f"/api/players/{players['bob']}", json={"name": "anna"})
    assert taken.status_code == 409


def test_game_type_settings_change_recomputes(client, gt, players):
    client.post("/api/games", json=two_v_two(gt, list(players.values()), 10, 4))
    before = rating(client, players["ann"])["overall"]["mu"]
    assert client.patch(f"/api/game-types/{gt}", json={"beta": 1.0}).status_code == 200
    assert rating(client, players["ann"])["overall"]["mu"] > before  # less noise, bigger update
    assert client.patch(f"/api/game-types/{gt}", json={"beta": -1}).status_code == 422


def test_predict_and_balance(client, gt):
    ids = {
        "ace": add_player(client, "ace", game_type_id=gt, prior={"level": 2})["id"],
        "avg": add_player(client, "avg")["id"],
        "new": add_player(client, "new", game_type_id=gt, prior={"level": -1})["id"],
        "mid": add_player(client, "mid")["id"],
    }
    match = {
        "game_type_id": gt,
        "side_a": [{"player_id": ids["ace"]}],
        "side_b": [{"player_id": ids["new"]}],
    }
    pred = client.post("/api/predict", json=match).json()
    assert pred["p_a"] > 0.8
    assert pred["score"][0] == 10
    low, high = pred["margin_interval"]
    assert low < pred["expected_margin"] < high

    options = client.post(
        "/api/balance", json={"game_type_id": gt, "player_ids": list(ids.values())}
    ).json()
    assert len(options) == 12
    fairest = {s["player_id"] for s in options[0]["side_a"]}
    assert ids["ace"] in fairest and ids["new"] in fairest

    too_many = client.post(
        "/api/balance", json={"game_type_id": gt, "player_ids": [*ids.values()] * 2}
    )
    assert too_many.status_code == 422


def test_recompute_endpoint(client, gt, players):
    client.post("/api/games", json=two_v_two(gt, list(players.values()), 10, 4))
    assert client.post("/api/recompute", json={"game_type_id": gt}).json() == {"games": 1}
