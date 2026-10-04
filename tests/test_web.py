import pytest


def add_player(client, name, **fields):
    response = client.post(
        "/players/new", data={"game_type_id": 1, "name": name, **fields}, follow_redirects=False
    )
    assert response.status_code == 303, response.text
    return response.headers["location"]


@pytest.fixture
def four(client):
    for name in ("ann", "bob", "cat", "dan"):
        add_player(client, name)
    return {"ann": "1", "bob": "2", "cat": "3", "dan": "4"}


def game_form(a, b, *scores, **extra):
    """Form data for games between the same teams; scores like "10-6", default one game."""
    scores = scores or ("10-6",)
    return {
        "game_type_id": 1,
        "format": "2v2" if len(a) == 2 else "1v1",
        "a": a,
        "b": b,
        "row": [str(i) for i in range(len(scores))],
        "score": list(scores),
        **extra,
    }


def test_empty_leaderboard(client):
    page = client.get("/").text
    assert "Add your first player" in page


def test_add_player(client):
    location = add_player(client, "ann", level="2", stronger="attack", certainty="high")
    page = client.get(location).text
    assert "Added <strong>ann</strong>" in page
    assert "provisional" in page
    assert "+3.0" in page  # very good = 2 steps of 1.5 goals


def test_add_player_errors_keep_the_form(client):
    add_player(client, "ann")
    response = client.post("/players/new", data={"game_type_id": 1, "name": "ann"})
    assert response.status_code == 422
    assert "already exists" in response.text
    assert 'value="ann"' in response.text

    partial = {"game_type_id": 1, "name": "eve", "mu": ["1.5", ""], "sigma": ["", ""]}
    response = client.post("/players/new", data=partial)
    assert response.status_code == 422
    assert "every role" in response.text


def test_add_player_with_advanced_values(client):
    fields = {"mu": ["2", "-1"], "sigma": ["1", "1"]}
    page = client.get(add_player(client, "eve", **fields)).text
    assert "+0.5" in page  # overall = mean of attack and defence


def test_record_2v2_game_changes_the_ranking(client, four):
    response = client.post(
        "/games/new",
        data=game_form([four["ann"], four["bob"]], [four["cat"], four["dan"]]),
        follow_redirects=False,
    )
    assert response.status_code == 303, response.text
    page = client.get(response.headers["location"]).text
    assert "ann &amp; bob beat cat &amp; dan 10–6" in page
    assert page.index("<td>ann") < page.index("<td>cat")
    assert "▲" in page and "▼" in page


def test_record_1v1_game(client, four):
    form = client.get("/games/new?format=1v1").text
    assert form.count('<select name="a"') == 1

    data = game_form([four["cat"]], [four["dan"]], "9-10")
    page = client.post("/games/new", data=data).text
    assert "dan beat cat 10–9" in page


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"a": ["1", "1"]}, "only one seat"),
        ({"b": ["3", ""]}, "pick every player"),
        ({"score": ["ten-four"]}, "e.g. 10-4"),
        ({"score": ["10-10"]}, "loser 0-9"),
        ({"score": ["8-6"]}, "winner needs exactly 10"),
        ({"played_at": "yesterday"}, "date and time"),
    ],
)
def test_game_errors_keep_the_form(client, four, change, message):
    data = {**game_form(["1", "2"], ["3", "4"]), **change}
    response = client.post("/games/new", data=data)
    assert response.status_code == 422
    assert message in response.text
    assert '<option value="1" selected>' in response.text  # the choices are kept


@pytest.mark.parametrize("score", ["10-4", "10 - 4", "10–4", "10:4", " 10-4 "])
def test_score_formats(client, four, score):
    response = client.post("/games/new", data=game_form(["1", "2"], ["3", "4"], score))
    assert "beat cat &amp; dan 10–4" in response.text


def test_several_games_with_role_swaps(client, four):
    data = game_form(
        ["1", "2"], ["3", "4"], "10-4", "7-10", "10-8", swap_a=["1"], swap_b=["1", "2"]
    )
    page = client.post("/games/new", data=data).text
    assert "Game 1: ann &amp; bob beat cat &amp; dan 10–4" in page
    assert "Game 2: cat &amp; dan beat ann &amp; bob 10–7" in page
    assert "Game 3: ann &amp; bob beat cat &amp; dan 10–8" in page

    games = client.get("/api/games").json()[::-1]  # oldest first
    roles = [{s["name"]: s["role"] for s in (*g["side_a"], *g["side_b"])} for g in games]
    assert roles[0] == {"ann": "attack", "bob": "defence", "cat": "attack", "dan": "defence"}
    assert roles[1] == {"ann": "defence", "bob": "attack", "cat": "defence", "dan": "attack"}
    assert roles[2] == {"ann": "attack", "bob": "defence", "cat": "defence", "dan": "attack"}
    ann = client.get(f"/api/players/{four['ann']}").json()["ratings"][0]
    assert (ann["wins"], ann["losses"]) == (2, 1)


def test_an_invalid_game_saves_nothing(client, four):
    data = game_form(["1", "2"], ["3", "4"], "10-4", "10-10", swap_a=["1"])
    response = client.post("/games/new", data=data)
    assert response.status_code == 422
    assert "Game 2: the winner needs exactly 10" in response.text
    assert 'value="10-10"' in response.text and 'name="swap_a" value="1" checked' in response.text
    assert client.get("/api/games").json() == []


def test_game_with_a_date(client, four):
    data = game_form(["1", "2"], ["3", "4"], played_at="2026-05-01T20:30")
    client.post("/games/new", data=data)
    [game] = client.get("/api/games").json()
    assert game["played_at"].startswith("2026-05-01T20:30")


def test_game_form_needs_two_players(client):
    assert "at least two players" in client.get("/games/new").text


def test_role_tabs(client, four):
    client.post("/games/new", data=game_form(["1", "2"], ["3", "4"]))
    page = client.get("/?tab=attack").text
    assert 'tab=attack" aria-current="page"' in page
    assert client.get("/?tab=nonsense").status_code == 200
