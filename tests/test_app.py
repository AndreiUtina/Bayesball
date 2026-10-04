def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_leaderboard_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Leaderboard" in response.text
