import os
import re

from app import web


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_leaderboard_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Leaderboard" in response.text


def test_static_files_have_fingerprinted_addresses(visitor):
    page = visitor.get("/").text
    css = re.search(r'href="(/static/style\.css\?v=[0-9a-f]{12})"', page)[1]
    assert re.search(r'src="/static/logo\.jpg\?v=[0-9a-f]{12}"', page)
    versioned = visitor.get(css)
    assert versioned.status_code == 200 and ".viz" in versioned.text
    assert versioned.headers["cache-control"] == "public, max-age=31536000, immutable"
    assert visitor.get("/static/style.css").headers["cache-control"] == "no-cache"


def test_fingerprint_follows_the_contents(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "STATIC", tmp_path)
    file = tmp_path / "style.css"
    file.write_text("a {}")
    first = web.static_url("style.css")
    file.write_text("a { color: red }")
    os.utime(file, ns=(file.stat().st_atime_ns, file.stat().st_mtime_ns + 1_000_000))
    assert web.static_url("style.css") != first
