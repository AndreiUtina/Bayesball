"""Settings from environment variables."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def database_url(url: str) -> str:
    """Neon (and others) hand out postgres:// addresses; SQLAlchemy also needs the driver name."""
    for prefix in ("postgres://", "postgresql://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url.removeprefix(prefix)
    return url


DATABASE_URL = database_url(
    os.environ.get("DATABASE_URL") or f"sqlite:///{ROOT / 'data' / 'bayesball.db'}"
)

# Times typed into the web forms are in this timezone (e.g. "Europe/Amsterdam").
TIMEZONE = os.environ.get("TIMEZONE", "UTC")

# Signs the login cookie. Must be a long random secret online; the fallback only suits local use.
SECRET_KEY = os.environ.get("SECRET_KEY", "")
if not SECRET_KEY:
    if not DATABASE_URL.startswith("sqlite"):
        raise RuntimeError("set the SECRET_KEY environment variable")
    SECRET_KEY = "local-development-only"

# Only send the login cookie over HTTPS (set to "true" online).
SECURE_COOKIES = os.environ.get("SECURE_COOKIES", "").lower() == "true"

# The site's address, used in setup and invite links. Render sets RENDER_EXTERNAL_URL.
PUBLIC_URL = (
    os.environ.get("PUBLIC_URL") or os.environ.get("RENDER_EXTERNAL_URL") or "http://localhost:8000"
).rstrip("/")
