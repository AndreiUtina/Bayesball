"""Settings from environment variables."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATABASE_URL = os.environ.get("DATABASE_URL", f"sqlite:///{ROOT / 'data' / 'bayesball.db'}")

# Times typed into the web forms are in this timezone (e.g. "Europe/Amsterdam").
TIMEZONE = os.environ.get("TIMEZONE", "UTC")
