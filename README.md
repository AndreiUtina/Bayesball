# Bayesball

<p align="center">
  <img src="app/static/logo.jpg" alt="Bayesball logo" width="420">
</p>

Bayesian ratings and match predictions for foosball (1v1 and 2v2 with attack/defence roles),
built to support other games too. See [PLAN.md](PLAN.md) for the full design.

## Run locally

Requires [uv](https://docs.astral.sh/uv/).

```sh
uv sync                                   # install dependencies
uv run uvicorn app.main:app --reload      # start the site at http://localhost:8000
```

On the first start the log shows a one-time link (`No owner account yet. Create it here: …`).
Open it to create your owner account. Anyone can view the site; adding or changing players and
games needs a login, and the owner invites other admins from the **Admins** page.

The JSON API is documented (and can be tried out) at http://localhost:8000/docs; log in on the
site first to use the write endpoints. Locally the data lives in `data/bayesball.db`; set
`DATABASE_URL` to use another database.

| Setting | Meaning |
|---|---|
| `DATABASE_URL` | database address (default: the local SQLite file) |
| `SECRET_KEY` | long random secret that signs login cookies (required online) |
| `SECURE_COOKIES` | `true` to send the login cookie over HTTPS only (online) |
| `PUBLIC_URL` | the site's address, used in setup and invite links |
| `TIMEZONE` | timezone for times typed into forms, e.g. `Europe/Amsterdam` |

## Develop

```sh
uv run pytest            # tests
uv run ruff check .      # lint
uv run ruff format .     # format
```
