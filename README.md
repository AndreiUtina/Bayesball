# Bayesball

Bayesian ratings and match predictions for foosball (1v1 and 2v2 with attack/defence roles),
built to support other games too. See [PLAN.md](PLAN.md) for the full design.

## Run locally

Requires [uv](https://docs.astral.sh/uv/).

```sh
uv sync                                   # install dependencies
uv run uvicorn app.main:app --reload      # start the site at http://localhost:8000
```

The JSON API is documented (and can be tried out) at http://localhost:8000/docs. Locally the
data lives in `data/bayesball.db`; set `DATABASE_URL` to use another database.

## Develop

```sh
uv run pytest            # tests
uv run ruff check .      # lint
uv run ruff format .     # format
```
