# Bayesball

Bayesian ratings and match predictions for foosball (1v1 and 2v2 with attack/defence roles),
built to support other games too. See [PLAN.md](PLAN.md) for the full design.

## Run locally

Requires [uv](https://docs.astral.sh/uv/).

```sh
uv sync                                   # install dependencies
uv run uvicorn app.main:app --reload      # start the site at http://localhost:8000
```

## Develop

```sh
uv run pytest            # tests
uv run ruff check .      # lint
uv run ruff format .     # format
```
