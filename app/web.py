"""Website pages (PLAN.md §7): leaderboard, add player, record game. Logins come in Phase 5."""

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any
from zoneinfo import ZoneInfo

import numpy as np
from fastapi import APIRouter, Depends, Form, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from sqlmodel import Session, col, select

from app.config import TIMEZONE
from app.db import get_session
from app.models import Game, GameParticipant, GameType, Player, PlayerRating
from app.rating.margin import Seat, role_weights
from app.ratings import (
    PROVISIONAL_GAMES,
    RANKING_K,
    RuleError,
    add_player,
    record_game,
    to_rating,
)

templates = Jinja2Templates(directory=Path(__file__).parent / "templates")
router = APIRouter(include_in_schema=False)
SessionDep = Annotated[Session, Depends(get_session)]

EXPERIENCE = [("-1", "Beginner"), ("0", "Average"), ("1", "Good"), ("2", "Very good")]
CERTAINTY = {"low": 1.0, "medium": 0.7, "high": 0.4}  # how sure → prior sigma factor


def render(
    request: Request, session: Session, template: str, status: int = 200, **context: Any
) -> HTMLResponse:
    game_types = session.exec(select(GameType).order_by(col(GameType.id))).all()
    return templates.TemplateResponse(
        request, template, {"game_types": game_types, **context}, status_code=status
    )


def pick_game_type(session: Session, game_type_id: int | None) -> GameType:
    if game_type_id is not None:
        game_type = session.get(GameType, game_type_id)
    else:
        game_type = session.exec(select(GameType).order_by(col(GameType.id))).first()
    if game_type is None:
        raise HTTPException(404, "game type not found")
    return game_type


def to_int(value: str, what: str) -> int:
    try:
        return int(value)
    except ValueError:
        raise RuleError(f"{what} must be a whole number") from None


# --- Leaderboard -------------------------------------------------------------------------------


@dataclass(frozen=True)
class BoardRow:
    rank: int
    name: str
    score: float  # mu − k·sigma, what the table is sorted by
    mu: float
    sigma: float
    games: int
    wins: int
    losses: int
    draws: int
    provisional: bool
    change: float | None  # change in mu in the player's most recent game


@dataclass(frozen=True)
class SeatResult:
    name: str
    role: str | None  # None if the player switched roles between the games
    change: float  # total over the games


@dataclass(frozen=True)
class GamesResult:
    headlines: list[str]  # e.g. "ann & bob beat cat & dan 10–6", one per game
    seats: list[SeatResult]


def last_changes(session: Session, game_type: GameType, weights: np.ndarray) -> dict[int, float]:
    """Each player's rating change in their most recent game."""
    parts = session.exec(
        select(GameParticipant)
        .join(Game)
        .where(Game.game_type_id == game_type.id)
        .order_by(col(Game.played_at).desc(), col(Game.id).desc())
    )
    changes: dict[int, float] = {}
    for part in parts:
        if part.player_id not in changes and part.mu_after is not None:
            delta = np.array(part.mu_after) - np.array(part.mu_before)
            changes[part.player_id] = float(weights @ delta)
    return changes


def leaderboard_rows(session: Session, game_type: GameType, role: str | None) -> list[BoardRow]:
    settings = game_type.settings()
    changes = last_changes(session, game_type, role_weights(settings.roles, role))
    entries = []
    for rating, player in session.exec(
        select(PlayerRating, Player).where(
            PlayerRating.player_id == Player.id,
            PlayerRating.game_type_id == game_type.id,
            col(Player.active).is_(True),
        )
    ):
        mu, sigma = to_rating(rating, settings).skill(role)
        entries.append((mu - RANKING_K * sigma, mu, sigma, rating, player))
    entries.sort(key=lambda e: -e[0])
    return [
        BoardRow(
            rank=i,
            name=player.name,
            score=score,
            mu=mu,
            sigma=sigma,
            games=rating.games_played,
            wins=rating.wins,
            losses=rating.losses,
            draws=rating.draws,
            provisional=rating.games_played < PROVISIONAL_GAMES,
            change=changes.get(player.id),
        )
        for i, (score, mu, sigma, rating, player) in enumerate(entries, 1)
    ]


def headline(game: Game, names: dict[int, str]) -> str:
    teams: dict[str, list[str]] = {"A": [], "B": []}
    for part in game.participants:
        teams[part.side].append(names[part.player_id])
    a, b = " & ".join(teams["A"]), " & ".join(teams["B"])
    if game.outcome == "A":
        return f"{a} beat {b} {game.score_a}–{game.score_b}"
    if game.outcome == "B":
        return f"{b} beat {a} {game.score_b}–{game.score_a}"
    return f"{a} and {b} drew {game.score_a}–{game.score_b}"


def games_result(session: Session, games: list[Game], game_type: GameType) -> GamesResult:
    """What just happened: each game's result and each player's total rating change."""
    names = dict(session.exec(select(Player.id, Player.name)).all())
    weights = role_weights(game_type.roles, None)
    headlines = []
    changes: dict[int, float] = {}
    roles: dict[int, set[str | None]] = {}
    for n, game in enumerate(games, 1):
        line = headline(game, names)
        headlines.append(f"Game {n}: {line}" if len(games) > 1 else line)
        for part in game.participants:
            delta = np.array(part.mu_after) - np.array(part.mu_before)
            changes[part.player_id] = changes.get(part.player_id, 0.0) + float(weights @ delta)
            roles.setdefault(part.player_id, set()).add(part.role)
    seats = [
        SeatResult(names[p], next(iter(roles[p])) if len(roles[p]) == 1 else None, change)
        for p, change in changes.items()
    ]
    return GamesResult(headlines, seats)


@router.get("/", response_class=HTMLResponse)
def leaderboard(
    request: Request,
    session: SessionDep,
    game_type_id: int | None = None,
    tab: str = "overall",
    game: Annotated[list[int] | None, Query()] = None,  # show these games (just recorded)
    added: int | None = None,  # show "player added" (after adding them)
) -> HTMLResponse:
    game_type = pick_game_type(session, game_type_id)
    roles = game_type.roles
    if tab not in roles:
        tab = "overall"
    tabs = [("overall", "Overall"), *((r, r.capitalize()) for r in roles)] if len(roles) > 1 else []

    recorded = [g for g in (session.get(Game, i) for i in game or []) if g is not None]
    recorded = [g for g in recorded if g.game_type_id == game_type.id]
    result = games_result(session, recorded, game_type) if recorded else None
    return render(
        request,
        session,
        "index.html",
        page="board",
        game_type=game_type,
        tab=tab,
        tabs=tabs,
        rows=leaderboard_rows(session, game_type, None if tab == "overall" else tab),
        result=result,
        added=session.get(Player, added) if added is not None else None,
        provisional_games=PROVISIONAL_GAMES,
        ranking_k=RANKING_K,
    )


# --- Add player --------------------------------------------------------------------------------


def player_form(
    request: Request,
    session: Session,
    game_type: GameType,
    values: dict[str, Any],
    error: str | None = None,
) -> HTMLResponse:
    n = len(game_type.roles)
    values = {"level": "0", "stronger": "", "certainty": "low", **values}
    values["mu"] = values.get("mu") or [""] * n
    values["sigma"] = values.get("sigma") or [""] * n
    return render(
        request,
        session,
        "player_form.html",
        status=422 if error else 200,
        page="player",
        game_type=game_type,
        values=values,
        error=error,
        experience=EXPERIENCE,
    )


def parse_prior(
    level: str, stronger: str, certainty: str, mu: list[str], sigma: list[str]
) -> dict[str, Any]:
    """Turn the form answers into keyword arguments for app.rating.margin.prior."""
    if level not in dict(EXPERIENCE) or certainty not in CERTAINTY:
        raise RuleError("pick an experience level and how sure you are")
    inputs: dict[str, Any] = {"level": float(level), "sigma_factor": CERTAINTY[certainty]}
    if stronger:
        inputs["stronger"] = stronger
    for key, symbol, raw in (("mu", "μ", mu), ("sigma", "σ", sigma)):
        filled = [v.strip() for v in raw]
        if not any(filled):
            continue
        if not all(filled):
            raise RuleError(f"fill in {symbol} for every role, or leave all of them empty")
        try:
            inputs[key] = [float(v) for v in filled]
        except ValueError:
            raise RuleError(f"{symbol} values must be numbers") from None
    return inputs


@router.get("/players/new", response_class=HTMLResponse)
def new_player(request: Request, session: SessionDep, game_type_id: int | None = None):
    return player_form(request, session, pick_game_type(session, game_type_id), {})


@router.post("/players/new", response_class=HTMLResponse)
def create_player(
    request: Request,
    session: SessionDep,
    game_type_id: Annotated[int, Form()],
    name: Annotated[str, Form()] = "",
    level: Annotated[str, Form()] = "0",
    stronger: Annotated[str, Form()] = "",
    certainty: Annotated[str, Form()] = "low",
    mu: Annotated[list[str] | None, Form()] = None,  # advanced: one per role, in role order
    sigma: Annotated[list[str] | None, Form()] = None,
) -> Response:
    game_type = pick_game_type(session, game_type_id)
    values = {
        "name": name,
        "level": level,
        "stronger": stronger,
        "certainty": certainty,
        "mu": mu,
        "sigma": sigma,
    }
    try:
        inputs = parse_prior(level, stronger, certainty, mu or [], sigma or [])
        player = add_player(session, name.strip(), game_type, inputs)
        session.commit()
    except RuleError as e:
        session.rollback()
        return player_form(request, session, game_type, values, str(e))
    return RedirectResponse(f"/?game_type_id={game_type.id}&added={player.id}", status_code=303)


# --- Record games ------------------------------------------------------------------------------

SCORE = re.compile(r"\s*(\d+)\s*[-–—:]\s*(\d+)\s*")


def team_formats(game_type: GameType) -> list[str]:
    """ "1v1", plus one player per role (e.g. "2v2") when the game type has several roles."""
    n = len(game_type.roles)
    return ["1v1", f"{n}v{n}"] if n > 1 else ["1v1"]


def seat_roles(game_type: GameType, fmt: str) -> list[str | None]:
    return [None] if fmt == "1v1" else list(game_type.roles)


def game_form(
    request: Request,
    session: Session,
    game_type: GameType,
    fmt: str,
    values: dict[str, Any],
    error: str | None = None,
) -> HTMLResponse:
    players = session.exec(
        select(Player).where(col(Player.active).is_(True)).order_by(col(Player.name))
    ).all()
    roles = seat_roles(game_type, fmt)
    return render(
        request,
        session,
        "game_form.html",
        status=422 if error else 200,
        page="game",
        game_type=game_type,
        fmt=fmt,
        formats=team_formats(game_type),
        seat_roles=roles,
        swappable=len(roles) == 2,
        players=players,
        values={"a": [], "b": [], "games": [{"row": "0", "score": ""}], **values},
        error=error,
    )


def parse_side(ids: list[str], roles: list[str | None], team: str) -> list[Seat]:
    if len(ids) != len(roles) or not all(ids):
        raise RuleError(f"pick every player for {team}")
    return [(to_int(i, "a player id"), role) for i, role in zip(ids, roles, strict=True)]


def swap_roles(side: list[Seat]) -> list[Seat]:
    """The same players with their roles swapped (attacker plays defence and vice versa)."""
    players, roles = zip(*side, strict=True)
    return list(zip(players, reversed(roles), strict=True))


def parse_score(text: str) -> tuple[int, int]:
    """ "10-4" (also "10–4", "10:4") → (10, 4), Team A's score first."""
    match = SCORE.fullmatch(text)
    if match is None:
        raise RuleError("write the score as Team A–Team B, e.g. 10-4")
    return int(match[1]), int(match[2])


def parse_time(value: str) -> datetime | None:
    """Empty means now. A time without a timezone is in the TIMEZONE setting."""
    if not value.strip():
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        raise RuleError("the date and time aren't valid") from None
    return dt if dt.tzinfo else dt.replace(tzinfo=ZoneInfo(TIMEZONE))


@router.get("/games/new", response_class=HTMLResponse)
def new_game(
    request: Request,
    session: SessionDep,
    game_type_id: int | None = None,
    fmt: Annotated[str | None, Query(alias="format")] = None,
):
    game_type = pick_game_type(session, game_type_id)
    formats = team_formats(game_type)
    fmt = fmt if fmt in formats else formats[-1]  # teams by default: foosball is mainly 2v2
    return game_form(request, session, game_type, fmt, {})


@router.post("/games/new", response_class=HTMLResponse)
def create_games(
    request: Request,
    session: SessionDep,
    game_type_id: Annotated[int, Form()],
    fmt: Annotated[str, Form(alias="format")] = "",
    a: Annotated[list[str] | None, Form()] = None,  # Team A's player ids, in role order
    b: Annotated[list[str] | None, Form()] = None,
    row: Annotated[list[str] | None, Form()] = None,  # one id per game row in the form
    score: Annotated[list[str] | None, Form()] = None,  # one per row, e.g. "10-4"
    swap_a: Annotated[list[str] | None, Form()] = None,  # ids of rows where Team A swapped roles
    swap_b: Annotated[list[str] | None, Form()] = None,
    played_at: Annotated[str, Form()] = "",
    notes: Annotated[str, Form()] = "",
) -> Response:
    """Record one or more games between the same two teams, in the order they were played."""
    game_type = pick_game_type(session, game_type_id)
    formats = team_formats(game_type)
    fmt = fmt if fmt in formats else formats[-1]
    rows, scores = row or [], score or []
    swapped_a, swapped_b = set(swap_a or []), set(swap_b or [])
    values = {
        "a": a or [],
        "b": b or [],
        "games": [
            {"row": r, "score": s, "swap_a": r in swapped_a, "swap_b": r in swapped_b}
            for r, s in zip(rows, scores, strict=False)
        ],
        "played_at": played_at,
        "notes": notes,
    }
    try:
        if not rows or len(rows) != len(scores):
            raise RuleError("enter the score of at least one game")
        roles = seat_roles(game_type, fmt)
        side_a = parse_side(a or [], roles, "Team A")
        side_b = parse_side(b or [], roles, "Team B")
        when = parse_time(played_at)
        games = []
        for n, (row_id, text) in enumerate(zip(rows, scores, strict=True), 1):
            try:
                games.append(
                    record_game(
                        session,
                        game_type,
                        swap_roles(side_a) if row_id in swapped_a else side_a,
                        swap_roles(side_b) if row_id in swapped_b else side_b,
                        *parse_score(text),
                        played_at=when,
                        notes=notes.strip(),
                    )
                )
            except RuleError as e:
                raise RuleError(f"Game {n}: {e}" if len(rows) > 1 else str(e)) from e
        session.commit()
    except RuleError as e:
        session.rollback()
        return game_form(request, session, game_type, fmt, values, str(e))
    ids = "".join(f"&game={g.id}" for g in games)
    return RedirectResponse(f"/?game_type_id={game_type.id}{ids}", status_code=303)
