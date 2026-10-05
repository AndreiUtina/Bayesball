"""Website pages (PLAN.md §7). Logins come in Phase 5."""

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import cache
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

import numpy as np
from fastapi import APIRouter, Depends, Form, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from sqlmodel import Session, col, func, select

from app.auth import AdminDep, current_user
from app.config import TIMEZONE
from app.db import get_session
from app.models import Game, GameParticipant, GameType, Player, PlayerRating, User
from app.rating.margin import Seat, role_weights
from app.rating.predict import Prediction, arrangements, balance, predict, role_options
from app.ratings import (
    PROVISIONAL_GAMES,
    RANKING_K,
    RuleError,
    add_player,
    check_match,
    check_players,
    current_ratings,
    delete_game,
    record_game,
    sides,
    to_rating,
    update_game,
    update_player,
)
from app.stats import (
    GameView,
    game_views,
    games_query,
    pair_records,
    player_games,
    rating_change,
    rating_chart,
)

templates = Jinja2Templates(directory=Path(__file__).parent / "templates")
STATIC = Path(__file__).parent / "static"


@cache
def _fingerprint(file: Path, modified_ns: int) -> str:
    return hashlib.sha256(file.read_bytes()).hexdigest()[:12]


def static_url(path: str) -> str:
    """A static file's address with a fingerprint of its contents, e.g. /static/style.css?v=1a2b…

    When the file changes, so does the address, so browsers never mix a new page with an old
    stylesheet they kept from before a deploy.
    """
    file = STATIC / path
    return f"/static/{path}?v={_fingerprint(file, file.stat().st_mtime_ns)}"


templates.env.globals["static_url"] = static_url


def local_time(dt: datetime) -> str:
    """E.g. "Sun 4 Oct, 20:15" in the TIMEZONE setting; the year only if it isn't this year."""
    zone = ZoneInfo(TIMEZONE)
    dt = (dt if dt.tzinfo else dt.replace(tzinfo=UTC)).astimezone(zone)
    year = "" if dt.year == datetime.now(zone).year else f" {dt.year}"
    return f"{dt:%a} {dt.day} {dt:%b}{year}, {dt:%H:%M}"


templates.env.filters["when"] = local_time

router = APIRouter(include_in_schema=False)
SessionDep = Annotated[Session, Depends(get_session)]

EXPERIENCE = [("-1", "Beginner"), ("0", "Average"), ("1", "Good"), ("2", "Very good")]
CERTAINTY = {"low": 1.0, "medium": 0.7, "high": 0.4}  # how sure → prior sigma factor


def render(
    request: Request, session: Session, template: str, status: int = 200, **context: Any
) -> HTMLResponse:
    game_types = session.exec(select(GameType).order_by(col(GameType.id))).all()
    user = current_user(request, session)
    return templates.TemplateResponse(
        request,
        template,
        {"game_types": game_types, "user": user, **context},
        status_code=status,
    )


def pick_game_type(session: Session, game_type_id: int | None) -> GameType:
    if game_type_id is not None:
        game_type = session.get(GameType, game_type_id)
    else:
        game_type = session.exec(select(GameType).order_by(col(GameType.id))).first()
    if game_type is None:
        raise HTTPException(404, "game type not found")
    return game_type


def active_players(session: Session) -> list[Player]:
    return list(
        session.exec(select(Player).where(col(Player.active).is_(True)).order_by(col(Player.name)))
    )


def to_int(value: str, what: str) -> int:
    try:
        return int(value)
    except ValueError:
        raise RuleError(f"{what} must be a whole number") from None


# --- Leaderboard -------------------------------------------------------------------------------


@dataclass(frozen=True)
class BoardRow:
    rank: int
    player_id: int
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
            changes[part.player_id] = rating_change(part, weights)
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
            player_id=player.id,
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
            change = rating_change(part, weights)
            changes[part.player_id] = changes.get(part.player_id, 0.0) + change
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
    editing: Player | None = None,
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
        editing=editing,
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
def new_player(
    request: Request, session: SessionDep, user: AdminDep, game_type_id: int | None = None
):
    return player_form(request, session, pick_game_type(session, game_type_id), {})


@router.post("/players/new", response_class=HTMLResponse)
def create_player(
    request: Request,
    session: SessionDep,
    user: AdminDep,
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
        player = add_player(session, name.strip(), game_type, inputs, user_id=user.id)
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


def pick_format(game_type: GameType, fmt: str | None) -> str:
    formats = team_formats(game_type)
    return fmt if fmt in formats else formats[-1]  # teams by default: foosball is mainly 2v2


def seat_roles(game_type: GameType, fmt: str) -> list[str | None]:
    return [None] if fmt == "1v1" else list(game_type.roles)


def matchup_url(
    path: str, game_type: GameType, side_a: Sequence[Seat], side_b: Sequence[Seat]
) -> str:
    """A link to /predict or /games/new with these teams filled in."""
    fmt = "1v1" if len(side_a) == 1 else team_formats(game_type)[-1]
    query = {
        "game_type_id": game_type.id,
        "format": fmt,
        "a": [p for p, _ in side_a],
        "b": [p for p, _ in side_b],
    }
    return f"{path}?{urlencode(query, doseq=True)}"


def game_form(
    request: Request,
    session: Session,
    game_type: GameType,
    fmt: str,
    values: dict[str, Any],
    error: str | None = None,
) -> HTMLResponse:
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
        players=active_players(session),
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
    user: AdminDep,
    game_type_id: int | None = None,
    fmt: Annotated[str | None, Query(alias="format")] = None,
    a: Annotated[list[str] | None, Query()] = None,  # prefill teams, e.g. from /predict
    b: Annotated[list[str] | None, Query()] = None,
):
    game_type = pick_game_type(session, game_type_id)
    fmt = pick_format(game_type, fmt)
    return game_form(request, session, game_type, fmt, {"a": a or [], "b": b or []})


@router.post("/games/new", response_class=HTMLResponse)
def create_games(
    request: Request,
    session: SessionDep,
    user: AdminDep,
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
    fmt = pick_format(game_type, fmt)
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
                        user_id=user.id,
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


# --- Predict -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class MatchupView:
    team_a: str  # e.g. "ann & bob"
    team_b: str
    prediction: Prediction
    low: float  # 90% range of the goal difference (Team A − Team B)
    high: float
    provisional: bool  # someone has few games, so the prediction is rough
    rows: list[list[str]]  # Team A's role arrangements, as "name: role" lines
    cols: list[list[str]]  # Team B's
    grid: list[list[float]]  # P(Team A wins) for [Team A arrangement][Team B arrangement]
    advice: list[str]  # which arrangement suits each team best
    record_url: str


def arrangement_lines(option: Sequence[Seat], names: dict[int, str]) -> list[str]:
    return [f"{names[p]}: {role}" for p, role in option]


def role_advice(team: str, option: Sequence[Seat], gain: float, names: dict[int, str]) -> str:
    """What a team gains (in expected goal difference) from its best role arrangement."""
    if gain < 0.05:
        return f"{team}: the roles as picked are their best choice."
    if gain < 0.25:
        return f"{team}: swapping roles makes almost no difference."
    best = " and ".join(f"{names[p]} in {role}" for p, role in option)
    return f"{team}: about {gain:.1f} goals stronger with {best}."


def matchup_view(
    session: Session, game_type: GameType, side_a: list[Seat], side_b: list[Seat]
) -> MatchupView:
    settings = game_type.settings()
    names = dict(session.exec(select(Player.id, Player.name)).all())
    ratings = current_ratings(session, game_type)
    pred = predict(settings, ratings, side_a, side_b)
    half = 1.645 * pred.margin_sd
    low, high = pred.expected_margin - half, pred.expected_margin + half
    if game_type.points_to_win:
        cap = game_type.points_to_win
        low, high = max(low, -cap), min(high, cap)

    games_played = dict(
        session.exec(
            select(PlayerRating.player_id, PlayerRating.games_played).where(
                PlayerRating.game_type_id == game_type.id
            )
        ).all()
    )
    seats = [*side_a, *side_b]
    options_a, options_b = arrangements(side_a), arrangements(side_b)
    grid = role_options(settings, ratings, side_a, side_b)
    margins_a = [row[0].expected_margin for row in grid]  # Team B as picked
    margins_b = [-p.expected_margin for p in grid[0]]  # Team A as picked
    best_a = int(np.argmax(margins_a))
    best_b = int(np.argmax(margins_b))
    advice = []
    if len(options_a) > 1:
        gain_a = margins_a[best_a] - margins_a[0]
        advice.append(role_advice("Team A", options_a[best_a], gain_a, names))
        gain_b = margins_b[best_b] - margins_b[0]
        advice.append(role_advice("Team B", options_b[best_b], gain_b, names))

    return MatchupView(
        team_a=" & ".join(names[p] for p, _ in side_a),
        team_b=" & ".join(names[p] for p, _ in side_b),
        prediction=pred,
        low=low,
        high=high,
        provisional=any(games_played.get(p, 0) < PROVISIONAL_GAMES for p, _ in seats),
        rows=[arrangement_lines(o, names) for o in options_a],
        cols=[arrangement_lines(o, names) for o in options_b],
        grid=[[p.p_a for p in row] for row in grid],
        advice=advice,
        record_url=matchup_url("/games/new", game_type, side_a, side_b),
    )


@router.get("/predict", response_class=HTMLResponse)
def predict_page(
    request: Request,
    session: SessionDep,
    game_type_id: int | None = None,
    fmt: Annotated[str | None, Query(alias="format")] = None,
    a: Annotated[list[str] | None, Query()] = None,  # Team A's player ids, in role order
    b: Annotated[list[str] | None, Query()] = None,
) -> HTMLResponse:
    game_type = pick_game_type(session, game_type_id)
    fmt = pick_format(game_type, fmt)
    roles = seat_roles(game_type, fmt)
    matchup, error = None, None
    if a or b:
        try:
            side_a = parse_side(a or [], roles, "Team A")
            side_b = parse_side(b or [], roles, "Team B")
            check_match(session, game_type.settings(), side_a, side_b)
            matchup = matchup_view(session, game_type, side_a, side_b)
        except RuleError as e:
            error = str(e)
    return render(
        request,
        session,
        "predict.html",
        status=422 if error else 200,
        page="predict",
        game_type=game_type,
        fmt=fmt,
        formats=team_formats(game_type),
        seat_roles=roles,
        players=active_players(session),
        values={"a": a or [], "b": b or []},
        matchup=matchup,
        error=error,
    )


# --- Balance teams -----------------------------------------------------------------------------


@dataclass(frozen=True)
class OptionView:
    side_a: list[tuple[str, str | None]]  # (name, role)
    side_b: list[tuple[str, str | None]]
    p_a: float
    p_b: float
    score: tuple[int, int] | None
    predict_url: str
    record_url: str


@router.get("/balance", response_class=HTMLResponse)
def balance_page(
    request: Request,
    session: SessionDep,
    game_type_id: int | None = None,
    p: Annotated[list[str] | None, Query()] = None,  # ids of the players at the table
) -> HTMLResponse:
    game_type = pick_game_type(session, game_type_id)
    sizes = sorted({2, 2 * len(game_type.roles)})
    options, error = None, None
    if p is not None:
        try:
            ids = list(dict.fromkeys(to_int(i, "a player id") for i in p))
            if len(ids) not in sizes:
                counts = " or ".join(str(n) for n in sizes)
                raise RuleError(f"pick {counts} players (you picked {len(ids)})")
            check_players(session, ids)
            names = dict(session.exec(select(Player.id, Player.name)).all())
            matchups = balance(game_type.settings(), current_ratings(session, game_type), ids)
            options = [
                OptionView(
                    side_a=[(names[pid], role) for pid, role in m.side_a],
                    side_b=[(names[pid], role) for pid, role in m.side_b],
                    p_a=m.prediction.p_a,
                    p_b=m.prediction.p_b,
                    score=m.prediction.score,
                    predict_url=matchup_url("/predict", game_type, m.side_a, m.side_b),
                    record_url=matchup_url("/games/new", game_type, m.side_a, m.side_b),
                )
                for m in matchups
            ]
        except RuleError as e:
            error = str(e)
    return render(
        request,
        session,
        "balance.html",
        status=422 if error else 200,
        page="predict",
        game_type=game_type,
        sizes=sizes,
        players=active_players(session),
        picked=set(p or []),
        options=options,
        error=error,
    )


# --- Player profile ----------------------------------------------------------------------------

RECENT_GAMES = 10


@router.get("/players/{player_id:int}", response_class=HTMLResponse)
def player_page(
    request: Request,
    session: SessionDep,
    player_id: int,
    game_type_id: int | None = None,
    saved: bool = False,  # show "saved" (after editing the player)
) -> HTMLResponse:
    game_type = pick_game_type(session, game_type_id)
    player = session.get(Player, player_id)
    row = session.exec(
        select(PlayerRating).where(
            PlayerRating.player_id == player_id, PlayerRating.game_type_id == game_type.id
        )
    ).first()
    if player is None or row is None:
        raise HTTPException(404, "player not found")

    rating = to_rating(row, game_type.settings())
    keys: list[str | None] = [None, *game_type.roles] if len(game_type.roles) > 1 else [None]
    skills = [(key.capitalize() if key else "Overall", *rating.skill(key)) for key in keys]
    board = leaderboard_rows(session, game_type, None)
    rank = next((r.rank for r in board if r.player_id == player_id), None)

    games = player_games(session, game_type, player_id)
    views = game_views(session, games, game_type)
    teammates, opponents = pair_records(views, player_id)
    chart = None
    if games:
        chart = rating_chart(games, player_id, game_type)
        chart["titles"] = ["Starting rating"] + [
            f"{local_time(v.played_at)} · {v.result_for(player_id)} "
            + "–".join(map(str, v.score_for(player_id)))
            for v in views
        ]
    return render(
        request,
        session,
        "player.html",
        page="profile",
        game_type=game_type,
        player=player,
        rating=row,
        skills=skills,
        rank=rank,
        board_size=len(board),
        provisional=row.games_played < PROVISIONAL_GAMES,
        chart=chart,
        recent=views[::-1][:RECENT_GAMES],
        more_games=len(views) > RECENT_GAMES,
        teammates=teammates,
        opponents=opponents,
        saved=saved,
    )


# --- Game history ------------------------------------------------------------------------------

GAMES_PER_PAGE = 50


@router.get("/games", response_class=HTMLResponse)
def history_page(
    request: Request,
    session: SessionDep,
    game_type_id: int | None = None,
    player: str = "",  # a player id, or empty for everyone
    page: Annotated[int, Query(ge=1)] = 1,
    notice: str = "",  # "saved" or "deleted" after editing a game
) -> HTMLResponse:
    game_type = pick_game_type(session, game_type_id)
    player_id = int(player) if player.isdigit() else None
    query = games_query(game_type, player_id)
    total = session.exec(select(func.count()).select_from(query.subquery())).one()
    games = session.exec(query.offset((page - 1) * GAMES_PER_PAGE).limit(GAMES_PER_PAGE)).all()
    views: list[GameView] = game_views(session, games, game_type)
    players = session.exec(select(Player).order_by(col(Player.name))).all()
    return render(
        request,
        session,
        "games.html",
        page="history",
        game_type=game_type,
        games=views,
        players=players,
        player_id=player_id,
        total=total,
        first=(page - 1) * GAMES_PER_PAGE + 1,
        page_num=page,
        has_newer=page > 1,
        has_older=page * GAMES_PER_PAGE < total,
        notice=notice,
    )


# --- Edit players (admins) ---------------------------------------------------------------------


def prior_values(inputs: dict[str, Any], n_roles: int) -> dict[str, Any]:
    """The form answers that give these stored prior inputs (the reverse of parse_prior)."""
    level = float(inputs.get("level", 0.0))
    shown_level = str(int(level)) if level.is_integer() else "0"
    factor = inputs.get("sigma_factor", 1.0)
    return {
        "level": shown_level if shown_level in dict(EXPERIENCE) else "0",
        "stronger": inputs.get("stronger") or "",
        "certainty": next((k for k, v in CERTAINTY.items() if v == factor), "low"),
        "mu": [str(v) for v in inputs.get("mu") or []] or [""] * n_roles,
        "sigma": [str(v) for v in inputs.get("sigma") or []] or [""] * n_roles,
    }


def player_and_rating(
    session: Session, player_id: int, game_type: GameType
) -> tuple[Player, PlayerRating]:
    player = session.get(Player, player_id)
    row = session.exec(
        select(PlayerRating).where(
            PlayerRating.player_id == player_id, PlayerRating.game_type_id == game_type.id
        )
    ).first()
    if player is None or row is None:
        raise HTTPException(404, "player not found")
    return player, row


@router.get("/players/{player_id:int}/edit", response_class=HTMLResponse)
def edit_player(
    request: Request,
    session: SessionDep,
    user: AdminDep,
    player_id: int,
    game_type_id: int | None = None,
) -> HTMLResponse:
    game_type = pick_game_type(session, game_type_id)
    player, row = player_and_rating(session, player_id, game_type)
    values = {
        "name": player.name,
        "active": player.active,
        **prior_values(row.prior, len(game_type.roles)),
    }
    return player_form(request, session, game_type, values, editing=player)


@router.post("/players/{player_id:int}/edit", response_class=HTMLResponse)
def save_player(
    request: Request,
    session: SessionDep,
    user: AdminDep,
    player_id: int,
    game_type_id: Annotated[int, Form()],
    name: Annotated[str, Form()] = "",
    active: Annotated[str, Form()] = "",  # "on" when ticked
    level: Annotated[str, Form()] = "0",
    stronger: Annotated[str, Form()] = "",
    certainty: Annotated[str, Form()] = "low",
    mu: Annotated[list[str] | None, Form()] = None,
    sigma: Annotated[list[str] | None, Form()] = None,
) -> Response:
    game_type = pick_game_type(session, game_type_id)
    player, row = player_and_rating(session, player_id, game_type)
    values = {
        "name": name,
        "active": bool(active),
        "level": level,
        "stronger": stronger,
        "certainty": certainty,
        "mu": mu,
        "sigma": sigma,
    }
    try:
        inputs = parse_prior(level, stronger, certainty, mu or [], sigma or [])
        shown = prior_values(row.prior, len(game_type.roles))
        unchanged = parse_prior(
            shown["level"], shown["stronger"], shown["certainty"], shown["mu"], shown["sigma"]
        )
        update_player(
            session,
            player,
            name=name.strip(),
            active=bool(active),
            game_type=game_type,
            # Only a changed prior replaces the stored one (and replays the history).
            prior_inputs=None if inputs == unchanged else inputs,
            user_id=user.id,
        )
        session.commit()
    except RuleError as e:
        session.rollback()
        return player_form(request, session, game_type, values, str(e), editing=player)
    return RedirectResponse(
        f"/players/{player_id}?game_type_id={game_type.id}&saved=true", status_code=303
    )


# --- Edit and delete games (admins) ------------------------------------------------------------


def game_and_type(session: Session, game_id: int) -> tuple[Game, GameType]:
    game = session.get(Game, game_id)
    if game is None:
        raise HTTPException(404, "game not found")
    return game, pick_game_type(session, game.game_type_id)


def form_time(dt: datetime) -> str:
    """A stored time as a datetime-local form value, in the TIMEZONE setting."""
    return (
        (dt if dt.tzinfo else dt.replace(tzinfo=UTC))
        .astimezone(ZoneInfo(TIMEZONE))
        .strftime("%Y-%m-%dT%H:%M")
    )


def game_values(game: Game, game_type: GameType) -> dict[str, Any]:
    """The edit form's fields for a recorded game; team seats in role order."""
    side_a, side_b = sides(game)
    fmt = "1v1" if len(side_a) == 1 else team_formats(game_type)[-1]
    roles = seat_roles(game_type, fmt)

    def ids(side: list[Seat]) -> list[str]:
        by_role = {role: player for player, role in side}
        return [str(by_role[role]) for role in roles]

    return {
        "format": fmt,
        "a": ids(side_a),
        "b": ids(side_b),
        "score": f"{game.score_a}-{game.score_b}",
        "played_at": form_time(game.played_at),
        "notes": game.notes,
    }


def game_edit_form(
    request: Request,
    session: Session,
    game: Game,
    game_type: GameType,
    values: dict[str, Any],
    error: str | None = None,
) -> HTMLResponse:
    in_game = {p.player_id for p in game.participants}
    players = session.exec(
        select(Player)
        .where(col(Player.active).is_(True) | col(Player.id).in_(in_game))
        .order_by(col(Player.name))
    ).all()
    usernames = dict(session.exec(select(User.id, User.username)).all())
    return render(
        request,
        session,
        "game_edit.html",
        status=422 if error else 200,
        page="history",
        game=game,
        game_type=game_type,
        seat_roles=seat_roles(game_type, values["format"]),
        players=players,
        values=values,
        created_by=usernames.get(game.created_by),
        updated_by=usernames.get(game.updated_by),
        error=error,
    )


@router.get("/games/{game_id:int}/edit", response_class=HTMLResponse)
def edit_game(request: Request, session: SessionDep, user: AdminDep, game_id: int):
    game, game_type = game_and_type(session, game_id)
    return game_edit_form(request, session, game, game_type, game_values(game, game_type))


@router.post("/games/{game_id:int}/edit", response_class=HTMLResponse)
def save_game(
    request: Request,
    session: SessionDep,
    user: AdminDep,
    game_id: int,
    a: Annotated[list[str] | None, Form()] = None,
    b: Annotated[list[str] | None, Form()] = None,
    score: Annotated[str, Form()] = "",
    played_at: Annotated[str, Form()] = "",
    notes: Annotated[str, Form()] = "",
) -> Response:
    game, game_type = game_and_type(session, game_id)
    shown = game_values(game, game_type)
    roles = seat_roles(game_type, shown["format"])
    values = {**shown, "a": a or [], "b": b or [], "score": score, "played_at": played_at}
    values["notes"] = notes
    try:
        update_game(
            session,
            game_type,
            game,
            parse_side(a or [], roles, "Team A"),
            parse_side(b or [], roles, "Team B"),
            *parse_score(score),
            # The form shows minutes only, so an untouched time keeps its exact stored value
            # (and the game keeps its place among games played in the same minute).
            played_at=None if played_at == shown["played_at"] else parse_time(played_at),
            notes=notes.strip(),
            user_id=user.id,
        )
        session.commit()
    except RuleError as e:
        session.rollback()
        return game_edit_form(request, session, game, game_type, values, str(e))
    return RedirectResponse(f"/games?game_type_id={game_type.id}&notice=saved", status_code=303)


@router.post("/games/{game_id:int}/delete")
def remove_game(request: Request, session: SessionDep, user: AdminDep, game_id: int) -> Response:
    game, game_type = game_and_type(session, game_id)
    delete_game(session, game_type, game)
    session.commit()
    return RedirectResponse(f"/games?game_type_id={game_type.id}&notice=deleted", status_code=303)
