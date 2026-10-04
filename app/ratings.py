"""Connects the database to the rating engine: game rules, priors and recomputing ratings."""

from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

import numpy as np
from sqlalchemy.orm import selectinload
from sqlmodel import Session, col, select

from app.models import Game, GameParticipant, GameType, Player, PlayerRating
from app.rating.margin import GameSettings, Rating, Seat, prior, update

RANKING_K = 2.0  # the leaderboard sorts by mu − k·sigma
PROVISIONAL_GAMES = 5  # players with fewer games are labelled provisional


class RuleError(ValueError):
    """A request breaks a game rule. The API returns it as a 422 with this message."""


def as_utc(dt: datetime) -> datetime:
    """Times without a timezone are taken to be UTC."""
    return dt.astimezone(UTC) if dt.tzinfo else dt.replace(tzinfo=UTC)


def outcome(score_a: int, score_b: int) -> str:
    return "A" if score_a > score_b else "B" if score_b > score_a else "DRAW"


# --- Rules -------------------------------------------------------------------------------------


def check_prior(settings: GameSettings, inputs: dict[str, Any]) -> None:
    roles = settings.roles
    if inputs.get("stronger") not in (None, *roles):
        raise RuleError(f"stronger must be one of: {', '.join(roles)}")
    for key in ("mu", "sigma"):
        values = inputs.get(key)
        if values is not None and len(values) != len(roles):
            raise RuleError(f"{key} needs one value per role: {', '.join(roles)}")
    if any(s <= 0 for s in inputs.get("sigma") or ()):
        raise RuleError("sigma values must be positive")


def check_players(session: Session, player_ids: Sequence[int]) -> None:
    known = set(session.exec(select(Player.id).where(col(Player.id).in_(player_ids))))
    missing = set(player_ids) - known
    if missing:
        raise RuleError(f"unknown player id(s): {sorted(missing)}")


def check_match(
    session: Session, settings: GameSettings, side_a: Sequence[Seat], side_b: Sequence[Seat]
) -> None:
    """Both sides are the same size, nobody plays twice, and roles fit the team size."""
    if not side_a or len(side_a) != len(side_b):
        raise RuleError("both sides need the same number of players (at least one)")
    players = [p for p, _ in (*side_a, *side_b)]
    if len(set(players)) != len(players):
        raise RuleError("a player can take only one seat per game")
    roles = settings.roles
    for side in (side_a, side_b):
        given = [r for _, r in side]
        if len(side) == 1 or len(roles) == 1:
            if any(given):
                raise RuleError("roles are only used when a team has one player per role")
        elif Counter(given) != Counter(roles):
            raise RuleError(f"each team needs exactly one player per role: {', '.join(roles)}")
    check_players(session, players)


def check_score(game_type: GameType, score_a: int, score_b: int) -> None:
    if game_type.scoring != "margin":
        raise RuleError("only score-margin game types are supported so far")
    if min(score_a, score_b) < 0:
        raise RuleError("scores can't be negative")
    win = game_type.points_to_win
    if win is not None:
        if max(score_a, score_b) != win or min(score_a, score_b) >= win:
            raise RuleError(f"the winner needs exactly {win} points and the loser 0-{win - 1}")
    elif score_a == score_b and not game_type.allow_draws:
        raise RuleError(f"{game_type.name} games can't end in a draw")


def check_game(
    session: Session,
    game_type: GameType,
    side_a: Sequence[Seat],
    side_b: Sequence[Seat],
    score_a: int,
    score_b: int,
) -> None:
    check_score(game_type, score_a, score_b)
    check_match(session, game_type.settings(), side_a, side_b)


# --- Ratings -----------------------------------------------------------------------------------


def to_rating(row: PlayerRating, settings: GameSettings) -> Rating:
    return Rating(settings.roles, np.array(row.mu), np.array(row.cov))


def _store(row: PlayerRating, rating: Rating) -> None:
    row.mu, row.cov = rating.mu.tolist(), rating.cov.tolist()


def new_rating_row(
    player_id: int, game_type: GameType, prior_inputs: dict[str, Any] | None = None
) -> PlayerRating:
    settings = game_type.settings()
    inputs = prior_inputs or {}
    check_prior(settings, inputs)
    row = PlayerRating(player_id=player_id, game_type_id=game_type.id, prior=inputs)
    _store(row, prior(settings, **inputs))
    return row


def current_ratings(session: Session, game_type: GameType) -> dict[int, Rating]:
    settings = game_type.settings()
    rows = session.exec(select(PlayerRating).where(PlayerRating.game_type_id == game_type.id))
    return {row.player_id: to_rating(row, settings) for row in rows}


def make_participants(side_a: Sequence[Seat], side_b: Sequence[Seat]) -> list[GameParticipant]:
    return [
        GameParticipant(player_id=player, side=side, role=role)
        for side, seats in (("A", side_a), ("B", side_b))
        for player, role in seats
    ]


def sides(game: Game) -> tuple[list[Seat], list[Seat]]:
    seats: dict[str, list[Seat]] = {"A": [], "B": []}
    for part in game.participants:
        seats[part.side].append((part.player_id, part.role))
    return seats["A"], seats["B"]


def recompute(session: Session, game_type: GameType) -> int:
    """Replay every game of this type in date order, starting from each player's prior.

    Runs after every change (new, edited or deleted game, changed prior or setting), so a game
    entered late still lands in the right place. Returns the number of games replayed.
    """
    settings = game_type.settings()
    rows = {
        row.player_id: row
        for row in session.exec(
            select(PlayerRating).where(PlayerRating.game_type_id == game_type.id)
        )
    }
    for player_id in session.exec(select(Player.id)):
        if player_id not in rows:
            rows[player_id] = new_rating_row(player_id, game_type)
            session.add(rows[player_id])

    ratings = {player_id: prior(settings, **row.prior) for player_id, row in rows.items()}
    for row in rows.values():
        row.games_played = row.wins = row.losses = row.draws = 0
        row.last_played_at = None

    games = session.exec(
        select(Game)
        .where(Game.game_type_id == game_type.id)
        .order_by(col(Game.played_at), col(Game.id))
        .options(selectinload(Game.participants))  # type: ignore[arg-type]
    ).all()
    for game in games:
        side_a, side_b = sides(game)
        new = update(settings, ratings, side_a, side_b, game.score_a - game.score_b)
        for part in game.participants:
            before, after = ratings[part.player_id], new[part.player_id]
            part.mu_before, part.cov_before = before.mu.tolist(), before.cov.tolist()
            part.mu_after, part.cov_after = after.mu.tolist(), after.cov.tolist()
            session.add(part)

            row = rows[part.player_id]
            row.games_played += 1
            row.last_played_at = game.played_at
            if game.outcome == "DRAW":
                row.draws += 1
            elif game.outcome == part.side:
                row.wins += 1
            else:
                row.losses += 1
        ratings.update(new)

    for player_id, row in rows.items():
        _store(row, ratings[player_id])
        session.add(row)
    return len(games)
