"""Read-only views of past games for the profile and history pages (PLAN.md G5)."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
from sqlalchemy.orm import selectinload
from sqlmodel import Session, col, select

from app.models import Game, GameParticipant, GameType, Player
from app.rating.margin import Rating, role_weights


def rating_change(part: GameParticipant, weights: np.ndarray) -> float:
    """How much a player's skill (weighted over roles) moved in one game."""
    return float(weights @ (np.array(part.mu_after) - np.array(part.mu_before)))


@dataclass(frozen=True)
class SeatView:
    player_id: int
    name: str
    role: str | None
    change: float  # change in overall skill


@dataclass(frozen=True)
class GameView:
    id: int
    played_at: datetime
    score_a: int
    score_b: int
    outcome: str  # "A", "B" or "DRAW"
    notes: str
    side_a: list[SeatView]
    side_b: list[SeatView]

    def side_of(self, player_id: int) -> str:
        return "A" if any(s.player_id == player_id for s in self.side_a) else "B"

    def seat(self, player_id: int) -> SeatView:
        return next(s for s in (*self.side_a, *self.side_b) if s.player_id == player_id)

    def result_for(self, player_id: int) -> str:
        """ "W", "L" or "D" from this player's point of view."""
        if self.outcome == "DRAW":
            return "D"
        return "W" if self.outcome == self.side_of(player_id) else "L"

    def score_for(self, player_id: int) -> tuple[int, int]:
        """(own team's score, other team's score)."""
        a, b = self.score_a, self.score_b
        return (a, b) if self.side_of(player_id) == "A" else (b, a)

    def teams_for(self, player_id: int) -> tuple[list[SeatView], list[SeatView]]:
        """(teammates, opponents) of this player."""
        own, other = (
            (self.side_a, self.side_b)
            if self.side_of(player_id) == "A"
            else (self.side_b, self.side_a)
        )
        return [s for s in own if s.player_id != player_id], other


def game_views(session: Session, games: Sequence[Game], game_type: GameType) -> list[GameView]:
    names = dict(session.exec(select(Player.id, Player.name)).all())
    weights = role_weights(game_type.roles, None)

    def seats(game: Game, side: str) -> list[SeatView]:
        return [
            SeatView(p.player_id, names[p.player_id], p.role, rating_change(p, weights))
            for p in game.participants
            if p.side == side
        ]

    return [
        GameView(
            id=g.id,
            played_at=g.played_at,
            score_a=g.score_a,
            score_b=g.score_b,
            outcome=g.outcome,
            notes=g.notes,
            side_a=seats(g, "A"),
            side_b=seats(g, "B"),
        )
        for g in games
    ]


def games_query(game_type: GameType, player_id: int | None = None):
    """Games of this type (optionally only one player's), newest first."""
    query = select(Game).where(Game.game_type_id == game_type.id)
    if player_id is not None:
        query = query.where(
            Game.participants.any(GameParticipant.player_id == player_id)  # type: ignore[attr-defined]
        )
    return query.order_by(col(Game.played_at).desc(), col(Game.id).desc()).options(
        selectinload(Game.participants)  # type: ignore[arg-type]
    )


def player_games(session: Session, game_type: GameType, player_id: int) -> list[Game]:
    """All of a player's games of this type, oldest first."""
    return list(reversed(session.exec(games_query(game_type, player_id)).all()))


def rating_chart(games: Sequence[Game], player_id: int, game_type: GameType) -> dict[str, Any]:
    """The player's skill before their first game and after each game, for Chart.js.

    One series per role, plus overall; each with the best guess (mu) and its uncertainty (sd).
    """
    roles = tuple(game_type.roles)
    parts = [next(p for p in g.participants if p.player_id == player_id) for g in games]
    points = [(parts[0].mu_before, parts[0].cov_before)] + [
        (p.mu_after, p.cov_after) for p in parts
    ]
    ratings = [Rating(roles, np.array(mu), np.array(cov)) for mu, cov in points]
    keys: list[str | None] = [*roles, None] if len(roles) > 1 else [None]
    series = []
    for role in keys:
        skills = [r.skill(role) for r in ratings]
        series.append(
            {
                "name": role.capitalize() if role else "Overall",
                "band": role is not None or len(keys) == 1,  # shade ± sd around this line
                "mu": [round(mu, 3) for mu, _ in skills],
                "sd": [round(sd, 3) for _, sd in skills],
            }
        )
    return {"labels": ["Start", *(str(i) for i in range(1, len(games) + 1))], "series": series}


@dataclass
class PairRecord:
    """A player's record with (or against) one other player."""

    player_id: int
    name: str
    games: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0

    @property
    def win_rate(self) -> float:
        return self.wins / self.games if self.games else 0.0


def pair_records(
    views: Sequence[GameView], player_id: int
) -> tuple[list[PairRecord], list[PairRecord]]:
    """(teammates best first, opponents most played first) of one player."""
    mates: dict[int, PairRecord] = {}
    opponents: dict[int, PairRecord] = {}
    for view in views:
        result = view.result_for(player_id)
        own, other = view.teams_for(player_id)
        for records, seats in ((mates, own), (opponents, other)):
            for seat in seats:
                record = records.setdefault(seat.player_id, PairRecord(seat.player_id, seat.name))
                record.games += 1
                record.wins += result == "W"
                record.losses += result == "L"
                record.draws += result == "D"
    return (
        sorted(mates.values(), key=lambda r: (-r.win_rate, -r.games, r.name)),
        sorted(opponents.values(), key=lambda r: (-r.games, -r.win_rate, r.name)),
    )
