"""Recompute ratings from scratch by replaying games in order (PLAN.md §6)."""

from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass

from app.rating.margin import GameSettings, Rating, Side, update


@dataclass(frozen=True)
class GameResult:
    side_a: Side
    side_b: Side
    margin: float  # score_A − score_B


def replay(
    settings: GameSettings, priors: Mapping[Hashable, Rating], games: Iterable[GameResult]
) -> dict[Hashable, Rating]:
    ratings = dict(priors)
    for game in games:
        ratings.update(update(settings, ratings, game.side_a, game.side_b, game.margin))
    return ratings
