"""Win probabilities, expected scoreline and the team & role balancer (PLAN.md §2.7)."""

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations, permutations

import numpy as np
from scipy.stats import norm

from app.rating.margin import GameSettings, Rating, Side, predict_margin


@dataclass(frozen=True)
class Prediction:
    p_a: float
    p_draw: float
    p_b: float
    expected_margin: float  # score_A − score_B
    margin_sd: float
    score: tuple[int, int] | None  # expected scoreline (A, B) for race-to games


@dataclass(frozen=True)
class Matchup:
    side_a: Side
    side_b: Side
    prediction: Prediction


def expected_score(margin: float, points_to_win: int | None) -> tuple[int, int] | None:
    if points_to_win is None:
        return None
    loser = int(np.clip(round(points_to_win - abs(margin)), 0, points_to_win - 1))
    return (points_to_win, loser) if margin >= 0 else (loser, points_to_win)


def predict(
    settings: GameSettings, ratings: Mapping[Hashable, Rating], side_a: Side, side_b: Side
) -> Prediction:
    m, v = predict_margin(settings, ratings, side_a, side_b)
    sd = float(np.sqrt(v))
    if settings.allow_draws:
        p_b = float(norm.cdf((-0.5 - m) / sd))
        p_draw = float(norm.cdf((0.5 - m) / sd)) - p_b
    else:
        p_b = float(norm.cdf(-m / sd))
        p_draw = 0.0
    return Prediction(
        p_a=1.0 - p_b - p_draw,
        p_draw=p_draw,
        p_b=p_b,
        expected_margin=m,
        margin_sd=sd,
        score=expected_score(m, settings.points_to_win),
    )


def balance(
    settings: GameSettings, ratings: Mapping[Hashable, Rating], players: Sequence[Hashable]
) -> list[Matchup]:
    """Every split of the players into two teams and every role assignment, fairest first.

    For 4 foosball players that is 3 splits × 2 × 2 role choices = 12 matchups.
    """
    if len(players) < 2 or len(players) % 2:
        raise ValueError("need an even number of players, at least 2")
    team_size = len(players) // 2
    if team_size == 1:
        role_options = [(None,)]
    elif team_size == len(settings.roles):
        role_options = list(permutations(settings.roles))
    else:
        raise ValueError(f"teams of {team_size} don't match roles {settings.roles}")

    first, rest = players[0], players[1:]
    matchups = []
    for mates in combinations(rest, team_size - 1):  # fixing `first` in team A skips mirrors
        team_a = (first, *mates)
        team_b = tuple(p for p in rest if p not in mates)
        for roles_a in role_options:
            for roles_b in role_options:
                side_a = tuple(zip(team_a, roles_a, strict=True))
                side_b = tuple(zip(team_b, roles_b, strict=True))
                pred = predict(settings, ratings, side_a, side_b)
                matchups.append(Matchup(side_a, side_b, pred))
    return sorted(matchups, key=lambda m: abs(m.prediction.p_a - 0.5))
