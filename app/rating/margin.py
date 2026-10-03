"""Gaussian score-margin model with one skill per role (PLAN.md §2.1-2.3).

Every game is a linear observation  margin = hᵀs + ε,  ε ~ Normal(0, β²),  where s stacks the
role skills of the players involved and h holds the weight each skill has in the game. Prior and
likelihood are Gaussian, so the update is exact and in closed form (a Kalman filter step).
"""

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.linalg import block_diag

# A seat in a game: (player id, role). Role None means the player covers every role (e.g. 1v1).
Seat = tuple[Hashable, str | None]
Side = Sequence[Seat]


@dataclass(frozen=True)
class GameSettings:
    roles: tuple[str, ...]
    beta: float  # game randomness, in goals of margin
    default_sigma: float  # prior uncertainty per role skill
    role_correlation: float  # prior correlation between a player's roles
    skill_scale: float  # size of one experience step in the prior
    tau: float  # skill drift added before each game
    points_to_win: int | None = None
    allow_draws: bool = False
    team_strength: str = "mean"  # "mean" or "sum" of the players' skills


FOOSBALL = GameSettings(
    roles=("attack", "defence"),
    beta=2.5,
    default_sigma=2.5,
    role_correlation=0.7,
    skill_scale=1.5,
    tau=0.1,
    points_to_win=10,
)


@dataclass(frozen=True)
class Rating:
    roles: tuple[str, ...]
    mu: np.ndarray  # one mean per role
    cov: np.ndarray  # role covariance matrix

    def skill(self, role: str | None = None) -> tuple[float, float]:
        """Mean and sd of one role skill, or of the overall skill if role is None."""
        w = role_weights(self.roles, role)
        return float(w @ self.mu), float(np.sqrt(w @ self.cov @ w))


def role_weights(roles: Sequence[str], role: str | None) -> np.ndarray:
    if role is None:
        return np.full(len(roles), 1.0 / len(roles))
    w = np.zeros(len(roles))
    w[roles.index(role)] = 1.0
    return w


def prior(
    settings: GameSettings,
    *,
    level: float = 0.0,
    stronger: str | None = None,
    sigma_factor: float = 1.0,
    mu: Sequence[float] | None = None,
    sigma: Sequence[float] | None = None,
) -> Rating:
    """Starting belief about a new player (PLAN.md §2.5).

    level: experience steps (-1 beginner, 0 average, +1 good, +2 very good).
    stronger: the player's better role; shifts it up by half a step and the others down.
    sigma_factor: how unsure we are (1.0 low, 0.7 medium, 0.4 high confidence).
    mu / sigma: raw per-role values that override the simple inputs.
    """
    n = len(settings.roles)
    scale = settings.skill_scale
    if mu is None:
        mean = np.full(n, level * scale)
        if stronger is not None and n > 1:
            shift = -0.5 * scale / (n - 1) * np.ones(n)
            shift[settings.roles.index(stronger)] = 0.5 * scale
            mean += shift
    else:
        mean = np.asarray(mu, dtype=float)
    if sigma is None:
        sd = np.full(n, settings.default_sigma * sigma_factor)
    else:
        sd = np.asarray(sigma, dtype=float)
    rho = settings.role_correlation
    corr = rho * np.ones((n, n)) + (1 - rho) * np.eye(n)
    return Rating(settings.roles, mean, corr * np.outer(sd, sd))


def _observation(
    settings: GameSettings, ratings: Mapping[Hashable, Rating], side_a: Side, side_b: Side
) -> tuple[list[Hashable], np.ndarray, np.ndarray, np.ndarray]:
    """Stack the skills of everyone in the game: (players, h, mean, covariance incl. drift)."""
    players: list[Hashable] = []
    h_parts = []
    for sign, side in ((1.0, side_a), (-1.0, side_b)):
        share = 1.0 / len(side) if settings.team_strength == "mean" else 1.0
        for player, role in side:
            players.append(player)
            h_parts.append(sign * share * role_weights(settings.roles, role))
    if len(set(players)) != len(players):
        raise ValueError("a player can take only one seat per game")

    drift = settings.tau**2 * np.eye(len(settings.roles))
    mu = np.concatenate([ratings[p].mu for p in players])
    cov = block_diag(*[ratings[p].cov + drift for p in players])
    return players, np.concatenate(h_parts), mu, cov


def predict_margin(
    settings: GameSettings, ratings: Mapping[Hashable, Rating], side_a: Side, side_b: Side
) -> tuple[float, float]:
    """Mean and variance of margin = score_A − score_B."""
    _, h, mu, cov = _observation(settings, ratings, side_a, side_b)
    return float(h @ mu), float(h @ cov @ h + settings.beta**2)


def update(
    settings: GameSettings,
    ratings: Mapping[Hashable, Rating],
    side_a: Side,
    side_b: Side,
    margin: float,
) -> dict[Hashable, Rating]:
    """Posterior ratings of the players in a game with the observed margin (score_A − score_B).

    Only each player's own covariance is kept, not the correlations between players
    (assumed-density filtering).
    """
    players, h, mu, cov = _observation(settings, ratings, side_a, side_b)
    v = h @ cov @ h + settings.beta**2
    gain = cov @ h / v
    mu = mu + gain * (margin - h @ mu)
    cov = cov - np.outer(gain, gain) * v

    n = len(settings.roles)
    new = {}
    for i, player in enumerate(players):
        block = slice(i * n, (i + 1) * n)
        c = cov[block, block]
        new[player] = Rating(settings.roles, mu[block], (c + c.T) / 2)
    return new
