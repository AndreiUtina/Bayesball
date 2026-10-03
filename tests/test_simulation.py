"""Phase 1 validation (PLAN.md §9): recover known skills from simulated race-to-10 games.

20 players and 500 games per run. One run is noisy (even the exact joint posterior gives an
overall Spearman of ~0.85-0.95 depending on the seed), so results are pooled over a few seeds.
"""

import numpy as np
import pytest
from scipy.stats import spearmanr

from app.rating.margin import FOOSBALL, prior, update
from app.rating.predict import predict

S = FOOSBALL
N_PLAYERS, N_GAMES, TRUE_SD = 20, 500, 1.5
SEEDS = range(4)


def race_to_10(true_margin: float) -> int:
    """Round a continuous margin to a valid race-to-10 result: winner 10, loser 0-9."""
    m = int(round(true_margin)) or (1 if true_margin >= 0 else -1)
    return int(np.clip(m, -10, 10))


def simulate(seed: int):
    rng = np.random.default_rng(seed)
    rho = S.role_correlation
    true_cov = TRUE_SD**2 * np.array([[1, rho], [rho, 1]])
    true = rng.multivariate_normal([0, 0], true_cov, size=N_PLAYERS)
    true -= true.mean(axis=0)  # the model only sees skill differences, so centre the truth

    def strength(side):
        return np.mean(
            [true[p].mean() if r is None else true[p, S.roles.index(r)] for p, r in side]
        )

    ratings = {p: prior(S) for p in range(N_PLAYERS)}
    log = []  # (P(A wins), A won, expected margin, actual margin), predicted before each game
    for _ in range(N_GAMES):
        if rng.random() < 0.8:
            a1, a2, b1, b2 = rng.choice(N_PLAYERS, 4, replace=False)
            side_a = list(zip((a1, a2), rng.permutation(S.roles), strict=True))
            side_b = list(zip((b1, b2), rng.permutation(S.roles), strict=True))
        else:
            a, b = rng.choice(N_PLAYERS, 2, replace=False)
            side_a, side_b = [(a, None)], [(b, None)]
        margin = race_to_10(strength(side_a) - strength(side_b) + rng.normal(0, S.beta))

        pred = predict(S, ratings, side_a, side_b)
        log.append((pred.p_a, margin > 0, pred.expected_margin, margin))
        ratings.update(update(S, ratings, side_a, side_b, margin))
    return true, ratings, np.array(log, dtype=float)


@pytest.fixture(scope="module")
def runs():
    return [simulate(seed) for seed in SEEDS]


def test_ranking_is_recovered(runs):
    overall, roles = [], []
    for true, ratings, _ in runs:
        est = np.array([ratings[p].mu for p in range(N_PLAYERS)])
        overall.append(spearmanr(true.mean(axis=1), est.mean(axis=1)).statistic)
        roles += [spearmanr(true[:, r], est[:, r]).statistic for r in range(len(S.roles))]
    assert np.mean(overall) > 0.85
    assert np.mean(roles) > 0.8


def test_win_probabilities_are_calibrated(runs):
    log = np.concatenate([log[100:] for *_, log in runs])  # skip the warm-up games
    p_a, a_won = log[:, 0], log[:, 1]
    p_fav = np.maximum(p_a, 1 - p_a)
    fav_won = np.where(p_a >= 0.5, a_won, 1 - a_won)
    bins = np.digitize(p_fav, [0.6, 0.7, 0.8, 0.9])
    error = sum(
        abs(fav_won[bins == b].mean() - p_fav[bins == b].mean()) * (bins == b).mean()
        for b in np.unique(bins)
    )
    assert error < 0.05


def test_margin_beats_predicting_zero(runs):
    log = np.concatenate([log for *_, log in runs])
    expected, actual = log[:, 2], log[:, 3]
    assert np.mean((actual - expected) ** 2) < 0.9 * np.mean(actual**2)


def test_uncertainty_intervals_cover_the_truth(runs):
    z = [
        (true[p, r] - ratings[p].mu[r]) / np.sqrt(ratings[p].cov[r, r])
        for true, ratings, _ in runs
        for p in range(N_PLAYERS)
        for r in range(len(S.roles))
    ]
    coverage = np.mean(np.abs(z) < 1.645)  # 90% interval
    # Usually above 90%: the engine adds drift (tau) each game but the simulated skills are fixed.
    assert coverage >= 0.8
