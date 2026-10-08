from dataclasses import replace

import numpy as np
import pytest

from app.rating.margin import FOOSBALL, prior
from app.rating.predict import (
    arrangements,
    balance,
    expected_score,
    predict,
    role_options,
    winning_margin,
)

S = FOOSBALL


def test_equal_teams_are_a_coin_flip():
    ratings = {p: prior(S) for p in "ab"}
    pred = predict(S, ratings, [("a", None)], [("b", None)])
    assert pred.p_a == pytest.approx(0.5)
    assert pred.p_draw == 0
    assert pred.score == (10, 7)  # someone still wins, typically by about 3 goals


def test_stronger_side_is_favoured_and_swapping_sides_mirrors():
    ratings = {"a": prior(S, level=2), "b": prior(S)}
    ab = predict(S, ratings, [("a", None)], [("b", None)])
    ba = predict(S, ratings, [("b", None)], [("a", None)])
    assert ab.p_a > 0.5
    assert ab.p_a == pytest.approx(ba.p_b)
    assert ab.expected_margin == pytest.approx(-ba.expected_margin)


def test_draws_when_allowed():
    s = replace(S, allow_draws=True, points_to_win=None)
    ratings = {p: prior(s) for p in "ab"}
    pred = predict(s, ratings, [("a", None)], [("b", None)])
    assert pred.p_draw > 0
    assert pred.p_a == pytest.approx(pred.p_b)
    assert pred.p_a + pred.p_draw + pred.p_b == pytest.approx(1)
    assert pred.score is None


@pytest.mark.parametrize(
    ("margin", "sd", "score"),
    [
        (0.0, 2.5, (10, 8)),  # an even game: the winner still wins by about 2
        (0.3, 2.9, (10, 8)),  # a near-even game is no longer shown as 10-9
        (3.2, 2.5, (10, 6)),
        (-2.6, 2.5, (7, 10)),  # Team B favoured
        (14.0, 2.5, (10, 0)),
    ],
)
def test_expected_score(margin, sd, score):
    assert expected_score(margin, sd, 10) == score
    assert expected_score(margin, sd, None) is None


@pytest.mark.parametrize(("margin", "sd"), [(0.0, 2.5), (1.0, 3.0), (-3.0, 3.4)])
def test_winning_margin_matches_simulated_games(margin, sd):
    goal_differences = np.random.default_rng(0).normal(abs(margin), sd, 400_000)
    wins = goal_differences[goal_differences > 0]
    assert winning_margin(margin, sd) == pytest.approx(wins.mean(), abs=0.02)


def test_balance_four_players():
    ratings = {
        "ace": prior(S, level=2, stronger="attack"),
        "wall": prior(S, level=1, stronger="defence"),
        "newbie": prior(S, level=-1),
        "avg": prior(S),
    }
    options = balance(S, ratings, list(ratings))
    assert len(options) == 12
    gaps = [abs(o.prediction.p_a - 0.5) for o in options]
    assert gaps == sorted(gaps)
    for o in options:
        assert sorted(p for p, _ in (*o.side_a, *o.side_b)) == sorted(ratings)
    # The two strongest players should not end up together in the fairest matchup.
    best = {p for p, _ in options[0].side_a}
    assert best != {"ace", "wall"} and best != {"newbie", "avg"}


def test_role_options_for_fixed_teams():
    ratings = {
        "striker": prior(S, stronger="attack"),
        "keeper": prior(S, stronger="defence"),
        "c": prior(S),
        "d": prior(S),
    }
    wrong_way = [("striker", "defence"), ("keeper", "attack")]
    assert arrangements(wrong_way) == [
        (("striker", "defence"), ("keeper", "attack")),
        (("striker", "attack"), ("keeper", "defence")),
    ]
    assert arrangements([("c", None)]) == [(("c", None),)]

    grid = role_options(S, ratings, wrong_way, [("c", "attack"), ("d", "defence")])
    assert len(grid) == 2 and len(grid[0]) == 2
    assert grid[1][0].p_a > 0.5 > grid[0][0].p_a  # swapping puts both in their strong role
