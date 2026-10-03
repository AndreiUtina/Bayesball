import numpy as np
import pytest

from app.rating.margin import FOOSBALL, predict_margin, prior, update

S = FOOSBALL
TWO_V_TWO = ((("a", "attack"), ("b", "defence")), (("c", "attack"), ("d", "defence")))


def fresh(*players):
    return {p: prior(S) for p in players}


def test_default_prior():
    r = prior(S)
    sd, rho = S.default_sigma, S.role_correlation
    np.testing.assert_allclose(r.mu, [0, 0])
    np.testing.assert_allclose(r.cov, sd**2 * np.array([[1, rho], [rho, 1]]))


def test_prior_from_form_inputs():
    r = prior(S, level=1, stronger="attack", sigma_factor=0.4)
    np.testing.assert_allclose(r.mu, [1.5 + 0.75, 1.5 - 0.75])
    assert r.skill("attack")[1] == pytest.approx(0.4 * S.default_sigma)
    assert r.skill()[0] == pytest.approx(1.5)


def test_win_moves_winners_up_losers_down_and_shrinks_uncertainty():
    ratings = fresh("a", "b", "c", "d")
    new = update(S, ratings, *TWO_V_TWO, margin=10 - 4)
    for p in "ab":
        assert new[p].skill()[0] > 0
    for p in "cd":
        assert new[p].skill()[0] < 0
    for p in "abcd":
        assert new[p].skill()[1] < ratings[p].skill()[1]


def test_used_role_moves_most_and_other_role_follows_through_correlation():
    new = update(S, fresh("a", "b", "c", "d"), *TWO_V_TWO, margin=6)
    attack, defence = new["a"].mu
    assert attack > defence > 0


def test_expected_result_changes_nothing():
    ratings = fresh("a", "b", "c", "d")
    ratings["a"] = prior(S, level=2)
    m, _ = predict_margin(S, ratings, *TWO_V_TWO)
    new = update(S, ratings, *TWO_V_TWO, margin=m)
    for p in "abcd":
        np.testing.assert_allclose(new[p].mu, ratings[p].mu)


def test_one_v_one_uses_overall_skill():
    new = update(S, fresh("a", "b"), [("a", None)], [("b", None)], margin=-5)
    attack, defence = new["a"].mu
    assert attack == pytest.approx(defence)
    assert attack < 0


def test_same_player_twice_is_rejected():
    with pytest.raises(ValueError):
        update(S, fresh("a", "b"), [("a", "attack"), ("b", "defence")], [("a", None)], margin=1)
