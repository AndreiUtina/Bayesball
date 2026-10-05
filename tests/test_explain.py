import pytest

from app.explain import HEIGHT, WIDTH, margin_figure, worked_example
from app.models import GameType
from app.rating.margin import FOOSBALL


@pytest.fixture
def example():
    return worked_example(GameType.from_settings("Foosball", FOOSBALL))


def test_worked_example_follows_the_model(example):
    assert example["first"].p_a == pytest.approx(0.5)
    ann_attack, ann_defence, cat_attack, _ = example["game1"]
    assert ann_attack.change == pytest.approx(example["margin1"] * example["gain"])
    assert ann_attack.change > ann_defence.change > 0  # the unplayed role follows, less
    assert cat_attack.change == pytest.approx(-ann_attack.change)
    assert ann_attack.after[1] < ann_attack.before[1]  # less uncertain after a game
    # The rematch: favourites win by less than expected, so they go down.
    assert example["rematch"].p_a > 0.5 and example["surprise2"] < 0
    ann, bob, cat = example["game2"]
    assert ann.change < 0 and bob.change < 0 and cat.change > 0


def test_figures_stay_inside_the_chart(example):
    for figure in (example["skill_figure"], example["margin_figure"]):
        assert len(figure.curves) == 2
        for x, y, _, _ in figure.labels:
            assert 0 <= x <= WIDTH and 0 <= y <= HEIGHT
        assert all(0 <= x <= WIDTH for x, _ in figure.ticks)


def test_margin_figure_labels_the_chances(example):
    figure = margin_figure(example["rematch"], "A", "B")
    texts = [text for *_, text in figure.labels]
    assert texts == [
        f"B win: {example['rematch'].p_b:.0%}",
        f"A win: {example['rematch'].p_a:.0%}",
    ]


def test_how_it_works_page(visitor):
    page = visitor.get("/how-it-works").text
    assert "<h1>How the ratings work</h1>" in page
    assert "P(skill | result)" in page
    assert page.count('<svg viewBox="0 0 480 200"') == 2
    assert "+1.50" in page and "81%" in page and "10–7" in page  # the live worked example
    assert 'href="/how-it-works"' in visitor.get("/").text  # linked from the nav


def test_how_it_works_uses_the_live_settings(client):
    before = client.get("/how-it-works").text
    client.patch("/api/game-types/1", json={"beta": 1.0})
    after = client.get("/how-it-works").text
    assert "β = 2.5 goals" in before and "β = 1.0 goals" in after
    assert "+1.50" in before and "+1.50" not in after  # less luck: the same game says more
