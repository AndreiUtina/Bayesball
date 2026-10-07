"""The "How it works" page: the rating model explained with worked examples.

The examples are not written out by hand: they run through the real rating engine with the
site's current settings, so every number on the page matches what the site does.
"""

import math
from dataclasses import dataclass
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from app.models import GameType
from app.rating.margin import GameSettings, Rating, Side, prior, update
from app.rating.predict import Prediction, predict
from app.ratings import PROVISIONAL_GAMES, RANKING_K
from app.web import CERTAINTY, EXPERIENCE, SessionDep, pick_game_type, render

router = APIRouter(include_in_schema=False)

# --- Small SVG charts of bell curves (drawn on the server; a script adds the hover readout) ---

WIDTH, HEIGHT = 480, 200
LEFT, RIGHT, TOP, BOTTOM = 12, 12, 30, 28  # plot insets; the bottom band holds the x labels


@dataclass(frozen=True)
class Curve:
    key: str  # CSS hook for its colour: "before", "after", "a" or "b"
    line: str  # SVG path data
    area: str = ""  # SVG path data of the shaded area under it, if any


@dataclass(frozen=True)
class Figure:
    curves: list[Curve]
    ticks: list[tuple[float, str]]  # (svg x, label) along the x axis
    labels: list[tuple[float, float, str, str]]  # direct labels: (svg x, svg y, anchor, text)
    marker: float | None  # svg x of a dotted reference line
    hover: dict[str, Any]  # what the hover script needs to compute its readout
    baseline: float = HEIGHT - BOTTOM


class Scale:
    def __init__(self, x0: float, x1: float, ymax: float):
        self.x0, self.x1, self.ymax = x0, x1, ymax

    def px(self, x: float) -> float:
        return LEFT + (x - self.x0) / (self.x1 - self.x0) * (WIDTH - LEFT - RIGHT)

    def py(self, y: float) -> float:
        return HEIGHT - BOTTOM - y / self.ymax * (HEIGHT - TOP - BOTTOM)


def pdf(x: float, mu: float, sd: float) -> float:
    return math.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * math.sqrt(2 * math.pi))


def bell(scale: Scale, mu: float, sd: float, lo: float, hi: float, n: int = 120) -> list:
    xs = [lo + (hi - lo) * i / n for i in range(n + 1)]
    return [(scale.px(x), scale.py(pdf(x, mu, sd))) for x in xs]


def line_path(points: list) -> str:
    return "M" + "L".join(f"{x:.1f},{y:.1f}" for x, y in points)


def area_path(points: list) -> str:
    base = HEIGHT - BOTTOM
    return f"{line_path(points)}L{points[-1][0]:.1f},{base}L{points[0][0]:.1f},{base}Z"


def ticks(scale: Scale, step: int = 2) -> list[tuple[float, str]]:
    first = math.ceil(scale.x0 / step) * step
    return [
        (scale.px(x), f"{x:+d}" if x else "0") for x in range(first, math.floor(scale.x1) + 1, step)
    ]


def skill_figure(before: tuple[float, float], after: tuple[float, float]) -> Figure:
    """One skill's bell curve before and after a game."""
    x0, x1 = -8.0, 8.0
    peak = max(pdf(before[0], *before), pdf(after[0], *after))
    scale = Scale(x0, x1, peak * 1.12)
    (mu_b, sd_b), (mu_a, sd_a) = before, after
    after_points = bell(scale, mu_a, sd_a, x0, x1)
    return Figure(
        curves=[
            Curve("before", line_path(bell(scale, mu_b, sd_b, x0, x1))),
            Curve("after", line_path(after_points), area_path(after_points)),
        ],
        ticks=ticks(scale),
        labels=[
            # Labels sit beside each peak, on opposite sides, so they never collide.
            (
                scale.px(mu_b) - 8,
                scale.py(pdf(mu_b, mu_b, sd_b)) - 6,
                "end",
                f"before: {mu_b:+.1f} ± {sd_b:.1f}",
            ),
            (
                scale.px(mu_a) + 8,
                scale.py(pdf(mu_a, mu_a, sd_a)) - 6,
                "start",
                f"after: {mu_a:+.1f} ± {sd_a:.1f}",
            ),
        ],
        marker=None,
        hover={
            "kind": "skill",
            "x0": x0,
            "x1": x1,
            "series": [
                {"key": "before", "label": "before the game", "mu": mu_b, "sd": sd_b},
                {"key": "after", "label": "after the game", "mu": mu_a, "sd": sd_a},
            ],
        },
    )


def margin_figure(pred: Prediction, team_a: str, team_b: str) -> Figure:
    """The predicted goal difference, split into "Team A wins" and "Team B wins"."""
    x0, x1 = -10.0, 10.0
    mu, sd = pred.expected_margin, pred.margin_sd
    scale = Scale(x0, x1, pdf(mu, mu, sd) * 1.12)
    b_side, a_side = bell(scale, mu, sd, x0, 0, 60), bell(scale, mu, sd, 0, x1, 60)
    return Figure(
        curves=[
            Curve("b", line_path(b_side), area_path(b_side)),
            Curve("a", line_path(a_side), area_path(a_side)),
        ],
        ticks=ticks(scale),
        labels=[
            (scale.px(-5), 16, "middle", f"{team_b} win: {pred.p_b:.0%}"),
            (scale.px(5), 16, "middle", f"{team_a} win: {pred.p_a:.0%}"),
        ],
        marker=scale.px(mu),
        hover={
            "kind": "margin",
            "x0": x0,
            "x1": x1,
            "teams": [team_a, team_b],
            "series": [{"key": "a", "label": "", "mu": mu, "sd": sd}],
        },
    )


# --- The worked examples -----------------------------------------------------------------------


@dataclass(frozen=True)
class SkillChange:
    who: str
    role: str
    before: tuple[float, float]  # (mu, sigma)
    after: tuple[float, float]

    @property
    def change(self) -> float:
        return self.after[0] - self.before[0]


def changes(
    before: dict[str, Rating], after: dict[str, Rating], rows: list[tuple[str, str | None]]
) -> list[SkillChange]:
    return [
        SkillChange(who, role or "overall", before[who].skill(role), after[who].skill(role))
        for who, role in rows
    ]


def worked_example(game_type: GameType) -> dict[str, Any]:
    """Four new players play twice; every number comes from the rating engine itself."""
    settings = game_type.settings()
    att, dfn = settings.roles[0], settings.roles[-1]
    win = game_type.points_to_win or 10
    new = {name: prior(settings) for name in ("Ann", "Bob", "Cat", "Dan")}
    side_a: Side = [("Ann", att), ("Bob", dfn)]
    side_b: Side = [("Cat", att), ("Dan", dfn)]
    team_a, team_b = "Ann & Bob", "Cat & Dan"

    # Game 1: everyone is new, Ann & Bob win 10–4.
    first = predict(settings, new, side_a, side_b)
    score1 = (win, win - 6)
    margin1 = score1[0] - score1[1]
    after1 = {**new, **update(settings, new, side_a, side_b, margin1)}

    # Game 2: the rematch. Ann & Bob are now favourites, and win 10–8: by less than expected.
    rematch = predict(settings, after1, side_a, side_b)
    score2 = (win, win - 2)
    after2 = {**after1, **update(settings, after1, side_a, side_b, score2[0] - score2[1])}

    rows = [("Ann", att), ("Ann", dfn), ("Cat", att), ("Cat", dfn)]
    ann_attack = changes(new, after1, [("Ann", att)])[0]
    return {
        "att": att,
        "dfn": dfn,
        "team_a": team_a,
        "team_b": team_b,
        "first": first,
        "score1": score1,
        "margin1": margin1,
        "game1": changes(new, after1, rows),
        "gain": ann_attack.change / margin1,
        "rematch": rematch,
        "score2": score2,
        "surprise2": score2[0] - score2[1] - rematch.expected_margin,
        "game2": changes(after1, after2, [("Ann", None), ("Bob", None), ("Cat", None)]),
        "skill_figure": skill_figure(ann_attack.before, ann_attack.after),
        "margin_figure": margin_figure(rematch, team_a, team_b),
    }


# --- Common questions --------------------------------------------------------------------------

REGULAR_SIGMA = 1.0  # the questions use "regular" players, each skill known to about ±1 goal


def regulars_game(
    settings: GameSettings, opponents: float, score: tuple[int, int]
) -> tuple[float, float]:
    """Ann & Bob (average regulars) play Cat & Dan (regulars `opponents` goals better).

    Returns (expected goal difference, change in Ann's skill in her role).
    """
    att, dfn = settings.roles[0], settings.roles[-1]
    sd = [REGULAR_SIGMA] * len(settings.roles)
    ratings = {p: prior(settings, mu=[0.0] * len(sd), sigma=sd) for p in ("Ann", "Bob")}
    ratings |= {p: prior(settings, mu=[opponents] * len(sd), sigma=sd) for p in ("Cat", "Dan")}
    side_a: Side = [("Ann", att), ("Bob", dfn)]
    side_b: Side = [("Cat", att), ("Dan", dfn)]
    expected = predict(settings, ratings, side_a, side_b).expected_margin
    after = update(settings, ratings, side_a, side_b, score[0] - score[1])
    return expected, after["Ann"].skill(att)[0] - ratings["Ann"].skill(att)[0]


def half_life(settings: GameSettings) -> float | None:
    """Roughly after how many later games an old game's influence halves, for a regular player.

    Follows one skill of a player who keeps playing 2v2 with and against equally well-known
    players, until their uncertainty settles; then each new game keeps (1 − gain/2) of the
    weight of every older one. None if skills never drift (then all games count equally).
    """
    if settings.tau <= 0:
        return None
    variance = settings.default_sigma**2
    for _ in range(10_000):
        before = variance + settings.tau**2
        # The four skills in play each count ½; luck adds β².
        gain = 0.5 * before / (before + settings.beta**2)
        variance = before - 0.5 * gain * before
    return math.log(0.5) / math.log(1 - 0.5 * gain)


def common_questions(game_type: GameType) -> dict[str, Any]:
    settings = game_type.settings()
    win = game_type.points_to_win or 10
    scores = [(win, win - 1), (win, win - 2), (win, win - 5), (win, win - 8), (win - 2, win)]
    return {
        "sigma": REGULAR_SIGMA,
        "by_score": [(score, regulars_game(settings, 0.0, score)[1]) for score in scores],
        "by_opponent": [
            (strength, *regulars_game(settings, strength, (win, win - 2)))
            for strength in (-2.0, 0.0, 2.0, 4.0)
        ],
        "close_win": (win, win - 2),
        "half_life": half_life(settings),
    }


@router.get("/how-it-works", response_class=HTMLResponse)
def how_it_works(
    request: Request, session: SessionDep, game_type_id: int | None = None
) -> HTMLResponse:
    game_type = pick_game_type(session, game_type_id)
    settings = game_type.settings()
    new = prior(settings)
    return render(
        request,
        session,
        "how.html",
        page="how",
        game_type=game_type,
        s=settings,
        k=RANKING_K,
        provisional_games=PROVISIONAL_GAMES,
        new_overall_sd=new.skill()[1],
        experience=[(label, float(level) * settings.skill_scale) for level, label in EXPERIENCE],
        certainty=[(key, f * settings.default_sigma) for key, f in CERTAINTY.items()],
        ex=worked_example(game_type),
        faq=common_questions(game_type),
    )
