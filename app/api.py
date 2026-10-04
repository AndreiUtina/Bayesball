"""JSON API (PLAN.md §7). No logins yet: writes are open until Phase 5 adds admins."""

from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlmodel import Session, SQLModel, col, select

from app.db import get_session
from app.models import Game, GameParticipant, GameType, Player, PlayerRating, utcnow
from app.rating.margin import Seat
from app.rating.predict import Prediction, balance, predict
from app.ratings import (
    PROVISIONAL_GAMES,
    RANKING_K,
    RuleError,
    as_utc,
    check_game,
    check_match,
    check_players,
    current_ratings,
    make_participants,
    new_rating_row,
    outcome,
    recompute,
    sides,
    to_rating,
)

router = APIRouter(prefix="/api", tags=["api"])
SessionDep = Annotated[Session, Depends(get_session)]


def get_or_404[T: SQLModel](session: Session, model: type[T], id_: int) -> T:
    obj = session.get(model, id_)
    if obj is None:
        raise HTTPException(404, f"{model.__name__} {id_} not found")
    return obj


# --- Schemas -----------------------------------------------------------------------------------


class SeatIn(BaseModel):
    player_id: int
    role: str | None = None  # leave empty in 1v1


class PriorIn(BaseModel):
    """Starting belief about a player (PLAN.md §2.5)."""

    level: float = Field(0.0, description="-1 beginner, 0 average, 1 good, 2 very good")
    stronger: str | None = Field(None, description="the player's better role, if any")
    sigma_factor: float = Field(1.0, gt=0, description="1.0 low, 0.7 medium, 0.4 high certainty")
    mu: list[float] | None = Field(None, description="advanced: raw mean per role")
    sigma: list[float] | None = Field(None, description="advanced: raw sd per role")


class PlayerCreate(BaseModel):
    name: str = Field(min_length=1, max_length=50)
    game_type_id: int | None = Field(None, description="the game type `prior` applies to")
    prior: PriorIn = PriorIn()


class PlayerUpdate(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=50)
    active: bool | None = None
    game_type_id: int | None = Field(None, description="the game type `prior` applies to")
    prior: PriorIn | None = None


class SkillOut(BaseModel):
    mu: float
    sigma: float


class RatingOut(BaseModel):
    game_type_id: int
    overall: SkillOut
    roles: dict[str, SkillOut]
    ranking_score: float  # overall mu − k·sigma
    games_played: int
    wins: int
    losses: int
    draws: int
    provisional: bool
    prior: dict[str, Any]


class PlayerOut(BaseModel):
    id: int
    name: str
    active: bool
    created_at: datetime
    ratings: list[RatingOut]


class GameCreate(BaseModel):
    game_type_id: int
    side_a: list[SeatIn]
    side_b: list[SeatIn]
    score_a: int
    score_b: int
    played_at: datetime | None = None  # defaults to now
    notes: str = ""


class GameUpdate(BaseModel):
    side_a: list[SeatIn] | None = None
    side_b: list[SeatIn] | None = None
    score_a: int | None = None
    score_b: int | None = None
    played_at: datetime | None = None
    notes: str | None = None


class SeatOut(BaseModel):
    player_id: int
    name: str
    role: str | None
    mu_before: dict[str, float] | None
    mu_after: dict[str, float] | None


class GameOut(BaseModel):
    id: int
    game_type_id: int
    played_at: datetime
    score_a: int
    score_b: int
    outcome: str
    notes: str
    side_a: list[SeatOut]
    side_b: list[SeatOut]


class GameTypeUpdate(BaseModel):
    name: str | None = Field(None, min_length=1)
    beta: float | None = Field(None, gt=0)
    default_sigma: float | None = Field(None, gt=0)
    role_correlation: float | None = Field(None, ge=0, lt=1)
    skill_scale: float | None = Field(None, ge=0)
    tau: float | None = Field(None, ge=0)


class MatchIn(BaseModel):
    game_type_id: int
    side_a: list[SeatIn]
    side_b: list[SeatIn]


class PredictionOut(BaseModel):
    p_a: float
    p_draw: float
    p_b: float
    expected_margin: float
    margin_interval: tuple[float, float]  # 90% interval of score_A − score_B
    score: tuple[int, int] | None


class BalanceIn(BaseModel):
    game_type_id: int
    player_ids: list[int]


class MatchupOut(BaseModel):
    side_a: list[SeatIn]
    side_b: list[SeatIn]
    prediction: PredictionOut


class RecomputeIn(BaseModel):
    game_type_id: int


# --- Conversions -------------------------------------------------------------------------------


def to_seats(side: list[SeatIn]) -> list[Seat]:
    return [(s.player_id, s.role) for s in side]


def from_seats(side: Sequence[Seat]) -> list[SeatIn]:
    return [SeatIn(player_id=p, role=r) for p, r in side]


def rating_out(row: PlayerRating, game_type: GameType) -> RatingOut:
    rating = to_rating(row, game_type.settings())

    def skill(role: str | None = None) -> SkillOut:
        mu, sigma = rating.skill(role)
        return SkillOut(mu=mu, sigma=sigma)

    overall = skill()
    return RatingOut(
        game_type_id=row.game_type_id,
        overall=overall,
        roles={role: skill(role) for role in game_type.roles},
        ranking_score=overall.mu - RANKING_K * overall.sigma,
        games_played=row.games_played,
        wins=row.wins,
        losses=row.losses,
        draws=row.draws,
        provisional=row.games_played < PROVISIONAL_GAMES,
        prior=row.prior,
    )


def players_out(
    session: Session, players: Sequence[Player], game_type_id: int | None = None
) -> list[PlayerOut]:
    query = select(PlayerRating, GameType).where(
        PlayerRating.game_type_id == GameType.id,
        col(PlayerRating.player_id).in_([p.id for p in players]),
    )
    if game_type_id is not None:
        query = query.where(GameType.id == game_type_id)
    ratings = defaultdict(list)
    for row, game_type in session.exec(query):
        ratings[row.player_id].append(rating_out(row, game_type))
    return [
        PlayerOut(
            id=p.id, name=p.name, active=p.active, created_at=p.created_at, ratings=ratings[p.id]
        )
        for p in players
    ]


def games_out(session: Session, games: Sequence[Game]) -> list[GameOut]:
    names = dict(session.exec(select(Player.id, Player.name)).all())
    roles = {gt.id: gt.roles for gt in session.exec(select(GameType))}

    def seats_out(game: Game, side: str) -> list[SeatOut]:
        def by_role(values: list[float] | None) -> dict[str, float] | None:
            return (
                None if values is None else dict(zip(roles[game.game_type_id], values, strict=True))
            )

        return [
            SeatOut(
                player_id=part.player_id,
                name=names[part.player_id],
                role=part.role,
                mu_before=by_role(part.mu_before),
                mu_after=by_role(part.mu_after),
            )
            for part in game.participants
            if part.side == side
        ]

    return [
        GameOut(
            id=game.id,
            game_type_id=game.game_type_id,
            played_at=game.played_at,
            score_a=game.score_a,
            score_b=game.score_b,
            outcome=game.outcome,
            notes=game.notes,
            side_a=seats_out(game, "A"),
            side_b=seats_out(game, "B"),
        )
        for game in games
    ]


def prediction_out(pred: Prediction) -> PredictionOut:
    m, half = pred.expected_margin, 1.645 * pred.margin_sd
    return PredictionOut(
        p_a=pred.p_a,
        p_draw=pred.p_draw,
        p_b=pred.p_b,
        expected_margin=m,
        margin_interval=(m - half, m + half),
        score=pred.score,
    )


# --- Game types --------------------------------------------------------------------------------


@router.get("/game-types")
def list_game_types(session: SessionDep) -> list[GameType]:
    return list(session.exec(select(GameType).order_by(col(GameType.id))))


@router.patch("/game-types/{game_type_id}")
def update_game_type(game_type_id: int, body: GameTypeUpdate, session: SessionDep) -> GameType:
    game_type = get_or_404(session, GameType, game_type_id)
    changes = body.model_dump(exclude_unset=True, exclude_none=True)
    if (
        "name" in changes
        and session.exec(
            select(GameType).where(GameType.name == changes["name"], GameType.id != game_type_id)
        ).first()
    ):
        raise HTTPException(409, f"a game type called {changes['name']!r} already exists")
    for key, value in changes.items():
        setattr(game_type, key, value)
    session.add(game_type)
    recompute(session, game_type)
    session.commit()
    session.refresh(game_type)
    return game_type


# --- Players -----------------------------------------------------------------------------------


def _check_name_free(session: Session, name: str, player_id: int | None = None) -> None:
    existing = session.exec(select(Player).where(Player.name == name)).first()
    if existing is not None and existing.id != player_id:
        raise HTTPException(409, f"a player called {name!r} already exists")


@router.get("/players")
def list_players(
    session: SessionDep,
    game_type_id: Annotated[
        int | None, Query(description="only this game type's ratings, best ranked first")
    ] = None,
) -> list[PlayerOut]:
    players = session.exec(select(Player).order_by(col(Player.name))).all()
    out = players_out(session, players, game_type_id)
    if game_type_id is not None:
        out.sort(key=lambda p: -p.ratings[0].ranking_score if p.ratings else float("inf"))
    return out


@router.get("/players/{player_id}")
def get_player(player_id: int, session: SessionDep) -> PlayerOut:
    return players_out(session, [get_or_404(session, Player, player_id)])[0]


@router.post("/players", status_code=201)
def create_player(body: PlayerCreate, session: SessionDep) -> PlayerOut:
    _check_name_free(session, body.name)
    if body.game_type_id is not None:
        get_or_404(session, GameType, body.game_type_id)
    elif "prior" in body.model_fields_set:
        raise RuleError("say which game type the prior is for (game_type_id)")

    player = Player(name=body.name)
    session.add(player)
    session.flush()
    for game_type in session.exec(select(GameType)):
        inputs = (
            body.prior.model_dump(exclude_none=True) if game_type.id == body.game_type_id else {}
        )
        session.add(new_rating_row(player.id, game_type, inputs))
    session.commit()
    return players_out(session, [player])[0]


@router.patch("/players/{player_id}")
def update_player(player_id: int, body: PlayerUpdate, session: SessionDep) -> PlayerOut:
    player = get_or_404(session, Player, player_id)
    if body.name is not None:
        _check_name_free(session, body.name, player_id)
        player.name = body.name
    if body.active is not None:
        player.active = body.active
    session.add(player)

    if body.prior is not None:
        if body.game_type_id is None:
            raise RuleError("say which game type the prior is for (game_type_id)")
        game_type = get_or_404(session, GameType, body.game_type_id)
        row = session.exec(
            select(PlayerRating).where(
                PlayerRating.player_id == player_id, PlayerRating.game_type_id == game_type.id
            )
        ).first()
        new_row = new_rating_row(player_id, game_type, body.prior.model_dump(exclude_none=True))
        if row is None:
            session.add(new_row)
        else:
            row.prior = new_row.prior
            session.add(row)
        recompute(session, game_type)
    session.commit()
    return players_out(session, [player])[0]


# --- Games -------------------------------------------------------------------------------------


@router.get("/games")
def list_games(
    session: SessionDep,
    game_type_id: int | None = None,
    player_id: int | None = None,
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[GameOut]:
    """Newest first."""
    query = select(Game).order_by(col(Game.played_at).desc(), col(Game.id).desc())
    if game_type_id is not None:
        query = query.where(Game.game_type_id == game_type_id)
    if player_id is not None:
        query = query.where(Game.participants.any(GameParticipant.player_id == player_id))  # type: ignore[attr-defined]
    return games_out(session, session.exec(query.offset(offset).limit(limit)).all())


@router.get("/games/{game_id}")
def get_game(game_id: int, session: SessionDep) -> GameOut:
    return games_out(session, [get_or_404(session, Game, game_id)])[0]


@router.post("/games", status_code=201)
def create_game(body: GameCreate, session: SessionDep) -> GameOut:
    game_type = get_or_404(session, GameType, body.game_type_id)
    side_a, side_b = to_seats(body.side_a), to_seats(body.side_b)
    check_game(session, game_type, side_a, side_b, body.score_a, body.score_b)

    game = Game(
        game_type_id=game_type.id,
        played_at=as_utc(body.played_at) if body.played_at else utcnow(),
        score_a=body.score_a,
        score_b=body.score_b,
        outcome=outcome(body.score_a, body.score_b),
        notes=body.notes,
        participants=make_participants(side_a, side_b),
    )
    session.add(game)
    recompute(session, game_type)
    session.commit()
    return games_out(session, [game])[0]


@router.patch("/games/{game_id}")
def update_game(game_id: int, body: GameUpdate, session: SessionDep) -> GameOut:
    game = get_or_404(session, Game, game_id)
    game_type = get_or_404(session, GameType, game.game_type_id)
    old_a, old_b = sides(game)
    side_a = to_seats(body.side_a) if body.side_a is not None else old_a
    side_b = to_seats(body.side_b) if body.side_b is not None else old_b
    score_a = body.score_a if body.score_a is not None else game.score_a
    score_b = body.score_b if body.score_b is not None else game.score_b
    check_game(session, game_type, side_a, side_b, score_a, score_b)

    if body.side_a is not None or body.side_b is not None:
        game.participants = make_participants(side_a, side_b)
    game.score_a, game.score_b, game.outcome = score_a, score_b, outcome(score_a, score_b)
    if body.played_at is not None:
        game.played_at = as_utc(body.played_at)
    if body.notes is not None:
        game.notes = body.notes
    game.updated_at = utcnow()
    session.add(game)
    recompute(session, game_type)
    session.commit()
    return games_out(session, [game])[0]


@router.delete("/games/{game_id}", status_code=204)
def delete_game(game_id: int, session: SessionDep) -> None:
    game = get_or_404(session, Game, game_id)
    game_type = get_or_404(session, GameType, game.game_type_id)
    session.delete(game)
    session.flush()
    recompute(session, game_type)
    session.commit()


# --- Predictions -------------------------------------------------------------------------------


@router.post("/predict")
def predict_match(body: MatchIn, session: SessionDep) -> PredictionOut:
    game_type = get_or_404(session, GameType, body.game_type_id)
    settings = game_type.settings()
    side_a, side_b = to_seats(body.side_a), to_seats(body.side_b)
    check_match(session, settings, side_a, side_b)
    ratings = current_ratings(session, game_type)
    return prediction_out(predict(settings, ratings, side_a, side_b))


@router.post("/balance")
def balance_teams(body: BalanceIn, session: SessionDep) -> list[MatchupOut]:
    """Every split into two teams and every role assignment, fairest first."""
    game_type = get_or_404(session, GameType, body.game_type_id)
    if len(set(body.player_ids)) != len(body.player_ids):
        raise RuleError("each player can be listed only once")
    check_players(session, body.player_ids)
    ratings = current_ratings(session, game_type)
    try:
        options = balance(game_type.settings(), ratings, body.player_ids)
    except ValueError as e:
        raise RuleError(str(e)) from e
    return [
        MatchupOut(
            side_a=from_seats(o.side_a),
            side_b=from_seats(o.side_b),
            prediction=prediction_out(o.prediction),
        )
        for o in options
    ]


@router.post("/recompute")
def recompute_ratings(body: RecomputeIn, session: SessionDep) -> dict[str, int]:
    game_type = get_or_404(session, GameType, body.game_type_id)
    games = recompute(session, game_type)
    session.commit()
    return {"games": games}
