"""Database tables (PLAN.md §6)."""

from dataclasses import asdict, fields
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, UniqueConstraint
from sqlmodel import Field, Relationship, SQLModel

from app.rating.margin import GameSettings


def utcnow() -> datetime:
    return datetime.now(UTC)


class GameType(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(unique=True)
    scoring: str = "margin"  # "margin" or "win_draw_loss"
    roles: list[str] = Field(sa_type=JSON)
    points_to_win: int | None = None
    allow_draws: bool = False
    team_strength: str = "mean"
    beta: float
    default_sigma: float
    role_correlation: float
    skill_scale: float
    tau: float

    @classmethod
    def from_settings(cls, name: str, settings: GameSettings) -> "GameType":
        return cls(name=name, **{**asdict(settings), "roles": list(settings.roles)})

    def settings(self) -> GameSettings:
        values = {f.name: getattr(self, f.name) for f in fields(GameSettings)}
        return GameSettings(**{**values, "roles": tuple(self.roles)})


class User(SQLModel, table=True):
    """Someone who can log in: the owner or an admin (PLAN.md §5)."""

    __tablename__ = "users"  # "user" is a reserved word in Postgres

    id: int | None = Field(default=None, primary_key=True)
    username: str = Field(unique=True)
    password_hash: str
    role: str = "admin"  # "owner" or "admin"
    active: bool = True  # False once removed; kept for the audit trail
    session_version: int = 0  # bumped to log the user out everywhere
    created_at: datetime = Field(default_factory=utcnow)


class Invite(SQLModel, table=True):
    """A single-use link to create an account. Only a hash of the token is stored."""

    id: int | None = Field(default=None, primary_key=True)
    token_hash: str = Field(unique=True)
    role: str = "admin"  # "owner" for the first-start setup link
    created_by: int | None = Field(default=None, foreign_key="users.id")
    created_at: datetime = Field(default_factory=utcnow)
    expires_at: datetime
    used_by: int | None = Field(default=None, foreign_key="users.id")
    used_at: datetime | None = None


class Player(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(unique=True)
    active: bool = True
    created_at: datetime = Field(default_factory=utcnow)
    created_by: int | None = Field(default=None, foreign_key="users.id")
    updated_by: int | None = Field(default=None, foreign_key="users.id")


class PlayerRating(SQLModel, table=True):
    """One per (player, game type): the prior inputs and the current (cached) posterior."""

    __table_args__ = (UniqueConstraint("player_id", "game_type_id"),)

    id: int | None = Field(default=None, primary_key=True)
    player_id: int = Field(foreign_key="player.id", index=True)
    game_type_id: int = Field(foreign_key="gametype.id", index=True)
    # Keyword arguments for app.rating.margin.prior (level, stronger, sigma_factor, mu, sigma).
    # Stored as inputs, not as numbers, so a change to the game-type settings reaches the priors.
    prior: dict[str, Any] = Field(default_factory=dict, sa_type=JSON)
    mu: list[float] = Field(default_factory=list, sa_type=JSON)
    cov: list[list[float]] = Field(default_factory=list, sa_type=JSON)
    games_played: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    last_played_at: datetime | None = None


class Game(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    game_type_id: int = Field(foreign_key="gametype.id", index=True)
    played_at: datetime = Field(default_factory=utcnow, index=True)
    score_a: int
    score_b: int
    outcome: str  # "A", "B" or "DRAW"
    notes: str = ""
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    created_by: int | None = Field(default=None, foreign_key="users.id")
    updated_by: int | None = Field(default=None, foreign_key="users.id")

    participants: list["GameParticipant"] = Relationship(
        back_populates="game",
        sa_relationship_kwargs={"cascade": "all, delete-orphan", "order_by": "GameParticipant.id"},
    )


class GameParticipant(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    game_id: int = Field(foreign_key="game.id", index=True)
    player_id: int = Field(foreign_key="player.id", index=True)
    side: str  # "A" or "B"
    role: str | None = None  # None when the player covers every role (1v1)
    # The player's rating just before and after this game, for history charts.
    mu_before: list[float] | None = Field(default=None, sa_type=JSON)
    cov_before: list[list[float]] | None = Field(default=None, sa_type=JSON)
    mu_after: list[float] | None = Field(default=None, sa_type=JSON)
    cov_after: list[list[float]] | None = Field(default=None, sa_type=JSON)

    game: Game | None = Relationship(back_populates="participants")
