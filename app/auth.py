"""Logins, roles and invite links (PLAN.md §5).

Visitors can view everything. Admins can change players and games. The owner can also invite
and remove admins. There is no sign-up: every account comes from a single-use invite link,
and on first start the app logs a setup link for the owner's account.
"""

import hashlib
import re
import secrets
import time
from datetime import timedelta
from functools import cache
from typing import Annotated

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError
from fastapi import Depends, HTTPException, Request
from sqlmodel import Session, col, func, select

from app.config import PUBLIC_URL
from app.db import get_session
from app.models import Invite, User, utcnow
from app.ratings import RuleError

INVITE_DAYS = 7
SETUP_DAYS = 1
MIN_PASSWORD = 10
USERNAME = re.compile(r"[A-Za-z0-9._-]{3,30}")
MAX_FAILURES = 5  # failed logins per username and address ...
FAILURE_WINDOW = 15 * 60  # ... within this many seconds, before logins are refused

hasher = PasswordHasher()


class LoginRequired(Exception):
    """The request needs a logged-in admin: pages redirect to /login, the API returns 401."""


# --- Tokens and passwords ----------------------------------------------------------------------


def hash_token(token: str) -> str:
    """Invite tokens are long and random, so a plain SHA-256 is enough to store them safely."""
    return hashlib.sha256(token.encode()).hexdigest()


@cache
def _dummy_hash() -> str:
    return hasher.hash(secrets.token_hex(16))


def verify_password(password_hash: str | None, password: str) -> bool:
    try:
        # Unknown usernames still pay for a hash check, so timing doesn't reveal which exist.
        return hasher.verify(password_hash or _dummy_hash(), password) and password_hash is not None
    except (VerificationError, InvalidHashError):
        return False


def find_user(session: Session, username: str) -> User | None:
    return session.exec(
        select(User).where(func.lower(User.username) == username.strip().lower())
    ).first()


def check_new_account(session: Session, username: str, password: str) -> None:
    if not USERNAME.fullmatch(username):
        raise RuleError("usernames are 3-30 letters, digits, dots, dashes or underscores")
    if len(password) < MIN_PASSWORD:
        raise RuleError(f"the password needs at least {MIN_PASSWORD} characters")
    if find_user(session, username) is not None:
        raise RuleError(f"the username {username!r} is taken")


# --- Logging in --------------------------------------------------------------------------------

_failures: dict[str, list[float]] = {}  # (username, address) → times of recent failed logins


def _recent_failures(key: str) -> list[float]:
    now = time.monotonic()
    recent = [t for t in _failures.pop(key, []) if now - t < FAILURE_WINDOW]
    if recent:
        _failures[key] = recent
    return recent


def authenticate(session: Session, username: str, password: str, address: str) -> User:
    """The active user with these credentials, or a RuleError (also after too many failures)."""
    key = f"{username.strip().lower()}|{address}"
    if len(_recent_failures(key)) >= MAX_FAILURES:
        raise RuleError("too many failed logins; try again in 15 minutes")
    user = find_user(session, username)
    valid = verify_password(user.password_hash if user else None, password)
    if user is None or not valid or not user.active:
        _failures.setdefault(key, []).append(time.monotonic())
        raise RuleError("wrong username or password")
    _failures.pop(key, None)
    if hasher.check_needs_rehash(user.password_hash):
        user.password_hash = hasher.hash(password)
        session.add(user)
        session.commit()
    return user


def log_in(request: Request, user: User) -> None:
    request.session["user"] = {"id": user.id, "v": user.session_version}


def log_out(request: Request) -> None:
    request.session.pop("user", None)


def current_user(request: Request, session: Session) -> User | None:
    """The logged-in user, if their account is still active and their login wasn't revoked."""
    if hasattr(request.state, "user"):
        return request.state.user
    data = request.session.get("user")
    user = session.get(User, data.get("id")) if isinstance(data, dict) else None
    if user is not None and (not user.active or user.session_version != data.get("v")):
        user = None
    if user is None and data is not None:
        log_out(request)
    request.state.user = user
    return user


def admin_user(request: Request, session: Annotated[Session, Depends(get_session)]) -> User:
    user = current_user(request, session)
    if user is None:
        raise LoginRequired()
    return user


AdminDep = Annotated[User, Depends(admin_user)]


def owner_user(user: AdminDep) -> User:
    if user.role != "owner":
        raise HTTPException(403, "only the owner can do this")
    return user


OwnerDep = Annotated[User, Depends(owner_user)]


# --- Invites and accounts (the caller commits) -------------------------------------------------


def create_invite(
    session: Session, created_by: User | None, role: str = "admin", days: int = INVITE_DAYS
) -> str:
    """A new single-use invite; returns the token, which is shown once and never stored."""
    token = secrets.token_urlsafe(32)
    session.add(
        Invite(
            token_hash=hash_token(token),
            role=role,
            created_by=created_by.id if created_by else None,
            expires_at=utcnow() + timedelta(days=days),
        )
    )
    return token


def invite_url(token: str) -> str:
    return f"{PUBLIC_URL}/invite/{token}"


def find_invite(session: Session, token: str) -> Invite:
    invite = session.exec(select(Invite).where(Invite.token_hash == hash_token(token))).first()
    if invite is None or invite.used_at is not None or invite.expires_at < utcnow():
        raise RuleError("this link is not valid any more; ask for a new one")
    return invite


def accept_invite(session: Session, invite: Invite, username: str, password: str) -> User:
    username = username.strip()
    check_new_account(session, username, password)
    if invite.role == "owner" and active_owner(session) is not None:
        raise RuleError("this site already has an owner")
    user = User(username=username, password_hash=hasher.hash(password), role=invite.role)
    session.add(user)
    session.flush()
    invite.used_by, invite.used_at = user.id, utcnow()
    session.add(invite)
    return user


def remove_user(session: Session, user: User) -> None:
    """Take an admin's access away; this also ends their logins on every device."""
    if user.role == "owner":
        raise RuleError("the owner can't be removed")
    user.active = False
    user.session_version += 1
    session.add(user)


def active_owner(session: Session) -> User | None:
    return session.exec(
        select(User).where(User.role == "owner", col(User.active).is_(True))
    ).first()


def owner_setup_link(session: Session) -> str | None:
    """While the site has no owner, a fresh one-time link to create the owner account.

    Each start replaces the previous link, so only the newest one in the log works.
    """
    if active_owner(session) is not None:
        return None
    for old in session.exec(
        select(Invite).where(Invite.role == "owner", col(Invite.used_at).is_(None))
    ):
        session.delete(old)
    token = create_invite(session, None, role="owner", days=SETUP_DAYS)
    session.commit()
    return invite_url(token)
