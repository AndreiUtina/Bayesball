"""Account pages: log in and out, accept an invite, and the owner's list of admins."""

from typing import Annotated

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from sqlmodel import Session, col, select

from app.auth import (
    OwnerDep,
    accept_invite,
    authenticate,
    create_invite,
    current_user,
    find_invite,
    invite_url,
    log_in,
    log_out,
    remove_user,
)
from app.models import Invite, User, utcnow
from app.ratings import RuleError
from app.web import SessionDep, render

router = APIRouter(include_in_schema=False)


def safe_next(target: str) -> str:
    """Where to go after logging in: only pages of this site."""
    return target if target.startswith("/") and not target.startswith(("//", "/\\")) else "/"


# --- Log in and out ----------------------------------------------------------------------------


@router.get("/login", response_class=HTMLResponse)
def login_page(request: Request, session: SessionDep, next: str = "/") -> Response:
    if current_user(request, session) is not None:
        return RedirectResponse(safe_next(next), status_code=303)
    return render(request, session, "login.html", page="login", next=safe_next(next), username="")


@router.post("/login", response_class=HTMLResponse)
def login(
    request: Request,
    session: SessionDep,
    username: Annotated[str, Form()] = "",
    password: Annotated[str, Form()] = "",
    next: Annotated[str, Form()] = "/",
) -> Response:
    address = request.client.host if request.client else "unknown"
    try:
        user = authenticate(session, username, password, address)
    except RuleError as e:
        return render(
            request,
            session,
            "login.html",
            status=401,
            page="login",
            next=safe_next(next),
            username=username,
            error=str(e),
        )
    log_in(request, user)
    return RedirectResponse(safe_next(next), status_code=303)


@router.post("/logout")
def logout(request: Request) -> Response:
    log_out(request)
    return RedirectResponse("/", status_code=303)


# --- Accept an invite --------------------------------------------------------------------------


@router.get("/invite/{token}", response_class=HTMLResponse)
def invite_page(request: Request, session: SessionDep, token: str) -> HTMLResponse:
    try:
        invite = find_invite(session, token)
    except RuleError as e:
        return render(request, session, "invite.html", status=404, invite=None, error=str(e))
    return render(request, session, "invite.html", invite=invite, username="")


@router.post("/invite/{token}", response_class=HTMLResponse)
def accept(
    request: Request,
    session: SessionDep,
    token: str,
    username: Annotated[str, Form()] = "",
    password: Annotated[str, Form()] = "",
    password2: Annotated[str, Form()] = "",
) -> Response:
    try:
        invite = find_invite(session, token)
    except RuleError as e:
        return render(request, session, "invite.html", status=404, invite=None, error=str(e))
    try:
        if password != password2:
            raise RuleError("the two passwords don't match")
        user = accept_invite(session, invite, username, password)
        session.commit()
    except RuleError as e:
        session.rollback()
        return render(
            request,
            session,
            "invite.html",
            status=422,
            invite=find_invite(session, token),
            username=username,
            error=str(e),
        )
    log_in(request, user)
    return RedirectResponse("/", status_code=303)


# --- The owner's list of admins ----------------------------------------------------------------


def users_page(
    request: Request, session: Session, owner: User, new_link: str | None = None
) -> HTMLResponse:
    users = session.exec(select(User).order_by(col(User.created_at))).all()
    pending = session.exec(
        select(Invite)
        .where(Invite.role == "admin", col(Invite.used_at).is_(None), Invite.expires_at > utcnow())
        .order_by(col(Invite.created_at))
    ).all()
    return render(
        request,
        session,
        "users.html",
        page="users",
        users=users,
        pending=pending,
        new_link=new_link,
    )


@router.get("/admin/users", response_class=HTMLResponse)
def list_users(request: Request, session: SessionDep, owner: OwnerDep) -> HTMLResponse:
    return users_page(request, session, owner)


@router.post("/admin/users/invite", response_class=HTMLResponse)
def new_invite(request: Request, session: SessionDep, owner: OwnerDep) -> HTMLResponse:
    token = create_invite(session, owner)
    session.commit()
    # Only a hash is stored, so this is the one time the link can be shown.
    return users_page(request, session, owner, new_link=invite_url(token))


@router.post("/admin/users/{user_id:int}/remove")
def remove(request: Request, session: SessionDep, owner: OwnerDep, user_id: int) -> Response:
    user = session.get(User, user_id)
    if user is not None:
        remove_user(session, user)
        session.commit()
    return RedirectResponse("/admin/users", status_code=303)


@router.post("/admin/invites/{invite_id:int}/cancel")
def cancel_invite(
    request: Request, session: SessionDep, owner: OwnerDep, invite_id: int
) -> Response:
    invite = session.get(Invite, invite_id)
    if invite is not None and invite.used_at is None:
        session.delete(invite)
        session.commit()
    return RedirectResponse("/admin/users", status_code=303)
