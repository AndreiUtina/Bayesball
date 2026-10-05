import logging
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import quote, urlsplit

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from sqlmodel import Session
from starlette.middleware.sessions import SessionMiddleware

from app import accounts, api, explain, web
from app.auth import LoginRequired, owner_setup_link
from app.config import SECRET_KEY, SECURE_COOKIES
from app.db import engine, init_db
from app.ratings import NameTaken, RuleError

BASE_DIR = Path(__file__).parent
log = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db(engine)
    with Session(engine) as session:
        link = owner_setup_link(session)
    if link:
        log.warning("No owner account yet. Create it here (valid for 1 day, one use): %s", link)
    yield


app = FastAPI(title="Bayesball", lifespan=lifespan)
app.add_middleware(
    SessionMiddleware,
    secret_key=SECRET_KEY,
    session_cookie="bayesball_session",
    max_age=30 * 24 * 3600,  # stay logged in for 30 days
    same_site="lax",
    https_only=SECURE_COOKIES,
)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
app.include_router(api.router)
app.include_router(web.router)
app.include_router(accounts.router)
app.include_router(explain.router)

UNSAFE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}


def same_origin(request: Request) -> bool:
    """Whether a request comes from this site's own pages (not another site's form or script)."""
    if request.headers.get("sec-fetch-site") in ("cross-site", "same-site"):
        return False
    origin = request.headers.get("origin")
    return origin is None or urlsplit(origin).netloc == request.headers.get("host")


@app.middleware("http")
async def security(request: Request, call_next) -> Response:
    if request.method in UNSAFE_METHODS and not same_origin(request):
        return PlainTextResponse("Cross-site request blocked.", status_code=403)
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "same-origin")
    return response


@app.exception_handler(RuleError)
def rule_error(request: Request, exc: RuleError) -> JSONResponse:
    status = 409 if isinstance(exc, NameTaken) else 422
    return JSONResponse(status_code=status, content={"detail": str(exc)})


@app.exception_handler(LoginRequired)
def login_required(request: Request, exc: LoginRequired) -> Response:
    if request.url.path.startswith("/api/"):
        return JSONResponse(status_code=401, content={"detail": "log in first"})
    target = request.url.path + (f"?{request.url.query}" if request.url.query else "")
    if request.method != "GET":
        target = "/"  # a form posted after the login expired: start again from the home page
    return RedirectResponse(f"/login?next={quote(target)}", status_code=303)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/robots.txt", include_in_schema=False)
def robots() -> PlainTextResponse:
    """The site is public, but search engines are asked not to list it."""
    return PlainTextResponse("User-agent: *\nDisallow: /\n")
