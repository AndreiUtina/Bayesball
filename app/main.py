from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app import api
from app.db import engine, init_db
from app.ratings import RuleError

BASE_DIR = Path(__file__).parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db(engine)
    yield


app = FastAPI(title="Bayesball", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
app.include_router(api.router)
templates = Jinja2Templates(directory=BASE_DIR / "templates")


@app.exception_handler(RuleError)
def rule_error(request: Request, exc: RuleError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.get("/", response_class=HTMLResponse)
def leaderboard(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}
