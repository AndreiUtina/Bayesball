from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app import api, web
from app.db import engine, init_db
from app.ratings import NameTaken, RuleError

BASE_DIR = Path(__file__).parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db(engine)
    yield


app = FastAPI(title="Bayesball", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
app.include_router(api.router)
app.include_router(web.router)


@app.exception_handler(RuleError)
def rule_error(request: Request, exc: RuleError) -> JSONResponse:
    status = 409 if isinstance(exc, NameTaken) else 422
    return JSONResponse(status_code=status, content={"detail": str(exc)})


@app.get("/health")
def health():
    return {"status": "ok"}
