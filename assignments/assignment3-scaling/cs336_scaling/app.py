import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from cs336_scaling.api.internal import router as internal_router
from cs336_scaling.api.public import router as public_router
from cs336_scaling.db import init_db
from cs336_scaling.log.setup import configure_logging

logger = logging.getLogger(__name__)
DASHBOARD_PATH = Path(__file__).resolve().parent.parent / "examples" / "dashboard.html"


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("starting_api")
    init_db()
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="CS336 Assignment 3 Backend", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "null",
            "http://localhost:8000",
            "http://127.0.0.1:8000",
            "http://localhost:8080",
            "http://127.0.0.1:8080",
            "http://hyperturing.stanford.edu:8000",
            "http://172.24.75.170:8000",
        ],
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["X-API-Key", "Content-Type"],
        allow_private_network=True,
    )

    @app.get("/dashboard", include_in_schema=False)
    async def dashboard() -> FileResponse:
        return FileResponse(DASHBOARD_PATH)

    configure_logging(app=app, log_file="logs/api/api.jsonl")
    app.include_router(public_router)
    app.include_router(internal_router)
    return app


app = create_app()
