"""FastAPI application (FR-25: Swagger docs at ``/docs``).

Run with ``uvicorn src.api.app:app`` or ``python -m src.api.app``.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from src.api.dependencies import Services, build_services
from src.api.routes import admin_router, public_router
from src.config import get_settings

logger = logging.getLogger(__name__)


def create_app(services: Services | None = None) -> FastAPI:
    """Build the app. Without ``services``, real ones are created from settings at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if getattr(app.state, "services", None) is None:
            settings = get_settings()
            logging.basicConfig(level=settings.log_level.upper())
            app.state.services = build_services(settings)
            if not settings.admin_enabled:
                logger.warning("ADMIN_PASSWORD is not set: admin endpoints are disabled.")
        yield

    app = FastAPI(
        title="InnovaTech AI Assistant API",
        description="Answers employee questions from company documents, with citations.",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.services = services
    app.include_router(public_router)
    app.include_router(admin_router)

    @app.exception_handler(Exception)
    async def unhandled_error(request: Request, exc: Exception) -> JSONResponse:
        # Never leak internal details (paths, document text) to the client (NFR-05).
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Something went wrong on the server. Please try again."},
        )

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run("src.api.app:app", host=settings.api_host, port=settings.api_port)
