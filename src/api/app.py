"""FastAPI application (FR-25: Swagger docs at ``/docs``).

Run with ``uvicorn src.api.app:app`` or ``python -m src.api.app``.

The endpoints are served under ``/api``. If the React app has been built
(``frontend/dist``), it is served too, so one server hosts everything.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import FileResponse, JSONResponse

from src.api.dependencies import Services, ServicesDep, build_services
from src.api.routes import admin_router, auth_router, public_router
from src.config import get_settings

logger = logging.getLogger(__name__)

API_PREFIX = "/api"
_ROUTERS = (public_router, auth_router, admin_router)


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
        models = app.state.services.models
        models.activate()  # start loading the local model if it's the saved choice
        try:
            yield
        finally:
            models.shutdown()  # stop the local model's server

    app = FastAPI(
        title="InnovaTech AI Assistant API",
        description="Answers employee questions from company documents, with citations.",
        version="1.1.0",
        lifespan=lifespan,
    )
    app.state.services = services
    for router in _ROUTERS:
        app.include_router(router, prefix=API_PREFIX)

    @app.exception_handler(Exception)
    async def unhandled_error(request: Request, exc: Exception) -> JSONResponse:
        # Never leak internal details (paths, document text) to the client (NFR-05).
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Something went wrong on the server. Please try again."},
        )

    @app.get("/{path:path}", include_in_schema=False)
    def web_app(path: str, services: ServicesDep) -> FileResponse:
        """Serve the built React app; unknown paths get index.html (client-side routing)."""
        dist = services.settings.frontend_dist_dir.resolve()
        index = dist / "index.html"
        if path.startswith("api/") or not index.is_file():
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Not found.")
        file = (dist / path).resolve()
        if path and file.is_file() and file.is_relative_to(dist):
            return FileResponse(file)
        return FileResponse(index, headers={"Cache-Control": "no-cache"})

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run("src.api.app:app", host=settings.api_host, port=settings.api_port)
