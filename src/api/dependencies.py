"""Service wiring and FastAPI dependencies (Dependency Injection).

``build_services`` creates the real services once at startup. Tests pass their own
``Services`` (with fake models) to ``create_app`` instead.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Header, HTTPException, Request, status
from langchain_core.embeddings import Embeddings

from src.api.auth import SESSION_COOKIE, LoginRateLimiter, SessionManager
from src.config import Settings
from src.data_pipeline.confidential import ConfidentialStore
from src.data_pipeline.ingestion import IngestionService
from src.data_pipeline.upload_limit import UploadLimit
from src.data_pipeline.vector_store import VectorStoreRepository
from src.local_llm import LocalLLMServer
from src.model_factory import ChatModel, EmbeddingFactory, LLMFactory
from src.rag_engine.instructions import InstructionsStore
from src.rag_engine.llm_chain import RAGChain
from src.rag_engine.model_selector import LocalModel, ModelSelector
from src.rag_engine.retriever import DocumentRetriever

ADMIN_HEADER = "X-Admin-Password"


@dataclass
class Services:
    settings: Settings
    repository: VectorStoreRepository
    ingestion: IngestionService
    instructions: InstructionsStore
    chain: RAGChain
    models: ModelSelector
    upload_limit: UploadLimit
    confidential: ConfidentialStore
    sessions: SessionManager
    login_limiter: LoginRateLimiter


def build_services(
    settings: Settings,
    embeddings: Embeddings | None = None,
    llm: ChatModel | None = None,
    repository: VectorStoreRepository | None = None,
    local_server: LocalLLMServer | None = None,
) -> Services:
    """Create every service from settings. Any argument given replaces the real one.

    A given ``llm`` answers for whichever model is selected (used by tests).
    """
    repository = repository or VectorStoreRepository.from_settings(
        embeddings or EmbeddingFactory.create(settings), settings
    )
    instructions = InstructionsStore.from_settings(settings)
    models = ModelSelector(
        settings.available_models,
        factory=(lambda _model: llm) if llm else (lambda model: LLMFactory.create(settings, model)),
        store_path=settings.model_selection_file,
        local=_local_model(settings, local_server),
        idle_seconds=settings.local_llm_idle_minutes * 60,
    )
    upload_limit = UploadLimit.from_settings(settings)
    confidential = ConfidentialStore.from_settings(settings)
    chain = RAGChain.from_settings(
        DocumentRetriever.from_settings(repository, settings),
        models,
        instructions,
        settings,
        confidential=confidential,
    )
    return Services(
        settings=settings,
        repository=repository,
        ingestion=IngestionService.from_settings(
            repository, settings, max_size_bytes=upload_limit.max_bytes
        ),
        instructions=instructions,
        chain=chain,
        models=models,
        upload_limit=upload_limit,
        confidential=confidential,
        sessions=SessionManager(ttl_seconds=int(settings.admin_session_hours * 3600)),
        login_limiter=LoginRateLimiter(),
    )


def _local_model(settings: Settings, server: LocalLLMServer | None) -> LocalModel | None:
    """The local model, if LOCAL_LLM_SERVER and LOCAL_LLM_MODEL are set."""
    server = server or LocalLLMServer.from_settings(settings)
    if server is None or settings.local_model_id is None:
        return None
    label = settings.local_llm_label.strip() or settings.local_llm_model.stem
    return LocalModel(id=settings.local_model_id, label=label, server=server)


def get_services(request: Request) -> Services:
    return request.app.state.services


ServicesDep = Annotated[Services, Depends(get_services)]
AdminPasswordHeader = Annotated[str | None, Header(alias=ADMIN_HEADER)]


def ensure_admin_enabled(services: Services) -> None:
    if not services.settings.admin_enabled:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Admin access is disabled. Set ADMIN_PASSWORD in the server's .env file.",
        )


def password_matches(services: Services, password: str | None) -> bool:
    """Constant-time check, so response timing reveals nothing about the password."""
    expected = services.settings.admin_password
    if expected is None or password is None:
        return False
    return secrets.compare_digest(
        password.encode("utf-8"), expected.get_secret_value().encode("utf-8")
    )


def require_admin(
    request: Request, services: ServicesDep, x_admin_password: AdminPasswordHeader = None
) -> None:
    """Allow the request with a valid session cookie (web app) or password header (scripts)."""
    ensure_admin_enabled(services)
    if services.sessions.verify(request.cookies.get(SESSION_COOKIE)):
        return
    if password_matches(services, x_admin_password):
        return
    raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Incorrect admin password.")
