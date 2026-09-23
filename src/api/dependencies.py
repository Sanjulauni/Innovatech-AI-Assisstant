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
from langchain_core.language_models import BaseChatModel

from src.config import Settings
from src.data_pipeline.ingestion import IngestionService
from src.data_pipeline.vector_store import VectorStoreRepository
from src.model_factory import EmbeddingFactory, LLMFactory
from src.rag_engine.instructions import InstructionsStore
from src.rag_engine.llm_chain import RAGChain
from src.rag_engine.retriever import DocumentRetriever

ADMIN_HEADER = "X-Admin-Password"


@dataclass
class Services:
    settings: Settings
    repository: VectorStoreRepository
    ingestion: IngestionService
    instructions: InstructionsStore
    chain: RAGChain


def build_services(
    settings: Settings,
    embeddings: Embeddings | None = None,
    llm: BaseChatModel | None = None,
    repository: VectorStoreRepository | None = None,
) -> Services:
    """Create every service from settings. Any argument given replaces the real one."""
    repository = repository or VectorStoreRepository.from_settings(
        embeddings or EmbeddingFactory.create(settings), settings
    )
    instructions = InstructionsStore.from_settings(settings)
    chain = RAGChain.from_settings(
        DocumentRetriever.from_settings(repository, settings),
        llm or LLMFactory.create(settings),
        instructions,
        settings,
    )
    return Services(
        settings=settings,
        repository=repository,
        ingestion=IngestionService.from_settings(repository, settings),
        instructions=instructions,
        chain=chain,
    )


def get_services(request: Request) -> Services:
    return request.app.state.services


ServicesDep = Annotated[Services, Depends(get_services)]


def require_admin(
    services: ServicesDep,
    x_admin_password: Annotated[str | None, Header(alias=ADMIN_HEADER)] = None,
) -> None:
    """Allow the request only if it carries the correct admin password."""
    expected = services.settings.admin_password
    if expected is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Admin access is disabled. Set ADMIN_PASSWORD in the server's .env file.",
        )
    # Constant-time comparison, so response timing reveals nothing about the password.
    if x_admin_password is None or not secrets.compare_digest(
        x_admin_password.encode("utf-8"), expected.get_secret_value().encode("utf-8")
    ):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Incorrect admin password.")
