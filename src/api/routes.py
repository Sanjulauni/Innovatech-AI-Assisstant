"""REST endpoints (FR-19 – FR-24, FR-31, FR-32).

Employee endpoints: ``POST /chat``, ``GET /health``.
Admin endpoints (``X-Admin-Password`` header): everything under ``/admin``.

Blocking work (embedding, LLM calls) runs in FastAPI's thread pool: endpoints are
plain ``def`` functions, and the upload endpoint uses ``run_in_threadpool``.
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Path, Response, UploadFile, status
from fastapi.concurrency import run_in_threadpool

from src.api.dependencies import ServicesDep, require_admin
from src.api.schemas import (
    ChatRequest,
    ChatResponse,
    DocumentOut,
    HealthResponse,
    InstructionsIn,
    InstructionsOut,
    LoginResponse,
    SourceOut,
    UploadResponse,
)
from src.data_pipeline.ingestion import (
    FileTooLargeError,
    IngestionError,
    IngestionStatus,
    UnsupportedTypeError,
)
from src.rag_engine.llm_chain import ServiceUnavailableError
from src.rag_engine.models import ChatMessage

logger = logging.getLogger(__name__)

DocId = Annotated[str, Path(pattern=r"^[0-9a-f]{64}$", description="SHA-256 of the file.")]

public_router = APIRouter(tags=["employee"])
admin_router = APIRouter(prefix="/admin", tags=["admin"], dependencies=[Depends(require_admin)])


# --- Employee ------------------------------------------------------------------


@public_router.get("/health", response_model=HealthResponse)
def health(services: ServicesDep) -> HealthResponse:
    settings = services.settings
    try:
        documents = len(services.repository.list_documents())
        store_ok = True
    except Exception as exc:
        logger.error("Vector store health check failed: %s", exc)
        documents, store_ok = 0, False
    return HealthResponse(
        status="ok" if store_ok else "degraded",
        vector_store="ok" if store_ok else "error",
        documents=documents,
        llm_provider=settings.llm_provider.value,
        llm_model=settings.llm_model_name,
        embedding_model=settings.embedding_model_name,
        admin_enabled=settings.admin_enabled,
    )


@public_router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, services: ServicesDep) -> ChatResponse:
    history = [ChatMessage(role=m.role, content=m.content) for m in request.history]
    try:
        answer = services.chain.ask(request.question, history)
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
    except ServiceUnavailableError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    return ChatResponse(
        answer=answer.answer,
        sources=[SourceOut(**vars(source)) for source in answer.sources],
    )


# --- Admin ---------------------------------------------------------------------


@admin_router.post("/login", response_model=LoginResponse)
def login() -> LoginResponse:
    """Check the admin password. ``require_admin`` has already rejected a wrong one."""
    return LoginResponse(authenticated=True)


@admin_router.get("/documents", response_model=list[DocumentOut])
def list_documents(services: ServicesDep) -> list[DocumentOut]:
    return [
        DocumentOut(
            doc_id=d.doc_id, source=d.source, chunk_count=d.chunk_count, ingested_at=d.ingested_at
        )
        for d in services.repository.list_documents()
    ]


@admin_router.post(
    "/documents",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    responses={200: {"description": "Already indexed; skipped."}},
)
async def upload_document(
    services: ServicesDep,
    response: Response,
    file: Annotated[UploadFile, File(description="PDF, DOCX, TXT or MD file.")],
) -> UploadResponse:
    limit = services.ingestion.max_size_bytes
    content = await file.read(limit + 1)  # one extra byte is enough to detect "too large"
    filename = file.filename or ""
    try:
        result = await run_in_threadpool(services.ingestion.ingest_upload, filename, content)
    except UnsupportedTypeError as exc:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, str(exc)) from exc
    except FileTooLargeError as exc:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, str(exc)) from exc
    except IngestionError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
    except Exception as exc:
        logger.exception("Indexing %r failed", filename)
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "The document could not be indexed because the embedding service is unavailable. "
            "Please try again shortly.",
        ) from exc

    if result.status is IngestionStatus.DUPLICATE:
        response.status_code = status.HTTP_200_OK
        message = f"'{result.source}' is already in the knowledge base; skipped."
    else:
        message = f"Indexed '{result.source}' ({result.chunk_count} chunks)."
    return UploadResponse(
        doc_id=result.doc_id,
        source=result.source,
        status=result.status,
        chunk_count=result.chunk_count,
        message=message,
    )


@admin_router.delete("/documents/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(doc_id: DocId, services: ServicesDep) -> Response:
    if not services.ingestion.delete(doc_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found.")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@admin_router.get("/instructions", response_model=InstructionsOut)
def get_instructions(services: ServicesDep) -> InstructionsOut:
    instructions = services.instructions.get()
    return InstructionsOut(text=instructions.text, updated_at=instructions.updated_at)


@admin_router.put("/instructions", response_model=InstructionsOut)
def update_instructions(body: InstructionsIn, services: ServicesDep) -> InstructionsOut:
    instructions = services.instructions.update(body.text)
    return InstructionsOut(text=instructions.text, updated_at=instructions.updated_at)
