"""REST endpoints (FR-18 – FR-24, FR-31, FR-32).

Employee endpoints: ``POST /chat``, ``POST /chat/stream``, ``GET /health``.
Admin endpoints: everything under ``/admin``. They accept either the session cookie
set by ``POST /admin/login`` (web app) or the ``X-Admin-Password`` header (Streamlit).

Blocking work (embedding, LLM calls) runs in FastAPI's thread pool: endpoints are
plain ``def`` functions, and the upload endpoint uses ``run_in_threadpool``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from typing import Annotated

from fastapi import (
    APIRouter,
    Body,
    Depends,
    File,
    HTTPException,
    Path,
    Request,
    Response,
    UploadFile,
    status,
)
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse

from src.api.auth import SESSION_COOKIE
from src.api.dependencies import (
    AdminPasswordHeader,
    ServicesDep,
    ensure_admin_enabled,
    password_matches,
    require_admin,
)
from src.api.schemas import (
    ChatRequest,
    ChatResponse,
    DocumentOut,
    HealthResponse,
    InstructionsIn,
    InstructionsOut,
    LoginRequest,
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
from src.model_factory import is_rate_limited
from src.rag_engine.llm_chain import ServiceUnavailableError
from src.rag_engine.models import Answer, ChatMessage

logger = logging.getLogger(__name__)

DocId = Annotated[str, Path(pattern=r"^[0-9a-f]{64}$", description="SHA-256 of the file.")]

public_router = APIRouter(tags=["employee"])
auth_router = APIRouter(prefix="/admin", tags=["admin"])
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


def _history(request: ChatRequest) -> list[ChatMessage]:
    return [ChatMessage(role=m.role, content=m.content) for m in request.history]


def _response(answer: Answer) -> ChatResponse:
    return ChatResponse(
        answer=answer.answer,
        sources=[SourceOut(**vars(source)) for source in answer.sources],
    )


@public_router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, services: ServicesDep) -> ChatResponse:
    try:
        answer = services.chain.ask(request.question, _history(request))
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
    except ServiceUnavailableError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    return _response(answer)


def _line(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False) + "\n"


def _event(item: str | Answer) -> str:
    if isinstance(item, Answer):
        return _line({"type": "done", **_response(item).model_dump()})
    return _line({"type": "token", "text": item})


@public_router.post(
    "/chat/stream",
    response_class=StreamingResponse,
    responses={
        200: {
            "content": {"application/x-ndjson": {}},
            "description": "One JSON object per line: `token` events with a piece of the "
            "answer, then a `done` event with the full answer and sources, or an `error` "
            "event if the model fails part-way.",
        }
    },
)
def chat_stream(request: ChatRequest, services: ServicesDep) -> StreamingResponse:
    """Stream the answer as it is generated (FR-18)."""
    events = services.chain.stream(request.question, _history(request))
    try:
        # Validation and retrieval errors surface here, as normal HTTP errors.
        first = next(events)
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
    except ServiceUnavailableError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc

    def lines() -> Iterator[str]:
        try:
            yield _event(first)
            for item in events:
                yield _event(item)
        except ServiceUnavailableError as exc:
            yield _line({"type": "error", "message": str(exc)})
        except Exception:
            logger.exception("Streaming answer failed")
            message = "Something went wrong while answering. Please try again."
            yield _line({"type": "error", "message": message})

    return StreamingResponse(lines(), media_type="application/x-ndjson")


# --- Admin session -------------------------------------------------------------


def _client(request: Request) -> str:
    return request.client.host if request.client else "unknown"


@auth_router.post(
    "/login",
    response_model=LoginResponse,
    responses={401: {"description": "Wrong password."}, 429: {"description": "Too many tries."}},
)
def login(
    request: Request,
    response: Response,
    services: ServicesDep,
    body: Annotated[LoginRequest | None, Body()] = None,
    x_admin_password: AdminPasswordHeader = None,
) -> LoginResponse:
    """Check the admin password (JSON body or header) and start a session cookie."""
    ensure_admin_enabled(services)
    client = _client(request)
    if services.login_limiter.is_blocked(client):
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            "Too many failed attempts. Please wait a minute and try again.",
        )
    password = body.password if body and body.password is not None else x_admin_password
    if not password_matches(services, password):
        services.login_limiter.record_failure(client)
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Incorrect admin password.")

    services.login_limiter.reset(client)
    response.set_cookie(
        SESSION_COOKIE,
        services.sessions.issue(),
        max_age=services.sessions.ttl_seconds,
        httponly=True,  # page scripts can't read it
        samesite="strict",  # other sites can't send it
        secure=request.url.scheme == "https",
        path="/",
    )
    return LoginResponse(authenticated=True, expires_in=services.sessions.ttl_seconds)


@auth_router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(request: Request, services: ServicesDep) -> Response:
    services.sessions.revoke(request.cookies.get(SESSION_COOKIE))
    response = Response(status_code=status.HTTP_204_NO_CONTENT)
    response.delete_cookie(SESSION_COOKIE, path="/")
    return response


@admin_router.get("/session", response_model=LoginResponse)
def session() -> LoginResponse:
    """Whether the caller is logged in. ``require_admin`` rejects everyone else."""
    return LoginResponse(authenticated=True)


# --- Admin documents -----------------------------------------------------------


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
        if is_rate_limited(exc):
            detail = (
                "The embedding service's usage limit was reached while indexing this "
                "document. Wait a minute, then upload it again."
            )
        else:
            detail = (
                "The document could not be indexed because the embedding service is "
                "unavailable. Please try again shortly."
            )
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail) from exc

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
