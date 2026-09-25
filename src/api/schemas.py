"""Request and response models for the REST API (FR-24)."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from src.data_pipeline.ingestion import IngestionStatus
from src.rag_engine.instructions import MAX_INSTRUCTIONS_LENGTH

MAX_QUESTION_LENGTH = 2000
MAX_HISTORY_MESSAGES = 50
MAX_MESSAGE_LENGTH = 8000


# --- Chat ----------------------------------------------------------------------


class ChatMessageIn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=MAX_MESSAGE_LENGTH)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_LENGTH)
    history: list[ChatMessageIn] = Field(
        default_factory=list,
        max_length=MAX_HISTORY_MESSAGES,
        description="Earlier messages in this conversation, oldest first.",
    )


class SourceOut(BaseModel):
    index: int = Field(description="The number used to cite this source in the answer, e.g. [1].")
    doc_id: str
    source: str = Field(description="Document file name.")
    page: int | None = None
    snippet: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceOut]


# --- Health --------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    vector_store: Literal["ok", "error"]
    documents: int
    llm_provider: str
    llm_model: str
    embedding_model: str
    admin_enabled: bool


# --- Admin ---------------------------------------------------------------------


class LoginRequest(BaseModel):
    password: str | None = Field(default=None, max_length=512)


class LoginResponse(BaseModel):
    authenticated: bool
    expires_in: int | None = Field(default=None, description="Session lifetime in seconds.")


class DocumentOut(BaseModel):
    doc_id: str
    source: str
    chunk_count: int
    ingested_at: str


class UploadResponse(BaseModel):
    doc_id: str
    source: str
    status: IngestionStatus
    chunk_count: int
    message: str


class ReindexResponse(BaseModel):
    indexed: int = Field(description="Files newly indexed.")
    skipped: int = Field(description="Files that were already indexed.")
    failed: dict[str, str] = Field(description="File name -> reason it could not be indexed.")


class ModelOptionOut(BaseModel):
    id: str
    label: str
    description: str


class ModelsOut(BaseModel):
    current: str
    options: list[ModelOptionOut]


class ModelSelectIn(BaseModel):
    model: str = Field(min_length=1, max_length=200)


class UploadLimitOut(BaseModel):
    max_upload_size_mb: int = Field(description="Largest file accepted now, in MB.")
    default_mb: int = Field(description="The limit from MAX_UPLOAD_SIZE_MB in .env.")
    max_allowed_mb: int = Field(description="The highest limit that can be set.")


class UploadLimitIn(BaseModel):
    max_upload_size_mb: int = Field(ge=1)


class InstructionsIn(BaseModel):
    text: str = Field(max_length=MAX_INSTRUCTIONS_LENGTH)


class InstructionsOut(BaseModel):
    text: str
    updated_at: datetime | None = None
