"""Centralized application settings.

All tunable values are read from environment variables or a local ``.env`` file
(see ``.env.example``). Nothing configurable should be hard-coded elsewhere (NFR-15).

Chat answers come from the Groq API (free tier) or, if configured, a local GGUF
model served by llama.cpp's ``llama-server`` on this machine; the admin picks
which one to use. Document embeddings are computed locally with FastEmbed, so
indexing and search need no API and no quota.
"""

from __future__ import annotations

import re
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Model IDs of local models start with this, e.g. "local:gemma-4-E2B-it-Q4_K_M".
LOCAL_MODEL_PREFIX = "local:"
# The local model server only listens on this machine, never on the network.
LOCAL_LLM_HOST = "127.0.0.1"


class ModelProvider(str, Enum):
    """Chat model providers. Add new ones (e.g. OLLAMA) here in later versions."""

    GROQ = "groq"


class EmbeddingProvider(str, Enum):
    """Embedding providers. FastEmbed runs locally on the CPU."""

    FASTEMBED = "fastembed"


_ENV_CONFIG = SettingsConfigDict(
    env_file=PROJECT_ROOT / ".env",
    env_file_encoding="utf-8",
    case_sensitive=False,
    env_ignore_empty=True,  # `KEY=` in .env counts as "not set"
    extra="ignore",
)

_PATH_FIELDS = (
    "raw_data_dir",
    "vector_db_dir",
    "instructions_file",
    "model_selection_file",
    "upload_limit_file",
    "local_llm_log_file",
    "embedding_cache_dir",
    "frontend_dist_dir",
)


class Settings(BaseSettings):
    """Application settings, populated from the environment and ``.env``."""

    model_config = _ENV_CONFIG

    # --- Application -------------------------------------------------------
    app_name: str = "InnovaTech AI Assistant"
    log_level: str = "INFO"

    # --- Chat model (Groq) -------------------------------------------------
    llm_provider: ModelProvider = ModelProvider.GROQ
    llm_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    groq_api_key: SecretStr | None = None
    # Models the admin can choose from (comma-separated in .env). The first is the default.
    groq_models: Annotated[list[str], NoDecode] = [
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "qwen/qwen3.8-27b",
    ]

    # --- Local chat model (optional) ---------------------------------------
    # A GGUF model the API runs with llama.cpp's llama-server. Set both paths to
    # offer it in the Admin page's model list; the server starts when it's selected.
    local_llm_server: Path | None = None  # path to llama-server(.exe)
    local_llm_model: Path | None = None  # path to the .gguf file
    local_llm_label: str = ""  # name shown to the admin; defaults to the file name
    local_llm_port: int = Field(default=8080, gt=0, lt=65536)
    local_llm_context_size: int = Field(default=8192, gt=0)
    # Loading a model from disk can take a while on a slow drive or CPU.
    local_llm_startup_timeout: float = Field(default=180, gt=0)
    # Extra llama-server options, e.g. "--threads 4". They override the defaults.
    local_llm_args: str = ""
    # A CPU reads a prompt slowly (~30 tokens/s on a laptop), so the local model gets
    # fewer excerpts and less history than the cloud models.
    local_llm_top_k: int = Field(default=2, gt=0)
    local_llm_history_limit: int = Field(default=2, ge=0)
    local_llm_log_file: Path = PROJECT_ROOT / "data" / "logs" / "llama-server.log"

    # --- Embeddings (local) ------------------------------------------------
    embedding_provider: EmbeddingProvider = EmbeddingProvider.FASTEMBED
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    # Where the embedding model is downloaded to on first use (~70 MB).
    embedding_cache_dir: Path = PROJECT_ROOT / "data" / "models"

    # --- Data & vector store -----------------------------------------------
    raw_data_dir: Path = PROJECT_ROOT / "data" / "raw"
    vector_db_dir: Path = PROJECT_ROOT / "data" / "vector_db"
    collection_prefix: str = "innovatech"
    instructions_file: Path = PROJECT_ROOT / "data" / "agent_instructions.json"
    # The chat model the admin selected.
    model_selection_file: Path = PROJECT_ROOT / "data" / "model_selection.json"

    # --- Ingestion ---------------------------------------------------------
    chunk_size: int = Field(default=1000, gt=0)
    chunk_overlap: int = Field(default=200, ge=0)
    allowed_extensions: set[str] = {".pdf", ".docx", ".txt", ".md"}
    # Largest upload accepted until the admin changes it on the Admin page.
    max_upload_size_mb: int = Field(default=20, gt=0)
    # The highest limit the admin may set. Uploads are held in memory while indexed.
    max_upload_size_cap_mb: int = Field(default=200, gt=0)
    # The upload limit the admin set.
    upload_limit_file: Path = PROJECT_ROOT / "data" / "upload_limit.json"

    # --- Retrieval ---------------------------------------------------------
    retriever_top_k: int = Field(default=4, gt=0)
    # Number of previous chat messages (user + assistant) sent with each question.
    chat_history_limit: int = Field(default=6, ge=0)

    # --- API ---------------------------------------------------------------
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    # Built React app (``npm run build`` in frontend/). Served by the API if it exists.
    frontend_dist_dir: Path = PROJECT_ROOT / "frontend" / "dist"

    # --- Access control ----------------------------------------------------
    # Password for the admin endpoints and page. Admin features are disabled if unset.
    admin_password: SecretStr | None = None
    # How long an admin stays logged in to the web app.
    admin_session_hours: float = Field(default=8, gt=0)

    @field_validator("groq_models", mode="before")
    @classmethod
    def _split_models(cls, value: object) -> object:
        if isinstance(value, str):
            return [part.strip() for part in value.split(",")]
        return value

    @model_validator(mode="after")
    def _validate(self) -> Settings:
        if self.llm_provider is ModelProvider.GROQ and not (
            self.groq_api_key and self.groq_api_key.get_secret_value().strip()
        ):
            raise ValueError(
                "GROQ_API_KEY is not set. Add it to your .env file "
                "(get a free key at https://console.groq.com/keys)."
            )

        # Drop blanks and duplicates, keeping the order (the first model is the default).
        self.groq_models = list(dict.fromkeys(m.strip() for m in self.groq_models if m.strip()))
        if not self.groq_models:
            raise ValueError("GROQ_MODELS must list at least one model.")

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")

        if self.max_upload_size_mb > self.max_upload_size_cap_mb:
            raise ValueError("MAX_UPLOAD_SIZE_MB must not exceed MAX_UPLOAD_SIZE_CAP_MB.")

        if self.admin_password is not None and not self.admin_password.get_secret_value().strip():
            self.admin_password = None

        if (self.local_llm_server is None) != (self.local_llm_model is None):
            raise ValueError(
                "Set both LOCAL_LLM_SERVER and LOCAL_LLM_MODEL to use a local model, or neither."
            )

        # Relative paths in .env are relative to the project root, not the working directory.
        for field in (*_PATH_FIELDS, "local_llm_server", "local_llm_model"):
            path = getattr(self, field)
            if path is not None and not path.is_absolute():
                setattr(self, field, PROJECT_ROOT / path)

        self.allowed_extensions = {
            ext.lower() if ext.startswith(".") else f".{ext.lower()}"
            for ext in self.allowed_extensions
        }
        return self

    @property
    def admin_enabled(self) -> bool:
        """Whether admin features are available (``ADMIN_PASSWORD`` is set)."""
        return self.admin_password is not None

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def local_model_id(self) -> str | None:
        """Model ID of the local model, or None if none is configured."""
        if self.local_llm_model is None:
            return None
        return LOCAL_MODEL_PREFIX + self.local_llm_model.stem

    @property
    def local_llm_base_url(self) -> str:
        """OpenAI-compatible endpoint of the local model server."""
        return f"http://{LOCAL_LLM_HOST}:{self.local_llm_port}/v1"

    @property
    def available_models(self) -> list[str]:
        """Chat models the admin can choose from: the Groq models, then the local one."""
        local = [self.local_model_id] if self.local_model_id else []
        return [*self.groq_models, *local]

    @property
    def default_model(self) -> str:
        return self.groq_models[0]

    @property
    def embedding_model_name(self) -> str:
        return self.embedding_model

    @property
    def collection_name(self) -> str:
        """ChromaDB collection name, unique per embedding provider and model (FR-36).

        Different embedding models produce incompatible vectors, so each gets its
        own collection. Changing the model then just requires re-indexing.
        """
        slug = re.sub(r"[^a-z0-9]+", "-", self.embedding_model_name.lower()).strip("-")
        name = f"{self.collection_prefix}_{self.embedding_provider.value}_{slug}"
        return name[:63].rstrip("-_")  # ChromaDB: max 63 chars, must end alphanumeric


@lru_cache
def get_settings() -> Settings:
    """Return the shared settings instance (Singleton)."""
    return Settings()
