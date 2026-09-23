"""Centralized application settings.

All tunable values are read from environment variables or a local ``.env`` file
(see ``.env.example``). Nothing configurable should be hard-coded elsewhere (NFR-15).

Version 1 supports a single model provider: the Google Gemini API (FR-34).
"""

from __future__ import annotations

import re
from enum import Enum
from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class ModelProvider(str, Enum):
    """Supported model providers. Add new ones (e.g. OLLAMA) here in later versions."""

    GEMINI = "gemini"


_ENV_CONFIG = SettingsConfigDict(
    env_file=PROJECT_ROOT / ".env",
    env_file_encoding="utf-8",
    case_sensitive=False,
    env_ignore_empty=True,  # `KEY=` in .env counts as "not set"
    extra="ignore",
)


class Settings(BaseSettings):
    """Application settings, populated from the environment and ``.env``."""

    model_config = _ENV_CONFIG

    # --- Application -------------------------------------------------------
    app_name: str = "InnovaTech AI Assistant"
    log_level: str = "INFO"

    # --- Model provider ----------------------------------------------------
    llm_provider: ModelProvider = ModelProvider.GEMINI
    llm_temperature: float = Field(default=0.1, ge=0.0, le=2.0)

    # --- Gemini API --------------------------------------------------------
    google_api_key: SecretStr | None = None
    gemini_llm_model: str = "gemini-2.5-flash"
    gemini_embedding_model: str = "models/gemini-embedding-001"

    # --- Data & vector store -----------------------------------------------
    raw_data_dir: Path = PROJECT_ROOT / "data" / "raw"
    vector_db_dir: Path = PROJECT_ROOT / "data" / "vector_db"
    collection_prefix: str = "innovatech"
    instructions_file: Path = PROJECT_ROOT / "data" / "agent_instructions.json"

    # --- Ingestion ---------------------------------------------------------
    chunk_size: int = Field(default=1000, gt=0)
    chunk_overlap: int = Field(default=200, ge=0)
    allowed_extensions: set[str] = {".pdf", ".docx", ".txt", ".md"}
    max_upload_size_mb: int = Field(default=20, gt=0)

    # --- Retrieval ---------------------------------------------------------
    retriever_top_k: int = Field(default=4, gt=0)
    # Number of previous chat messages (user + assistant) sent with each question.
    chat_history_limit: int = Field(default=6, ge=0)

    # --- API ---------------------------------------------------------------
    api_host: str = "127.0.0.1"
    api_port: int = 8000

    # --- Access control ----------------------------------------------------
    # Password for the admin endpoints and page. Admin features are disabled if unset.
    admin_password: SecretStr | None = None

    @model_validator(mode="after")
    def _validate(self) -> Settings:
        if self.llm_provider is ModelProvider.GEMINI and not (
            self.google_api_key and self.google_api_key.get_secret_value().strip()
        ):
            raise ValueError(
                "GOOGLE_API_KEY is not set. Add it to your .env file "
                "(get a key at https://aistudio.google.com/apikey)."
            )

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")

        if self.admin_password is not None and not self.admin_password.get_secret_value().strip():
            self.admin_password = None

        # Relative paths in .env are relative to the project root, not the working directory.
        for field in ("raw_data_dir", "vector_db_dir", "instructions_file"):
            path = getattr(self, field)
            if not path.is_absolute():
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
    def llm_model_name(self) -> str:
        """Model name for the active LLM provider."""
        return self.gemini_llm_model

    @property
    def embedding_model_name(self) -> str:
        """Model name for the active embedding provider."""
        return self.gemini_embedding_model

    @property
    def collection_name(self) -> str:
        """ChromaDB collection name, unique per provider and embedding model (FR-36).

        Different embedding models produce incompatible vectors, so each gets its
        own collection. Changing the model then just requires re-ingesting.
        """
        slug = re.sub(r"[^a-z0-9]+", "-", self.embedding_model_name.lower()).strip("-")
        name = f"{self.collection_prefix}_{self.llm_provider.value}_{slug}"
        return name[:63].rstrip("-_")  # ChromaDB: max 63 chars, must end alphanumeric


class UISettings(BaseSettings):
    """The few settings the Streamlit UI needs. It never needs the Gemini key."""

    model_config = _ENV_CONFIG

    api_base_url: str = "http://127.0.0.1:8000"
    request_timeout_seconds: float = Field(default=120.0, gt=0)


@lru_cache
def get_settings() -> Settings:
    """Return the shared settings instance (Singleton)."""
    return Settings()
