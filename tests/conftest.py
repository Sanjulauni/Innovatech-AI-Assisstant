"""Shared pytest fixtures."""

import uuid

import chromadb
import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from src.config import Settings
from src.data_pipeline.vector_store import VectorStoreRepository

_ENV_VARS = ("LLM_PROVIDER", "GOOGLE_API_KEY", "ADMIN_PASSWORD")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Make tests independent of the developer's shell environment."""
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def make_settings(**overrides) -> Settings:
    """Build settings from explicit values only, ignoring the real .env file."""
    overrides.setdefault("google_api_key", "test-key")
    return Settings(_env_file=None, **overrides)


@pytest.fixture
def repository() -> VectorStoreRepository:
    """An in-memory vector store with fake embeddings (no Gemini calls).

    The in-memory Chroma client is shared by the whole process, so each test gets
    its own uniquely named collection.
    """
    return VectorStoreRepository.create(
        DeterministicFakeEmbedding(size=64),
        collection_name=f"test-{uuid.uuid4().hex}",
        client=chromadb.EphemeralClient(),
    )
