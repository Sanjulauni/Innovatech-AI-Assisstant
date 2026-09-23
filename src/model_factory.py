"""Factories that build the chat model and the embedding model.

The rest of the application depends only on LangChain's chat-model (``Runnable``) and
``Embeddings`` interfaces, so adding a provider (e.g. Ollama) means writing one
builder function and registering it (FR-37, NFR-16).
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable

from src.config import EmbeddingProvider, ModelProvider, Settings, get_settings

logger = logging.getLogger(__name__)

# Anything with invoke/stream that returns a chat message.
ChatModel = Runnable[LanguageModelInput, BaseMessage]


def is_rate_limited(exc: BaseException) -> bool:
    """Whether the provider rejected a call for exceeding its quota (HTTP 429)."""
    text = str(exc)
    return "429" in text or "RESOURCE_EXHAUSTED" in text or "rate limit" in text.lower()


# --- Embeddings ------------------------------------------------------------------


class FastEmbedEmbeddings(Embeddings):
    """Local embeddings with FastEmbed (ONNX, CPU). No API calls and no quota.

    The model is downloaded to ``cache_dir`` and loaded on first use, so starting
    the app (and running tests) stays fast.
    """

    def __init__(self, model_name: str, cache_dir: Path, batch_size: int = 32) -> None:
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self._batch_size = batch_size
        self._model: Any = None
        self._lock = threading.Lock()

    def _load(self) -> Any:
        with self._lock:
            if self._model is None:
                # Windows without Developer Mode can't make symlinks; the cache still works.
                os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
                from fastembed import TextEmbedding

                logger.info("Loading embedding model %s (first use downloads it)", self.model_name)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                self._model = TextEmbedding(self.model_name, cache_dir=str(self.cache_dir))
            return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._load()
        return [vector.tolist() for vector in model.embed(texts, batch_size=self._batch_size)]

    def embed_query(self, text: str) -> list[float]:
        # BGE models expect queries with a search prefix; query_embed adds it.
        return next(iter(self._load().query_embed(text))).tolist()


# --- Builders --------------------------------------------------------------------


def _build_groq_llm(settings: Settings, model: str) -> BaseChatModel:
    from langchain_groq import ChatGroq

    options: dict[str, Any] = {}
    if model.startswith("qwen/"):
        # Qwen reasoning models otherwise put their <think> notes in the answer.
        options["reasoning_format"] = "hidden"
    return ChatGroq(
        model=model,
        api_key=settings.groq_api_key,
        temperature=settings.llm_temperature,
        max_retries=2,
        **options,
    )


def _build_fastembed(settings: Settings) -> Embeddings:
    return FastEmbedEmbeddings(settings.embedding_model, settings.embedding_cache_dir)


# --- Factories -------------------------------------------------------------------


class LLMFactory:
    """Creates a chat model for the configured ``LLM_PROVIDER``."""

    _BUILDERS: dict[ModelProvider, Callable[[Settings, str], ChatModel]] = {
        ModelProvider.GROQ: _build_groq_llm,
    }

    @classmethod
    def create(cls, settings: Settings | None = None, model: str | None = None) -> ChatModel:
        settings = settings or get_settings()
        builder = cls._BUILDERS.get(settings.llm_provider)
        if builder is None:
            raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")
        model = model or settings.default_model
        logger.info("Using LLM provider=%s model=%s", settings.llm_provider.value, model)
        return builder(settings, model)


class EmbeddingFactory:
    """Creates the embedding model for the configured ``EMBEDDING_PROVIDER``."""

    _BUILDERS: dict[EmbeddingProvider, Callable[[Settings], Embeddings]] = {
        EmbeddingProvider.FASTEMBED: _build_fastembed,
    }

    @classmethod
    def create(cls, settings: Settings | None = None) -> Embeddings:
        settings = settings or get_settings()
        builder = cls._BUILDERS.get(settings.embedding_provider)
        if builder is None:
            raise ValueError(f"Unsupported embedding provider: {settings.embedding_provider}")
        logger.info(
            "Using embedding provider=%s model=%s",
            settings.embedding_provider.value,
            settings.embedding_model_name,
        )
        return builder(settings)
