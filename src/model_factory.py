"""Factories that build the chat model and embedding model for the configured provider.

The rest of the application depends only on LangChain's ``BaseChatModel`` and
``Embeddings`` interfaces, so adding a provider (e.g. Ollama in v2) means writing
one builder function and registering it in ``_BUILDERS`` (FR-37, NFR-16).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import TypeVar

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from src.config import ModelProvider, Settings, get_settings

logger = logging.getLogger(__name__)

T = TypeVar("T")


def is_rate_limited(exc: BaseException) -> bool:
    """Whether the provider rejected a call for exceeding its quota (HTTP 429)."""
    text = str(exc)
    return "429" in text or "RESOURCE_EXHAUSTED" in text


class RetryingEmbeddings(Embeddings):
    """Wraps an embedding model: sends documents in small batches and, when the
    provider says "too many requests", waits and retries instead of failing.

    Free-tier quotas are per minute, so a large document (or several uploaded in a
    row) would otherwise fail part-way even though waiting briefly fixes it.
    """

    _FIRST_DELAY = 5.0
    _MAX_DELAY = 30.0
    _QUERY_MAX_WAIT = 15.0  # chat questions shouldn't hang for minutes

    def __init__(
        self,
        inner: Embeddings,
        batch_size: int = 20,
        max_wait_seconds: float = 120.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.inner = inner
        self._batch_size = batch_size
        self._max_wait = max_wait_seconds
        self._sleep = sleep

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            embedded = self._retry(lambda b=batch: self.inner.embed_documents(b), self._max_wait)
            vectors.extend(embedded)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._retry(
            lambda: self.inner.embed_query(text), min(self._max_wait, self._QUERY_MAX_WAIT)
        )

    def _retry(self, call: Callable[[], T], max_wait: float) -> T:
        waited, delay = 0.0, self._FIRST_DELAY
        while True:
            try:
                return call()
            except Exception as exc:
                if not is_rate_limited(exc) or waited + delay > max_wait:
                    raise
                logger.warning("Embedding quota reached; retrying in %.0fs", delay)
                self._sleep(delay)
                waited += delay
                delay = min(delay * 2, self._MAX_DELAY)


# --- Builders ----------------------------------------------------------------


def _build_gemini_llm(settings: Settings) -> BaseChatModel:
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=settings.gemini_llm_model,
        google_api_key=settings.google_api_key,
        temperature=settings.llm_temperature,
    )


def _build_gemini_embeddings(settings: Settings) -> Embeddings:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    return GoogleGenerativeAIEmbeddings(
        model=settings.gemini_embedding_model,
        google_api_key=settings.google_api_key,
    )


# --- Factories ---------------------------------------------------------------


class LLMFactory:
    """Creates the chat model for the configured ``LLM_PROVIDER``."""

    _BUILDERS: dict[ModelProvider, Callable[[Settings], BaseChatModel]] = {
        ModelProvider.GEMINI: _build_gemini_llm,
    }

    @classmethod
    def create(cls, settings: Settings | None = None) -> BaseChatModel:
        settings = settings or get_settings()
        builder = cls._BUILDERS.get(settings.llm_provider)
        if builder is None:
            raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")

        logger.info(
            "Using LLM provider=%s model=%s",
            settings.llm_provider.value,
            settings.llm_model_name,
        )
        return builder(settings)


class EmbeddingFactory:
    """Creates the embedding model for the configured provider."""

    _BUILDERS: dict[ModelProvider, Callable[[Settings], Embeddings]] = {
        ModelProvider.GEMINI: _build_gemini_embeddings,
    }

    @classmethod
    def create(cls, settings: Settings | None = None) -> Embeddings:
        settings = settings or get_settings()
        builder = cls._BUILDERS.get(settings.llm_provider)
        if builder is None:
            raise ValueError(f"Unsupported embedding provider: {settings.llm_provider}")

        logger.info(
            "Using embedding provider=%s model=%s",
            settings.llm_provider.value,
            settings.embedding_model_name,
        )
        return RetryingEmbeddings(
            builder(settings),
            batch_size=settings.embedding_batch_size,
            max_wait_seconds=settings.embedding_retry_seconds,
        )
