"""Factories that build the chat model and embedding model for the configured provider.

The rest of the application depends only on LangChain's ``BaseChatModel`` and
``Embeddings`` interfaces, so adding a provider (e.g. Ollama in v2) means writing
one builder function and registering it in ``_BUILDERS`` (FR-37, NFR-16).
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from src.config import ModelProvider, Settings, get_settings

logger = logging.getLogger(__name__)


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
        return builder(settings)
