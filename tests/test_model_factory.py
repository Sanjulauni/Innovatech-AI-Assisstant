"""Tests for the model factories (FR-37, NFR-16). No network calls are made."""

import pytest
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from src.model_factory import EmbeddingFactory, LLMFactory
from tests.conftest import make_settings


def test_llm_factory_builds_gemini():
    settings = make_settings(llm_temperature=0.3)
    llm = LLMFactory.create(settings)
    assert isinstance(llm, ChatGoogleGenerativeAI)
    assert settings.gemini_llm_model in llm.model
    assert llm.temperature == 0.3


def test_embedding_factory_builds_gemini():
    settings = make_settings()
    embeddings = EmbeddingFactory.create(settings)
    assert isinstance(embeddings, GoogleGenerativeAIEmbeddings)
    assert settings.gemini_embedding_model in embeddings.model


def test_llm_factory_rejects_unregistered_provider(monkeypatch):
    monkeypatch.setattr(LLMFactory, "_BUILDERS", {})
    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        LLMFactory.create(make_settings())


def test_embedding_factory_rejects_unregistered_provider(monkeypatch):
    monkeypatch.setattr(EmbeddingFactory, "_BUILDERS", {})
    with pytest.raises(ValueError, match="Unsupported embedding provider"):
        EmbeddingFactory.create(make_settings())
