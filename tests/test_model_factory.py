"""Tests for the model factories (FR-37, NFR-16). No network calls are made."""

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from src.model_factory import EmbeddingFactory, LLMFactory, RetryingEmbeddings, is_rate_limited
from tests.conftest import make_settings


def test_llm_factory_builds_gemini():
    settings = make_settings(llm_temperature=0.3)
    llm = LLMFactory.create(settings)
    assert isinstance(llm, ChatGoogleGenerativeAI)
    assert settings.gemini_llm_model in llm.model
    assert llm.temperature == 0.3


def test_embedding_factory_builds_gemini_with_retries():
    settings = make_settings(embedding_batch_size=7, embedding_retry_seconds=30)
    embeddings = EmbeddingFactory.create(settings)
    assert isinstance(embeddings, RetryingEmbeddings)
    assert isinstance(embeddings.inner, GoogleGenerativeAIEmbeddings)
    assert settings.gemini_embedding_model in embeddings.inner.model
    assert (embeddings._batch_size, embeddings._max_wait) == (7, 30)


def test_llm_factory_rejects_unregistered_provider(monkeypatch):
    monkeypatch.setattr(LLMFactory, "_BUILDERS", {})
    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        LLMFactory.create(make_settings())


def test_embedding_factory_rejects_unregistered_provider(monkeypatch):
    monkeypatch.setattr(EmbeddingFactory, "_BUILDERS", {})
    with pytest.raises(ValueError, match="Unsupported embedding provider"):
        EmbeddingFactory.create(make_settings())


# --- RetryingEmbeddings ------------------------------------------------------------


class FlakyEmbeddings(DeterministicFakeEmbedding):
    """Fails with a quota error for the first ``failures`` calls, and records batches."""

    failures: int = 0
    batches: list = []

    def embed_documents(self, texts):
        self.batches.append(len(texts))
        if self.failures > 0:
            self.failures -= 1
            raise RuntimeError("429 RESOURCE_EXHAUSTED: quota exceeded")
        return super().embed_documents(texts)

    def embed_query(self, text):
        if self.failures > 0:
            self.failures -= 1
            raise RuntimeError("429 RESOURCE_EXHAUSTED")
        return super().embed_query(text)


def make_retrying(failures=0, batch_size=20, max_wait=120.0):
    inner = FlakyEmbeddings(size=8, failures=failures, batches=[])
    sleeps = []
    return RetryingEmbeddings(inner, batch_size, max_wait, sleep=sleeps.append), inner, sleeps


def test_embeds_in_batches():
    embeddings, inner, sleeps = make_retrying(batch_size=20)
    texts = [f"text {i}" for i in range(54)]
    vectors = embeddings.embed_documents(texts)
    assert inner.batches == [20, 20, 14]
    assert sleeps == []
    assert vectors == DeterministicFakeEmbedding(size=8).embed_documents(texts)  # order kept


def test_retries_quota_errors_with_backoff():
    embeddings, inner, sleeps = make_retrying(failures=3)
    assert len(embeddings.embed_documents(["a", "b"])) == 2
    assert sleeps == [5, 10, 20]


def test_gives_up_after_retry_budget():
    embeddings, _, sleeps = make_retrying(failures=100, max_wait=40)
    with pytest.raises(RuntimeError, match="429"):
        embeddings.embed_documents(["a"])
    assert sum(sleeps) <= 40


def test_other_errors_are_not_retried():
    class Broken(DeterministicFakeEmbedding):
        def embed_documents(self, texts):
            raise ConnectionError("network down")

    sleeps = []
    embeddings = RetryingEmbeddings(Broken(size=8), sleep=sleeps.append)
    with pytest.raises(ConnectionError):
        embeddings.embed_documents(["a"])
    assert sleeps == []


def test_query_retries_briefly():
    embeddings, _, sleeps = make_retrying(failures=1)
    assert len(embeddings.embed_query("q")) == 8
    assert sleeps == [5]

    embeddings, _, sleeps = make_retrying(failures=100)
    with pytest.raises(RuntimeError):
        embeddings.embed_query("q")
    assert sum(sleeps) <= 15


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (RuntimeError("429 Too Many Requests"), True),
        (RuntimeError("Error (RESOURCE_EXHAUSTED)"), True),
        (ConnectionError("refused"), False),
    ],
)
def test_is_rate_limited(error, expected):
    assert is_rate_limited(error) is expected
