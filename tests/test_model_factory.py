"""Tests for the model factories (FR-37, NFR-16).

No network calls are made and no embedding model is downloaded: building a ChatGroq
client does not contact Groq, and FastEmbed is replaced with a fake.
"""

import sys
import types

import numpy as np
import pytest
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from src.model_factory import (
    EmbeddingFactory,
    FastEmbedEmbeddings,
    LLMFactory,
    is_rate_limited,
)
from tests.conftest import make_settings

# --- Chat models -------------------------------------------------------------------


def test_llm_factory_builds_default_groq_model():
    settings = make_settings(llm_temperature=0.3)
    llm = LLMFactory.create(settings)
    assert isinstance(llm, ChatGroq)
    assert llm.model_name == "openai/gpt-oss-120b"
    assert llm.temperature == 0.3


def test_llm_factory_builds_requested_model():
    llm = LLMFactory.create(make_settings(), "openai/gpt-oss-20b")
    assert llm.model_name == "openai/gpt-oss-20b"
    assert llm.reasoning_format is None


def test_qwen_hides_its_reasoning_notes():
    llm = LLMFactory.create(make_settings(), "qwen/qwen3.8-27b")
    assert llm.reasoning_format == "hidden"


def test_local_model_ids_build_a_client_for_the_local_server():
    llm = LLMFactory.create(make_settings(local_llm_port=9001), "local:gemma-4-E2B")
    assert isinstance(llm, ChatOpenAI)
    assert llm.openai_api_base == "http://127.0.0.1:9001/v1"
    assert llm.model_name == "gemma-4-E2B"
    assert llm.max_retries == 0


def test_llm_factory_rejects_unregistered_provider(monkeypatch):
    monkeypatch.setattr(LLMFactory, "_BUILDERS", {})
    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        LLMFactory.create(make_settings())


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (RuntimeError("Error code: 429 - {'error': {'message': 'Rate limit reached'}}"), True),
        (RuntimeError("Rate limit reached for model"), True),
        (RuntimeError("RESOURCE_EXHAUSTED"), True),
        (ConnectionError("refused"), False),
    ],
)
def test_is_rate_limited(error, expected):
    assert is_rate_limited(error) is expected


# --- Local embeddings ----------------------------------------------------------------


class FakeTextEmbedding:
    """Stands in for fastembed.TextEmbedding: records how it was used."""

    instances: list["FakeTextEmbedding"] = []

    def __init__(self, model_name, cache_dir=None):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.document_batches = []
        FakeTextEmbedding.instances.append(self)

    def embed(self, texts, batch_size=32):
        self.document_batches.append(batch_size)
        return (np.array([float(len(t)), 1.0]) for t in texts)

    def query_embed(self, text):
        return iter([np.array([float(len(text)), 2.0])])


@pytest.fixture
def fake_fastembed(monkeypatch):
    FakeTextEmbedding.instances = []
    fake_module = types.SimpleNamespace(TextEmbedding=FakeTextEmbedding)
    monkeypatch.setitem(sys.modules, "fastembed", fake_module)
    return FakeTextEmbedding


def test_embedding_factory_builds_fastembed(tmp_path):
    settings = make_settings(embedding_model="BAAI/bge-small-en-v1.5", embedding_cache_dir=tmp_path)
    embeddings = EmbeddingFactory.create(settings)
    assert isinstance(embeddings, FastEmbedEmbeddings)
    assert embeddings.model_name == "BAAI/bge-small-en-v1.5"
    assert embeddings.cache_dir == tmp_path


def test_fastembed_loads_model_lazily_once(fake_fastembed, tmp_path):
    embeddings = FastEmbedEmbeddings("some/model", tmp_path / "models")
    assert fake_fastembed.instances == []  # nothing loaded at startup

    assert embeddings.embed_documents(["ab", "abcd"]) == [[2.0, 1.0], [4.0, 1.0]]
    assert embeddings.embed_query("abc") == [3.0, 2.0]  # query path uses query_embed

    [model] = fake_fastembed.instances
    assert model.model_name == "some/model"
    assert model.cache_dir == str(tmp_path / "models")
    assert (tmp_path / "models").is_dir()


def test_fastembed_empty_input_does_not_load_model(fake_fastembed, tmp_path):
    assert FastEmbedEmbeddings("m", tmp_path).embed_documents([]) == []
    assert fake_fastembed.instances == []


def test_embedding_factory_rejects_unregistered_provider(monkeypatch):
    monkeypatch.setattr(EmbeddingFactory, "_BUILDERS", {})
    with pytest.raises(ValueError, match="Unsupported embedding provider"):
        EmbeddingFactory.create(make_settings())
