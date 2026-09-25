"""Tests for the vector store repository (FR-05, FR-08, FR-09, NFR-19)."""

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from src.data_pipeline.vector_store import VectorStoreRepository
from tests.conftest import make_settings


def chunks(source, *texts, ingested_at="2026-01-01T00:00:00+00:00"):
    return [
        Document(
            page_content=text,
            metadata={
                "source": source,
                "file_name": f"abc_{source}",
                "ingested_at": ingested_at,
                "chunk_index": i,
            },
        )
        for i, text in enumerate(texts)
    ]


def test_add_and_list_documents(repository):
    repository.add_document("doc-a", chunks("a.txt", "alpha one", "alpha two"))
    repository.add_document(
        "doc-b", chunks("b.pdf", "beta", ingested_at="2026-02-01T00:00:00+00:00")
    )

    documents = repository.list_documents()

    assert [d.doc_id for d in documents] == ["doc-b", "doc-a"]  # newest first
    assert documents[1].source == "a.txt"
    assert documents[1].file_name == "abc_a.txt"
    assert documents[1].chunk_count == 2
    assert repository.count_chunks() == 3


def test_add_empty_chunk_list_is_noop(repository):
    assert repository.add_document("doc-a", []) == 0
    assert repository.list_documents() == []


def test_adding_same_document_twice_does_not_duplicate(repository):
    repository.add_document("doc-a", chunks("a.txt", "one", "two"))
    repository.add_document("doc-a", chunks("a.txt", "one", "two"))
    assert repository.count_chunks() == 2


def test_has_and_get_document(repository):
    repository.add_document("doc-a", chunks("a.txt", "one"))

    assert repository.has_document("doc-a")
    assert not repository.has_document("missing")
    assert repository.get_document("doc-a").source == "a.txt"
    assert repository.get_document("missing") is None


def test_delete_document_removes_only_its_chunks(repository):
    repository.add_document("doc-a", chunks("a.txt", "one", "two"))
    repository.add_document("doc-b", chunks("b.txt", "three"))

    assert repository.delete_document("doc-a") == 2
    assert repository.delete_document("doc-a") == 0
    assert [d.doc_id for d in repository.list_documents()] == ["doc-b"]


def test_search_returns_best_match_first_with_metadata(repository):
    repository.add_document("doc-a", chunks("a.txt", "annual leave is 14 days"))
    repository.add_document("doc-b", chunks("b.txt", "the office opens at 9"))

    results = repository.search("annual leave is 14 days", k=2)

    assert len(results) == 2
    assert results[0].document.page_content == "annual leave is 14 days"
    assert results[0].document.metadata["doc_id"] == "doc-a"
    assert results[0].score == 1.0
    assert all(0.0 <= r.score <= 1.0 for r in results)


def test_search_empty_store_returns_nothing(repository):
    assert repository.search("anything", k=4) == []


def test_persists_across_instances(tmp_path):
    settings = make_settings(vector_db_dir=tmp_path / "db")
    embeddings = DeterministicFakeEmbedding(size=16)

    first = VectorStoreRepository.from_settings(embeddings, settings)
    first.add_document("doc-a", chunks("a.txt", "persisted text"))
    del first

    second = VectorStoreRepository.from_settings(embeddings, settings)
    assert second.has_document("doc-a")


def test_large_documents_are_stored_in_batches(repository, monkeypatch):
    monkeypatch.setattr(repository, "_max_batch_size", lambda: 3)
    calls = []
    original = repository._store.add_documents
    monkeypatch.setattr(
        repository._store,
        "add_documents",
        lambda docs, ids: calls.append(len(docs)) or original(docs, ids=ids),
    )

    count = repository.add_document("doc-a", chunks("a.txt", *[f"part {n}" for n in range(7)]))

    assert count == 7
    assert calls == [3, 3, 1]
    assert repository.get_document("doc-a").chunk_count == 7


def test_failed_batch_leaves_no_partial_document(repository, monkeypatch):
    monkeypatch.setattr(repository, "_max_batch_size", lambda: 2)
    original = repository._store.add_documents
    calls = []

    def flaky(docs, ids):
        calls.append(ids)
        if len(calls) == 2:
            raise RuntimeError("disk full")
        return original(docs, ids=ids)

    monkeypatch.setattr(repository._store, "add_documents", flaky)

    with pytest.raises(RuntimeError, match="disk full"):
        repository.add_document("doc-a", chunks("a.txt", "one", "two", "three", "four"))
    assert not repository.has_document("doc-a")
    assert repository.count_chunks() == 0


def test_max_batch_size_comes_from_chromadb(repository):
    assert repository._max_batch_size() > 1000
