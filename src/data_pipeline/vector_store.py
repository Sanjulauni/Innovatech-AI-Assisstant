"""Repository over the ChromaDB vector store (FR-04, FR-05, FR-08, FR-09, NFR-19).

Chunks of one uploaded file share a ``doc_id`` (the SHA-256 of the file's bytes), so
a document can be listed, found or deleted as a whole. The rest of the application
talks to ``VectorStoreRepository`` only and never imports ChromaDB directly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.config import Settings, get_settings

logger = logging.getLogger(__name__)

# Metadata keys written on every chunk.
DOC_ID = "doc_id"
SOURCE = "source"
FILE_NAME = "file_name"
INGESTED_AT = "ingested_at"


@dataclass(frozen=True)
class StoredDocument:
    """Summary of one indexed document, aggregated from its chunks' metadata."""

    doc_id: str
    source: str
    file_name: str
    chunk_count: int
    ingested_at: str


@dataclass(frozen=True)
class SearchResult:
    """A retrieved chunk with its relevance score (0 = unrelated, 1 = identical)."""

    document: Document
    score: float


class VectorStoreRepository:
    """Adds, lists, deletes and searches documents in one Chroma collection."""

    def __init__(self, store: Chroma) -> None:
        self._store = store

    @classmethod
    def create(
        cls,
        embeddings: Embeddings,
        collection_name: str,
        persist_directory: str | None = None,
        client=None,
    ) -> VectorStoreRepository:
        """Open (or create) a collection. Without a directory or client it lives in memory."""
        store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_directory,
            client=client,
            # Cosine distance gives scores that don't depend on vector length.
            collection_metadata={"hnsw:space": "cosine"},
        )
        return cls(store)

    @classmethod
    def from_settings(
        cls, embeddings: Embeddings, settings: Settings | None = None
    ) -> VectorStoreRepository:
        settings = settings or get_settings()
        settings.vector_db_dir.mkdir(parents=True, exist_ok=True)
        return cls.create(
            embeddings,
            collection_name=settings.collection_name,
            persist_directory=str(settings.vector_db_dir),
        )

    # --- Writes --------------------------------------------------------------

    def add_document(self, doc_id: str, chunks: list[Document]) -> int:
        """Store the chunks of one document. Returns the number of chunks stored.

        Chunk ids are ``<doc_id>:<n>``, so adding the same document twice overwrites
        its chunks instead of duplicating them.
        """
        if not chunks:
            return 0
        for chunk in chunks:
            chunk.metadata[DOC_ID] = doc_id
        ids = [f"{doc_id}:{n}" for n in range(len(chunks))]
        self._store.add_documents(chunks, ids=ids)
        logger.info("Stored %d chunks for document %s", len(chunks), doc_id[:12])
        return len(chunks)

    def delete_document(self, doc_id: str) -> int:
        """Delete every chunk of a document. Returns how many chunks were removed."""
        ids = self._store.get(where={DOC_ID: doc_id}, include=[])["ids"]
        if ids:
            self._store.delete(ids=ids)
            logger.info("Deleted %d chunks for document %s", len(ids), doc_id[:12])
        return len(ids)

    # --- Reads ---------------------------------------------------------------

    def has_document(self, doc_id: str) -> bool:
        return bool(self._store.get(where={DOC_ID: doc_id}, limit=1, include=[])["ids"])

    def get_document(self, doc_id: str) -> StoredDocument | None:
        metadatas = self._store.get(where={DOC_ID: doc_id}, include=["metadatas"])["metadatas"]
        documents = self._summarize(metadatas)
        return documents[0] if documents else None

    def list_documents(self) -> list[StoredDocument]:
        """All indexed documents, newest first."""
        metadatas = self._store.get(include=["metadatas"])["metadatas"]
        return self._summarize(metadatas)

    def count_chunks(self) -> int:
        return self._store._collection.count()

    def search(self, query: str, k: int) -> list[SearchResult]:
        """Return the ``k`` chunks most similar to ``query``, best first."""
        if self.count_chunks() == 0:
            return []
        results = self._store.similarity_search_with_score(query, k=k)
        # Chroma returns cosine *distance* (0 = identical, 2 = opposite).
        return [
            SearchResult(document=doc, score=max(0.0, min(1.0, 1.0 - distance)))
            for doc, distance in results
        ]

    @staticmethod
    def _summarize(metadatas: list[dict]) -> list[StoredDocument]:
        by_id: dict[str, dict] = {}
        for metadata in metadatas:
            doc_id = metadata.get(DOC_ID)
            if not doc_id:
                continue
            entry = by_id.setdefault(doc_id, {"metadata": metadata, "count": 0})
            entry["count"] += 1

        documents = [
            StoredDocument(
                doc_id=doc_id,
                source=str(entry["metadata"].get(SOURCE, "")),
                file_name=str(entry["metadata"].get(FILE_NAME, "")),
                chunk_count=entry["count"],
                ingested_at=str(entry["metadata"].get(INGESTED_AT, "")),
            )
            for doc_id, entry in by_id.items()
        ]
        return sorted(documents, key=lambda d: d.ingested_at, reverse=True)
