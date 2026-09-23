"""Splits loaded documents into overlapping chunks for embedding (FR-03)."""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import Settings, get_settings


class DocumentSplitter:
    """Splits documents into chunks, trying paragraph, line, then word boundaries first."""

    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True,  # records each chunk's character offset in its source
        )

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> DocumentSplitter:
        settings = settings or get_settings()
        return cls(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)

    def split(self, documents: list[Document]) -> list[Document]:
        """Split documents into chunks, numbering them in order with ``chunk_index``."""
        chunks = self._splitter.split_documents(documents)
        for index, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = index
        return chunks
