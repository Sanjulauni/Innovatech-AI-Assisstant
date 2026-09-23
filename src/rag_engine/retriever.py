"""Finds the chunks most relevant to a question (FR-12)."""

from __future__ import annotations

from src.config import Settings, get_settings
from src.data_pipeline.vector_store import SearchResult, VectorStoreRepository
from src.rag_engine.models import ChatMessage


def build_search_query(question: str, history: list[ChatMessage] | None = None) -> str:
    """Add the previous user message to the query so follow-ups find the right chunks.

    A follow-up such as "and for contractors?" means little on its own; combined with
    the earlier "How many leave days do employees get?" it retrieves the leave policy.
    """
    previous = next((m.content for m in reversed(history or []) if m.role == "user"), "")
    return f"{previous}\n{question}".strip() if previous else question.strip()


class DocumentRetriever:
    """Returns the top-k chunks for a question from the vector store."""

    def __init__(self, repository: VectorStoreRepository, top_k: int) -> None:
        self._repository = repository
        self._top_k = top_k

    @classmethod
    def from_settings(
        cls, repository: VectorStoreRepository, settings: Settings | None = None
    ) -> DocumentRetriever:
        settings = settings or get_settings()
        return cls(repository, top_k=settings.retriever_top_k)

    def retrieve(
        self, question: str, history: list[ChatMessage] | None = None
    ) -> list[SearchResult]:
        return self._repository.search(build_search_query(question, history), k=self._top_k)
