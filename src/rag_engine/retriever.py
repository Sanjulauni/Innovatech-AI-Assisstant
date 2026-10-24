"""Finds the chunks most relevant to a question (FR-12, FR-16)."""

from __future__ import annotations

import math
from collections.abc import Collection

from src.config import Settings, get_settings
from src.data_pipeline.vector_store import SearchResult, VectorStoreRepository
from src.rag_engine.models import ChatMessage

# How much the previous question counts in the search, relative to the new one (1.0).
# Enough for follow-ups such as "and for contractors?" to find the earlier topic, but
# small enough that a question on a new topic finds that new topic.
FOLLOW_UP_WEIGHT = 0.3


def previous_question(history: list[ChatMessage] | None) -> str:
    """The last question the user asked before this one, or "" if there is none."""
    return next((m.content.strip() for m in reversed(history or []) if m.role == "user"), "")


def _unit(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector)) or 1.0
    return [x / norm for x in vector]


class DocumentRetriever:
    """Returns the top-k chunks for a question from the vector store.

    A follow-up such as "and for contractors?" means little on its own, so the search
    also takes the previous question into account: the two questions are embedded
    separately and combined, with the new question weighted most. (Joining the texts
    instead let the previous question dominate, so a question on a new topic still
    found the old topic's documents.)
    """

    def __init__(
        self,
        repository: VectorStoreRepository,
        top_k: int,
        follow_up_weight: float = FOLLOW_UP_WEIGHT,
    ) -> None:
        self._repository = repository
        self._top_k = top_k
        self._follow_up_weight = follow_up_weight

    @classmethod
    def from_settings(
        cls, repository: VectorStoreRepository, settings: Settings | None = None
    ) -> DocumentRetriever:
        settings = settings or get_settings()
        return cls(repository, top_k=settings.retriever_top_k)

    def retrieve(
        self,
        question: str,
        history: list[ChatMessage] | None = None,
        k: int | None = None,
        exclude_doc_ids: Collection[str] = (),
    ) -> list[SearchResult]:
        """The ``k`` best chunks (default: the configured top-k), best first.

        Without ``history`` this searches for the question alone. Chunks of the
        documents in ``exclude_doc_ids`` are never returned.
        """
        vector = _unit(self._repository.embed_query(question.strip()))
        previous = previous_question(history)
        if previous and self._follow_up_weight > 0:
            earlier = _unit(self._repository.embed_query(previous))
            weight = self._follow_up_weight
            vector = _unit([v + weight * e for v, e in zip(vector, earlier, strict=True)])
        return self._repository.search_by_vector(
            vector, k=k or self._top_k, exclude_doc_ids=exclude_doc_ids
        )
