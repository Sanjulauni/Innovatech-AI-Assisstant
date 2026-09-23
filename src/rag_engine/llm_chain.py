"""The RAG facade: one ``ask(question, history)`` call over retrieval, prompting and generation.

Callers (the API) never deal with the vector store, prompts or the LLM directly.
"""

from __future__ import annotations

import logging
import re

from langchain_core.language_models import BaseChatModel

from src.config import Settings, get_settings
from src.data_pipeline.vector_store import SearchResult
from src.rag_engine.instructions import InstructionsStore
from src.rag_engine.models import Answer, ChatMessage, Source
from src.rag_engine.prompts import NOT_FOUND_MESSAGE, build_messages
from src.rag_engine.retriever import DocumentRetriever

logger = logging.getLogger(__name__)

_SNIPPET_LENGTH = 300
_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


class RAGError(Exception):
    """Base class for errors while answering a question."""


class ServiceUnavailableError(RAGError):
    """The embedding model or LLM could not be reached (NFR-20)."""


class RAGChain:
    """Answers questions from the indexed documents, with numbered source citations."""

    def __init__(
        self,
        retriever: DocumentRetriever,
        llm: BaseChatModel,
        instructions: InstructionsStore | None = None,
        history_limit: int = 6,
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._instructions = instructions
        self._history_limit = history_limit

    @classmethod
    def from_settings(
        cls,
        retriever: DocumentRetriever,
        llm: BaseChatModel,
        instructions: InstructionsStore | None = None,
        settings: Settings | None = None,
    ) -> RAGChain:
        settings = settings or get_settings()
        return cls(retriever, llm, instructions, history_limit=settings.chat_history_limit)

    def ask(self, question: str, history: list[ChatMessage] | None = None) -> Answer:
        question = question.strip()
        if not question:
            raise ValueError("The question is empty.")
        history = self._trim(history or [])

        try:
            results = self._retriever.retrieve(question, history)
        except Exception as exc:
            logger.error("Retrieval failed: %s", exc)
            raise ServiceUnavailableError(
                "The document search is unavailable right now. Please try again shortly."
            ) from exc

        if not results:
            # Nothing is indexed: answer without calling the LLM (FR-15, UC-01 3a).
            return Answer(answer=NOT_FOUND_MESSAGE)

        admin_instructions = self._instructions.get().text if self._instructions else ""
        messages = build_messages(question, results, history, admin_instructions)
        try:
            response = self._llm.invoke(messages)
        except Exception as exc:
            logger.error("LLM call failed: %s", exc)
            if _is_rate_limited(exc):
                raise ServiceUnavailableError(
                    "The AI model is busy (usage limit reached). "
                    "Please wait a minute and try again."
                ) from exc
            raise ServiceUnavailableError(
                "The AI model is unavailable right now. Please try again shortly."
            ) from exc

        text = response.text.strip() or NOT_FOUND_MESSAGE
        return Answer(answer=text, sources=self._select_sources(text, results))

    def _trim(self, history: list[ChatMessage]) -> list[ChatMessage]:
        if self._history_limit == 0:
            return []
        return [m for m in history if m.content.strip()][-self._history_limit :]

    @staticmethod
    def _select_sources(answer: str, results: list[SearchResult]) -> list[Source]:
        """Return the sources cited in the answer.

        A "not found" answer has no sources. If the model answered but forgot to
        cite, all retrieved chunks are returned so the employee can still check.
        """
        cited = {
            int(number)
            for group in _CITATION.findall(answer)
            for number in group.split(",")
            if 1 <= int(number) <= len(results)
        }
        if not cited:
            if is_not_found(answer):
                return []
            cited = set(range(1, len(results) + 1))

        sources = []
        for index in sorted(cited):
            result = results[index - 1]
            metadata = result.document.metadata
            page = metadata.get("page")
            sources.append(
                Source(
                    index=index,
                    doc_id=str(metadata.get("doc_id", "")),
                    source=str(metadata.get("source", "unknown")),
                    page=int(page) if page is not None else None,
                    snippet=_snippet(result.document.page_content),
                    score=round(result.score, 4),
                )
            )
        return sources


def _is_rate_limited(exc: Exception) -> bool:
    """Whether the provider rejected the call for exceeding its quota (HTTP 429)."""
    text = str(exc)
    return "429" in text or "RESOURCE_EXHAUSTED" in text


def is_not_found(answer: str) -> bool:
    """Whether the answer is the "not found" reply (ignoring case and apostrophe style)."""

    def normalize(text: str) -> str:
        return text.replace("\u2019", "'").strip().lower()  # curly -> straight apostrophe

    return normalize(answer).startswith(normalize(NOT_FOUND_MESSAGE).rstrip("."))


def _snippet(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= _SNIPPET_LENGTH else text[: _SNIPPET_LENGTH - 1] + "…"
