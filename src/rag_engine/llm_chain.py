"""The RAG facade: one ``ask(question, history)`` call over retrieval, prompting and generation.

Callers (the API) never deal with the vector store, prompts or the LLM directly.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass

from langchain_core.messages import BaseMessage

from src.config import Settings, get_settings
from src.data_pipeline.confidential import ConfidentialStore
from src.data_pipeline.vector_store import SearchResult
from src.model_factory import ChatModel, ModelUnavailableError, is_rate_limited
from src.rag_engine.instructions import InstructionsStore
from src.rag_engine.model_selector import NO_LOCAL_MODEL_MESSAGE, ModelSelector
from src.rag_engine.models import Answer, ChatMessage, Route, Source
from src.rag_engine.prompts import NOT_FOUND_MESSAGE, build_messages
from src.rag_engine.retriever import DocumentRetriever

logger = logging.getLogger(__name__)

_SNIPPET_LENGTH = 300
_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
# Some models (e.g. GPT-OSS) cite with full-width brackets (U+3010/U+3011), such as
# "【1】", "【1, 2】" or "【1†source】". The comma may be full-width too (U+FF0C).
_WIDE_CITATION = re.compile(r"【\s*(\d+(?:\s*[,，]\s*\d+)*)[^】]*】")
_COMMA = re.compile(r"\s*[,，]\s*")


def normalize_citations(text: str) -> str:
    """Rewrite full-width citation markers as [1] / [1, 2]."""
    return _WIDE_CITATION.sub(lambda m: "[" + ", ".join(_COMMA.split(m.group(1))) + "]", text)


class RAGError(Exception):
    """Base class for errors while answering a question."""


class ServiceUnavailableError(RAGError):
    """The embedding model or LLM could not be reached (NFR-20)."""


@dataclass(frozen=True)
class _Plan:
    """What one question is answered with, decided after searching the documents."""

    question: str
    history: list[ChatMessage]
    results: list[SearchResult]
    route: Route
    confidential: frozenset[str]  # confidential doc IDs at the time of the search


def _doc_id(result: SearchResult) -> str:
    return str(result.document.metadata.get("doc_id", ""))


def _without_confidential_turns(history: list[ChatMessage]) -> list[ChatMessage]:
    """Drop answers based on confidential documents, and the questions that led to them."""
    kept: list[ChatMessage] = []
    for message in history:
        if message.confidential:
            if kept and kept[-1].role == "user":
                kept.pop()  # the question belongs to the confidential answer
            continue
        kept.append(message)
    return kept


class RAGChain:
    """Answers questions from the indexed documents, with numbered source citations.

    If any retrieved excerpt comes from a document the admin marked confidential, the
    question is answered by the local model, whichever model is selected. If there is no
    local model, or it can't be started, the question is refused: confidential text is
    never sent to a cloud model. Earlier answers based on confidential documents are
    likewise left out of the history sent to a cloud model.
    """

    def __init__(
        self,
        retriever: DocumentRetriever,
        llm: ChatModel | ModelSelector,
        instructions: InstructionsStore | None = None,
        history_limit: int = 6,
        local_top_k: int | None = None,
        local_history_limit: int | None = None,
        confidential: ConfidentialStore | None = None,
    ) -> None:
        self._retriever = retriever
        # A selector means the admin's current choice is used for each question.
        self._llm = llm
        self._instructions = instructions
        self._history_limit = history_limit
        # Smaller prompts for the local model, which reads prompts slowly on a CPU.
        self._local_top_k = local_top_k
        self._local_history_limit = local_history_limit
        self._confidential = confidential

    @classmethod
    def from_settings(
        cls,
        retriever: DocumentRetriever,
        llm: ChatModel | ModelSelector,
        instructions: InstructionsStore | None = None,
        settings: Settings | None = None,
        confidential: ConfidentialStore | None = None,
    ) -> RAGChain:
        settings = settings or get_settings()
        return cls(
            retriever,
            llm,
            instructions,
            history_limit=settings.chat_history_limit,
            local_top_k=settings.local_llm_top_k,
            local_history_limit=settings.local_llm_history_limit,
            confidential=confidential,
        )

    def ask(self, question: str, history: list[ChatMessage] | None = None) -> Answer:
        plan = self._plan(question, history)
        if not plan.results:
            # Nothing is indexed: answer without calling the LLM (FR-15, UC-01 3a).
            return Answer(answer=NOT_FOUND_MESSAGE)

        try:
            response = self._llm_for(plan.route).invoke(self._messages(plan))
        except Exception as exc:
            raise _llm_error(exc) from exc
        return self._finish(response.text, plan)

    def stream(
        self, question: str, history: list[ChatMessage] | None = None
    ) -> Iterator[Route | str | Answer]:
        """Like ``ask``, but yields the answer text piece by piece (FR-18).

        Yields the ``Route`` (which model answers) as soon as the documents are searched,
        then ``str`` pieces, then one final ``Answer`` with the full text and the cited
        sources. Retrieval errors are raised by the first ``next()`` call, before any
        text, so callers can still report them as a normal error response.
        """
        plan = self._plan(question, history)
        yield plan.route
        if not plan.results:
            yield NOT_FOUND_MESSAGE
            yield Answer(answer=NOT_FOUND_MESSAGE)
            return

        parts: list[str] = []
        try:
            for chunk in self._llm_for(plan.route).stream(self._messages(plan)):
                if chunk.text:
                    parts.append(chunk.text)
                    yield chunk.text
        except Exception as exc:
            raise _llm_error(exc) from exc

        answer = self._finish("".join(parts), plan)
        if not parts or not "".join(parts).strip():
            yield answer.answer
        yield answer

    def _plan(self, question: str, history: list[ChatMessage] | None) -> _Plan:
        """Search the documents, then decide which model answers and with what context."""
        question = question.strip()
        if not question:
            raise ValueError("The question is empty.")
        history = [m for m in history or [] if m.content.strip()]
        local_selected = isinstance(self._llm, ModelSelector) and self._llm.local_selected()
        try:
            results = self._retriever.retrieve(
                question, history, k=self._local_top_k if local_selected else None
            )
        except Exception as exc:
            logger.error("Retrieval failed: %s", exc)
            raise ServiceUnavailableError(
                "The document search is unavailable right now. Please try again shortly."
            ) from exc

        confidential = self._confidential.ids() if self._confidential else frozenset()
        private = any(_doc_id(result) in confidential for result in results)
        local = local_selected or private
        if private and not local_selected and self._local_top_k:
            results = results[: self._local_top_k]  # the local model's smaller prompt
        if local:
            limit = self._local_history_limit
            if limit is None:
                limit = self._history_limit
        else:
            history = _without_confidential_turns(history)
            limit = self._history_limit
        route = Route(local=local, private=private)
        return _Plan(question, self._trim(history, limit), results, route, confidential)

    def _messages(self, plan: _Plan) -> list[BaseMessage]:
        admin_instructions = self._instructions.get().text if self._instructions else ""
        return build_messages(plan.question, plan.results, plan.history, admin_instructions)

    def _finish(self, text: str, plan: _Plan) -> Answer:
        text = normalize_citations(text).strip() or NOT_FOUND_MESSAGE
        sources = self._select_sources(text, plan.results, plan.confidential)
        return Answer(answer=text, sources=sources, private=plan.route.private)

    def _llm_for(self, route: Route) -> ChatModel:
        if not isinstance(self._llm, ModelSelector):
            if route.private:  # a bare model can't be known to be local
                raise ModelUnavailableError(NO_LOCAL_MODEL_MESSAGE)
            return self._llm
        if route.private and not self._llm.local_selected():
            return self._llm.local_llm()
        return self._llm.llm()

    @staticmethod
    def _trim(history: list[ChatMessage], limit: int) -> list[ChatMessage]:
        if limit == 0:
            return []
        return history[-limit:]

    @staticmethod
    def _select_sources(
        answer: str, results: list[SearchResult], confidential: frozenset[str] = frozenset()
    ) -> list[Source]:
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
                    confidential=_doc_id(result) in confidential,
                )
            )
        return sources


def _llm_error(exc: Exception) -> ServiceUnavailableError:
    logger.error("LLM call failed: %s", exc)
    if isinstance(exc, ModelUnavailableError):
        return ServiceUnavailableError(str(exc))  # already written for the user
    if is_rate_limited(exc):
        return ServiceUnavailableError(
            "The AI model is busy (usage limit reached). Please wait a minute and try again."
        )
    if _is_overloaded(exc):
        return ServiceUnavailableError(
            "The AI model is overloaded right now (high demand). Please try again in a minute."
        )
    return ServiceUnavailableError(
        "The AI model is unavailable right now. Please try again shortly."
    )


def _is_overloaded(exc: Exception) -> bool:
    text = str(exc)
    return "503" in text or "UNAVAILABLE" in text or "overloaded" in text.lower()


def is_not_found(answer: str) -> bool:
    """Whether the answer is the "not found" reply (ignoring case and apostrophe style)."""

    def normalize(text: str) -> str:
        return text.replace("\u2019", "'").strip().lower()  # curly -> straight apostrophe

    return normalize(answer).startswith(normalize(NOT_FOUND_MESSAGE).rstrip("."))


def _snippet(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= _SNIPPET_LENGTH else text[: _SNIPPET_LENGTH - 1] + "…"
