"""Plain data types shared by the RAG engine, the API and the UI."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Role = Literal["user", "assistant"]


@dataclass(frozen=True)
class ChatMessage:
    """One earlier message in the conversation."""

    role: Role
    content: str


@dataclass(frozen=True)
class Source:
    """A document chunk the answer is based on, numbered as cited in the answer."""

    index: int
    doc_id: str
    source: str
    page: int | None
    snippet: str
    score: float


@dataclass(frozen=True)
class Answer:
    answer: str
    sources: list[Source] = field(default_factory=list)
