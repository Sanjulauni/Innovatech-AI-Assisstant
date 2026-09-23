"""Fake models used in tests, so no real model or API is ever called."""

from typing import Any

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from pydantic import Field


class RecordingChatModel(FakeListChatModel):
    """Fake LLM that returns canned responses and remembers every prompt."""

    prompts: list[list[Any]] = Field(default_factory=list)

    def _call(self, messages, *args, **kwargs):
        self.prompts.append(messages)
        return super()._call(messages, *args, **kwargs)

    def _stream(self, messages, *args, **kwargs):
        self.prompts.append(messages)
        yield from super()._stream(messages, *args, **kwargs)


class BrokenChatModel(FakeListChatModel):
    def _call(self, *args, **kwargs):
        raise ConnectionError("Groq unreachable")

    def _stream(self, *args, **kwargs):
        raise ConnectionError("Groq unreachable")
        yield  # pragma: no cover  (makes this a generator)


class RateLimitedChatModel(FakeListChatModel):
    def _call(self, *args, **kwargs):
        raise RuntimeError("429 RESOURCE_EXHAUSTED: quota exceeded")


class MidStreamFailureChatModel(FakeListChatModel):
    """Streams a few pieces of text, then loses the connection."""

    def _stream(self, *args, **kwargs):
        yield from list(super()._stream(*args, **kwargs))[:3]
        raise ConnectionError("connection reset")


class OverloadedChatModel(FakeListChatModel):
    def _call(self, *args, **kwargs):
        raise RuntimeError("503 UNAVAILABLE: This model is currently experiencing high demand.")
