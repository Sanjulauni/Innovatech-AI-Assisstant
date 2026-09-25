"""Fake models used in tests, so no real model or API is ever called."""

from typing import Any

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from pydantic import Field

from src.local_llm import ServerState, ServerStatus
from src.model_factory import ModelUnavailableError


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


class StubLocalServer:
    """Stands in for ``LocalLLMServer`` without starting a process; records calls."""

    def __init__(self, state=None, error: str | None = None) -> None:
        self._state = state or ServerState(ServerStatus.STOPPED)
        self._error = error  # raised by wait_until_ready, if set
        self.calls: list[str] = []

    def start(self) -> None:
        self.calls.append("start")
        self._state = ServerState(ServerStatus.STARTING)

    def stop(self) -> None:
        self.calls.append("stop")
        self._state = ServerState(ServerStatus.STOPPED)

    def wait_until_ready(self) -> None:
        self.calls.append("wait")
        if self._error:
            raise ModelUnavailableError(self._error)
        self._state = ServerState(ServerStatus.READY)

    def refresh(self):
        return self._state
