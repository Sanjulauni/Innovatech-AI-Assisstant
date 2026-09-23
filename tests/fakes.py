"""Fake models used in tests, so no Gemini calls are made."""

from typing import Any

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from pydantic import Field


class RecordingChatModel(FakeListChatModel):
    """Fake LLM that returns canned responses and remembers every prompt."""

    prompts: list[list[Any]] = Field(default_factory=list)

    def _call(self, messages, *args, **kwargs):
        self.prompts.append(messages)
        return super()._call(messages, *args, **kwargs)


class BrokenChatModel(FakeListChatModel):
    def _call(self, *args, **kwargs):
        raise ConnectionError("Gemini unreachable")
