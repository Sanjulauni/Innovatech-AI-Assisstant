"""Tests for the admin's agent instructions store."""

import json
from datetime import datetime, timezone

import pytest

from src.rag_engine.instructions import (
    MAX_INSTRUCTIONS_LENGTH,
    AgentInstructions,
    InstructionsStore,
)
from tests.conftest import make_settings


@pytest.fixture
def store(tmp_path):
    return InstructionsStore(tmp_path / "nested" / "instructions.json")


def test_empty_when_no_file(store):
    assert store.get() == AgentInstructions(text="", updated_at=None)


def test_update_then_get(store):
    before = datetime.now(timezone.utc)

    saved = store.update("  Be friendly. Escalate IT issues to it-help@innovatech.example.  ")
    loaded = store.get()

    assert saved.text == "Be friendly. Escalate IT issues to it-help@innovatech.example."
    assert loaded == saved
    assert loaded.updated_at >= before
    assert loaded.updated_at.tzinfo is not None


def test_update_overwrites_previous_text(store):
    store.update("first")
    store.update("second")
    assert store.get().text == "second"


def test_file_is_json_with_unicode(store, tmp_path):
    store.update("Répondez poliment — merci")
    data = json.loads((tmp_path / "nested" / "instructions.json").read_text(encoding="utf-8"))
    assert data["text"] == "Répondez poliment — merci"
    assert "updated_at" in data
    assert not (tmp_path / "nested" / "instructions.json.tmp").exists()


def test_rejects_too_long_text(store):
    with pytest.raises(ValueError, match="too long"):
        store.update("x" * (MAX_INSTRUCTIONS_LENGTH + 1))
    assert store.get().text == ""


@pytest.mark.parametrize("content", ["not json", "[1, 2]", '{"updated_at": "not a date"}'])
def test_corrupt_file_returns_empty(tmp_path, content):
    path = tmp_path / "instructions.json"
    path.write_text(content, encoding="utf-8")
    assert InstructionsStore(path).get() == AgentInstructions()


def test_from_settings(tmp_path):
    settings = make_settings(instructions_file=tmp_path / "i.json")
    InstructionsStore.from_settings(settings).update("hello")
    assert (tmp_path / "i.json").exists()
