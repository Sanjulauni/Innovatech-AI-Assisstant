"""Tests for the admin's chat model choice."""

import json

import pytest

from src.rag_engine.model_selector import ModelSelector, describe
from tests.fakes import RecordingChatModel

MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]


def make_selector(tmp_path=None, built=None):
    built = built if built is not None else []

    def factory(model_id):
        built.append(model_id)
        return RecordingChatModel(responses=[f"answer from {model_id}"])

    path = tmp_path / "selection.json" if tmp_path else None
    return ModelSelector(MODELS, factory, path), built


def test_defaults_to_first_model():
    selector, _ = make_selector()
    assert selector.current() == "openai/gpt-oss-120b"


def test_select_changes_current_and_the_model_used(tmp_path):
    selector, built = make_selector(tmp_path)

    selector.select("openai/gpt-oss-20b")

    assert selector.current() == "openai/gpt-oss-20b"
    assert selector.llm().invoke("hi").text == "answer from openai/gpt-oss-20b"
    assert built == ["openai/gpt-oss-20b"]


def test_models_are_built_once_and_reused(tmp_path):
    selector, built = make_selector(tmp_path)
    first = selector.llm()
    assert selector.llm() is first

    selector.select("qwen/qwen3.8-27b")
    selector.llm()
    selector.select("openai/gpt-oss-120b")
    assert selector.llm() is first
    assert built == ["openai/gpt-oss-120b", "qwen/qwen3.8-27b"]


def test_rejects_unknown_model(tmp_path):
    selector, _ = make_selector(tmp_path)
    with pytest.raises(ValueError, match="not one of the available models"):
        selector.select("llama-3.3-70b-versatile")
    assert selector.current() == "openai/gpt-oss-120b"


def test_choice_survives_restart(tmp_path):
    make_selector(tmp_path)[0].select("qwen/qwen3.8-27b")

    restarted, _ = make_selector(tmp_path)

    assert restarted.current() == "qwen/qwen3.8-27b"
    saved = json.loads((tmp_path / "selection.json").read_text(encoding="utf-8"))
    assert saved["model"] == "qwen/qwen3.8-27b"
    assert saved["updated_at"]


def test_model_removed_from_list_falls_back_to_default(tmp_path):
    (tmp_path / "selection.json").write_text('{"model": "old/model"}', encoding="utf-8")
    selector, _ = make_selector(tmp_path)
    assert selector.current() == "openai/gpt-oss-120b"


@pytest.mark.parametrize("content", ["not json", "[]", '{"other": 1}'])
def test_corrupt_file_falls_back_to_default(tmp_path, content):
    (tmp_path / "selection.json").write_text(content, encoding="utf-8")
    assert make_selector(tmp_path)[0].current() == "openai/gpt-oss-120b"


def test_options_have_friendly_labels():
    selector, _ = make_selector()
    options = selector.options()
    assert [o.id for o in options] == MODELS
    assert options[0].label == "GPT-OSS 120B"
    assert options[0].description


def test_unknown_models_are_shown_by_id():
    option = describe("vendor/new-model")
    assert (option.label, option.description) == ("vendor/new-model", "")


def test_needs_at_least_one_model():
    with pytest.raises(ValueError):
        ModelSelector([], lambda m: None)
