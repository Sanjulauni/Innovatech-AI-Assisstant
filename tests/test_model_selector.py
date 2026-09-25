"""Tests for the admin's chat model choice."""

import json

import pytest

from src.local_llm import ServerState, ServerStatus
from src.model_factory import ModelUnavailableError
from src.rag_engine.model_selector import LocalModel, ModelSelector, describe
from tests.fakes import RecordingChatModel, StubLocalServer

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


# --- Local model --------------------------------------------------------------------------

LOCAL = "local:gemma-4-E2B-it-Q4_K_M"


def make_local_selector(tmp_path, server=None):
    server = server or StubLocalServer()
    selector, built = make_selector(tmp_path)
    selector = ModelSelector(
        MODELS,
        selector._factory,
        tmp_path / "selection.json",
        local=LocalModel(id=LOCAL, label="Gemma 4 E2B", server=server),
    )
    return selector, server, built


def test_local_model_is_listed_last_with_its_state(tmp_path):
    selector, _, _ = make_local_selector(tmp_path)

    options = selector.options()

    assert [o.id for o in options] == [*MODELS, LOCAL]
    assert {o.kind for o in options[:3]} == {"cloud"}
    assert {o.status for o in options[:3]} == {"ready"}
    local = options[-1]
    assert (local.kind, local.label, local.status) == ("local", "Gemma 4 E2B", "stopped")
    assert "never leave the machine" in local.description


def test_local_error_is_shown_with_its_reason(tmp_path):
    server = StubLocalServer(ServerState(ServerStatus.ERROR, "The model file was not found."))
    selector, _, _ = make_local_selector(tmp_path, server)
    local = selector.options()[-1]
    assert (local.status, local.detail) == ("error", "The model file was not found.")


def test_selecting_local_starts_it_and_switching_away_stops_it(tmp_path):
    selector, server, _ = make_local_selector(tmp_path)

    selector.select(LOCAL)
    assert server.calls == ["start"]
    assert selector.options()[-1].status == "starting"

    selector.select("openai/gpt-oss-20b")
    assert server.calls == ["start", "stop"]


def test_local_model_waits_for_its_server(tmp_path):
    selector, server, built = make_local_selector(tmp_path)
    selector.select(LOCAL)

    assert selector.llm().invoke("hi").text == f"answer from {LOCAL}"
    assert server.calls == ["start", "wait"]
    assert built == [LOCAL]


def test_cloud_models_never_touch_the_local_server(tmp_path):
    selector, server, _ = make_local_selector(tmp_path)
    selector.llm()
    assert server.calls == []


def test_local_model_that_cannot_start_raises(tmp_path):
    server = StubLocalServer(error="The local model is still loading.")
    selector, _, built = make_local_selector(tmp_path, server)
    selector.select(LOCAL)

    with pytest.raises(ModelUnavailableError, match="still loading"):
        selector.llm()
    assert built == []


def test_activate_starts_the_saved_local_model_and_shutdown_stops_it(tmp_path):
    make_local_selector(tmp_path)[0].select(LOCAL)
    restarted, server, _ = make_local_selector(tmp_path)

    restarted.activate()
    restarted.shutdown()

    assert server.calls == ["start", "stop"]


def test_activate_does_nothing_for_cloud_models(tmp_path):
    selector, server, _ = make_local_selector(tmp_path)
    selector.activate()
    assert server.calls == []
