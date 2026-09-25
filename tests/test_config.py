"""Tests for settings loading and validation (FR-34 – FR-36)."""

import pytest
from pydantic import ValidationError

from src.config import PROJECT_ROOT, EmbeddingProvider, ModelProvider, Settings
from tests.conftest import make_settings


def test_defaults_to_groq_and_local_embeddings():
    settings = make_settings()
    assert settings.llm_provider is ModelProvider.GROQ
    assert settings.embedding_provider is EmbeddingProvider.FASTEMBED
    assert settings.default_model == "openai/gpt-oss-120b"
    assert settings.available_models == [
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "qwen/qwen3.8-27b",
    ]
    assert settings.embedding_model_name == "BAAI/bge-small-en-v1.5"


def test_api_key_read_from_environment(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "env-key")
    settings = Settings(_env_file=None)
    assert settings.groq_api_key.get_secret_value() == "env-key"


@pytest.mark.parametrize("key", [None, "", "   "])
def test_missing_api_key_is_rejected(key):
    with pytest.raises(ValidationError, match="GROQ_API_KEY"):
        make_settings(groq_api_key=key)


def test_api_key_is_hidden_in_output():
    settings = make_settings(groq_api_key="super-secret")
    assert "super-secret" not in repr(settings)
    assert "super-secret" not in str(settings.groq_api_key)


def test_models_from_comma_separated_env(monkeypatch):
    monkeypatch.setenv("GROQ_MODELS", " model-b , model-a,, model-b ")
    settings = Settings(_env_file=None, groq_api_key="k")
    assert settings.available_models == ["model-b", "model-a"]  # order kept, no blanks/dupes
    assert settings.default_model == "model-b"


def test_empty_model_list_is_rejected():
    with pytest.raises(ValidationError, match="GROQ_MODELS"):
        make_settings(groq_models=[" "])


def test_unsupported_provider_rejected():
    with pytest.raises(ValidationError):
        make_settings(llm_provider="ollama")  # not supported yet


def test_chunk_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValidationError, match="CHUNK_OVERLAP"):
        make_settings(chunk_size=100, chunk_overlap=100)


def test_allowed_extensions_are_normalized():
    settings = make_settings(allowed_extensions={"PDF", ".Txt"})
    assert settings.allowed_extensions == {".pdf", ".txt"}


def test_collection_name_includes_embedding_model():
    assert make_settings().collection_name == "innovatech_fastembed_baai-bge-small-en-v1-5"


def test_collection_name_changes_with_embedding_model():
    a = make_settings(embedding_model="org/model-a")
    b = make_settings(embedding_model="org/model-b")
    assert a.collection_name != b.collection_name


def test_collection_name_respects_chroma_limits():
    settings = make_settings(embedding_model="x" * 100)
    assert len(settings.collection_name) <= 63
    assert settings.collection_name[-1].isalnum()


def test_admin_disabled_without_password():
    assert make_settings().admin_enabled is False


@pytest.mark.parametrize("password", ["", "   "])
def test_blank_admin_password_disables_admin(password):
    settings = make_settings(admin_password=password)
    assert settings.admin_password is None
    assert settings.admin_enabled is False


def test_admin_password_read_from_environment_and_hidden(monkeypatch):
    monkeypatch.setenv("ADMIN_PASSWORD", "s3cret-admin")
    settings = Settings(_env_file=None, groq_api_key="test-key")
    assert settings.admin_enabled is True
    assert settings.admin_password.get_secret_value() == "s3cret-admin"
    assert "s3cret-admin" not in repr(settings)


def test_chat_history_limit_default_and_validation():
    assert make_settings().chat_history_limit == 6
    assert make_settings(chat_history_limit=0).chat_history_limit == 0
    with pytest.raises(ValidationError):
        make_settings(chat_history_limit=-1)


def test_relative_paths_are_resolved_against_project_root():
    settings = make_settings(instructions_file="data/custom.json", raw_data_dir="uploads")
    assert settings.instructions_file == PROJECT_ROOT / "data" / "custom.json"
    assert settings.raw_data_dir == PROJECT_ROOT / "uploads"


def test_absolute_paths_are_kept(tmp_path):
    settings = make_settings(instructions_file=tmp_path / "i.json")
    assert settings.instructions_file == tmp_path / "i.json"


def test_max_upload_size_bytes():
    assert make_settings(max_upload_size_mb=2).max_upload_size_bytes == 2 * 1024 * 1024


def test_upload_size_default_must_not_exceed_the_cap():
    assert make_settings().max_upload_size_cap_mb == 200
    with pytest.raises(ValidationError, match="MAX_UPLOAD_SIZE_CAP_MB"):
        make_settings(max_upload_size_mb=300)


# --- Local model -------------------------------------------------------------------------


def test_no_local_model_by_default():
    settings = make_settings()
    assert settings.local_model_id is None
    assert settings.available_models == settings.groq_models


def test_local_model_is_added_after_groq_models(tmp_path):
    settings = make_settings(
        local_llm_server=tmp_path / "llama-server.exe",
        local_llm_model=tmp_path / "gemma-4-E2B-it-Q4_K_M.gguf",
    )
    assert settings.local_model_id == "local:gemma-4-E2B-it-Q4_K_M"
    assert settings.available_models[-1] == "local:gemma-4-E2B-it-Q4_K_M"
    assert settings.default_model == "openai/gpt-oss-120b"  # the default stays a Groq model
    assert settings.local_llm_base_url == "http://127.0.0.1:8080/v1"


@pytest.mark.parametrize("field", ["local_llm_server", "local_llm_model"])
def test_local_model_needs_both_paths(tmp_path, field):
    with pytest.raises(ValidationError, match="both LOCAL_LLM_SERVER and LOCAL_LLM_MODEL"):
        make_settings(**{field: tmp_path / "x"})


def test_relative_local_paths_are_resolved_against_project_root():
    settings = make_settings(local_llm_server="bin/llama-server", local_llm_model="m/g.gguf")
    assert settings.local_llm_server == PROJECT_ROOT / "bin" / "llama-server"
    assert settings.local_llm_model == PROJECT_ROOT / "m" / "g.gguf"
