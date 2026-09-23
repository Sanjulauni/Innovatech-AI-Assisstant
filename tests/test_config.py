"""Tests for settings loading and validation (FR-34 – FR-36)."""

import pytest
from pydantic import ValidationError

from src.config import PROJECT_ROOT, ModelProvider, Settings
from tests.conftest import make_settings


def test_defaults_to_gemini():
    settings = make_settings()
    assert settings.llm_provider is ModelProvider.GEMINI
    assert settings.llm_model_name == settings.gemini_llm_model
    assert settings.embedding_model_name == settings.gemini_embedding_model


def test_api_key_read_from_environment(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "env-key")
    settings = Settings(_env_file=None)
    assert settings.google_api_key.get_secret_value() == "env-key"


@pytest.mark.parametrize("key", [None, "", "   "])
def test_missing_api_key_is_rejected(key):
    with pytest.raises(ValidationError, match="GOOGLE_API_KEY"):
        make_settings(google_api_key=key)


def test_api_key_is_hidden_in_output():
    settings = make_settings(google_api_key="super-secret")
    assert "super-secret" not in repr(settings)
    assert "super-secret" not in str(settings.google_api_key)


def test_unsupported_provider_rejected():
    with pytest.raises(ValidationError):
        make_settings(llm_provider="ollama")  # planned for v2


def test_chunk_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValidationError, match="CHUNK_OVERLAP"):
        make_settings(chunk_size=100, chunk_overlap=100)


def test_allowed_extensions_are_normalized():
    settings = make_settings(allowed_extensions={"PDF", ".Txt"})
    assert settings.allowed_extensions == {".pdf", ".txt"}


def test_collection_name_includes_embedding_model():
    assert make_settings().collection_name == "innovatech_gemini_models-gemini-embedding-001"


def test_collection_name_changes_with_embedding_model():
    a = make_settings(gemini_embedding_model="models/model-a")
    b = make_settings(gemini_embedding_model="models/model-b")
    assert a.collection_name != b.collection_name


def test_collection_name_respects_chroma_limits():
    settings = make_settings(gemini_embedding_model="x" * 100)
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
    settings = Settings(_env_file=None, google_api_key="test-key")
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
