"""Shared pytest fixtures."""

import pytest

from src.config import Settings

_ENV_VARS = ("LLM_PROVIDER", "GOOGLE_API_KEY")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Make tests independent of the developer's shell environment."""
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def make_settings(**overrides) -> Settings:
    """Build settings from explicit values only, ignoring the real .env file."""
    overrides.setdefault("google_api_key", "test-key")
    return Settings(_env_file=None, **overrides)
