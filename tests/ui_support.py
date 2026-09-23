"""Builds an ``ApiClient`` backed by the real FastAPI app with fake models."""

import uuid
from pathlib import Path

import chromadb
from fastapi.testclient import TestClient
from langchain_core.embeddings import DeterministicFakeEmbedding

from src.api.app import create_app
from src.api.dependencies import build_services
from src.data_pipeline.vector_store import VectorStoreRepository
from src.ui.api_client import ApiClient
from tests.conftest import make_settings
from tests.fakes import RecordingChatModel

ADMIN_PASSWORD = "ui-test-password"


def make_fake_api(tmp_dir: Path, responses: list[str]) -> ApiClient:
    settings = make_settings(
        admin_password=ADMIN_PASSWORD,
        raw_data_dir=tmp_dir / "raw",
        instructions_file=tmp_dir / "instructions.json",
    )
    embeddings = DeterministicFakeEmbedding(size=32)
    repository = VectorStoreRepository.create(
        embeddings, collection_name=f"ui-{uuid.uuid4().hex}", client=chromadb.EphemeralClient()
    )
    services = build_services(
        settings, embeddings, RecordingChatModel(responses=responses), repository
    )
    return ApiClient("http://testserver", http_client=TestClient(create_app(services)))
