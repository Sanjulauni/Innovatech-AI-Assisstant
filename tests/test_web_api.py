"""Tests for the web-app side of the API: sessions, login throttling, streaming chat,
``/api`` paths and serving the built React app."""

import json
import time

import pytest
from fastapi.testclient import TestClient
from langchain_core.embeddings import DeterministicFakeEmbedding

from src.api.app import create_app
from src.api.auth import SESSION_COOKIE, LoginRateLimiter, SessionManager
from src.api.dependencies import build_services
from src.rag_engine.prompts import NOT_FOUND_MESSAGE
from tests.conftest import make_settings
from tests.fakes import BrokenChatModel, MidStreamFailureChatModel, RecordingChatModel

PASSWORD = "web-admin-password"


@pytest.fixture
def settings(tmp_path):
    return make_settings(
        admin_password=PASSWORD,
        raw_data_dir=tmp_path / "raw",
        instructions_file=tmp_path / "instructions.json",
        model_selection_file=tmp_path / "model.json",
        frontend_dist_dir=tmp_path / "dist",
    )


def make_client(settings, repository, llm=None):
    llm = llm or RecordingChatModel(responses=["Leave is 14 days [1]."])
    services = build_services(settings, DeterministicFakeEmbedding(size=32), llm, repository)
    return TestClient(create_app(services))


@pytest.fixture
def client(settings, repository):
    return make_client(settings, repository)


def login(client, password=PASSWORD):
    return client.post("/api/admin/login", json={"password": password})


def upload(client, name="leave.txt", content=b"Leave is 14 days."):
    return client.post("/api/admin/documents", files={"file": (name, content)})


def stream_events(response):
    return [json.loads(line) for line in response.text.splitlines() if line]


# --- SessionManager & LoginRateLimiter ---------------------------------------------


def test_session_tokens_verify_and_expire(monkeypatch):
    sessions = SessionManager(ttl_seconds=60)
    token = sessions.issue()
    assert sessions.verify(token)

    now = time.time()
    monkeypatch.setattr(time, "time", lambda: now + 61)
    assert not sessions.verify(token)


@pytest.mark.parametrize("bad", [None, "", "garbage", "1.2.3.4", "9999999999.abc.deadbeef"])
def test_session_rejects_malformed_tokens(bad):
    assert not SessionManager(ttl_seconds=60).verify(bad)


def test_session_rejects_tampered_and_foreign_tokens():
    sessions = SessionManager(ttl_seconds=60)
    expires, nonce, signature = sessions.issue().split(".")
    assert not sessions.verify(f"{int(expires) + 1000}.{nonce}.{signature}")
    assert not sessions.verify(SessionManager(ttl_seconds=60).issue())  # other key


def test_session_revoke():
    sessions = SessionManager(ttl_seconds=60)
    token = sessions.issue()
    sessions.revoke(token)
    assert not sessions.verify(token)


def test_rate_limiter_blocks_and_resets(monkeypatch):
    limiter = LoginRateLimiter(max_failures=2, window_seconds=10)
    limiter.record_failure("a")
    assert not limiter.is_blocked("a")
    limiter.record_failure("a")
    assert limiter.is_blocked("a")
    assert not limiter.is_blocked("b")

    limiter.reset("a")
    assert not limiter.is_blocked("a")


def test_rate_limiter_forgets_old_failures(monkeypatch):
    limiter = LoginRateLimiter(max_failures=1, window_seconds=10)
    limiter.record_failure("a")
    now = time.monotonic()
    monkeypatch.setattr(time, "monotonic", lambda: now + 11)
    assert not limiter.is_blocked("a")


# --- Login, session cookie, logout -------------------------------------------------------


def test_login_sets_httponly_session_cookie(client):
    response = login(client)

    assert response.status_code == 200
    assert response.json() == {"authenticated": True, "expires_in": 8 * 3600}
    cookie = response.headers["set-cookie"]
    assert cookie.startswith(f"{SESSION_COOKIE}=")
    assert "HttpOnly" in cookie
    assert "SameSite=strict" in cookie
    assert PASSWORD not in cookie


def test_session_cookie_grants_admin_access(client):
    assert client.get("/api/admin/session").status_code == 401
    login(client)

    assert client.get("/api/admin/session").json()["authenticated"] is True
    assert upload(client).status_code == 201
    assert len(client.get("/api/admin/documents").json()) == 1


def test_wrong_password_sets_no_cookie(client):
    response = login(client, "wrong")
    assert response.status_code == 401
    assert "set-cookie" not in response.headers
    assert client.get("/api/admin/documents").status_code == 401


def test_logout_ends_session(client):
    login(client)
    token = client.cookies.get(SESSION_COOKIE)

    assert client.post("/api/admin/logout").status_code == 204
    assert client.get("/api/admin/session").status_code == 401

    # The old token no longer works even if someone kept a copy.
    client.cookies.set(SESSION_COOKIE, token)
    assert client.get("/api/admin/session").status_code == 401


def test_forged_cookie_is_rejected(client):
    client.cookies.set(SESSION_COOKIE, "9999999999.nonce.forged")
    assert client.get("/api/admin/documents").status_code == 401


def test_login_is_throttled_after_repeated_failures(client):
    for _ in range(5):
        assert login(client, "wrong").status_code == 401

    response = login(client)  # even the right password is refused for now

    assert response.status_code == 429
    assert "Too many failed attempts" in response.json()["detail"]


def test_successful_login_resets_failure_count(client):
    for _ in range(4):
        login(client, "wrong")
    assert login(client).status_code == 200
    for _ in range(4):
        assert login(client, "wrong").status_code == 401


def test_login_disabled_without_admin_password(tmp_path, repository):
    client = make_client(make_settings(instructions_file=tmp_path / "i.json"), repository)
    response = login(client)
    assert response.status_code == 503
    assert "ADMIN_PASSWORD" in response.json()["detail"]


def test_session_lifetime_from_settings(tmp_path, repository):
    settings = make_settings(
        admin_password=PASSWORD, admin_session_hours=0.5, instructions_file=tmp_path / "i.json"
    )
    client = make_client(settings, repository)
    assert login(client).json()["expires_in"] == 1800


# --- Streaming chat --------------------------------------------------------------------


def test_chat_stream_sends_tokens_then_done(client):
    login(client)
    upload(client)

    response = client.post("/api/chat/stream", json={"question": "Leave is 14 days."})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    events = stream_events(response)
    tokens = [e for e in events if e["type"] == "token"]
    assert len(tokens) > 1
    assert "".join(e["text"] for e in tokens) == "Leave is 14 days [1]."
    done = events[-1]
    assert done["type"] == "done"
    assert done["answer"] == "Leave is 14 days [1]."
    assert [s["source"] for s in done["sources"]] == ["leave.txt"]


def test_chat_stream_empty_knowledge_base(client):
    events = stream_events(client.post("/api/chat/stream", json={"question": "Hi?"}))
    assert events == [
        {"type": "token", "text": NOT_FOUND_MESSAGE},
        {"type": "done", "answer": NOT_FOUND_MESSAGE, "sources": []},
    ]


def test_chat_stream_validates_request(client):
    assert client.post("/api/chat/stream", json={"question": ""}).status_code == 422
    assert client.post("/api/chat/stream", json={"question": "   "}).status_code == 422


def test_chat_stream_retrieval_failure_is_503(client, repository, monkeypatch):
    def fail(*_args, **_kwargs):
        raise ConnectionError("embeddings down")

    monkeypatch.setattr(repository, "search", fail)
    response = client.post("/api/chat/stream", json={"question": "Hi?"})
    assert response.status_code == 503


def test_chat_stream_llm_failure_before_text_is_503(settings, repository):
    client = make_client(settings, repository, BrokenChatModel(responses=["x"]))
    login(client)
    upload(client)

    response = client.post("/api/chat/stream", json={"question": "Leave?"})

    assert response.status_code == 503
    assert "unavailable" in response.json()["detail"]


def test_chat_stream_llm_failure_mid_answer_sends_error_event(settings, repository):
    client = make_client(settings, repository, MidStreamFailureChatModel(responses=["Hello"]))
    login(client)
    upload(client)

    events = stream_events(client.post("/api/chat/stream", json={"question": "Leave?"}))

    assert [e["type"] for e in events] == ["token", "token", "token", "error"]
    assert "unavailable" in events[-1]["message"]


# --- /api prefix and legacy paths -----------------------------------------------------------


def test_api_and_legacy_paths_both_work(client):
    assert client.get("/api/health").status_code == 200
    assert client.get("/health").status_code == 200
    assert client.post("/api/chat", json={"question": "Hi?"}).status_code == 200


# --- Serving the React app ----------------------------------------------------------------


@pytest.fixture
def built_frontend(settings):
    dist = settings.frontend_dist_dir
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<html>InnovaTech web app</html>", encoding="utf-8")
    (dist / "assets" / "app.js").write_text("console.log('app')", encoding="utf-8")
    return dist


def test_serves_index_and_assets(client, built_frontend):
    assert "InnovaTech web app" in client.get("/").text
    assert client.get("/assets/app.js").text == "console.log('app')"


def test_client_routes_fall_back_to_index(client, built_frontend):
    response = client.get("/admin")
    assert response.status_code == 200
    assert "InnovaTech web app" in response.text


def test_unknown_api_paths_are_not_swallowed(client, built_frontend):
    assert client.get("/api/does-not-exist").status_code == 404


def test_cannot_read_files_outside_dist(client, built_frontend, settings):
    secret = settings.frontend_dist_dir.parent / "secret.txt"
    secret.write_text("top secret", encoding="utf-8")
    for path in ("/../secret.txt", "/%2e%2e/secret.txt", "/assets/..%2f..%2fsecret.txt"):
        assert "top secret" not in client.get(path).text


def test_no_frontend_build_returns_404(client):
    assert client.get("/").status_code == 404
