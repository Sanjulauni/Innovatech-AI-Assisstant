"""Tests for the Streamlit UI and its API client (FR-26 – FR-30)."""

from pathlib import Path

import httpx
import pytest
from streamlit.testing.v1 import AppTest

from src.ui.api_client import ADMIN_HEADER, ApiClient, ApiError
from src.ui.components.admin_page import PASSWORD_KEY
from src.ui.components.chat_page import MESSAGES_KEY
from tests.ui_support import ADMIN_PASSWORD, make_fake_api

HARNESS = str(Path(__file__).with_name("ui_harness.py"))


# --- ApiClient -----------------------------------------------------------------------


def mock_client(handler) -> ApiClient:
    return ApiClient(
        "http://api.test", http_client=httpx.Client(base_url="http://api.test", transport=handler)
    )


def test_client_sends_admin_header_and_json():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["header"] = request.headers.get(ADMIN_HEADER)
        seen["body"] = request.content
        return httpx.Response(200, json={"text": "hi", "updated_at": None})

    mock_client(httpx.MockTransport(handler)).update_instructions("pw", "hi")

    assert seen["header"] == "pw"
    assert b'"text":"hi"' in seen["body"].replace(b" ", b"")


def test_client_uses_server_error_detail():
    transport = httpx.MockTransport(
        lambda r: httpx.Response(401, json={"detail": "Incorrect admin password."})
    )
    with pytest.raises(ApiError) as error:
        mock_client(transport).login("wrong")
    assert error.value.status_code == 401
    assert error.value.message == "Incorrect admin password."


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (httpx.Response(500, text="<html>boom</html>"), r"server returned an error \(500\)"),
        (httpx.Response(422, json={"detail": [{"msg": "bad"}]}), "request was not valid"),
    ],
)
def test_client_falls_back_to_generic_messages(response, expected):
    with pytest.raises(ApiError, match=expected):
        mock_client(httpx.MockTransport(lambda r: response)).chat("q")


def test_client_reports_unreachable_server():
    def handler(request):
        raise httpx.ConnectError("refused")

    with pytest.raises(ApiError, match="Can't reach the assistant server at http://api.test"):
        mock_client(httpx.MockTransport(handler)).health()


def test_client_reports_timeout():
    def handler(request):
        raise httpx.ReadTimeout("slow")

    with pytest.raises(ApiError, match="took too long"):
        mock_client(httpx.MockTransport(handler)).chat("q")


def test_client_against_real_api(tmp_path):
    api = make_fake_api(tmp_path, ["Leave is 14 days [1]."])

    api.login(ADMIN_PASSWORD)
    uploaded = api.upload_document(ADMIN_PASSWORD, "leave.txt", b"Leave is 14 days.")
    assert uploaded["status"] == "ingested"
    assert [d["source"] for d in api.list_documents(ADMIN_PASSWORD)] == ["leave.txt"]

    reply = api.chat("Leave is 14 days.", [{"role": "user", "content": "hi"}])
    assert reply["answer"] == "Leave is 14 days [1]."
    assert reply["sources"][0]["source"] == "leave.txt"

    api.delete_document(ADMIN_PASSWORD, uploaded["doc_id"])
    assert api.list_documents(ADMIN_PASSWORD) == []
    assert api.health()["documents"] == 0


# --- Streamlit pages ---------------------------------------------------------------------


def run_page(tmp_path, page="chat", responses=None, **state) -> AppTest:
    at = AppTest.from_file(HARNESS, default_timeout=30)
    at.session_state["_tmp"] = str(tmp_path)
    at.session_state["_page"] = page
    if responses:
        at.session_state["_responses"] = responses
    for key, value in state.items():
        at.session_state[key] = value
    return at.run()


def test_chat_page_shows_answer_and_sources(tmp_path):
    at = run_page(tmp_path, responses=["Leave is 14 days [1]."])
    at.session_state["_api"].upload_document(ADMIN_PASSWORD, "leave.txt", b"Leave is 14 days.")

    at.chat_input[0].set_value("Leave is 14 days.").run()

    assert not at.exception
    assert [m.name for m in at.chat_message] == ["user", "assistant"]
    assert "Leave is 14 days [1]." in at.chat_message[1].markdown[0].value
    assert at.expander[0].label == "Sources (1)"
    assert "[1] leave.txt" in at.expander[0].markdown[0].value
    messages = at.session_state[MESSAGES_KEY]
    assert [m["role"] for m in messages] == ["user", "assistant"]


def test_chat_page_clear_button(tmp_path):
    at = run_page(tmp_path)
    at.chat_input[0].set_value("Hello?").run()
    assert len(at.session_state[MESSAGES_KEY]) == 2

    at.sidebar.button[0].click().run()

    assert at.session_state[MESSAGES_KEY] == []
    assert len(at.chat_message) == 0


def test_chat_page_shows_friendly_error(tmp_path, monkeypatch):
    at = run_page(tmp_path)

    def fail(*_args, **_kwargs):
        raise ApiError("Can't reach the assistant server.")

    monkeypatch.setattr(at.session_state["_api"], "chat", fail)
    at.chat_input[0].set_value("Hello?").run()

    assert at.error[0].value == "Can't reach the assistant server."
    assert at.session_state[MESSAGES_KEY] == []


def test_admin_page_requires_login(tmp_path):
    at = run_page(tmp_path, page="admin")

    assert at.text_input[0].label == "Admin password"
    at.text_input[0].input("wrong")
    at.button[0].click().run()
    assert at.error[0].value == "Incorrect admin password."
    assert PASSWORD_KEY not in at.session_state

    at.text_input[0].input(ADMIN_PASSWORD)
    at.button[0].click().run()
    assert at.session_state[PASSWORD_KEY] == ADMIN_PASSWORD
    assert [t.label for t in at.tabs] == ["Documents", "Instructions"]


def test_admin_page_lists_documents_and_saves_instructions(tmp_path):
    at = run_page(tmp_path, page="admin", **{PASSWORD_KEY: ADMIN_PASSWORD})
    api = at.session_state["_api"]
    api.upload_document(ADMIN_PASSWORD, "handbook.md", b"# Handbook\nWelcome.")
    at.run()

    assert any("handbook.md" in m.value for m in at.markdown)

    at.text_area[0].input("Use a friendly tone.")
    save = next(b for b in at.button if b.label == "Save instructions")
    save.click().run()

    assert not at.exception
    assert any("Instructions saved" in s.value for s in at.success)
    assert api.get_instructions(ADMIN_PASSWORD)["text"] == "Use a friendly tone."


def test_admin_page_logs_out_when_password_is_rejected(tmp_path):
    at = run_page(tmp_path, page="admin", **{PASSWORD_KEY: "stale-password"})

    assert PASSWORD_KEY not in at.session_state
    assert at.text_input[0].label == "Admin password"
    assert any("session expired" in e.value for e in at.error)


def test_entry_point_renders_chat_page():
    app_path = Path(__file__).resolve().parents[1] / "src" / "ui" / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=30).run()

    assert not at.exception
    assert at.title[0].value == "InnovaTech Assistant"
    assert len(at.chat_input) == 1
