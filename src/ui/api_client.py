"""HTTP client the Streamlit UI uses to call the FastAPI backend.

Every failure is raised as ``ApiError`` with a plain-language message that the UI can
show as-is (NFR-12).
"""

from __future__ import annotations

from typing import Any

import httpx

ADMIN_HEADER = "X-Admin-Password"


class ApiError(Exception):
    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class ApiClient:
    def __init__(
        self,
        base_url: str,
        timeout: float = 120.0,
        http_client: httpx.Client | None = None,
    ) -> None:
        """``http_client`` replaces the default client (tests pass a FastAPI ``TestClient``)."""
        self._base_url = base_url.rstrip("/")
        self._client = http_client or httpx.Client(base_url=self._base_url, timeout=timeout)

    # --- Employee ------------------------------------------------------------------

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health").json()

    def chat(self, question: str, history: list[dict[str, str]] | None = None) -> dict[str, Any]:
        """Ask a question. ``history`` holds earlier ``{"role", "content"}`` messages."""
        payload = {"question": question, "history": history or []}
        return self._request("POST", "/chat", json=payload).json()

    # --- Admin ---------------------------------------------------------------------

    def login(self, password: str) -> None:
        """Raise ``ApiError`` (status 401) if the password is wrong."""
        self._request("POST", "/admin/login", password=password)

    def list_documents(self, password: str) -> list[dict[str, Any]]:
        return self._request("GET", "/admin/documents", password=password).json()

    def upload_document(self, password: str, filename: str, content: bytes) -> dict[str, Any]:
        files = {"file": (filename, content)}
        return self._request("POST", "/admin/documents", password=password, files=files).json()

    def delete_document(self, password: str, doc_id: str) -> None:
        self._request("DELETE", f"/admin/documents/{doc_id}", password=password)

    def get_instructions(self, password: str) -> dict[str, Any]:
        return self._request("GET", "/admin/instructions", password=password).json()

    def update_instructions(self, password: str, text: str) -> dict[str, Any]:
        return self._request(
            "PUT", "/admin/instructions", password=password, json={"text": text}
        ).json()

    # --- Internals -----------------------------------------------------------------

    def _request(
        self, method: str, path: str, password: str | None = None, **kwargs: Any
    ) -> httpx.Response:
        headers = {ADMIN_HEADER: password} if password is not None else None
        try:
            response = self._client.request(method, path, headers=headers, **kwargs)
        except httpx.TimeoutException as exc:
            raise ApiError("The server took too long to respond. Please try again.") from exc
        except httpx.HTTPError as exc:
            raise ApiError(
                f"Can't reach the assistant server at {self._base_url}. "
                "Make sure the API is running, then try again."
            ) from exc

        if response.is_error:
            raise ApiError(_error_message(response), response.status_code)
        return response


def _error_message(response: httpx.Response) -> str:
    try:
        detail = response.json().get("detail")
    except (ValueError, AttributeError):
        detail = None
    if isinstance(detail, str) and detail:
        return detail
    if response.status_code == 422:
        return "The request was not valid. Please check your input and try again."
    return f"The server returned an error ({response.status_code}). Please try again."
