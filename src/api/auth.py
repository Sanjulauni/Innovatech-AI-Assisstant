"""Admin sessions and login throttling.

After a correct password, ``/admin/login`` sets a signed session token in an
HttpOnly cookie, so the browser app never has to store the password. Tokens are
signed with a random key created at startup: restarting the API logs everyone out.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import threading
import time
from collections import defaultdict, deque

SESSION_COOKIE = "admin_session"


class SessionManager:
    """Issues and checks signed, expiring admin session tokens."""

    def __init__(self, ttl_seconds: int, secret: bytes | None = None) -> None:
        self.ttl_seconds = ttl_seconds
        self._secret = secret or secrets.token_bytes(32)
        self._revoked: set[str] = set()
        self._lock = threading.Lock()

    def issue(self) -> str:
        expires = int(time.time()) + self.ttl_seconds
        payload = f"{expires}.{secrets.token_urlsafe(16)}"
        return f"{payload}.{self._sign(payload)}"

    def verify(self, token: str | None) -> bool:
        if not token or token.count(".") != 2:
            return False
        payload, signature = token.rsplit(".", 1)
        if not hmac.compare_digest(signature, self._sign(payload)):
            return False
        expires = payload.split(".", 1)[0]
        if not expires.isdigit() or int(expires) < time.time():
            return False
        with self._lock:
            return token not in self._revoked

    def revoke(self, token: str | None) -> None:
        """Invalidate a token on logout (kept until the process restarts)."""
        if self.verify(token):
            with self._lock:
                self._revoked.add(token)

    def _sign(self, payload: str) -> str:
        return hmac.new(self._secret, payload.encode(), hashlib.sha256).hexdigest()


class LoginRateLimiter:
    """Blocks a client after too many wrong passwords in a short window."""

    def __init__(self, max_failures: int = 5, window_seconds: float = 60.0) -> None:
        self._max_failures = max_failures
        self._window = window_seconds
        self._failures: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def is_blocked(self, client: str) -> bool:
        with self._lock:
            return len(self._recent(client)) >= self._max_failures

    def record_failure(self, client: str) -> None:
        with self._lock:
            self._recent(client).append(time.monotonic())

    def reset(self, client: str) -> None:
        with self._lock:
            self._failures.pop(client, None)

    def _recent(self, client: str) -> deque[float]:
        attempts = self._failures[client]
        cutoff = time.monotonic() - self._window
        while attempts and attempts[0] < cutoff:
            attempts.popleft()
        return attempts
