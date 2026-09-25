"""The largest upload the admin allows, between 1 MB and ``MAX_UPLOAD_SIZE_CAP_MB``.

The limit is saved to a small JSON file (``UPLOAD_LIMIT_FILE``) so it survives
restarts, and it applies to the next upload without restarting the API. Until the
admin changes it, ``MAX_UPLOAD_SIZE_MB`` from ``.env`` is used.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

from src.config import Settings, get_settings

logger = logging.getLogger(__name__)

_MB = 1024 * 1024


class UploadLimit:
    """Knows the current upload size limit and lets the admin change it."""

    def __init__(self, default_mb: int, max_mb: int, store_path: Path | None = None) -> None:
        if not 1 <= default_mb <= max_mb:
            raise ValueError("The default upload limit must be between 1 MB and the maximum.")
        self.default_mb = default_mb
        self.max_mb = max_mb
        self._path = Path(store_path) if store_path else None
        self._selected: int | None = None  # used when there is no file to save to
        self._lock = threading.Lock()

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> UploadLimit:
        settings = settings or get_settings()
        return cls(
            default_mb=settings.max_upload_size_mb,
            max_mb=settings.max_upload_size_cap_mb,
            store_path=settings.upload_limit_file,
        )

    def current_mb(self) -> int:
        """The limit in MB; the default if none is saved or the saved one is out of range."""
        saved = self._read()
        return saved if saved is not None and 1 <= saved <= self.max_mb else self.default_mb

    def max_bytes(self) -> int:
        return self.current_mb() * _MB

    def set_mb(self, mb: int) -> int:
        if not 1 <= mb <= self.max_mb:
            raise ValueError(f"The upload limit must be between 1 and {self.max_mb} MB.")
        with self._lock:
            self._selected = mb
            if self._path is not None:
                payload = {"max_mb": mb, "updated_at": datetime.now(timezone.utc).isoformat()}
                self._path.parent.mkdir(parents=True, exist_ok=True)
                tmp = self._path.with_name(self._path.name + ".tmp")
                tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                os.replace(tmp, self._path)  # readers never see a half-written file
        logger.info("Upload size limit changed to %d MB", mb)
        return mb

    def _read(self) -> int | None:
        if self._path is None:
            return self._selected
        if not self._path.exists():
            return None
        try:
            value = json.loads(self._path.read_text(encoding="utf-8"))["max_mb"]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            logger.error("Could not read the upload limit from %s: %s", self._path, exc)
            return None
        # bool is an int subclass; a hand-edited `true` must not become 1 MB.
        return value if isinstance(value, int) and not isinstance(value, bool) else None
