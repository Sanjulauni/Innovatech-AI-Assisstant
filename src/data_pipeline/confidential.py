"""Which documents the admin marked as confidential (e.g. SOPs).

Questions that draw on a confidential document are answered only by the local model,
never by a cloud model (see ``RAGChain``). The marks are kept in a small JSON file
(``CONFIDENTIAL_FILE``), keyed by ``doc_id`` (the SHA-256 of the file), not in the
vector store, so re-indexing or changing the embedding model never loses them.

Because this protects private data, an unreadable file stops the API from starting
instead of silently treating every document as non-confidential.
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


class ConfidentialStore:
    """The set of confidential document IDs, saved to JSON on every change."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = Path(path) if path else None
        self._lock = threading.Lock()
        self._ids: set[str] = self._load()

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> ConfidentialStore:
        settings = settings or get_settings()
        return cls(settings.confidential_file)

    def ids(self) -> frozenset[str]:
        with self._lock:
            return frozenset(self._ids)

    def is_confidential(self, doc_id: str) -> bool:
        with self._lock:
            return doc_id in self._ids

    def set(self, doc_id: str, confidential: bool) -> None:
        with self._lock:
            if (doc_id in self._ids) == confidential:
                return
            if confidential:
                self._ids.add(doc_id)
            else:
                self._ids.discard(doc_id)
            self._save()
        logger.info(
            "Document %s marked as %s", doc_id[:12], "confidential" if confidential else "normal"
        )

    def _load(self) -> set[str]:
        if self._path is None or not self._path.exists():
            return set()
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            ids = data["doc_ids"]
            if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
                raise TypeError("doc_ids must be a list of strings")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ValueError(
                f"Could not read the confidential documents list at {self._path}: {exc}. "
                "Fix or restore the file; the API won't start without it, so that "
                "confidential documents are never sent to a cloud model by mistake."
            ) from exc
        return set(ids)

    def _save(self) -> None:
        if self._path is None:
            return
        payload = {
            "doc_ids": sorted(self._ids),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_name(self._path.name + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, self._path)  # readers never see a half-written file
