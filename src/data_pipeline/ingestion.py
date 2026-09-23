"""Ingestion service: validate, save, load, split and index documents (FR-05 – FR-10, NFR-03).

Each file is identified by the SHA-256 of its bytes (``doc_id``), so uploading the
same content twice is detected and skipped (FR-08). Uploads are saved to
``data/raw/`` as ``<doc_id prefix>_<file name>`` so different files with the same
name never overwrite each other.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from src.config import Settings, get_settings
from src.data_pipeline.document_loaders import DocumentLoaderFactory, DocumentLoadError
from src.data_pipeline.text_splitter import DocumentSplitter
from src.data_pipeline.vector_store import (
    FILE_NAME,
    INGESTED_AT,
    SOURCE,
    VectorStoreRepository,
)

logger = logging.getLogger(__name__)

_DOC_ID_PREFIX_LENGTH = 16
_MAX_FILENAME_LENGTH = 150


class IngestionError(Exception):
    """A file was rejected. The message is safe to show to the user."""


class UnsupportedTypeError(IngestionError):
    """The file extension is not allowed."""


class FileTooLargeError(IngestionError):
    """The file exceeds ``MAX_UPLOAD_SIZE_MB``."""


class IngestionStatus(str, Enum):
    INGESTED = "ingested"
    DUPLICATE = "duplicate"


@dataclass(frozen=True)
class IngestionResult:
    doc_id: str
    source: str
    status: IngestionStatus
    chunk_count: int


def compute_doc_id(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sanitize_filename(filename: str) -> str:
    """Keep only the base name and replace characters that are unsafe in file names."""
    name = Path(filename.replace("\\", "/")).name.strip()
    name = re.sub(r"[^\w.\- ()]+", "_", name).strip(" .")
    if len(name) > _MAX_FILENAME_LENGTH:
        stem, suffix = Path(name).stem, Path(name).suffix
        name = stem[: _MAX_FILENAME_LENGTH - len(suffix)] + suffix
    return name


class IngestionService:
    """Turns uploaded files into indexed chunks in the vector store."""

    def __init__(
        self,
        repository: VectorStoreRepository,
        splitter: DocumentSplitter,
        raw_dir: Path,
        allowed_extensions: set[str],
        max_size_bytes: int,
    ) -> None:
        self._repository = repository
        self._splitter = splitter
        self._raw_dir = Path(raw_dir)
        self._allowed_extensions = allowed_extensions & DocumentLoaderFactory.supported_extensions()
        self._max_size_bytes = max_size_bytes

    @classmethod
    def from_settings(
        cls, repository: VectorStoreRepository, settings: Settings | None = None
    ) -> IngestionService:
        settings = settings or get_settings()
        return cls(
            repository=repository,
            splitter=DocumentSplitter.from_settings(settings),
            raw_dir=settings.raw_data_dir,
            allowed_extensions=settings.allowed_extensions,
            max_size_bytes=settings.max_upload_size_bytes,
        )

    @property
    def max_size_bytes(self) -> int:
        return self._max_size_bytes

    # --- Public API ------------------------------------------------------------

    def ingest_upload(self, filename: str, content: bytes) -> IngestionResult:
        """Validate an uploaded file, save it to the raw folder and index it."""
        source = self._validate(filename, content)
        doc_id = compute_doc_id(content)
        if self._repository.has_document(doc_id):
            return self._duplicate(doc_id, source)

        self._raw_dir.mkdir(parents=True, exist_ok=True)
        path = self._raw_dir / f"{doc_id[:_DOC_ID_PREFIX_LENGTH]}_{source}"
        path.write_bytes(content)
        try:
            return self._index(path, doc_id, source)
        except Exception:
            path.unlink(missing_ok=True)
            raise

    def ingest_directory(self) -> tuple[list[IngestionResult], dict[str, str]]:
        """Index every supported file already in the raw folder (FR-06).

        Returns the results and a ``{file name: error}`` map for files that failed;
        one bad file never stops the others (FR-10).
        """
        results: list[IngestionResult] = []
        errors: dict[str, str] = {}
        if not self._raw_dir.is_dir():
            return results, errors

        for path in sorted(p for p in self._raw_dir.iterdir() if p.is_file()):
            if path.name.startswith("."):
                continue
            try:
                content = path.read_bytes()
                source = self._validate(self._display_name(path), content)
                doc_id = compute_doc_id(content)
                if self._repository.has_document(doc_id):
                    results.append(self._duplicate(doc_id, source))
                else:
                    results.append(self._index(path, doc_id, source))
            except (IngestionError, DocumentLoadError, OSError) as exc:
                logger.warning("Skipped %s: %s", path.name, exc)
                errors[path.name] = str(exc)
        return results, errors

    def delete(self, doc_id: str) -> bool:
        """Remove a document's chunks and its saved file. Returns False if it was unknown."""
        document = self._repository.get_document(doc_id)
        if document is None:
            return False
        self._repository.delete_document(doc_id)
        if document.file_name:
            (self._raw_dir / Path(document.file_name).name).unlink(missing_ok=True)
        return True

    # --- Helpers -----------------------------------------------------------------

    def _validate(self, filename: str, content: bytes) -> str:
        source = sanitize_filename(filename)
        extension = Path(source).suffix.lower()
        if not source or extension not in self._allowed_extensions:
            allowed = ", ".join(sorted(self._allowed_extensions))
            raise UnsupportedTypeError(
                f"'{filename}' is not a supported file type. Allowed types: {allowed}."
            )
        if len(content) > self._max_size_bytes:
            limit_mb = self._max_size_bytes / (1024 * 1024)
            raise FileTooLargeError(f"'{source}' is larger than the {limit_mb:g} MB limit.")
        if not content:
            raise IngestionError(f"'{source}' is empty.")
        return source

    def _index(self, path: Path, doc_id: str, source: str) -> IngestionResult:
        try:
            pages = DocumentLoaderFactory.load(path)
        except DocumentLoadError as exc:
            raise IngestionError(str(exc).replace(path.name, source)) from exc

        ingested_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        for page in pages:
            page.metadata.update({SOURCE: source, FILE_NAME: path.name, INGESTED_AT: ingested_at})
        chunks = self._splitter.split(pages)
        count = self._repository.add_document(doc_id, chunks)
        logger.info("Ingested %s (%d chunks)", source, count)
        return IngestionResult(doc_id, source, IngestionStatus.INGESTED, count)

    def _duplicate(self, doc_id: str, source: str) -> IngestionResult:
        existing = self._repository.get_document(doc_id)
        logger.info("Skipped duplicate %s (already indexed)", source)
        return IngestionResult(
            doc_id=doc_id,
            source=existing.source if existing else source,
            status=IngestionStatus.DUPLICATE,
            chunk_count=existing.chunk_count if existing else 0,
        )

    @staticmethod
    def _display_name(path: Path) -> str:
        """Strip the ``<doc_id prefix>_`` that ``ingest_upload`` adds to saved files."""
        prefix, sep, rest = path.name.partition("_")
        is_prefix = len(prefix) == _DOC_ID_PREFIX_LENGTH and all(
            c in "0123456789abcdef" for c in prefix
        )
        return rest if sep and is_prefix and rest else path.name
