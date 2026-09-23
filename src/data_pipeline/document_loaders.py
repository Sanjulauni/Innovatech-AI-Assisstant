"""Document loaders for the supported file types (FR-01, FR-02, FR-10).

``DocumentLoaderFactory`` picks the right loader from the file extension. To support
a new format, subclass ``BaseDocumentLoader`` and register it in ``_LOADERS``;
no other code changes (Open/Closed Principle).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from langchain_core.documents import Document


class DocumentLoadError(Exception):
    """Raised when a file cannot be read or contains no text."""


class UnsupportedFileTypeError(DocumentLoadError):
    """Raised when no loader is registered for a file extension."""


class BaseDocumentLoader(ABC):
    """Reads one file and returns its text as LangChain ``Document`` objects."""

    def load(self, path: Path) -> list[Document]:
        path = Path(path)
        try:
            documents = self._load(path)
        except DocumentLoadError:
            raise
        except Exception as exc:
            raise DocumentLoadError(f"Could not read '{path.name}': {exc}") from exc

        documents = [doc for doc in documents if doc.page_content.strip()]
        if not documents:
            raise DocumentLoadError(f"'{path.name}' contains no extractable text.")
        return documents

    @abstractmethod
    def _load(self, path: Path) -> list[Document]:
        """Read ``path`` and return its content. May raise any exception."""


class PdfLoader(BaseDocumentLoader):
    """Loads a PDF as one ``Document`` per page, with 1-based page numbers."""

    def _load(self, path: Path) -> list[Document]:
        from pypdf import PdfReader

        reader = PdfReader(path)
        return [
            Document(
                page_content=page.extract_text() or "",
                metadata={"source": path.name, "page": number},
            )
            for number, page in enumerate(reader.pages, start=1)
        ]


class DocxLoader(BaseDocumentLoader):
    """Loads a Word (.docx) document as a single ``Document``."""

    def _load(self, path: Path) -> list[Document]:
        import docx2txt

        text = docx2txt.process(str(path)) or ""
        return [Document(page_content=text, metadata={"source": path.name})]


class TextFileLoader(BaseDocumentLoader):
    """Loads plain text and Markdown files."""

    # utf-8-sig also handles files saved "UTF-8 with BOM" by Windows editors.
    ENCODINGS = ("utf-8-sig", "cp1252")

    def _load(self, path: Path) -> list[Document]:
        raw = path.read_bytes()
        for encoding in self.ENCODINGS:
            try:
                text = raw.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise DocumentLoadError(f"'{path.name}' is not a readable text file.")
        return [Document(page_content=text, metadata={"source": path.name})]


class DocumentLoaderFactory:
    """Selects the loader for a file based on its extension (Factory pattern)."""

    _LOADERS: dict[str, type[BaseDocumentLoader]] = {
        ".pdf": PdfLoader,
        ".docx": DocxLoader,
        ".txt": TextFileLoader,
        ".md": TextFileLoader,
    }

    @classmethod
    def supported_extensions(cls) -> set[str]:
        return set(cls._LOADERS)

    @classmethod
    def get_loader(cls, path: str | Path) -> BaseDocumentLoader:
        extension = Path(path).suffix.lower()
        loader_class = cls._LOADERS.get(extension)
        if loader_class is None:
            supported = ", ".join(sorted(cls._LOADERS))
            raise UnsupportedFileTypeError(
                f"Unsupported file type '{extension or '(none)'}'. Supported: {supported}."
            )
        return loader_class()

    @classmethod
    def load(cls, path: str | Path) -> list[Document]:
        """Load a file with the appropriate loader."""
        return cls.get_loader(path).load(Path(path))
