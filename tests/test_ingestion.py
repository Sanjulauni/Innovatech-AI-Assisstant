"""Tests for the ingestion service (FR-05 – FR-10, NFR-03)."""

import pytest

from src.data_pipeline.ingestion import (
    FileTooLargeError,
    IngestionError,
    IngestionService,
    IngestionStatus,
    UnsupportedTypeError,
    compute_doc_id,
    sanitize_filename,
)
from src.data_pipeline.text_splitter import DocumentSplitter
from tests.conftest import make_settings
from tests.test_loaders import write_docx, write_pdf


@pytest.fixture
def raw_dir(tmp_path):
    return tmp_path / "raw"


@pytest.fixture
def service(repository, raw_dir):
    return IngestionService(
        repository=repository,
        splitter=DocumentSplitter(chunk_size=200, chunk_overlap=20),
        raw_dir=raw_dir,
        allowed_extensions={".pdf", ".docx", ".txt", ".md"},
        max_size_bytes=1024,
    )


def test_ingest_upload_saves_and_indexes(service, repository, raw_dir):
    content = b"Annual leave is 14 days per year."

    result = service.ingest_upload("leave policy.txt", content)

    assert result.status is IngestionStatus.INGESTED
    assert result.doc_id == compute_doc_id(content)
    assert result.source == "leave policy.txt"
    assert result.chunk_count == 1

    saved = list(raw_dir.iterdir())
    assert len(saved) == 1
    assert saved[0].name.endswith("_leave policy.txt")
    assert saved[0].read_bytes() == content

    [stored] = repository.list_documents()
    assert stored.doc_id == result.doc_id
    assert stored.source == "leave policy.txt"
    assert stored.file_name == saved[0].name
    assert stored.ingested_at


def test_chunks_carry_citation_metadata(service, repository, tmp_path):
    pdf = tmp_path / "specs.pdf"
    write_pdf(pdf, ["Page one text", "Page two text"])

    service.ingest_upload("specs.pdf", pdf.read_bytes())

    [result] = repository.search("Page two text", k=1)
    metadata = result.document.metadata
    assert metadata["source"] == "specs.pdf"  # original name, not the saved name
    assert metadata["page"] == 2
    assert "chunk_index" in metadata


def test_ingest_docx(service, tmp_path):
    docx = tmp_path / "handbook.docx"
    write_docx(docx, "Remote work needs manager approval.")

    assert service.ingest_upload("handbook.docx", docx.read_bytes()).chunk_count == 1


def test_duplicate_upload_is_skipped(service, repository, raw_dir):
    first = service.ingest_upload("a.txt", b"same content")
    second = service.ingest_upload("renamed.txt", b"same content")

    assert second.status is IngestionStatus.DUPLICATE
    assert second.doc_id == first.doc_id
    assert second.source == "a.txt"
    assert second.chunk_count == 1
    assert len(list(raw_dir.iterdir())) == 1
    assert len(repository.list_documents()) == 1


def test_same_name_different_content_are_separate(service, repository):
    service.ingest_upload("policy.txt", b"version one")
    service.ingest_upload("policy.txt", b"version two")
    assert len(repository.list_documents()) == 2


@pytest.mark.parametrize("filename", ["virus.exe", "sheet.xlsx", "noextension", ""])
def test_rejects_unsupported_type(service, raw_dir, filename):
    with pytest.raises(UnsupportedTypeError, match="not a supported file type"):
        service.ingest_upload(filename, b"data")
    assert not raw_dir.exists()


def test_rejects_too_large_file(service, raw_dir):
    with pytest.raises(FileTooLargeError, match="larger than"):
        service.ingest_upload("big.txt", b"x" * 1025)
    assert not raw_dir.exists()


def test_rejects_empty_file(service):
    with pytest.raises(IngestionError, match="empty"):
        service.ingest_upload("empty.txt", b"")


def test_corrupt_file_is_rejected_and_not_kept(service, repository, raw_dir):
    with pytest.raises(IngestionError, match="Could not read 'broken.docx'"):
        service.ingest_upload("broken.docx", b"not a zip file")
    assert list(raw_dir.iterdir()) == []
    assert repository.list_documents() == []


def test_file_is_removed_if_indexing_fails(service, repository, raw_dir, monkeypatch):
    def fail(*_args):
        raise RuntimeError("embedding service down")

    monkeypatch.setattr(repository, "add_document", fail)
    with pytest.raises(RuntimeError):
        service.ingest_upload("a.txt", b"content")
    assert list(raw_dir.iterdir()) == []


def test_path_components_are_stripped_from_filename(service, raw_dir):
    result = service.ingest_upload("../../etc/evil.txt", b"content")
    assert result.source == "evil.txt"
    assert all(p.parent == raw_dir for p in raw_dir.iterdir())


@pytest.mark.parametrize(
    ("raw", "clean"),
    [
        ("report.pdf", "report.pdf"),
        ("C:\\Users\\me\\report.pdf", "report.pdf"),
        ("../secret.txt", "secret.txt"),
        ("we<ird>:name?.md", "we_ird_name_.md"),
        ("  spaced .txt ", "spaced .txt"),
    ],
)
def test_sanitize_filename(raw, clean):
    assert sanitize_filename(raw) == clean


def test_sanitize_filename_truncates_long_names():
    name = sanitize_filename("a" * 300 + ".pdf")
    assert len(name) <= 150
    assert name.endswith(".pdf")


def test_delete_removes_chunks_and_file(service, repository, raw_dir):
    result = service.ingest_upload("a.txt", b"content")

    assert service.delete(result.doc_id) is True
    assert repository.list_documents() == []
    assert list(raw_dir.iterdir()) == []
    assert service.delete(result.doc_id) is False


def test_ingest_directory_indexes_files_and_reports_errors(service, repository, raw_dir):
    raw_dir.mkdir()
    (raw_dir / "guide.md").write_text("# Guide\nWelcome aboard.", encoding="utf-8")
    (raw_dir / "notes.xyz").write_text("unsupported", encoding="utf-8")
    (raw_dir / "broken.docx").write_bytes(b"corrupt")
    (raw_dir / ".gitkeep").write_text("", encoding="utf-8")

    results, errors = service.ingest_directory()

    assert [r.source for r in results] == ["guide.md"]
    assert set(errors) == {"notes.xyz", "broken.docx"}
    assert repository.get_document(results[0].doc_id).file_name == "guide.md"


def test_ingest_directory_skips_already_uploaded_files(service, repository):
    uploaded = service.ingest_upload("policy.txt", b"uploaded content")

    results, errors = service.ingest_directory()

    assert errors == {}
    assert [(r.doc_id, r.status, r.source) for r in results] == [
        (uploaded.doc_id, IngestionStatus.DUPLICATE, "policy.txt")
    ]


def test_ingest_directory_without_folder(service):
    assert service.ingest_directory() == ([], {})


def test_from_settings(repository, tmp_path):
    settings = make_settings(raw_data_dir=tmp_path, max_upload_size_mb=3)
    service = IngestionService.from_settings(repository, settings)
    assert service.max_size_bytes == 3 * 1024 * 1024
