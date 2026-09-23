"""Tests for document loaders and the text splitter (FR-01 – FR-03, FR-10)."""

import zipfile

import pytest

from src.data_pipeline.document_loaders import (
    DocumentLoaderFactory,
    DocumentLoadError,
    DocxLoader,
    PdfLoader,
    TextFileLoader,
    UnsupportedFileTypeError,
)
from src.data_pipeline.text_splitter import DocumentSplitter
from tests.conftest import make_settings

# --- Helpers: build small test files without extra libraries ------------------


def write_docx(path, text):
    """Write a minimal .docx file (a zip containing word/document.xml)."""
    xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body><w:p><w:r><w:t>{text}</w:t></w:r></w:p></w:body></w:document>"
    )
    with zipfile.ZipFile(path, "w") as docx:
        docx.writestr("word/document.xml", xml)


def write_pdf(path, pages):
    """Write a minimal PDF with one line of text per page."""
    objects = ["<< /Type /Catalog /Pages 2 0 R >>"]
    kids = " ".join(f"{4 + i * 2} 0 R" for i in range(len(pages)))
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>")
    objects.append("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    for i, text in enumerate(pages):
        content_id = 5 + i * 2
        objects.append(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_id} 0 R >>"
        )
        stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET"
        objects.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")

    out = b"%PDF-1.4\n"
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n{body}\nendobj\n".encode("latin-1")
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += "".join(f"{offset:010d} 00000 n \n" for offset in offsets).encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode()
    path.write_bytes(out)


# --- Loader factory ------------------------------------------------------------


@pytest.mark.parametrize(
    ("filename", "loader_class"),
    [
        ("a.pdf", PdfLoader),
        ("a.docx", DocxLoader),
        ("a.txt", TextFileLoader),
        ("a.md", TextFileLoader),
        ("A.PDF", PdfLoader),
    ],
)
def test_factory_selects_loader_by_extension(filename, loader_class):
    assert isinstance(DocumentLoaderFactory.get_loader(filename), loader_class)


@pytest.mark.parametrize("filename", ["a.exe", "a.xlsx", "no_extension"])
def test_factory_rejects_unsupported_types(filename):
    with pytest.raises(UnsupportedFileTypeError, match="Unsupported file type"):
        DocumentLoaderFactory.get_loader(filename)


def test_supported_extensions_match_settings():
    assert DocumentLoaderFactory.supported_extensions() == make_settings().allowed_extensions


# --- Individual loaders --------------------------------------------------------


def test_loads_text_file(tmp_path):
    path = tmp_path / "policy.txt"
    path.write_text("Working hours are 9 to 5.", encoding="utf-8")

    docs = DocumentLoaderFactory.load(path)

    assert len(docs) == 1
    assert docs[0].page_content == "Working hours are 9 to 5."
    assert docs[0].metadata["source"] == "policy.txt"


def test_loads_markdown_with_bom(tmp_path):
    path = tmp_path / "guide.md"
    path.write_text("# Onboarding\nWelcome!", encoding="utf-8-sig")

    docs = DocumentLoaderFactory.load(path)

    assert docs[0].page_content.startswith("# Onboarding")


def test_loads_windows_encoded_text(tmp_path):
    path = tmp_path / "legacy.txt"
    path.write_bytes("Café policy – updated".encode("cp1252"))

    assert DocumentLoaderFactory.load(path)[0].page_content == "Café policy – updated"


def test_loads_docx(tmp_path):
    path = tmp_path / "handbook.docx"
    write_docx(path, "Annual leave is 14 days.")

    docs = DocumentLoaderFactory.load(path)

    assert "Annual leave is 14 days." in docs[0].page_content
    assert docs[0].metadata["source"] == "handbook.docx"


def test_loads_pdf_one_document_per_page(tmp_path):
    path = tmp_path / "specs.pdf"
    write_pdf(path, ["Page one text", "Page two text"])

    docs = DocumentLoaderFactory.load(path)

    assert [d.metadata["page"] for d in docs] == [1, 2]
    assert "Page two text" in docs[1].page_content
    assert all(d.metadata["source"] == "specs.pdf" for d in docs)


def test_empty_file_is_rejected(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("   \n  ", encoding="utf-8")

    with pytest.raises(DocumentLoadError, match="no extractable text"):
        DocumentLoaderFactory.load(path)


def test_corrupt_file_is_rejected(tmp_path):
    path = tmp_path / "broken.docx"
    path.write_bytes(b"this is not a zip file")

    with pytest.raises(DocumentLoadError, match="Could not read 'broken.docx'"):
        DocumentLoaderFactory.load(path)


def test_missing_file_is_rejected(tmp_path):
    with pytest.raises(DocumentLoadError):
        DocumentLoaderFactory.load(tmp_path / "missing.txt")


# --- Text splitter -------------------------------------------------------------


def test_splitter_creates_overlapping_numbered_chunks(tmp_path):
    path = tmp_path / "long.txt"
    path.write_text(" ".join(f"word{i}" for i in range(300)), encoding="utf-8")
    splitter = DocumentSplitter(chunk_size=200, chunk_overlap=50)

    chunks = splitter.split(DocumentLoaderFactory.load(path))

    assert len(chunks) > 1
    assert all(len(c.page_content) <= 200 for c in chunks)
    assert [c.metadata["chunk_index"] for c in chunks] == list(range(len(chunks)))
    assert all(c.metadata["source"] == "long.txt" for c in chunks)
    # Overlap: the end of one chunk appears at the start of the next.
    assert chunks[0].page_content.split()[-1] in chunks[1].page_content


def test_splitter_keeps_page_metadata(tmp_path):
    path = tmp_path / "specs.pdf"
    write_pdf(path, ["First page", "Second page"])

    chunks = DocumentSplitter(chunk_size=500, chunk_overlap=0).split(
        DocumentLoaderFactory.load(path)
    )

    assert [c.metadata["page"] for c in chunks] == [1, 2]


def test_splitter_from_settings():
    splitter = DocumentSplitter.from_settings(make_settings(chunk_size=300, chunk_overlap=30))
    assert splitter._splitter._chunk_size == 300
    assert splitter._splitter._chunk_overlap == 30
