import hashlib
from pathlib import Path

import pytest
from pypdf import PdfWriter

import app.services.document_ingestion as ingestion
from app.services.document_ingestion import (
    DocumentIngestionError,
    ingest_document,
)

def test_valid_text_document_is_ingested(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "policy.txt"
    content = (
        "Collision Coverage\n"
        "The maximum coverage is $10,000.\n"
        "A police report is required."
    )
    file_path.write_text(
        content,
        encoding="utf-8",
    )

    document = ingest_document(file_path)

    assert document.source_name == "policy.txt"
    assert document.media_type == "text/plain"
    assert document.page_count == 1
    assert document.pages[0].page_number == 1
    assert document.pages[0].text == content
    assert document.total_characters == len(content)

    expected_hash = hashlib.sha256(
        content.encode("utf-8")
    ).hexdigest()

    assert document.sha256 == expected_hash

def test_markdown_document_is_supported(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "policy.md"
    file_path.write_text(
        "# Policy\n\nRepair estimates are required.",
        encoding="utf-8",
    )

    document = ingest_document(file_path)
    assert document.media_type == "text/markdown"
    assert document.pages[0].text.startswith("# Policy")

def test_missing_document_is_rejected(
    tmp_path: Path,
) -> None:
    missing_path = tmp_path / "missing.pdf"

    with pytest.raises(
        DocumentIngestionError,
        match="does not exist",
    ):
        ingest_document(missing_path)
def test_unsupported_extension_is_rejected(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "unsafe.exe"
    file_path.write_bytes(b"not an executable")

    with pytest.raises(
        DocumentIngestionError,
        match="Unsupported document extension",
    ):
        ingest_document(file_path)

def test_empty_document_is_rejected(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "empty.txt"
    file_path.write_bytes(b"")

    with pytest.raises(
        DocumentIngestionError,
        match="document is empty",
    ):
        ingest_document(file_path)

def test_file_size_limit_is_enforced(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "large.txt"
    file_path.write_text(
        "123456",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        ingestion,
        "MAX_FILE_SIZE_BYTES",
        5,
    )

    with pytest.raises(
        DocumentIngestionError,
        match="file-size limit",
    ):
        ingest_document(file_path)
def test_non_utf8_text_is_rejected(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "invalid.txt"
    file_path.write_bytes(b"\xff\xfe\x00\x80")

    with pytest.raises(
        DocumentIngestionError,
        match="UTF-8",
    ):
        ingest_document(file_path)

def test_pdf_page_limit_is_enforced(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "too-many-pages.pdf"

    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.add_blank_page(width=612, height=792)

    with file_path.open("wb") as output:
        writer.write(output)

    monkeypatch.setattr(
        ingestion,
        "MAX_PDF_PAGES",
        1,
    )

    with pytest.raises(
        DocumentIngestionError,
        match="page limit",
    ):
        ingest_document(file_path)

def test_pdf_without_extractable_text_requires_visual_lane(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "scanned-policy.pdf"

    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    with file_path.open("wb") as output:
        writer.write(output)

    with pytest.raises(
        DocumentIngestionError,
        match="ColQwen",
    ):
        ingest_document(file_path)