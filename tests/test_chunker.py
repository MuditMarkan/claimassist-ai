from typing import Any

import pytest

from app.rag.chunker import chunk_document
from app.services.document_ingestion import (
    DocumentPage,
    IngestedDocument,
)

def make_document(
    page_texts: list[str],
    **overrides: Any,
) -> IngestedDocument:
    pages = [
        DocumentPage(
            page_number=index,
            text=text,
        )
        for index, text in enumerate(
            page_texts,
            start=1,
        )
    ]

    document_data = {
        "source_name": "test-policy.txt",
        "media_type": "text/plain",
        "sha256": "a" * 64,
        "file_size": 1000,
        "page_count": len(pages),
        "total_characters": sum(
            len(page.text)
            for page in pages
        ),
        "pages": pages,
    }
    document_data.update(overrides)
    return IngestedDocument(**document_data)

def test_small_document_creates_one_cited_chunk() -> None:
    document = make_document(
        [
            "Collision coverage has a maximum limit "
            "of $10,000."
        ]
    )

    chunks = chunk_document(document)

    assert len(chunks) == 1
    chunk = chunks[0]

    assert chunk.source_name == "test-policy.txt"
    assert chunk.source_sha256 == "a" * 64
    assert chunk.page_number == 1
    assert chunk.chunk_index == 1
    assert chunk.citation == "test-policy.txt, page 1"
    assert chunk.text == document.pages[0].text
    assert len(chunk.chunk_id) == 64

def test_long_document_creates_multiple_bounded_chunks() -> None:
    long_text = " ".join(
        f"policyword{index}"
        for index in range(200)
    )

    document = make_document([long_text])

    chunks = chunk_document(
        document,
        chunk_size=200,
        chunk_overlap=40,
    )

    assert len(chunks) > 1
    assert all(
        chunk.character_count <= 200
        for chunk in chunks
    )
def test_adjacent_chunks_contain_overlap() -> None:
    long_text = " ".join(
        f"word{index}"
        for index in range(100)
    )

    document = make_document([long_text])

    chunks = chunk_document(
        document,
        chunk_size=150,
        chunk_overlap=40,
    )
    assert len(chunks) > 1

    first_words = set(chunks[0].text.split())
    second_words = set(chunks[1].text.split())

    assert first_words.intersection(second_words)

def test_chunk_ids_are_deterministic() -> None:
    document = make_document(
        [
            "A repair estimate is required. "
            "A police report is also required."
        ]
    )
    first_run = chunk_document(document)
    second_run = chunk_document(document)

    assert [
        chunk.chunk_id
        for chunk in first_run
    ] == [
        chunk.chunk_id
        for chunk in second_run
    ]

def test_chunks_never_cross_page_boundaries() -> None:
    document = make_document(
        [
            "Page one discusses collision coverage.",
            "Page two discusses required documents.",
        ],
        media_type="application/pdf",
        source_name="policy.pdf",
    )

    chunks = chunk_document(document)

    assert len(chunks) == 2

    assert chunks[0].page_number == 1
    assert chunks[0].citation == "policy.pdf, page 1"
    assert "Page two" not in chunks[0].text

    assert chunks[1].page_number == 2
    assert chunks[1].citation == "policy.pdf, page 2"
    assert "Page one" not in chunks[1].text

@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap"),
    [
        (99, 10),
        (100, -1),
        (100, 100),
    ],
)
def test_invalid_chunk_settings_are_rejected(
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    document = make_document(["Synthetic policy text."])

    with pytest.raises(ValueError):
        chunk_document(
            document,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

def test_document_without_text_produces_no_chunks() -> None:
    document = IngestedDocument(
        source_name="blank.pdf",
        media_type="application/pdf",
        sha256="b" * 64,
        file_size=100,
        page_count=1,
        total_characters=1,
        pages=[
            DocumentPage(
                page_number=1,
                text="",
            )
        ],
    )

    with pytest.raises(
        ValueError,
        match="no searchable chunks",
    ):
        chunk_document(document)