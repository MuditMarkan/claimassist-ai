from pathlib import Path
from typing import Any

import pytest
from langchain_core.documents import Document

import app.rag.vector_store as vector_store_module
from app.rag.chunker import PolicyChunk
from app.rag.vector_store import (
    chunk_to_document,
    index_chunks,
    search_policy,
)

class FakeVectorStore:
    """Small test replacement for Chroma."""

    def __init__(self) -> None:
        self.added_batches: list[dict[str, Any]] = []
        self.search_results: list[
            tuple[Document, float]
        ] = []
        self.last_query: str | None = None
        self.last_k: int | None = None

    def add_documents(
        self,
        *,
        documents: list[Document],
        ids: list[str],
    ) -> None:
        self.added_batches.append(
            {
                "documents": documents,
                "ids": ids,
            }
        )
    def similarity_search_with_score(
        self,
        query: str,
        *,
        k: int,
    ) -> list[tuple[Document, float]]:
        self.last_query = query
        self.last_k = k
        return self.search_results[:k]

def make_chunk(index: int) -> PolicyChunk:
    text = (
        f"Policy chunk {index}: "
        "A repair estimate is required."
    )

    return PolicyChunk(
        chunk_id=f"{index:064x}",
        source_name="collision-policy.pdf",
        source_sha256="a" * 64,
        page_number=index,
        chunk_index=1,
        citation=(
            f"collision-policy.pdf, page {index}"
        ),
        text=text,
        character_count=len(text),
    )
def test_chunk_is_converted_to_langchain_document() -> None:
    chunk = make_chunk(1)

    document = chunk_to_document(chunk)

    assert document.page_content == chunk.text
    assert document.metadata == {
        "chunk_id": chunk.chunk_id,
        "source_name": "collision-policy.pdf",
        "source_sha256": "a" * 64,
        "page_number": 1,
        "chunk_index": 1,
        "citation": "collision-policy.pdf, page 1",
    }
def test_chunks_are_indexed_in_small_batches() -> None:
    fake_store = FakeVectorStore()
    chunks = [
        make_chunk(index)
        for index in range(1, 6)
    ]

    indexed_count = index_chunks(
        chunks,
        vector_store=fake_store,
        batch_size=2,
    )

    assert indexed_count == 5
    assert len(fake_store.added_batches) == 3
    assert [
        len(batch["documents"])
        for batch in fake_store.added_batches
    ] == [2, 2, 1]

    stored_ids = [
        chunk_id
        for batch in fake_store.added_batches
        for chunk_id in batch["ids"]
    ]

    assert stored_ids == [
        chunk.chunk_id
        for chunk in chunks
    ]

def test_empty_chunk_list_is_rejected() -> None:
    fake_store = FakeVectorStore()

    with pytest.raises(
        ValueError,
        match="no policy chunks",
    ):
        index_chunks(
            [],
            vector_store=fake_store,
        )

@pytest.mark.parametrize(
    "batch_size",
    [0, 65],
)
def test_invalid_embedding_batch_size_is_rejected(
    batch_size: int,
) -> None:
    fake_store = FakeVectorStore()

    with pytest.raises(
        ValueError,
        match="between 1 and 64",
    ):
        index_chunks(
            [make_chunk(1)],
            vector_store=fake_store,
            batch_size=batch_size,
        )

def test_search_returns_ranked_cited_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_store = FakeVectorStore()

    fake_store.search_results = [
        (
            Document(
                page_content=(
                    "A police report and repair estimate "
                    "are required."
                ),
                metadata={
                    "chunk_id": "1" * 64,
                    "source_name": "collision-policy.pdf",
                    "source_sha256": "a" * 64,
                    "page_number": 3,
                    "chunk_index": 1,
                    "citation": (
                        "collision-policy.pdf, page 3"
                    ),
                },
            ),
            0.123,
        )
    ]
    monkeypatch.setattr(
        vector_store_module,
        "create_vector_store",
        lambda persist_directory: fake_store,
    )

    evidence = search_policy(
        "What documents are required?",
        persist_directory=tmp_path / "chroma",
        top_k=3,
    )

    assert fake_store.last_query == (
        "What documents are required?"
    )
    assert fake_store.last_k == 3

    assert len(evidence) == 1
    assert evidence[0].rank == 1
    assert evidence[0].distance == 0.123
    assert evidence[0].page_number == 3
    assert evidence[0].citation == (
        "collision-policy.pdf, page 3"
    )
    assert "repair estimate" in evidence[0].text

def test_empty_retrieval_query_is_rejected(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        search_policy(
            "   ",
            persist_directory=tmp_path,
        )

@pytest.mark.parametrize(
    "top_k",
    [0, 11],
)
def test_invalid_top_k_is_rejected(
    top_k: int,
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="between 1 and 10",
    ):
        search_policy(
            "collision coverage",
            persist_directory=tmp_path,
            top_k=top_k,
        )