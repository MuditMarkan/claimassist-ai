import argparse
import hashlib
from pathlib import Path

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)
from pydantic import BaseModel, ConfigDict, Field

from app.services.document_ingestion import (
    IngestedDocument,
    ingest_document,
)

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150

class PolicyChunk(BaseModel):
    """One searchable policy chunk with citation metadata."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(min_length=64, max_length=64)
    source_name: str = Field(min_length=1)
    source_sha256: str = Field(min_length=64, max_length=64)
    page_number: int = Field(ge=1)
    chunk_index: int = Field(ge=1)
    citation: str = Field(min_length=1)
    text: str = Field(min_length=1)
    character_count: int = Field(ge=1)

def validate_chunk_settings(
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    """Reject unsafe or internally inconsistent settings."""

    if chunk_size < 100:
        raise ValueError(
            "Chunk size must be at least 100 characters."
        )
    if chunk_overlap < 0:
        raise ValueError(
            "Chunk overlap cannot be negative."
        )

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "Chunk overlap must be smaller than chunk size."
        )

def create_chunk_id(
    *,
    source_sha256: str,
    page_number: int,
    chunk_index: int,
    text: str,
) -> str:
    """Create a deterministic identifier for one chunk."""

    identity = (
        f"{source_sha256}:"
        f"{page_number}:"
        f"{chunk_index}:"
        f"{text}"
    )

    return hashlib.sha256(
        identity.encode("utf-8")
    ).hexdigest()
def chunk_document(
    document: IngestedDocument,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[PolicyChunk]:
    """
    Split each page separately so chunks never lose page citations.
    """

    validate_chunk_settings(
        chunk_size,
        chunk_overlap,
    )
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
        keep_separator=True,
    )
    chunks: list[PolicyChunk] = []

    for page in document.pages:
        if not page.text.strip():
            continue

        page_chunks = splitter.split_text(page.text)

        for chunk_index, chunk_text in enumerate(
            page_chunks,
            start=1,
        ):
            cleaned_chunk = chunk_text.strip()

            if not cleaned_chunk:
                continue

            chunk_id = create_chunk_id(
                source_sha256=document.sha256,
                page_number=page.page_number,
                chunk_index=chunk_index,
                text=cleaned_chunk,
            )

            chunks.append(
                PolicyChunk(
                    chunk_id=chunk_id,
                    source_name=document.source_name,
                    source_sha256=document.sha256,
                    page_number=page.page_number,
                    chunk_index=chunk_index,
                    citation=(
                        f"{document.source_name}, "
                        f"page {page.page_number}"
                    ),
                    text=cleaned_chunk,
                    character_count=len(cleaned_chunk),
                )
            )

    if not chunks:
        raise ValueError(
            "The document produced no searchable chunks."
        )
    return chunks

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split a policy into citation-preserving chunks."
    )

    parser.add_argument(
        "document",
        type=Path,
        help="Path to a supported policy document.",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
    )

    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
    )

    return parser.parse_args()

def main() -> None:
    args = parse_arguments()
    document = ingest_document(args.document)

    chunks = chunk_document(
        document,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    print(f"Source: {document.source_name}")
    print(f"Pages: {document.page_count}")
    print(f"Chunks: {len(chunks)}")
    print()

    for chunk in chunks:
        preview = chunk.text[:100].replace("\n", " ")
        print(
            f"{chunk.chunk_id[:12]} | "
            f"{chunk.citation} | "
            f"{chunk.character_count} characters"
        )
        print(f"  {preview}")

if __name__ == "__main__":
    main()
