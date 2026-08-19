import argparse
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from pydantic import BaseModel, ConfigDict, Field

from app.rag.chunker import PolicyChunk, chunk_document
from app.services.document_ingestion import ingest_document

EMBEDDING_MODEL = "nomic-embed-text:latest"
COLLECTION_NAME = "claimassist_policies"
CHROMA_DIRECTORY = Path("data/chroma")
DEFAULT_BATCH_SIZE = 16
DEFAULT_TOP_K = 4
class RetrievedEvidence(BaseModel):
    """One policy passage returned by semantic retrieval."""

    model_config = ConfigDict(extra="forbid")

    rank: int = Field(ge=1)
    distance: float
    chunk_id: str
    source_name: str
    page_number: int = Field(ge=1)
    citation: str
    text: str = Field(min_length=1)

def create_embeddings() -> OllamaEmbeddings:
    """Create the local Ollama embedding interface."""

    return OllamaEmbeddings(
        model=EMBEDDING_MODEL,
    )

def create_vector_store(
    persist_directory: Path = CHROMA_DIRECTORY,
) -> Chroma:
    """Open or create the persistent local Chroma collection."""

    persist_directory.mkdir(
        parents=True,
        exist_ok=True,
    )
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=create_embeddings(),
        persist_directory=str(persist_directory),
    )

def chunk_to_document(chunk: PolicyChunk) -> Document:
    """Convert a validated policy chunk into a LangChain document."""

    return Document(
        page_content=chunk.text,
        metadata={
            "chunk_id": chunk.chunk_id,
            "source_name": chunk.source_name,
            "source_sha256": chunk.source_sha256,
            "page_number": chunk.page_number,
            "chunk_index": chunk.chunk_index,
            "citation": chunk.citation,
        },
    )

def index_chunks(
    chunks: list[PolicyChunk],
    *,
    vector_store: Chroma,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> int:
    """
    Embed and store chunks in small sequential batches.
    Small batches limit memory spikes and prevent several embedding
    jobs from being launched simultaneously.
    """

    if not chunks:
        raise ValueError("There are no policy chunks to index.")

    if batch_size < 1 or batch_size > 64:
        raise ValueError(
            "Embedding batch size must be between 1 and 64."
        )

    indexed_count = 0
    for batch_start in range(
        0,
        len(chunks),
        batch_size,
    ):
        batch = chunks[
            batch_start : batch_start + batch_size
        ]

        documents = [
            chunk_to_document(chunk)
            for chunk in batch
        ]

        ids = [
            chunk.chunk_id
            for chunk in batch
        ]

        print(
            f"Embedding chunks "
            f"{batch_start + 1}-"
            f"{batch_start + len(batch)} "
            f"of {len(chunks)}",
            flush=True,
        )

        vector_store.add_documents(
            documents=documents,
            ids=ids,
        )

        indexed_count += len(batch)

    return indexed_count

def index_policy_document(
    document_path: Path,
    *,
    persist_directory: Path = CHROMA_DIRECTORY,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> int:
    """Ingest, split, embed, and persist one policy document."""

    document = ingest_document(document_path)
    chunks = chunk_document(document)
    vector_store = create_vector_store(persist_directory)

    return index_chunks(
        chunks,
        vector_store=vector_store,
        batch_size=batch_size,
    )

def search_policy(
    query: str,
    *,
    persist_directory: Path = CHROMA_DIRECTORY,
    top_k: int = DEFAULT_TOP_K,
) -> list[RetrievedEvidence]:
    """Retrieve the most relevant policy passages."""

    cleaned_query = query.strip()

    if not cleaned_query:
        raise ValueError("The retrieval query cannot be empty.")

    if top_k < 1 or top_k > 10:
        raise ValueError("top_k must be between 1 and 10.")

    vector_store = create_vector_store(persist_directory)

    results = vector_store.similarity_search_with_score(
        cleaned_query,
        k=top_k,
    )

    evidence: list[RetrievedEvidence] = []

    for rank, (document, distance) in enumerate(
        results,
        start=1,
    ):
        metadata = document.metadata

        evidence.append(
            RetrievedEvidence(
                rank=rank,
                distance=float(distance),
                chunk_id=str(metadata["chunk_id"]),
                source_name=str(metadata["source_name"]),
                page_number=int(metadata["page_number"]),
                citation=str(metadata["citation"]),
                text=document.page_content,
            )
        )

    return evidence

def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ClaimAssist local policy vector store."
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    index_parser = subparsers.add_parser(
        "index",
        help="Index a policy document.",
    )

    index_parser.add_argument(
        "document",
        type=Path,
    )

    index_parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )

    search_parser = subparsers.add_parser(
        "search",
        help="Search indexed policy evidence.",
    )

    search_parser.add_argument(
        "query",
        type=str,
    )
    search_parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
    )

    return parser

def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.command == "index":
        indexed_count = index_policy_document(
            args.document,
            batch_size=args.batch_size,
        )

        print()
        print(f"Indexed {indexed_count} policy chunks.")
        print(f"Vector database: {CHROMA_DIRECTORY.resolve()}")
        return

    if args.command == "search":
        evidence = search_policy(
            args.query,
            top_k=args.top_k,
        )

        if not evidence:
            print("No policy evidence was found.")
            return

        for item in evidence:
            print()
            print(
                f"Result {item.rank} | "
                f"{item.citation} | "
                f"distance={item.distance:.4f}"
            )
            print(item.text)

if __name__ == "__main__":
    main()