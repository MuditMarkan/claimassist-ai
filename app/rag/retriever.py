import argparse
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from app.rag.vector_store import (
    CHROMA_DIRECTORY,
    RetrievedEvidence,
    search_policy,
)

MAX_CLAIM_QUERY_CHARACTERS = 4000
MAX_EVIDENCE_CONTEXT_CHARACTERS = 6000
DEFAULT_TOP_K = 4

class EvidenceBundle(BaseModel):
    """Policy evidence prepared for the extraction pipeline."""

    model_config = ConfigDict(extra="forbid")

    retrieval_query: str = Field(min_length=1)
    evidence: list[RetrievedEvidence]
    citations: list[str]
    formatted_context: str
    has_evidence: bool

def build_retrieval_query(claim_text: str) -> str:
    """
    Build a deterministic search query without asking an LLM.

    Claim text is used only for semantic retrieval. Instructions
    appearing inside it do not become system instructions.
    """

    cleaned_claim = claim_text.strip()

    if not cleaned_claim:
        raise ValueError("The claim text cannot be empty.")

    bounded_claim = cleaned_claim[
        :MAX_CLAIM_QUERY_CHARACTERS
    ]

    return (
        "Find insurance policy provisions relevant to this claim. "
        "Prioritize coverage limits, deductibles, exclusions, "
        "required documents, eligibility conditions, and claim "
        "procedures.\n\n"
        f"Claim information:\n{bounded_claim}"
    )

def prepare_evidence_context(
    evidence: list[RetrievedEvidence],
    *,
    max_characters: int = MAX_EVIDENCE_CONTEXT_CHARACTERS,
) -> tuple[
    list[RetrievedEvidence],
    list[str],
    str,
]:
    """Format retrieved evidence within a strict context limit."""

    if max_characters < 500:
        raise ValueError(
            "Evidence context limit must be at least 500 characters."
        )
    selected_evidence: list[RetrievedEvidence] = []
    citations: list[str] = []
    blocks: list[str] = []
    used_characters = 0

    for item in evidence:
        header = (
            f"[POLICY EVIDENCE {item.rank}]\n"
            f"Citation: {item.citation}\n"
            f"Chunk ID: {item.chunk_id}\n"
            "Policy text:\n"
        )

        separator_length = 2 if blocks else 0
        remaining = (
            max_characters
            - used_characters
            - separator_length
        )

        if remaining <= len(header):
            break

        available_text_length = remaining - len(header)
        bounded_text = item.text[:available_text_length].strip()

        if not bounded_text:
            break

        block = header + bounded_text
        blocks.append(block)
        selected_evidence.append(item)

        if item.citation not in citations:
            citations.append(item.citation)

        used_characters += len(block) + separator_length

        if len(bounded_text) < len(item.text):
            break

    return (
        selected_evidence,
        citations,
        "\n\n".join(blocks),
    )

def retrieve_claim_evidence(
    claim_text: str,
    *,
    persist_directory: Path = CHROMA_DIRECTORY,
    top_k: int = DEFAULT_TOP_K,
    max_context_characters: int = (
        MAX_EVIDENCE_CONTEXT_CHARACTERS
    ),
) -> EvidenceBundle:
    """Retrieve and format policy evidence for one claim."""

    retrieval_query = build_retrieval_query(claim_text)
    retrieved_evidence = search_policy(
        retrieval_query,
        persist_directory=persist_directory,
        top_k=top_k,
    )

    evidence, citations, formatted_context = (
        prepare_evidence_context(
            retrieved_evidence,
            max_characters=max_context_characters,
        )
    )

    return EvidenceBundle(
        retrieval_query=retrieval_query,
        evidence=evidence,
        citations=citations,
        formatted_context=formatted_context,
        has_evidence=bool(evidence),
    )

def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Retrieve policy evidence for a claim."
    )

    parser.add_argument(
        "claim",
        type=str,
        help="Claim text or a short claim description.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
    )

    return parser

def main() -> None:
    args = build_argument_parser().parse_args()

    bundle = retrieve_claim_evidence(
        args.claim,
        top_k=args.top_k,
    )

    if not bundle.has_evidence:
        print("No policy evidence was retrieved.")
        print("Decision route: REVIEW")
        return

    print("Retrieved citations:")

    for citation in bundle.citations:
        print(f"- {citation}")

    print()
    print("Evidence context:")
    print(bundle.formatted_context)

if __name__ == "__main__":
    main()