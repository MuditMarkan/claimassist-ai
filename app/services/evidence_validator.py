import math
import re

from pydantic import BaseModel, ConfigDict

from app.business_rules import ClaimData
from app.claim_analyzer import (
    EvidenceReference,
    GroundedClaimExtraction,
)
from app.rag.retriever import EvidenceBundle

class EvidenceValidationResult(BaseModel):
    """Result of deterministic evidence verification."""

    model_config = ConfigDict(extra="forbid")

    is_valid: bool
    claim: ClaimData | None
    issues: list[str]
    verified_references: list[EvidenceReference]
    citations: list[str]
def normalize_evidence_text(text: str) -> str:
    """Normalize text for deterministic comparison."""

    return " ".join(text.casefold().split())

def extract_numeric_values(text: str) -> list[float]:
    """Extract ordinary currency and numeric values from evidence."""

    matches = re.findall(
        r"-?\$?\s*([0-9][0-9,]*(?:\.[0-9]+)?)",
        text,
    )

    values: list[float] = []
    for match in matches:
        try:
            values.append(
                float(match.replace(",", ""))
            )
        except ValueError:
            continue

    return values

def unique_strings(values: list[str]) -> list[str]:
    """Preserve order while removing duplicate strings."""

    return list(dict.fromkeys(values))
def validate_grounded_extraction(
    extraction: GroundedClaimExtraction,
    evidence_bundle: EvidenceBundle,
) -> EvidenceValidationResult:
    """
    Verify that extracted policy facts are supported by evidence.

    No LLM is used here.
    """

    issues: list[str] = []
    verified_references: list[EvidenceReference] = []

    if not evidence_bundle.has_evidence:
        issues.append("No policy evidence was retrieved.")

    if extraction.contradictions:
        issues.append(
            "Contradictory information was detected: "
            + "; ".join(extraction.contradictions)
        )

    if extraction.unsupported_fields:
        issues.append(
            "Unsupported fields were reported: "
            + ", ".join(extraction.unsupported_fields)
        )

    facts = extraction.facts
    required_fact_values = {
        "claim_id": facts.claim_id,
        "claim_amount": facts.claim_amount,
        "policy_limit": facts.policy_limit,
        "required_documents": facts.required_documents,
        "submitted_documents": facts.submitted_documents,
    }

    missing_fields = [
        field_name
        for field_name, value in required_fact_values.items()
        if value is None
    ]
    if missing_fields:
        issues.append(
            "Required facts are unavailable: "
            + ", ".join(missing_fields)
        )

    evidence_by_chunk_id = {
        item.chunk_id: item
        for item in evidence_bundle.evidence
    }

    for reference in extraction.evidence_references:
        retrieved_item = evidence_by_chunk_id.get(
            reference.chunk_id
        )
        if retrieved_item is None:
            issues.append(
                "Evidence reference uses an unknown chunk ID: "
                f"{reference.chunk_id}"
            )
            continue

        if reference.citation != retrieved_item.citation:
            issues.append(
                "Evidence citation does not match its chunk: "
                f"{reference.citation}"
            )
            continue
        normalized_excerpt = normalize_evidence_text(
            reference.excerpt
        )

        normalized_source = normalize_evidence_text(
            retrieved_item.text
        )

        if normalized_excerpt not in normalized_source:
            issues.append(
                "Evidence excerpt was not found in the "
                f"retrieved chunk: {reference.chunk_id}"
            )
            continue
        verified_references.append(reference)

    policy_limit_references = [
        reference
        for reference in verified_references
        if reference.field_name == "policy_limit"
    ]

    required_document_references = [
        reference
        for reference in verified_references
        if reference.field_name == "required_documents"
    ]

    if not policy_limit_references:
        issues.append(
            "The policy limit has no verified evidence reference."
        )

    if not required_document_references:
        issues.append(
            "Required documents have no verified evidence reference."
        )

    if (
        facts.policy_limit is not None
        and policy_limit_references
    ):
        policy_limit_values = [
            value
            for reference in policy_limit_references
            for value in extract_numeric_values(
                reference.excerpt
            )
        ]

        limit_is_supported = any(
            math.isclose(
                value,
                facts.policy_limit,
                rel_tol=0,
                abs_tol=0.01,
            )
            for value in policy_limit_values
        )
        if not limit_is_supported:
            issues.append(
                "The extracted policy limit does not appear "
                "in its cited evidence."
            )

    if (
        facts.required_documents is not None
        and required_document_references
    ):
        combined_document_evidence = (
            normalize_evidence_text(
                " ".join(
                    reference.excerpt
                    for reference
                    in required_document_references
                )
            )
        )

        unsupported_documents = [
            document
            for document in facts.required_documents
            if normalize_evidence_text(document)
            not in combined_document_evidence
        ]

        if unsupported_documents:
            issues.append(
                "Required documents were not found in their "
                "cited evidence: "
                + ", ".join(unsupported_documents)
            )

    issues = unique_strings(issues)

    if issues:
        return EvidenceValidationResult(
            is_valid=False,
            claim=None,
            issues=issues,
            verified_references=verified_references,
            citations=unique_strings(
                [
                    reference.citation
                    for reference in verified_references
                ]
            ),
        )

    # Every value has been checked for None above.
    claim = ClaimData(
        claim_id=facts.claim_id,
        claim_amount=facts.claim_amount,
        policy_limit=facts.policy_limit,
        required_documents=facts.required_documents,
        submitted_documents=facts.submitted_documents,
    )
    return EvidenceValidationResult(
        is_valid=True,
        claim=claim,
        issues=[],
        verified_references=verified_references,
        citations=unique_strings(
            [
                reference.citation
                for reference in verified_references
            ]
        ),
    )