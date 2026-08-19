from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError

from app.business_rules import (
    ClaimData,
    RuleDecision,
    evaluate_claim,
)
from app.claim_analyzer import extract_grounded_claim_data
from app.rag.retriever import retrieve_claim_evidence
from app.rag.vector_store import CHROMA_DIRECTORY
from app.services.evidence_validator import (
    validate_grounded_extraction,
)

class RagPipelineResult(BaseModel):
    """Complete result from the evidence-grounded pipeline."""

    model_config = ConfigDict(extra="forbid")

    claim: ClaimData | None
    decision: RuleDecision
    citations: list[str]
    issues: list[str]
    retrieved_evidence_count: int

def create_review_result(
    *,
    reasons: list[str],
    citations: list[str] | None = None,
    retrieved_evidence_count: int = 0,
) -> RagPipelineResult:
    """Create a safe human-review result."""

    cleaned_reasons = list(
        dict.fromkeys(
            reason.strip()
            for reason in reasons
            if reason.strip()
        )
    )

    if not cleaned_reasons:
        cleaned_reasons = [
            "The claim requires human review."
        ]

    decision = RuleDecision(
        decision="REVIEW",
        reasons=cleaned_reasons,
        missing_documents=[],
    )

    return RagPipelineResult(
        claim=None,
        decision=decision,
        citations=citations or [],
        issues=cleaned_reasons,
        retrieved_evidence_count=(
            retrieved_evidence_count
        ),
    )

def run_grounded_claim_pipeline(
    claim_text: str,
    *,
    persist_directory: Path = CHROMA_DIRECTORY,
    top_k: int = 4,
) -> RagPipelineResult:
    """
    Run the evidence-grounded ClaimAssist pipeline.

    Flow:
    1. Retrieve policy evidence.
    2. Extract nullable, source-linked facts.
    3. Validate evidence independently.
    4. Apply deterministic business rules.
    """

    if not claim_text.strip():
        return create_review_result(
            reasons=["The claim document is empty."]
        )

    print(
        "Step 1/4: Retrieving policy evidence...",
        flush=True,
    )

    try:
        evidence_bundle = retrieve_claim_evidence(
            claim_text,
            persist_directory=persist_directory,
            top_k=top_k,
        )
    except Exception as error:
        print(
            "Policy retrieval failed: "
            f"{type(error).__name__}: {error}",
            flush=True,
        )

        return create_review_result(
            reasons=[
                "Policy retrieval failed and requires "
                "human review."
            ]
        )

    evidence_count = len(evidence_bundle.evidence)
    if not evidence_bundle.has_evidence:
        return create_review_result(
            reasons=[
                "No relevant policy evidence was retrieved."
            ],
            retrieved_evidence_count=0,
        )

    print(
        f"Retrieved {evidence_count} evidence passages.",
        flush=True,
    )

    print(
        "Step 2/4: Extracting grounded claim facts...",
        flush=True,
    )

    try:
        extraction = extract_grounded_claim_data(
            claim_text,
            evidence_bundle.formatted_context,
        )
    except ValidationError as error:
        print(
            f"Extraction validation failed: {error}",
            flush=True,
        )

        return create_review_result(
            reasons=[
                "The grounded model response failed "
                "structural validation."
            ],
            citations=evidence_bundle.citations,
            retrieved_evidence_count=evidence_count,
        )
    except ValueError as error:
        return create_review_result(
            reasons=[str(error)],
            citations=evidence_bundle.citations,
            retrieved_evidence_count=evidence_count,
        )
    except Exception as error:
        print(
            "Grounded extraction failed: "
            f"{type(error).__name__}: {error}",
            flush=True,
        )

        return create_review_result(
            reasons=[
                "Grounded extraction failed and requires "
                "human review."
            ],
            citations=evidence_bundle.citations,
            retrieved_evidence_count=evidence_count,
        )

    print(
        "Step 3/4: Verifying evidence references...",
        flush=True,
    )

    validation = validate_grounded_extraction(
        extraction,
        evidence_bundle,
    )

    if not validation.is_valid or validation.claim is None:
        return create_review_result(
            reasons=validation.issues,
            citations=validation.citations,
            retrieved_evidence_count=evidence_count,
        )
    print(
        "Step 4/4: Applying deterministic rules...",
        flush=True,
    )

    decision = evaluate_claim(validation.claim)

    return RagPipelineResult(
        claim=validation.claim,
        decision=decision,
        citations=validation.citations,
        issues=[],
        retrieved_evidence_count=evidence_count,
    )
def main() -> None:
    claim_text = """
Claim ID: CLM-RAG-001
The customer requests $12,000 for collision repairs.

Submitted documents:
- Claim form
- Police report
"""

    result = run_grounded_claim_pipeline(claim_text)

    print()
    print("ClaimAssist grounded result:")
    print(result.model_dump_json(indent=2))

if __name__ == "__main__":
    main()