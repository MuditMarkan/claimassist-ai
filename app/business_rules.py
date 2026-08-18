from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ClaimData(BaseModel):
    """Validated claim information extracted from a document."""

    model_config = ConfigDict(extra="forbid")

    claim_id: str
    claim_amount: float = Field(ge=0)
    policy_limit: float = Field(ge=0)
    required_documents: list[str]
    submitted_documents: list[str]


class RuleDecision(BaseModel):
    """Final result produced by deterministic Python rules."""

    model_config = ConfigDict(extra="forbid")

    decision: Literal["APPROVE", "PEND", "REVIEW"]
    reasons: list[str]
    missing_documents: list[str]


def normalize_document_name(document: str) -> str:
    """
    Normalize a document name so comparisons ignore:
    - leading/trailing whitespace
    - multiple spaces
    - capitalization
    """

    return " ".join(document.split()).casefold()


def evaluate_claim(claim: ClaimData) -> RuleDecision:
    """
    Evaluate a validated claim using deterministic business rules.

    Rules:
    1. Claim amount cannot exceed the policy limit.
    2. All required documents must be submitted.
    3. If any rule fails, PEND the claim.
    4. If no rules fail, APPROVE the claim.
    """

    # Normalize submitted document names so that:
    #
    # "Police Report"
    # "police report"
    # "  POLICE   REPORT  "
    #
    # are treated as the same document.
    submitted_documents = {
        normalize_document_name(document)
        for document in claim.submitted_documents
    }

    # Find required documents that were not submitted.
    missing_documents = [
        document
        for document in claim.required_documents
        if normalize_document_name(document) not in submitted_documents
    ]

    reasons: list[str] = []

    # Rule 1:
    # Claim amount cannot exceed policy limit.
    if claim.claim_amount > claim.policy_limit:
        reasons.append(
            f"Claim amount ${claim.claim_amount:,.2f} exceeds "
            f"the policy limit of ${claim.policy_limit:,.2f}."
        )

    # Rule 2:
    # All required documents must be submitted.
    if missing_documents:
        reasons.append(
            "Missing required documents: "
            + ", ".join(missing_documents)
        )

    # If there are any rule failures, pend the claim.
    # Otherwise approve it.
    decision = "PEND" if reasons else "APPROVE"

    return RuleDecision(
        decision=decision,
        reasons=reasons,
        missing_documents=missing_documents,
    )


def main() -> None:
    """Run one synthetic ClaimAssist example."""

    claim = ClaimData(
        claim_id="CLM-001",
        claim_amount=12000,
        policy_limit=10000,
        required_documents=[
            "Claim form",
            "Police report",
            "Repair estimate",
        ],
        submitted_documents=[
            "claim form",
            "Police report",
        ],
    )

    result = evaluate_claim(claim)

    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()