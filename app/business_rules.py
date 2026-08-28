"""
ClaimAssist business rules — data models and normalization helpers.

The actual comparison logic lives in comparison_engine.py.
This module holds the shared Pydantic schemas and utility functions
used by both the extraction layer and the comparison engine.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Shared outcome type
# ---------------------------------------------------------------------------

Outcome = Literal["LIKELY_COVERED", "LIKELY_NOT_COVERED", "REVIEW_REQUIRED"]


# ---------------------------------------------------------------------------
# Extracted claim facts (from claim document)
# ---------------------------------------------------------------------------

class ClaimFacts(BaseModel):
    """
    Structured facts extracted from a submitted claim document.
    All fields are nullable — the model must not invent missing data.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    claim_id:            str   | None = Field(default=None, min_length=1)
    claim_amount:        float | None = Field(default=None, ge=0)
    loss_date:           str   | None = None   # YYYY-MM-DD
    loss_description:    str   | None = None
    claimant_name:       str   | None = None
    driver_name:         str   | None = None
    vehicle_vin:         str   | None = None
    vehicle_make_model:  str   | None = None
    submitted_documents: list[str] = Field(default_factory=list)
    unsupported_fields:  list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Comparison result
# ---------------------------------------------------------------------------

class RuleCheck(BaseModel):
    """One deterministic rule check with its outcome."""

    model_config = ConfigDict(extra="forbid")

    rule_name:   str
    passed:      bool
    reason:      str
    citation:    str | None = None


class ComparisonResult(BaseModel):
    """
    Complete result of comparing a claim against a policy version.
    Produced entirely by deterministic Python rules — no LLM involved.
    """

    model_config = ConfigDict(extra="forbid")

    outcome:            Outcome
    checks:             list[RuleCheck] = Field(default_factory=list)
    reasons:            list[str]       = Field(default_factory=list)
    missing_documents:  list[str]       = Field(default_factory=list)
    passed_checks:      list[str]       = Field(default_factory=list)
    failed_checks:      list[str]       = Field(default_factory=list)
    citations:          list[str]       = Field(default_factory=list)

    # Financials (None until rules confirm coverage is possible)
    eligible_gross:     float | None = None
    deductible_applied: float | None = None
    estimated_net:      float | None = None


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def normalize_name(value: str) -> str:
    """Casefold and collapse whitespace for fuzzy name matching."""
    return " ".join(value.split()).casefold()


def normalize_document_name(doc: str) -> str:
    """Normalize a document name for submitted-vs-required comparison."""
    return " ".join(doc.split()).casefold()


def normalize_vin(vin: str) -> str:
    """Strip spaces and uppercase a VIN for comparison."""
    return vin.replace(" ", "").upper()


def names_match(a: str | None, b: str | None) -> bool:
    """True when both are non-empty and normalize to the same string."""
    if not a or not b:
        return False
    return normalize_name(a) == normalize_name(b)


def vins_match(a: str | None, b: str | None) -> bool:
    """True when both are non-empty and normalize to the same VIN."""
    if not a or not b:
        return False
    return normalize_vin(a) == normalize_vin(b)


# ---------------------------------------------------------------------------
# Legacy shim — keeps older pipeline/tests working during transition
# ---------------------------------------------------------------------------

class ClaimData(BaseModel):
    """Legacy model kept for backward compatibility with pipeline.py."""

    model_config = ConfigDict(extra="forbid")

    claim_id:            str
    claim_amount:        float = Field(ge=0)
    policy_limit:        float = Field(ge=0)
    required_documents:  list[str]
    submitted_documents: list[str]


class RuleDecision(BaseModel):
    """Legacy decision model kept for backward compatibility."""

    model_config = ConfigDict(extra="forbid")

    decision:           Literal["LIKELY_COVERED", "LIKELY_NOT_COVERED", "REVIEW_REQUIRED"]
    reasons:            list[str]
    missing_documents:  list[str]


def evaluate_claim(claim: ClaimData) -> RuleDecision:
    """
    Legacy two-rule evaluator kept for pipeline.py / evaluation harness.
    New code should use comparison_engine.compare_claim_to_policy().
    """
    submitted = {
        normalize_document_name(d) for d in claim.submitted_documents
    }
    missing = [
        d for d in claim.required_documents
        if normalize_document_name(d) not in submitted
    ]
    reasons: list[str] = []

    if claim.claim_amount > claim.policy_limit:
        reasons.append(
            f"Claim amount ${claim.claim_amount:,.2f} exceeds "
            f"the policy limit of ${claim.policy_limit:,.2f}."
        )
    if missing:
        reasons.append("Missing required documents: " + ", ".join(missing))

    outcome: Outcome = "LIKELY_NOT_COVERED" if reasons else "LIKELY_COVERED"
    return RuleDecision(
        decision=outcome,
        reasons=reasons,
        missing_documents=missing,
    )
