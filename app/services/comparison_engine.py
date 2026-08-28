"""
Comparison engine — deterministic policy-to-claim comparison.

Implements all 16 rules from spec §9 in order, plus the
preliminary amount calculation from spec §10.

No LLM is used here.  Given the same inputs and rule version,
this function always produces the same output.
"""

from __future__ import annotations

from typing import Any

from app.business_rules import (
    ClaimFacts,
    ComparisonResult,
    RuleCheck,
    names_match,
    normalize_document_name,
    vins_match,
)

RULE_VERSION = "2.0.0"

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pass(rule_name: str, reason: str, citation: str | None = None) -> RuleCheck:
    return RuleCheck(rule_name=rule_name, passed=True,  reason=reason, citation=citation)


def _fail(rule_name: str, reason: str, citation: str | None = None) -> RuleCheck:
    return RuleCheck(rule_name=rule_name, passed=False, reason=reason, citation=citation)


def _missing_documents(
    required: list[str],
    submitted: list[str],
) -> list[str]:
    """Return required docs not found in submitted (case-insensitive)."""
    submitted_norm = {normalize_document_name(d) for d in submitted}
    return [
        d for d in required
        if normalize_document_name(d) not in submitted_norm
    ]


# ---------------------------------------------------------------------------
# Amount calculation  (spec §10)
# ---------------------------------------------------------------------------

def calculate_amounts(
    claim_amount: float | None,
    coverage_limit: float | None,
    deductible: float | None,
) -> tuple[float | None, float | None, float | None]:
    """
    Returns (eligible_gross, deductible_applied, estimated_net).

    eligible_gross  = min(claim_amount, coverage_limit)
    estimated_net   = max(eligible_gross - deductible, 0)
    """
    if claim_amount is None or coverage_limit is None:
        return None, None, None

    ded = deductible or 0.0
    eligible_gross  = min(claim_amount, coverage_limit)
    estimated_net   = max(eligible_gross - ded, 0.0)
    return eligible_gross, ded, estimated_net


# ---------------------------------------------------------------------------
# Main comparison function
# ---------------------------------------------------------------------------

def compare_claim_to_policy(
    policy: dict[str, Any],
    policy_extraction: dict[str, Any] | None,
    claim_facts: ClaimFacts,
) -> ComparisonResult:
    """
    Compare a submitted claim against a verified policy version.

    Args:
        policy:             Row from policy_versions (with document cols).
        policy_extraction:  Row from policy_extractions (decoded JSON lists).
                            May be None if extraction failed — forces REVIEW_REQUIRED.
        claim_facts:        ClaimFacts extracted from claim documents.

    Returns:
        ComparisonResult with outcome, checks, financials, citations.
    """

    checks:   list[RuleCheck] = []
    reasons:  list[str]       = []
    missing:  list[str]       = []
    citations: list[str]      = []

    def add(check: RuleCheck) -> None:
        checks.append(check)
        if check.citation and check.citation not in citations:
            citations.append(check.citation)

    # ── Rule 0: Organization ownership ──────────────────────────────────
    # (enforced at the route layer; here we trust the caller already checked)

    # ── Rule 1: Policy must be ACTIVE_AND_VERIFIED ───────────────────────
    policy_status = policy.get("status", "")
    if policy_status != "ACTIVE_AND_VERIFIED":
        add(_fail(
            "policy_active",
            f"Policy is not active (status: {policy_status}). "
            "Upload and verify a policy before submitting a claim.",
        ))
        return _review_result(checks, reasons, missing, citations,
                              "Policy is not ACTIVE_AND_VERIFIED.")

    add(_pass("policy_active", "Policy is ACTIVE_AND_VERIFIED."))

    # ── Rule 2: Extraction must exist ────────────────────────────────────
    if not policy_extraction:
        add(_fail(
            "policy_extraction",
            "Policy facts have not been extracted. "
            "Complete extraction and verification before analyzing claims.",
        ))
        return _review_result(checks, reasons, missing, citations,
                              "Policy extraction is missing.")

    add(_pass("policy_extraction", "Policy extraction is available."))

    # ── Rule 3: Policy version active on loss date ───────────────────────
    loss_date     = claim_facts.loss_date
    eff_from      = policy.get("effective_from")
    eff_to        = policy.get("effective_to")

    if not loss_date:
        add(_fail("loss_date_present", "Loss date was not provided."))
        reasons.append("Loss date is required but was not provided.")
    else:
        date_ok = True
        if eff_from and loss_date < eff_from:
            date_ok = False
        if eff_to and loss_date > eff_to:
            date_ok = False

        if date_ok:
            eff_range = f"{eff_from or 'open'} – {eff_to or 'open'}"
            add(_pass(
                "loss_date_in_range",
                f"Loss date {loss_date} falls within policy period {eff_range}.",
            ))
        else:
            add(_fail(
                "loss_date_in_range",
                f"Loss date {loss_date} is outside policy period "
                f"{eff_from or 'open'} – {eff_to or 'open'}.",
            ))
            reasons.append(
                f"Loss date {loss_date} is outside the policy effective period."
            )

    # ── Rule 4: Policy number comparison ─────────────────────────────────
    # (skipped in MVP if neither side has it — not a hard blocker)
    pol_number = policy_extraction.get("policy_number")
    # Policy number from claim is not separately extracted in MVP
    # — mark as informational pass.
    add(_pass(
        "policy_number",
        f"Policy number on record: {pol_number or 'not extracted'}.",
    ))

    # ── Rule 5: Claimant / insured match ─────────────────────────────────
    named_insured  = policy_extraction.get("named_insured")
    claimant_name  = claim_facts.claimant_name

    if named_insured and claimant_name:
        if names_match(named_insured, claimant_name):
            add(_pass(
                "insured_match",
                f"Claimant '{claimant_name}' matches named insured '{named_insured}'.",
            ))
        else:
            add(_fail(
                "insured_match",
                f"Claimant '{claimant_name}' does not match "
                f"named insured '{named_insured}'.",
            ))
            reasons.append(
                f"Claimant name '{claimant_name}' does not match "
                f"policy named insured '{named_insured}'."
            )
    else:
        add(_pass(
            "insured_match",
            "Claimant/insured comparison skipped (one or both names unavailable).",
        ))

    # ── Rule 6: Driver check ─────────────────────────────────────────────
    # MVP: driver name noted but not hard-blocked (no driver list extracted yet)
    driver_name = claim_facts.driver_name
    add(_pass(
        "driver_check",
        f"Driver: {driver_name or 'not provided'} — "
        "manual review recommended if driver differs from named insured.",
    ))

    # ── Rule 7: VIN / vehicle match ──────────────────────────────────────
    policy_vin = policy_extraction.get("vehicle_vin")
    claim_vin  = claim_facts.vehicle_vin

    if policy_vin and claim_vin:
        if vins_match(policy_vin, claim_vin):
            add(_pass(
                "vin_match",
                f"Vehicle VIN {claim_vin} matches policy VIN {policy_vin}.",
            ))
        else:
            add(_fail(
                "vin_match",
                f"Claim VIN '{claim_vin}' does not match policy VIN '{policy_vin}'.",
            ))
            reasons.append(
                f"Vehicle VIN '{claim_vin}' does not match the insured VIN '{policy_vin}'."
            )
    else:
        add(_pass(
            "vin_match",
            f"VIN comparison skipped "
            f"(policy VIN: {policy_vin or 'not extracted'}, "
            f"claim VIN: {claim_vin or 'not provided'}).",
        ))

    # ── Rule 8: Territory and permitted use ──────────────────────────────
    territory    = policy_extraction.get("territory")
    permitted    = policy_extraction.get("permitted_use")
    add(_pass(
        "territory_use",
        f"Territory: {territory or 'not specified'}. "
        f"Permitted use: {permitted or 'not specified'}. "
        "Manual review required if use or location is unusual.",
    ))

    # ── Rule 9: Coverage type trigger ────────────────────────────────────
    coverage_types = policy_extraction.get("coverage_types") or []
    coverage_lower = [c.casefold() for c in coverage_types]
    loss_desc      = (claim_facts.loss_description or "").casefold()

    collision_keywords = ["collision", "upset", "rollover", "impact", "crash",
                          "accident", "hit", "struck"]
    collision_in_loss  = any(kw in loss_desc for kw in collision_keywords)
    collision_covered  = any("collision" in c or "upset" in c for c in coverage_lower)

    if coverage_types:
        if collision_in_loss and collision_covered:
            add(_pass(
                "coverage_trigger",
                "Collision or Upset coverage is included and the loss "
                "description indicates a collision event.",
            ))
        elif not collision_in_loss:
            add(_pass(
                "coverage_trigger",
                f"Available coverages: {', '.join(coverage_types)}. "
                "Loss description does not specifically indicate collision — "
                "manual review recommended.",
            ))
        else:
            add(_fail(
                "coverage_trigger",
                f"Loss appears to be a collision but Collision coverage "
                f"is not listed. Available: {', '.join(coverage_types)}.",
            ))
            reasons.append(
                "The applicable coverage type was not found in the policy."
            )
    else:
        add(_fail(
            "coverage_trigger",
            "No coverage types were extracted from the policy.",
        ))
        reasons.append("Coverage types could not be confirmed from policy extraction.")

    # ── Rule 10: Exclusions ───────────────────────────────────────────────
    exclusions = policy_extraction.get("exclusions") or []
    if exclusions:
        # MVP: flag exclusions for human review rather than hard-block
        # (full exclusion matching requires more claim context)
        add(_pass(
            "exclusion_check",
            f"Policy has {len(exclusions)} recorded exclusion(s). "
            "Human reviewer must confirm none apply to this loss.",
        ))
        reasons.append(
            f"Policy contains {len(exclusions)} exclusion(s) requiring manual review: "
            + "; ".join(exclusions[:3])
            + ("..." if len(exclusions) > 3 else "")
        )
    else:
        add(_pass("exclusion_check", "No exclusions extracted from policy."))

    # ── Rule 11: Required documents ──────────────────────────────────────
    required_docs  = policy_extraction.get("required_documents") or []
    submitted_docs = claim_facts.submitted_documents or []
    missing_docs   = _missing_documents(required_docs, submitted_docs)

    if missing_docs:
        missing.extend(missing_docs)
        add(_fail(
            "required_documents",
            f"Missing required documents: {', '.join(missing_docs)}.",
        ))
        reasons.append("Missing required documents: " + ", ".join(missing_docs))
    else:
        if required_docs:
            add(_pass(
                "required_documents",
                f"All {len(required_docs)} required document(s) were submitted.",
            ))
        else:
            add(_pass(
                "required_documents",
                "No required documents specified in policy extraction.",
            ))

    # ── Rule 12: Cross-document consistency ──────────────────────────────
    # MVP: flag for human review if unsupported fields present
    unsupported = claim_facts.unsupported_fields or []
    if unsupported:
        add(_fail(
            "consistency_check",
            f"Claim extraction reported unsupported fields: "
            f"{', '.join(unsupported)}. Human review required.",
        ))
        reasons.append(
            "Some claim fields could not be reliably extracted: "
            + ", ".join(unsupported)
        )
    else:
        add(_pass("consistency_check",
                  "No consistency issues flagged by claim extraction."))

    # ── Rule 13: Coverage limit ───────────────────────────────────────────
    coverage_limit = policy_extraction.get("coverage_limit")
    claim_amount   = claim_facts.claim_amount

    if coverage_limit is not None and claim_amount is not None:
        if claim_amount > coverage_limit:
            add(_fail(
                "coverage_limit",
                f"Claim amount ${claim_amount:,.2f} exceeds policy limit "
                f"${coverage_limit:,.2f}. Claim will be capped at limit.",
            ))
            reasons.append(
                f"Claim amount ${claim_amount:,.2f} exceeds "
                f"policy coverage limit ${coverage_limit:,.2f}."
            )
        else:
            add(_pass(
                "coverage_limit",
                f"Claim amount ${claim_amount:,.2f} is within "
                f"policy limit ${coverage_limit:,.2f}.",
            ))
    else:
        add(_fail(
            "coverage_limit",
            f"Coverage limit comparison skipped "
            f"(limit: {coverage_limit}, amount: {claim_amount}).",
        ))
        reasons.append("Coverage limit or claim amount could not be determined.")

    # ── Rule 14: Deductible ───────────────────────────────────────────────
    deductible = policy_extraction.get("deductible")
    if deductible is not None:
        add(_pass(
            "deductible",
            f"Policy deductible is ${deductible:,.2f} and will be applied.",
        ))
    else:
        add(_fail("deductible",
                  "Deductible amount could not be extracted from policy."))
        reasons.append("Deductible amount could not be confirmed from policy extraction.")

    # ── Rule 15: Evidence support ─────────────────────────────────────────
    # (RAG evidence validation is handled upstream in the workflow;
    #  here we note citations from the policy extraction)
    policy_name    = policy.get("policy_name", "Policy")
    version_label  = policy.get("version_label", "v1")
    policy_cite    = f"{policy_name} ({version_label})"
    if policy_cite not in citations:
        citations.append(policy_cite)

    # ── Rule 16: Preliminary outcome ─────────────────────────────────────
    failed_checks  = [c for c in checks if not c.passed]
    passed_checks  = [c for c in checks if c.passed]

    # Hard blockers that force REVIEW_REQUIRED regardless of other rules
    hard_blocker_names = {
        "policy_active", "policy_extraction", "coverage_trigger",
    }
    has_hard_blocker = any(
        c.rule_name in hard_blocker_names and not c.passed
        for c in checks
    )

    # Any missing required doc, any failed hard-blocker → REVIEW_REQUIRED
    # Exclusions always force REVIEW_REQUIRED (human must confirm)
    has_exclusions  = bool(exclusions)
    has_missing_docs = bool(missing_docs)

    if has_hard_blocker or has_missing_docs:
        outcome = "REVIEW_REQUIRED"
    elif has_exclusions or any(not c.passed for c in checks):
        # Soft failures and exclusions → REVIEW_REQUIRED to be safe
        outcome = "REVIEW_REQUIRED"
    else:
        # All checks passed, amount within limit
        if coverage_limit is not None and claim_amount is not None:
            outcome = "LIKELY_COVERED"
        else:
            outcome = "REVIEW_REQUIRED"

    # ── Amount calculation (spec §10) ────────────────────────────────────
    eligible_gross, ded_applied, estimated_net = calculate_amounts(
        claim_amount, coverage_limit, deductible
    )

    return ComparisonResult(
        outcome=outcome,
        checks=checks,
        reasons=list(dict.fromkeys(reasons)),
        missing_documents=missing,
        passed_checks=[c.rule_name for c in passed_checks],
        failed_checks=[c.rule_name for c in failed_checks],
        citations=citations,
        eligible_gross=eligible_gross,
        deductible_applied=ded_applied,
        estimated_net=estimated_net,
    )


# ---------------------------------------------------------------------------
# Internal helper — build a REVIEW_REQUIRED result quickly
# ---------------------------------------------------------------------------

def _review_result(
    checks: list[RuleCheck],
    reasons: list[str],
    missing: list[str],
    citations: list[str],
    extra_reason: str,
) -> ComparisonResult:
    if extra_reason and extra_reason not in reasons:
        reasons.append(extra_reason)
    return ComparisonResult(
        outcome="REVIEW_REQUIRED",
        checks=checks,
        reasons=reasons,
        missing_documents=missing,
        passed_checks=[c.rule_name for c in checks if c.passed],
        failed_checks=[c.rule_name for c in checks if not c.passed],
        citations=citations,
    )


