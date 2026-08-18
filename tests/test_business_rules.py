import pytest
from pydantic import ValidationError

from app.business_rules import ClaimData, evaluate_claim

def make_claim(**overrides) -> ClaimData:
    """Create a valid default claim and optionally replace fields."""
    claim_data = {
        "claim_id": "CLM-TEST-001",
        "claim_amount": 5000,
        "policy_limit": 10000,
        "required_documents": [
            "Claim form",
            "Police report",
            "Repair estimate",
        ],
        "submitted_documents": [
            "Claim form",
            "Police report",
            "Repair estimate",
        ],
    }

    claim_data.update(overrides)
    return ClaimData(**claim_data)

def test_complete_claim_is_approved() -> None:
    claim = make_claim()

    result = evaluate_claim(claim)

    assert result.decision == "APPROVE"
    assert result.reasons == []
    assert result.missing_documents == []

def test_claim_above_policy_limit_is_pended() -> None:
    claim = make_claim(
        claim_amount=12000,
        policy_limit=10000,
    )

    result = evaluate_claim(claim)

    assert result.decision == "PEND"
    assert result.missing_documents == []
    assert any(
        "exceeds the policy limit" in reason
        for reason in result.reasons
    )

def test_missing_document_is_pended() -> None:
    claim = make_claim(
        submitted_documents=[
            "Claim form",
            "Police report",
        ]
    )

    result = evaluate_claim(claim)

    assert result.decision == "PEND"
    assert result.missing_documents == ["Repair estimate"]
    assert any(
        "Missing required documents" in reason
        for reason in result.reasons
    )
def test_multiple_rule_failures_are_all_reported() -> None:
    claim = make_claim(
        claim_amount=12000,
        policy_limit=10000,
        submitted_documents=["Claim form"],
    )

    result = evaluate_claim(claim)

    assert result.decision == "PEND"
    assert len(result.reasons) == 2
    assert result.missing_documents == [
        "Police report",
        "Repair estimate",
    ]

def test_document_comparison_ignores_case_and_whitespace() -> None:
    claim = make_claim(
        required_documents=["Police report"],
        submitted_documents=["   police    REPORT   "],
    )

    result = evaluate_claim(claim)

    assert result.decision == "APPROVE"
    assert result.missing_documents == []

def test_negative_claim_amount_fails_validation() -> None:
    with pytest.raises(ValidationError):
        make_claim(claim_amount=-100)

def test_negative_policy_limit_fails_validation() -> None:
    with pytest.raises(ValidationError):
        make_claim(policy_limit=-1)

def test_unknown_fields_fail_validation() -> None:
    with pytest.raises(ValidationError):
        make_claim(unexpected_field="not allowed")

def test_no_required_documents_can_be_approved() -> None:
    claim = make_claim(
        required_documents=[],
        submitted_documents=[],
    )
    result = evaluate_claim(claim)

    assert result.decision == "APPROVE"
    assert result.missing_documents == []

def test_rule_engine_is_deterministic() -> None:
    claim = make_claim(
        claim_amount=12000,
        submitted_documents=["Claim form"],
    )

    first_result = evaluate_claim(claim)
    second_result = evaluate_claim(claim)
    assert first_result == second_result