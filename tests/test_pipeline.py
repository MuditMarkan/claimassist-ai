from collections.abc import Callable

import pytest

import app.pipeline as pipeline
from app.business_rules import ClaimData

def complete_claim() -> ClaimData:
    return ClaimData(
        claim_id="CLM-APPROVE",
        claim_amount=5000,
        policy_limit=10000,
        required_documents=[
            "Claim form",
            "Police report",
        ],
        submitted_documents=[
            "Claim form",
            "Police report",
        ],
    )
def pending_claim() -> ClaimData:
    return ClaimData(
        claim_id="CLM-PEND",
        claim_amount=12000,
        policy_limit=10000,
        required_documents=[
            "Claim form",
            "Police report",
            "Repair estimate",
        ],
        submitted_documents=[
            "Claim form",
            "Police report",
        ],
    )
def replace_extractor(
    monkeypatch: pytest.MonkeyPatch,
    replacement: Callable[[str], ClaimData],
) -> None:
    monkeypatch.setattr(
        pipeline,
        "extract_claim_data",
        replacement,
    )

def test_valid_complete_claim_is_approved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_extractor(document_text: str) -> ClaimData:
        assert document_text == "Synthetic complete claim"
        return complete_claim()

    replace_extractor(monkeypatch, fake_extractor)

    claim, decision = pipeline.run_claim_pipeline(
        "Synthetic complete claim"
    )

    assert claim is not None
    assert claim.claim_id == "CLM-APPROVE"
    assert decision.decision == "APPROVE"
    assert decision.reasons == []

def test_valid_incomplete_claim_is_pended(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_extractor(document_text: str) -> ClaimData:
        return pending_claim()

    replace_extractor(monkeypatch, fake_extractor)

    claim, decision = pipeline.run_claim_pipeline(
        "Synthetic incomplete claim"
    )

    assert claim is not None
    assert decision.decision == "PEND"
    assert decision.missing_documents == ["Repair estimate"]
    assert len(decision.reasons) == 2
def test_invalid_extraction_returns_review(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_extractor(document_text: str) -> ClaimData:
        # This deliberately causes a Pydantic ValidationError.
        return ClaimData.model_validate(
            {
                "claim_id": "CLM-INVALID",
                "claim_amount": 5000,
            }
        )

    replace_extractor(monkeypatch, fake_extractor)
    claim, decision = pipeline.run_claim_pipeline(
        "Invalid extraction example"
    )

    assert claim is None
    assert decision.decision == "REVIEW"
    assert decision.missing_documents == []
    assert "failed validation" in decision.reasons[0]

def test_empty_document_returns_review(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_extractor(document_text: str) -> ClaimData:
        raise ValueError("The claim document is empty.")
    replace_extractor(monkeypatch, fake_extractor)

    claim, decision = pipeline.run_claim_pipeline("")

    assert claim is None
    assert decision.decision == "REVIEW"
    assert decision.reasons == ["The claim document is empty."]

def test_ollama_failure_returns_review(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_extractor(document_text: str) -> ClaimData:
        raise RuntimeError("Ollama is unavailable.")

    replace_extractor(monkeypatch, fake_extractor)
    claim, decision = pipeline.run_claim_pipeline(
        "Synthetic claim"
    )

    assert claim is None
    assert decision.decision == "REVIEW"
    assert decision.reasons == [
        "Claim processing failed and requires human review."
    ]