from pathlib import Path

import pytest

import app.pipeline as pipeline
from app.business_rules import ClaimData
from app.database import database_connection
from app.repositories import get_claim_snapshot

def approved_claim() -> ClaimData:
    return ClaimData(
        claim_id="CLM-PERSIST-001",
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

def test_successful_claim_is_fully_persisted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "test.db"

    def fake_extractor(document_text: str) -> ClaimData:
        assert document_text == "Complete synthetic claim"
        return approved_claim()

    monkeypatch.setattr(
        pipeline,
        "extract_claim_data",
        fake_extractor,
    )

    claim_record_id, claim, decision = (
        pipeline.process_and_persist_claim(
            "Complete synthetic claim",
            database_path,
        )
    )

    snapshot = get_claim_snapshot(
        claim_record_id,
        database_path,
    )
    assert claim is not None
    assert decision.decision == "APPROVE"

    assert snapshot["claim"]["processing_status"] == "COMPLETED"
    assert snapshot["claim"]["claim_reference"] == (
        "CLM-PERSIST-001"
    )

    assert snapshot["extracted_facts"] is not None
    assert snapshot["extracted_facts"]["claim_amount"] == 5000

    assert snapshot["latest_decision"] is not None
    assert snapshot["latest_decision"]["decision"] == "APPROVE"

    with database_connection(database_path) as connection:
        audit_events = connection.execute(
            """
            SELECT event_type
            FROM audit_events
            WHERE entity_type = 'claim'
              AND entity_id = ?
            ORDER BY id
            """,
            (claim_record_id,),
        ).fetchall()

    assert [row["event_type"] for row in audit_events] == [
        "CLAIM_CREATED",
        "CLAIM_PROCESSING_STARTED",
        "EXTRACTED_FACTS_SAVED",
        "DECISION_SAVED",
    ]

def test_extraction_failure_is_persisted_as_review(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "test.db"

    def failing_extractor(
        document_text: str,
    ) -> ClaimData:
        raise ValueError(
            "Required claim information is unavailable."
        )
    monkeypatch.setattr(
        pipeline,
        "extract_claim_data",
        failing_extractor,
    )

    claim_record_id, claim, decision = (
        pipeline.process_and_persist_claim(
            "Incomplete synthetic claim",
            database_path,
        )
    )

    snapshot = get_claim_snapshot(
        claim_record_id,
        database_path,
    )

    assert claim is None
    assert decision.decision == "REVIEW"
    assert snapshot["claim"]["processing_status"] == "REVIEW"
    assert snapshot["extracted_facts"] is None
    assert snapshot["latest_decision"]["decision"] == "REVIEW"

    with database_connection(database_path) as connection:
        open_reviews = connection.execute(
            """
            SELECT COUNT(*) AS count
            FROM human_reviews
            WHERE claim_id = ?
              AND review_status = 'OPEN'
            """,
            (claim_record_id,),
        ).fetchone()["count"]

    assert open_reviews == 1

def test_database_write_failure_marks_claim_failed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "test.db"

    monkeypatch.setattr(
        pipeline,
        "extract_claim_data",
        lambda document_text: approved_claim(),
    )

    def failing_save_decision(*args, **kwargs) -> int:
        raise RuntimeError("Synthetic database failure")

    monkeypatch.setattr(
        pipeline,
        "save_decision",
        failing_save_decision,
    )

    with pytest.raises(
        RuntimeError,
        match="Synthetic database failure",
    ):
        pipeline.process_and_persist_claim(
            "Synthetic database failure claim",
            database_path,
        )

    with database_connection(database_path) as connection:
        claim_row = connection.execute(
            """
            SELECT *
            FROM claims
            ORDER BY id DESC
            LIMIT 1
            """
        ).fetchone()

        failure_audit = connection.execute(
            """
            SELECT *
            FROM audit_events
            WHERE entity_id = ?
              AND event_type = 'CLAIM_PROCESSING_FAILED'
            """,
            (claim_row["id"],),
        ).fetchone()

    assert claim_row["processing_status"] == "FAILED"
    assert failure_audit is not None
    assert "RuntimeError" in failure_audit["details_json"]