from pathlib import Path

import pytest

from app.business_rules import ClaimData, RuleDecision
from app.database import database_connection
from app.repositories import (
    create_claim,
    get_claim_snapshot,
    mark_claim_failed,
    mark_claim_processing,
    save_decision,
    save_extracted_facts,
)

def sample_claim_data() -> ClaimData:
    return ClaimData(
        claim_id="CLM-DB-001",
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

def test_create_claim_writes_record_and_audit(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "test.db"

    claim_record_id = create_claim(
        "Synthetic claim document",
        database_path,
    )

    with database_connection(database_path) as connection:
        claim_row = connection.execute(
            "SELECT * FROM claims WHERE id = ?",
            (claim_record_id,),
        ).fetchone()

        audit_row = connection.execute(
            """
            SELECT *
            FROM audit_events
            WHERE entity_type = 'claim'
              AND entity_id = ?
              AND event_type = 'CLAIM_CREATED'
            """,
            (claim_record_id,),
        ).fetchone()

    assert claim_row is not None
    assert claim_row["raw_claim_text"] == (
        "Synthetic claim document"
    )
    assert claim_row["processing_status"] == "RECEIVED"
    assert audit_row is not None

def test_extracted_facts_are_saved(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "test.db"

    claim_record_id = create_claim(
        "Synthetic claim document",
        database_path,
    )

    mark_claim_processing(
        claim_record_id,
        database_path,
    )

    save_extracted_facts(
        claim_record_id,
        sample_claim_data(),
        database_path,
    )

    snapshot = get_claim_snapshot(
        claim_record_id,
        database_path,
    )

    assert snapshot["claim"]["claim_reference"] == "CLM-DB-001"
    assert snapshot["claim"]["processing_status"] == "PROCESSING"

    facts = snapshot["extracted_facts"]

    assert facts is not None
    assert facts["claim_amount"] == 12000
    assert facts["policy_limit"] == 10000
    assert facts["required_documents"] == [
        "Claim form",
        "Police report",
        "Repair estimate",
    ]
    assert facts["submitted_documents"] == [
        "Claim form",
        "Police report",
    ]

def test_pend_decision_is_saved(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "test.db"

    claim_record_id = create_claim(
        "Synthetic claim document",
        database_path,
    )

    decision = RuleDecision(
        decision="PEND",
        reasons=[
            "Claim amount exceeds the policy limit.",
            "Missing required documents: Repair estimate",
        ],
        missing_documents=["Repair estimate"],
    )
    decision_record_id = save_decision(
        claim_record_id,
        decision,
        database_path,
    )

    snapshot = get_claim_snapshot(
        claim_record_id,
        database_path,
    )

    assert decision_record_id > 0
    assert snapshot["claim"]["processing_status"] == "COMPLETED"
    saved_decision = snapshot["latest_decision"]

    assert saved_decision is not None
    assert saved_decision["decision"] == "PEND"
    assert saved_decision["missing_documents"] == [
        "Repair estimate"
    ]
    assert len(saved_decision["reasons"]) == 2

def test_review_decision_opens_one_human_review(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "test.db"

    claim_record_id = create_claim(
        "Claim with insufficient information",
        database_path,
    )

    review_decision = RuleDecision(
        decision="REVIEW",
        reasons=["Policy limit is unavailable."],
        missing_documents=[],
    )

    save_decision(
        claim_record_id,
        review_decision,
        database_path,
    )
    # Saving another REVIEW result must not create
    # a duplicate open review.
    save_decision(
        claim_record_id,
        review_decision,
        database_path,
    )

    with database_connection(database_path) as connection:
        review_rows = connection.execute(
            """
            SELECT *
            FROM human_reviews
            WHERE claim_id = ?
              AND review_status = 'OPEN'
            """,
            (claim_record_id,),
        ).fetchall()

        claim_row = connection.execute(
            "SELECT * FROM claims WHERE id = ?",
            (claim_record_id,),
        ).fetchone()

    assert len(review_rows) == 1
    assert claim_row["processing_status"] == "REVIEW"

def test_failed_claim_is_recorded_and_audited(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "test.db"

    claim_record_id = create_claim(
        "Synthetic failed claim",
        database_path,
    )

    mark_claim_failed(
        claim_record_id,
        "ModelTimeout",
        database_path,
    )

    with database_connection(database_path) as connection:
        claim_row = connection.execute(
            "SELECT * FROM claims WHERE id = ?",
            (claim_record_id,),
        ).fetchone()

        audit_row = connection.execute(
            """
            SELECT *
            FROM audit_events
            WHERE entity_id = ?
              AND event_type = 'CLAIM_PROCESSING_FAILED'
            """,
            (claim_record_id,),
        ).fetchone()
    assert claim_row["processing_status"] == "FAILED"
    assert audit_row is not None
    assert "ModelTimeout" in audit_row["details_json"]

def test_missing_claim_cannot_receive_facts(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "test.db"

    # Initialize the database with one legitimate claim.
    create_claim(
        "Legitimate synthetic claim",
        database_path,
    )
    with pytest.raises(
        LookupError,
        match="does not exist",
    ):
        save_extracted_facts(
            999999,
            sample_claim_data(),
            database_path,
        )

    with database_connection(database_path) as connection:
        orphan_count = connection.execute(
            """
            SELECT COUNT(*) AS count
            FROM extracted_facts
            WHERE claim_id = 999999
            """
        ).fetchone()["count"]

    assert orphan_count == 0