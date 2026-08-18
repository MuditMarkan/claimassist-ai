import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.business_rules import ClaimData, RuleDecision
from app.database import (
    DATABASE_PATH,
    database_connection,
    initialize_database,
)

RULE_VERSION = "1.0.0"

def utc_now() -> str:
    """Return a timezone-aware UTC timestamp."""

    return datetime.now(timezone.utc).isoformat()

def encode_json(value: Any) -> str:
    """Serialize a value consistently for SQLite storage."""
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    )

def decode_json(value: str | None) -> Any:
    """Deserialize stored JSON safely."""

    if value is None:
        return None

    return json.loads(value)
def _require_claim(
    connection: sqlite3.Connection,
    claim_record_id: int,
) -> None:
    """Raise an error when the internal claim record is missing."""

    row = connection.execute(
        "SELECT id FROM claims WHERE id = ?",
        (claim_record_id,),
    ).fetchone()

    if row is None:
        raise LookupError(
            f"Claim record {claim_record_id} does not exist."
        )
def _insert_audit_event(
    connection: sqlite3.Connection,
    *,
    event_type: str,
    entity_type: str,
    entity_id: int | None,
    details: dict[str, Any],
) -> None:
    """Insert an audit event within an existing transaction."""

    connection.execute(
        """
        INSERT INTO audit_events (
            event_type,
            entity_type,
            entity_id,
            details_json,
            created_at
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            event_type,
            entity_type,
            entity_id,
            encode_json(details),
            utc_now(),
        ),
    )
def create_claim(
    raw_claim_text: str,
    database_path: Path = DATABASE_PATH,
) -> int:
    """Create an internal claim record and return its database ID."""

    cleaned_text = raw_claim_text.strip()

    if not cleaned_text:
        raise ValueError("The claim text cannot be empty.")

    initialize_database(database_path)
    timestamp = utc_now()
    with database_connection(database_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO claims (
                claim_reference,
                raw_claim_text,
                processing_status,
                created_at,
                updated_at
            )
            VALUES (NULL, ?, 'RECEIVED', ?, ?)
            """,
            (
                cleaned_text,
                timestamp,
                timestamp,
            ),
        )

        claim_record_id = int(cursor.lastrowid)

        _insert_audit_event(
            connection,
            event_type="CLAIM_CREATED",
            entity_type="claim",
            entity_id=claim_record_id,
            details={
                "processing_status": "RECEIVED",
            },
        )
    return claim_record_id

def mark_claim_processing(
    claim_record_id: int,
    database_path: Path = DATABASE_PATH,
) -> None:
    """Mark a claim as actively processing."""

    with database_connection(database_path) as connection:
        _require_claim(connection, claim_record_id)

        connection.execute(
            """
            UPDATE claims
            SET processing_status = 'PROCESSING',
                updated_at = ?
            WHERE id = ?
            """,
            (
                utc_now(),
                claim_record_id,
            ),
        )

        _insert_audit_event(
            connection,
            event_type="CLAIM_PROCESSING_STARTED",
            entity_type="claim",
            entity_id=claim_record_id,
            details={
                "processing_status": "PROCESSING",
            },
        )

def save_extracted_facts(
    claim_record_id: int,
    claim: ClaimData,
    database_path: Path = DATABASE_PATH,
) -> None:
    """Insert or replace the validated facts for a claim."""

    timestamp = utc_now()

    with database_connection(database_path) as connection:
        _require_claim(connection, claim_record_id)

        connection.execute(
            """
            INSERT INTO extracted_facts (
                claim_id,
                extracted_claim_reference,
                claim_amount,
                policy_limit,
                required_documents_json,
                submitted_documents_json,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(claim_id) DO UPDATE SET
                extracted_claim_reference =
                    excluded.extracted_claim_reference,
                claim_amount = excluded.claim_amount,
                policy_limit = excluded.policy_limit,
                required_documents_json =
                    excluded.required_documents_json,
                submitted_documents_json =
                    excluded.submitted_documents_json,
                updated_at = excluded.updated_at
            """,
            (
                claim_record_id,
                claim.claim_id,
                claim.claim_amount,
                claim.policy_limit,
                encode_json(claim.required_documents),
                encode_json(claim.submitted_documents),
                timestamp,
                timestamp,
            ),
        )

        connection.execute(
            """
            UPDATE claims
            SET claim_reference = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (
                claim.claim_id,
                timestamp,
                claim_record_id,
            ),
        )

        _insert_audit_event(
            connection,
            event_type="EXTRACTED_FACTS_SAVED",
            entity_type="claim",
            entity_id=claim_record_id,
            details={
                "claim_reference": claim.claim_id,
                "extractor": "qwen3.5:4b",
            },
        )

def save_decision(
    claim_record_id: int,
    decision: RuleDecision,
    database_path: Path = DATABASE_PATH,
) -> int:
    """Save a decision and open human review when required."""

    timestamp = utc_now()

    processing_status = (
        "REVIEW"
        if decision.decision == "REVIEW"
        else "COMPLETED"
    )

    with database_connection(database_path) as connection:
        _require_claim(connection, claim_record_id)

        cursor = connection.execute(
            """
            INSERT INTO decisions (
                claim_id,
                decision,
                reasons_json,
                missing_documents_json,
                rule_version,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                claim_record_id,
                decision.decision,
                encode_json(decision.reasons),
                encode_json(decision.missing_documents),
                RULE_VERSION,
                timestamp,
            ),
        )

        decision_record_id = int(cursor.lastrowid)
        connection.execute(
            """
            UPDATE claims
            SET processing_status = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (
                processing_status,
                timestamp,
                claim_record_id,
            ),
        )

        if decision.decision == "REVIEW":
            existing_review = connection.execute(
                """
                SELECT id
                FROM human_reviews
                WHERE claim_id = ?
                  AND review_status = 'OPEN'
                LIMIT 1
                """,
                (claim_record_id,),
            ).fetchone()

            if existing_review is None:
                connection.execute(
                    """
                    INSERT INTO human_reviews (
                        claim_id,
                        review_status,
                        created_at
                    )
                    VALUES (?, 'OPEN', ?)
                    """,
                    (
                        claim_record_id,
                        timestamp,
                    ),
                )

        _insert_audit_event(
            connection,
            event_type="DECISION_SAVED",
            entity_type="claim",
            entity_id=claim_record_id,
            details={
                "decision_record_id": decision_record_id,
                "decision": decision.decision,
                "rule_version": RULE_VERSION,
            },
        )

    return decision_record_id

def mark_claim_failed(
    claim_record_id: int,
    error_type: str,
    database_path: Path = DATABASE_PATH,
) -> None:
    """Record a safe failure without storing sensitive error content."""

    with database_connection(database_path) as connection:
        _require_claim(connection, claim_record_id)

        connection.execute(
            """
            UPDATE claims
            SET processing_status = 'FAILED',
                updated_at = ?
            WHERE id = ?
            """,
            (
                utc_now(),
                claim_record_id,
            ),
        )

        _insert_audit_event(
            connection,
            event_type="CLAIM_PROCESSING_FAILED",
            entity_type="claim",
            entity_id=claim_record_id,
            details={
                "error_type": error_type,
            },
        )

def get_claim_snapshot(
    claim_record_id: int,
    database_path: Path = DATABASE_PATH,
) -> dict[str, Any]:
    """Return the claim, extracted facts, and latest decision."""

    with database_connection(database_path) as connection:
        claim_row = connection.execute(
            "SELECT * FROM claims WHERE id = ?",
            (claim_record_id,),
        ).fetchone()

        if claim_row is None:
            raise LookupError(
                f"Claim record {claim_record_id} does not exist."
            )
        facts_row = connection.execute(
            """
            SELECT *
            FROM extracted_facts
            WHERE claim_id = ?
            """,
            (claim_record_id,),
        ).fetchone()

        decision_row = connection.execute(
            """
            SELECT *
            FROM decisions
            WHERE claim_id = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (claim_record_id,),
        ).fetchone()

    claim_data = dict(claim_row)

    extracted_facts = None
    if facts_row is not None:
        extracted_facts = dict(facts_row)
        extracted_facts["required_documents"] = decode_json(
            extracted_facts.pop("required_documents_json")
        )
        extracted_facts["submitted_documents"] = decode_json(
            extracted_facts.pop("submitted_documents_json")
        )

    latest_decision = None
    if decision_row is not None:
        latest_decision = dict(decision_row)
        latest_decision["reasons"] = decode_json(
            latest_decision.pop("reasons_json")
        )
        latest_decision["missing_documents"] = decode_json(
            latest_decision.pop("missing_documents_json")
        )

    return {
        "claim": claim_data,
        "extracted_facts": extracted_facts,
        "latest_decision": latest_decision,
    }
