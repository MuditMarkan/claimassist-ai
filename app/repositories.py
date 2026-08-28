"""
ClaimAssist data-access layer.

All functions accept an optional database_path for testability.
Every write that changes state also appends an audit event in the
same transaction so the log is always consistent.
"""

import json
import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from app.database import (
    DATABASE_PATH,
    database_connection,
    initialize_database,
)

RULE_VERSION = "2.0.0"
DEFAULT_ORG_ID = 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def encode_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def decode_json(value: str | None) -> Any:
    if value is None:
        return None
    return json.loads(value)


def _insert_audit_event(
    connection: sqlite3.Connection,
    *,
    event_type: str,
    entity_type: str,
    entity_id: int | None,
    details: dict[str, Any],
    actor: str = "system",
) -> None:
    connection.execute(
        """
        INSERT INTO audit_events
            (event_type, entity_type, entity_id, actor, details_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (event_type, entity_type, entity_id, actor,
         encode_json(details), utc_now()),
    )


def _require_row(
    connection: sqlite3.Connection,
    table: str,
    row_id: int,
) -> None:
    row = connection.execute(
        f"SELECT id FROM {table} WHERE id = ?", (row_id,)
    ).fetchone()
    if row is None:
        raise LookupError(f"{table} record {row_id} does not exist.")


# ===========================================================================
# Organization
# ===========================================================================

def get_default_organization(
    database_path: Path = DATABASE_PATH,
) -> dict[str, Any]:
    """Return the default (seeded) organization."""
    with database_connection(database_path) as connection:
        row = connection.execute(
            "SELECT * FROM organizations WHERE id = ?",
            (DEFAULT_ORG_ID,),
        ).fetchone()
    if row is None:
        raise LookupError("Default organization not found. Run initialize_database() first.")
    return dict(row)


# ===========================================================================
# Policy documents (immutable file records)
# ===========================================================================

def create_policy_document(
    *,
    original_filename: str,
    stored_path: str,
    sha256: str,
    file_size: int,
    page_count: int,
    organization_id: int = DEFAULT_ORG_ID,
    database_path: Path = DATABASE_PATH,
) -> int:
    """Insert an immutable policy file record. Returns document ID."""
    timestamp = utc_now()
    with database_connection(database_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO policy_documents
                (organization_id, original_filename, stored_path,
                 sha256, file_size, page_count, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (organization_id, original_filename, stored_path,
             sha256, file_size, page_count, timestamp),
        )
        doc_id = int(cursor.lastrowid)
        _insert_audit_event(
            connection,
            event_type="POLICY_DOCUMENT_UPLOADED",
            entity_type="policy_document",
            entity_id=doc_id,
            details={"original_filename": original_filename, "sha256": sha256},
        )
    return doc_id


# ===========================================================================
# Policy versions (lifecycle)
# ===========================================================================

def create_policy_version(
    *,
    policy_document_id: int,
    policy_name: str,
    version_label: str = "v1",
    organization_id: int = DEFAULT_ORG_ID,
    database_path: Path = DATABASE_PATH,
) -> int:
    """Create a new policy version in UPLOADED state. Returns version ID."""
    timestamp = utc_now()
    with database_connection(database_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO policy_versions
                (organization_id, policy_document_id, policy_name,
                 version_label, status, index_status, created_at, updated_at)
            VALUES (?, ?, ?, ?, 'UPLOADED', 'NOT_INDEXED', ?, ?)
            """,
            (organization_id, policy_document_id, policy_name,
             version_label, timestamp, timestamp),
        )
        version_id = int(cursor.lastrowid)
        _insert_audit_event(
            connection,
            event_type="POLICY_VERSION_CREATED",
            entity_type="policy_version",
            entity_id=version_id,
            details={"policy_name": policy_name, "version_label": version_label},
        )
    return version_id


def update_policy_version_status(
    version_id: int,
    status: str,
    database_path: Path = DATABASE_PATH,
) -> None:
    """Advance or set a policy version's lifecycle status."""
    allowed = {
        "UPLOADED", "VALIDATING", "EXTRACTION_IN_PROGRESS",
        "NEEDS_VERIFICATION", "ACTIVE_AND_VERIFIED",
        "SUPERSEDED", "RETIRED", "FAILED_SAFE",
    }
    if status not in allowed:
        raise ValueError(f"Unknown policy status: {status!r}")
    with database_connection(database_path) as connection:
        _require_row(connection, "policy_versions", version_id)
        connection.execute(
            "UPDATE policy_versions SET status = ?, updated_at = ? WHERE id = ?",
            (status, utc_now(), version_id),
        )
        _insert_audit_event(
            connection,
            event_type="POLICY_STATUS_CHANGED",
            entity_type="policy_version",
            entity_id=version_id,
            details={"new_status": status},
        )


def update_policy_index_status(
    version_id: int,
    index_status: str,
    database_path: Path = DATABASE_PATH,
) -> None:
    """Update the ChromaDB index status of a policy version."""
    allowed = {"NOT_INDEXED", "INDEXING", "INDEXED", "FAILED"}
    if index_status not in allowed:
        raise ValueError(f"Unknown index status: {index_status!r}")
    with database_connection(database_path) as connection:
        connection.execute(
            "UPDATE policy_versions SET index_status = ?, updated_at = ? WHERE id = ?",
            (index_status, utc_now(), version_id),
        )
        _insert_audit_event(
            connection,
            event_type="POLICY_INDEX_STATUS_CHANGED",
            entity_type="policy_version",
            entity_id=version_id,
            details={"index_status": index_status},
        )


def activate_policy_version(
    version_id: int,
    *,
    effective_from: str | None = None,
    effective_to: str | None = None,
    database_path: Path = DATABASE_PATH,
) -> None:
    """
    Mark a policy version ACTIVE_AND_VERIFIED.
    Any previously active version for the same organization is
    automatically moved to SUPERSEDED.
    """
    with database_connection(database_path) as connection:
        _require_row(connection, "policy_versions", version_id)

        # Find org for this version
        row = connection.execute(
            "SELECT organization_id FROM policy_versions WHERE id = ?",
            (version_id,),
        ).fetchone()
        org_id = row["organization_id"]

        # Supersede any currently active versions for this org
        connection.execute(
            """
            UPDATE policy_versions
            SET status = 'SUPERSEDED', updated_at = ?
            WHERE organization_id = ?
              AND status = 'ACTIVE_AND_VERIFIED'
              AND id != ?
            """,
            (utc_now(), org_id, version_id),
        )

        # Activate this version
        connection.execute(
            """
            UPDATE policy_versions
            SET status = 'ACTIVE_AND_VERIFIED',
                effective_from = ?,
                effective_to   = ?,
                updated_at     = ?
            WHERE id = ?
            """,
            (effective_from, effective_to, utc_now(), version_id),
        )

        _insert_audit_event(
            connection,
            event_type="POLICY_ACTIVATED",
            entity_type="policy_version",
            entity_id=version_id,
            details={
                "effective_from": effective_from,
                "effective_to": effective_to,
            },
        )


def get_active_policy_version(
    organization_id: int = DEFAULT_ORG_ID,
    database_path: Path = DATABASE_PATH,
) -> dict[str, Any] | None:
    """Return the single ACTIVE_AND_VERIFIED policy version, or None."""
    with database_connection(database_path) as connection:
        row = connection.execute(
            """
            SELECT pv.*, pd.original_filename, pd.stored_path, pd.sha256
            FROM policy_versions pv
            JOIN policy_documents pd ON pd.id = pv.policy_document_id
            WHERE pv.organization_id = ?
              AND pv.status = 'ACTIVE_AND_VERIFIED'
            ORDER BY pv.id DESC
            LIMIT 1
            """,
            (organization_id,),
        ).fetchone()
    return dict(row) if row else None


def resolve_policy_version_for_date(
    loss_date: str,
    organization_id: int = DEFAULT_ORG_ID,
    database_path: Path = DATABASE_PATH,
) -> dict[str, Any] | None:
    """
    Find the ACTIVE_AND_VERIFIED policy version whose effective range
    covers loss_date.  Falls back to any active version if no dates set.
    """
    with database_connection(database_path) as connection:
        row = connection.execute(
            """
            SELECT pv.*, pd.original_filename, pd.stored_path, pd.sha256
            FROM policy_versions pv
            JOIN policy_documents pd ON pd.id = pv.policy_document_id
            WHERE pv.organization_id = ?
              AND pv.status = 'ACTIVE_AND_VERIFIED'
              AND (
                    pv.effective_from IS NULL
                 OR pv.effective_from <= ?
              )
              AND (
                    pv.effective_to IS NULL
                 OR pv.effective_to >= ?
              )
            ORDER BY pv.id DESC
            LIMIT 1
            """,
            (organization_id, loss_date, loss_date),
        ).fetchone()
    return dict(row) if row else None


def list_policy_versions(
    organization_id: int = DEFAULT_ORG_ID,
    database_path: Path = DATABASE_PATH,
) -> list[dict[str, Any]]:
    """Return all policy versions for an org, newest first."""
    with database_connection(database_path) as connection:
        rows = connection.execute(
            """
            SELECT pv.id, pv.policy_name, pv.version_label,
                   pv.status, pv.index_status,
                   pv.effective_from, pv.effective_to,
                   pv.created_at, pv.updated_at,
                   pd.original_filename
            FROM policy_versions pv
            JOIN policy_documents pd ON pd.id = pv.policy_document_id
            WHERE pv.organization_id = ?
            ORDER BY pv.id DESC
            """,
            (organization_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_policy_version(
    version_id: int,
    database_path: Path = DATABASE_PATH,
) -> dict[str, Any]:
    """Return full policy version row including document info."""
    with database_connection(database_path) as connection:
        row = connection.execute(
            """
            SELECT pv.*, pd.original_filename, pd.stored_path, pd.sha256,
                   pd.file_size, pd.page_count
            FROM policy_versions pv
            JOIN policy_documents pd ON pd.id = pv.policy_document_id
            WHERE pv.id = ?
            """,
            (version_id,),
        ).fetchone()
    if row is None:
        raise LookupError(f"Policy version {version_id} not found.")
    return dict(row)


# ===========================================================================
# Policy extractions (AI-extracted structured facts)
# ===========================================================================

def save_policy_extraction(
    *,
    policy_version_id: int,
    policy_number: str | None,
    named_insured: str | None,
    effective_from: str | None,
    effective_to: str | None,
    vehicle_year: str | None,
    vehicle_make_model: str | None,
    vehicle_vin: str | None,
    vehicle_plate: str | None,
    coverage_types: list[str],
    coverage_limit: float | None,
    deductible: float | None,
    valuation_basis: str | None,
    exclusions: list[str],
    required_documents: list[str],
    territory: str | None,
    permitted_use: str | None,
    raw_extraction_json: str,
    database_path: Path = DATABASE_PATH,
) -> int:
    """Insert extracted policy facts. Returns extraction ID."""
    timestamp = utc_now()
    with database_connection(database_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO policy_extractions (
                policy_version_id, policy_number, named_insured,
                effective_from, effective_to,
                vehicle_year, vehicle_make_model, vehicle_vin, vehicle_plate,
                coverage_types_json, coverage_limit, deductible, valuation_basis,
                exclusions_json, required_documents_json,
                territory, permitted_use,
                raw_extraction_json, extractor_model, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                policy_version_id, policy_number, named_insured,
                effective_from, effective_to,
                vehicle_year, vehicle_make_model, vehicle_vin, vehicle_plate,
                encode_json(coverage_types), coverage_limit, deductible, valuation_basis,
                encode_json(exclusions), encode_json(required_documents),
                territory, permitted_use,
                raw_extraction_json, "qwen3.5:4b", timestamp,
            ),
        )
        extraction_id = int(cursor.lastrowid)
        _insert_audit_event(
            connection,
            event_type="POLICY_EXTRACTION_SAVED",
            entity_type="policy_extraction",
            entity_id=extraction_id,
            details={"policy_version_id": policy_version_id},
        )
    return extraction_id


def get_policy_extraction(
    policy_version_id: int,
    database_path: Path = DATABASE_PATH,
) -> dict[str, Any] | None:
    """Return the latest extraction for a policy version, decoded."""
    with database_connection(database_path) as connection:
        row = connection.execute(
            """
            SELECT * FROM policy_extractions
            WHERE policy_version_id = ?
            ORDER BY id DESC LIMIT 1
            """,
            (policy_version_id,),
        ).fetchone()
    if row is None:
        return None
    result = dict(row)
    result["coverage_types"]     = decode_json(result.pop("coverage_types_json"))
    result["exclusions"]         = decode_json(result.pop("exclusions_json"))
    result["required_documents"] = decode_json(result.pop("required_documents_json"))
    return result


# ===========================================================================
# Claims
# ===========================================================================

def create_claim(
    *,
    raw_claim_text: str,
    loss_date: str | None = None,
    policy_version_id: int | None = None,
    organization_id: int = DEFAULT_ORG_ID,
    database_path: Path = DATABASE_PATH,
) -> int:
    """Create a claim record in DRAFT state. Returns claim ID."""
    cleaned = raw_claim_text.strip()
    if not cleaned:
        raise ValueError("Claim text cannot be empty.")
    initialize_database(database_path)
    timestamp = utc_now()
    with database_connection(database_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO claims
                (organization_id, policy_version_id, loss_date,
                 raw_claim_text, processing_status, created_at, updated_at)
            VALUES (?, ?, ?, ?, 'DRAFT', ?, ?)
            """,
            (organization_id, policy_version_id, loss_date,
             cleaned, timestamp, timestamp),
        )
        claim_id = int(cursor.lastrowid)
        _insert_audit_event(
            connection,
            event_type="CLAIM_CREATED",
            entity_type="claim",
            entity_id=claim_id,
            details={"loss_date": loss_date, "policy_version_id": policy_version_id},
        )
    return claim_id


def update_claim_status(
    claim_id: int,
    status: str,
    database_path: Path = DATABASE_PATH,
) -> None:
    allowed = {
        "DRAFT", "SUBMITTED", "VALIDATING", "EXTRACTING",
        "RETRIEVING_POLICY_EVIDENCE", "COMPARING",
        "LIKELY_COVERED", "LIKELY_NOT_COVERED", "REVIEW_REQUIRED",
        "HUMAN_REVIEWED", "CLOSED", "FAILED_SAFE", "FAILED",
    }
    if status not in allowed:
        raise ValueError(f"Unknown claim status: {status!r}")
    with database_connection(database_path) as connection:
        _require_row(connection, "claims", claim_id)
        connection.execute(
            "UPDATE claims SET processing_status = ?, updated_at = ? WHERE id = ?",
            (status, utc_now(), claim_id),
        )
        _insert_audit_event(
            connection,
            event_type="CLAIM_STATUS_CHANGED",
            entity_type="claim",
            entity_id=claim_id,
            details={"new_status": status},
        )


def set_claim_policy_version(
    claim_id: int,
    policy_version_id: int,
    database_path: Path = DATABASE_PATH,
) -> None:
    with database_connection(database_path) as connection:
        connection.execute(
            "UPDATE claims SET policy_version_id = ?, updated_at = ? WHERE id = ?",
            (policy_version_id, utc_now(), claim_id),
        )


def get_claim(
    claim_id: int,
    database_path: Path = DATABASE_PATH,
) -> dict[str, Any]:
    with database_connection(database_path) as connection:
        row = connection.execute(
            "SELECT * FROM claims WHERE id = ?", (claim_id,)
        ).fetchone()
    if row is None:
        raise LookupError(f"Claim {claim_id} not found.")
    return dict(row)


def list_claims(
    organization_id: int = DEFAULT_ORG_ID,
    database_path: Path = DATABASE_PATH,
) -> list[dict[str, Any]]:
    with database_connection(database_path) as connection:
        rows = connection.execute(
            """
            SELECT c.id, c.claim_reference, c.loss_date,
                   c.processing_status, c.created_at,
                   pv.policy_name
            FROM claims c
            LEFT JOIN policy_versions pv ON pv.id = c.policy_version_id
            WHERE c.organization_id = ?
            ORDER BY c.id DESC
            """,
            (organization_id,),
        ).fetchall()
    return [dict(r) for r in rows]


# ===========================================================================
# Claim documents
# ===========================================================================

def attach_claim_document(
    *,
    claim_id: int,
    original_filename: str,
    stored_path: str,
    mime_type: str,
    file_size: int,
    sha256: str,
    database_path: Path = DATABASE_PATH,
) -> int:
    timestamp = utc_now()
    with database_connection(database_path) as connection:
        _require_row(connection, "claims", claim_id)
        cursor = connection.execute(
            """
            INSERT INTO claim_documents
                (claim_id, original_filename, stored_path,
                 mime_type, file_size, sha256, version_number, created_at)
            VALUES (?, ?, ?, ?, ?, ?, 1, ?)
            """,
            (claim_id, original_filename, stored_path,
             mime_type, file_size, sha256, timestamp),
        )
        doc_id = int(cursor.lastrowid)
        _insert_audit_event(
            connection,
            event_type="CLAIM_DOCUMENT_ATTACHED",
            entity_type="claim_document",
            entity_id=doc_id,
            details={"claim_id": claim_id, "mime_type": mime_type},
        )
    return doc_id


# ===========================================================================
# Decisions
# ===========================================================================

def save_decision(
    *,
    claim_id: int,
    policy_version_id: int | None,
    outcome: str,
    eligible_gross: float | None,
    deductible_applied: float | None,
    estimated_net: float | None,
    reasons: list[str],
    missing_documents: list[str],
    passed_checks: list[str],
    failed_checks: list[str],
    citations: list[str],
    database_path: Path = DATABASE_PATH,
) -> int:
    allowed_outcomes = {"LIKELY_COVERED", "LIKELY_NOT_COVERED", "REVIEW_REQUIRED"}
    if outcome not in allowed_outcomes:
        raise ValueError(f"Invalid outcome: {outcome!r}")

    claim_status = (
        "REVIEW_REQUIRED" if outcome == "REVIEW_REQUIRED"
        else "LIKELY_COVERED" if outcome == "LIKELY_COVERED"
        else "LIKELY_NOT_COVERED"
    )

    timestamp = utc_now()
    with database_connection(database_path) as connection:
        _require_row(connection, "claims", claim_id)
        cursor = connection.execute(
            """
            INSERT INTO decisions (
                claim_id, policy_version_id, outcome,
                eligible_gross, deductible_applied, estimated_net,
                reasons_json, missing_documents_json,
                passed_checks_json, failed_checks_json,
                citations_json, rule_version, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                claim_id, policy_version_id, outcome,
                eligible_gross, deductible_applied, estimated_net,
                encode_json(reasons), encode_json(missing_documents),
                encode_json(passed_checks), encode_json(failed_checks),
                encode_json(citations), RULE_VERSION, timestamp,
            ),
        )
        decision_id = int(cursor.lastrowid)

        connection.execute(
            "UPDATE claims SET processing_status = ?, updated_at = ? WHERE id = ?",
            (claim_status, timestamp, claim_id),
        )

        # Open human review queue entry when needed
        if outcome == "REVIEW_REQUIRED":
            existing = connection.execute(
                "SELECT id FROM human_reviews WHERE claim_id = ? AND review_status = 'OPEN'",
                (claim_id,),
            ).fetchone()
            if existing is None:
                connection.execute(
                    "INSERT INTO human_reviews (claim_id, review_status, created_at) VALUES (?, 'OPEN', ?)",
                    (claim_id, timestamp),
                )

        _insert_audit_event(
            connection,
            event_type="DECISION_SAVED",
            entity_type="decision",
            entity_id=decision_id,
            details={"outcome": outcome, "rule_version": RULE_VERSION},
        )
    return decision_id


def get_latest_decision(
    claim_id: int,
    database_path: Path = DATABASE_PATH,
) -> dict[str, Any] | None:
    with database_connection(database_path) as connection:
        row = connection.execute(
            "SELECT * FROM decisions WHERE claim_id = ? ORDER BY id DESC LIMIT 1",
            (claim_id,),
        ).fetchone()
    if row is None:
        return None
    result = dict(row)
    result["reasons"]            = decode_json(result.pop("reasons_json"))
    result["missing_documents"]  = decode_json(result.pop("missing_documents_json"))
    result["passed_checks"]      = decode_json(result.pop("passed_checks_json"))
    result["failed_checks"]      = decode_json(result.pop("failed_checks_json"))
    result["citations"]          = decode_json(result.pop("citations_json"))
    return result


def delete_policy_version(
    version_id: int,
    database_path: Path = DATABASE_PATH,
) -> None:
    """
    Delete a policy version and its extraction record.

    Safety rules:
    - Cannot delete an ACTIVE_AND_VERIFIED policy.
    - Cannot delete if any claim references this version.
    Raises ValueError on safety violation, LookupError if not found.
    """
    with database_connection(database_path) as connection:
        _require_row(connection, "policy_versions", version_id)

        row = connection.execute(
            "SELECT status FROM policy_versions WHERE id = ?",
            (version_id,),
        ).fetchone()

        if row["status"] == "ACTIVE_AND_VERIFIED":
            raise ValueError(
                "Cannot delete an active policy version. "
                "Retire or supersede it first."
            )

        claims_using = connection.execute(
            "SELECT COUNT(*) AS n FROM claims WHERE policy_version_id = ?",
            (version_id,),
        ).fetchone()["n"]

        if claims_using > 0:
            raise ValueError(
                f"Cannot delete: {claims_using} claim(s) reference this policy version."
            )

        connection.execute(
            "DELETE FROM policy_extractions WHERE policy_version_id = ?",
            (version_id,),
        )
        connection.execute(
            "DELETE FROM policy_versions WHERE id = ?",
            (version_id,),
        )
        _insert_audit_event(
            connection,
            event_type="POLICY_VERSION_DELETED",
            entity_type="policy_version",
            entity_id=version_id,
            details={"version_id": version_id},
        )
