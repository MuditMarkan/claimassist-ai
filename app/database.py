"""
ClaimAssist database — schema, connection helpers, initialisation.

Tables
------
organizations           — top-level tenant
users                   — linked to one organization (MVP: one default user)
policy_documents        — immutable uploaded file record
policy_versions         — versioned policy entity with full lifecycle
policy_extractions      — structured facts extracted from a policy version
claims                  — claim submissions linked to a policy version
claim_documents         — files attached to a claim (with versioning)
decisions               — comparison results per analysis run
human_reviews           — adjuster review queue
workflow_runs           — LangGraph checkpoint tracking
audit_events            — append-only audit log
evaluation_runs         — synthetic evaluation harness results
"""

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATABASE_PATH      = Path("data/claimassist.db")
UPLOADS_DIRECTORY  = Path("data/uploads")
POLICIES_DIRECTORY = Path("data/uploads/policies")
EVIDENCE_DIRECTORY = Path("data/uploads/evidence")

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

SCHEMA_SQL = """
-- ── Organizations ────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS organizations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL,
    slug        TEXT    NOT NULL UNIQUE,
    created_at  TEXT    NOT NULL,
    updated_at  TEXT    NOT NULL
);

-- Seed a default organization so the MVP works without auth.
INSERT OR IGNORE INTO organizations (id, name, slug, created_at, updated_at)
VALUES (1, 'Default Organization', 'default',
        datetime('now'), datetime('now'));

-- ── Policy documents (immutable uploaded file records) ───────────────────
CREATE TABLE IF NOT EXISTS policy_documents (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    organization_id     INTEGER NOT NULL DEFAULT 1,
    original_filename   TEXT    NOT NULL,
    stored_path         TEXT    NOT NULL,
    sha256              TEXT    NOT NULL,
    file_size           INTEGER NOT NULL DEFAULT 0,
    page_count          INTEGER NOT NULL DEFAULT 0,
    created_at          TEXT    NOT NULL,
    FOREIGN KEY (organization_id) REFERENCES organizations(id)
);

-- ── Policy versions (one per upload, full lifecycle) ────────────────────
CREATE TABLE IF NOT EXISTS policy_versions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    organization_id     INTEGER NOT NULL DEFAULT 1,
    policy_document_id  INTEGER NOT NULL,
    policy_name         TEXT    NOT NULL,
    version_label       TEXT    NOT NULL DEFAULT 'v1',
    effective_from      TEXT,
    effective_to        TEXT,
    status              TEXT    NOT NULL DEFAULT 'UPLOADED'
        CHECK (status IN (
            'UPLOADED',
            'VALIDATING',
            'EXTRACTION_IN_PROGRESS',
            'NEEDS_VERIFICATION',
            'ACTIVE_AND_VERIFIED',
            'SUPERSEDED',
            'RETIRED',
            'FAILED_SAFE'
        )),
    index_status        TEXT    NOT NULL DEFAULT 'NOT_INDEXED'
        CHECK (index_status IN (
            'NOT_INDEXED',
            'INDEXING',
            'INDEXED',
            'FAILED'
        )),
    created_at          TEXT    NOT NULL,
    updated_at          TEXT    NOT NULL,
    FOREIGN KEY (organization_id)     REFERENCES organizations(id),
    FOREIGN KEY (policy_document_id)  REFERENCES policy_documents(id)
);

-- ── Policy extractions (structured facts extracted by AI) ────────────────
CREATE TABLE IF NOT EXISTS policy_extractions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    policy_version_id   INTEGER NOT NULL,
    -- Core identity
    policy_number       TEXT,
    named_insured       TEXT,
    effective_from      TEXT,
    effective_to        TEXT,
    -- Vehicle
    vehicle_year        TEXT,
    vehicle_make_model  TEXT,
    vehicle_vin         TEXT,
    vehicle_plate       TEXT,
    -- Coverage
    coverage_types_json TEXT    NOT NULL DEFAULT '[]',
    -- Financials
    coverage_limit      REAL,
    deductible          REAL,
    valuation_basis     TEXT,
    -- Rules
    exclusions_json         TEXT NOT NULL DEFAULT '[]',
    required_documents_json TEXT NOT NULL DEFAULT '[]',
    territory               TEXT,
    permitted_use           TEXT,
    -- Raw model output for auditing
    raw_extraction_json TEXT,
    extractor_model     TEXT    NOT NULL DEFAULT 'qwen3.5:4b',
    created_at          TEXT    NOT NULL,
    FOREIGN KEY (policy_version_id) REFERENCES policy_versions(id)
);

-- ── Claims ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS claims (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    organization_id     INTEGER NOT NULL DEFAULT 1,
    policy_version_id   INTEGER,               -- resolved on submit
    claim_reference     TEXT,
    loss_date           TEXT,
    raw_claim_text      TEXT    NOT NULL,
    processing_status   TEXT    NOT NULL DEFAULT 'DRAFT'
        CHECK (processing_status IN (
            'DRAFT',
            'SUBMITTED',
            'VALIDATING',
            'EXTRACTING',
            'RETRIEVING_POLICY_EVIDENCE',
            'COMPARING',
            'LIKELY_COVERED',
            'LIKELY_NOT_COVERED',
            'REVIEW_REQUIRED',
            'HUMAN_REVIEWED',
            'CLOSED',
            'FAILED_SAFE',
            'FAILED'
        )),
    created_at          TEXT    NOT NULL,
    updated_at          TEXT    NOT NULL,
    FOREIGN KEY (organization_id)   REFERENCES organizations(id),
    FOREIGN KEY (policy_version_id) REFERENCES policy_versions(id)
);

-- ── Claim documents ───────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS claim_documents (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    claim_id            INTEGER NOT NULL,
    original_filename   TEXT    NOT NULL,
    stored_path         TEXT    NOT NULL,
    mime_type           TEXT,
    file_size           INTEGER NOT NULL DEFAULT 0,
    sha256              TEXT    NOT NULL,
    version_number      INTEGER NOT NULL DEFAULT 1,
    parent_document_id  INTEGER,               -- self-ref for corrections
    created_at          TEXT    NOT NULL,
    FOREIGN KEY (claim_id)         REFERENCES claims(id) ON DELETE CASCADE,
    FOREIGN KEY (parent_document_id) REFERENCES claim_documents(id)
);

-- ── Decisions (comparison results) ───────────────────────────────────────
CREATE TABLE IF NOT EXISTS decisions (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    claim_id                INTEGER NOT NULL,
    policy_version_id       INTEGER,
    outcome                 TEXT    NOT NULL
        CHECK (outcome IN (
            'LIKELY_COVERED',
            'LIKELY_NOT_COVERED',
            'REVIEW_REQUIRED'
        )),
    -- Financials
    eligible_gross          REAL,
    deductible_applied      REAL,
    estimated_net           REAL,
    -- Detail lists (JSON arrays)
    reasons_json            TEXT    NOT NULL DEFAULT '[]',
    missing_documents_json  TEXT    NOT NULL DEFAULT '[]',
    passed_checks_json      TEXT    NOT NULL DEFAULT '[]',
    failed_checks_json      TEXT    NOT NULL DEFAULT '[]',
    citations_json          TEXT    NOT NULL DEFAULT '[]',
    rule_version            TEXT    NOT NULL DEFAULT '2.0.0',
    created_at              TEXT    NOT NULL,
    FOREIGN KEY (claim_id)          REFERENCES claims(id) ON DELETE CASCADE,
    FOREIGN KEY (policy_version_id) REFERENCES policy_versions(id)
);

-- ── Human reviews ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS human_reviews (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    claim_id            INTEGER NOT NULL,
    review_status       TEXT    NOT NULL DEFAULT 'OPEN'
        CHECK (review_status IN ('OPEN', 'RESOLVED')),
    reviewer_action     TEXT,
    reviewer_notes      TEXT,
    corrected_facts_json TEXT,
    created_at          TEXT    NOT NULL,
    resolved_at         TEXT,
    FOREIGN KEY (claim_id) REFERENCES claims(id) ON DELETE CASCADE
);

-- ── Workflow runs ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS workflow_runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    claim_id        INTEGER NOT NULL,
    thread_id       TEXT,
    workflow_status TEXT    NOT NULL,
    current_node    TEXT,
    error_message   TEXT,
    started_at      TEXT    NOT NULL,
    completed_at    TEXT,
    FOREIGN KEY (claim_id) REFERENCES claims(id) ON DELETE CASCADE
);

-- ── Audit events (append-only) ───────────────────────────────────────────
CREATE TABLE IF NOT EXISTS audit_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type  TEXT    NOT NULL,
    entity_type TEXT    NOT NULL,
    entity_id   INTEGER,
    actor       TEXT    NOT NULL DEFAULT 'system',
    details_json TEXT   NOT NULL DEFAULT '{}',
    created_at  TEXT    NOT NULL
);

-- ── Evaluation runs ───────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS evaluation_runs (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    total_cases         INTEGER NOT NULL,
    passed_cases        INTEGER NOT NULL,
    failed_cases        INTEGER NOT NULL,
    extraction_accuracy REAL    NOT NULL,
    decision_accuracy   REAL    NOT NULL,
    report_path         TEXT    NOT NULL,
    created_at          TEXT    NOT NULL
);

-- ── Indexes ───────────────────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_policy_versions_org
    ON policy_versions(organization_id);
CREATE INDEX IF NOT EXISTS idx_policy_versions_status
    ON policy_versions(status);
CREATE INDEX IF NOT EXISTS idx_policy_extractions_version
    ON policy_extractions(policy_version_id);
CREATE INDEX IF NOT EXISTS idx_claims_org
    ON claims(organization_id);
CREATE INDEX IF NOT EXISTS idx_claims_status
    ON claims(processing_status);
CREATE INDEX IF NOT EXISTS idx_claims_policy_version
    ON claims(policy_version_id);
CREATE INDEX IF NOT EXISTS idx_claim_documents_claim
    ON claim_documents(claim_id);
CREATE INDEX IF NOT EXISTS idx_decisions_claim
    ON decisions(claim_id);
CREATE INDEX IF NOT EXISTS idx_reviews_status
    ON human_reviews(review_status);
CREATE INDEX IF NOT EXISTS idx_workflow_claim
    ON workflow_runs(claim_id);
CREATE INDEX IF NOT EXISTS idx_audit_entity
    ON audit_events(entity_type, entity_id);
"""

# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------

def open_connection(
    database_path: Path = DATABASE_PATH,
) -> sqlite3.Connection:
    """Open a configured SQLite connection."""
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(database_path))
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA busy_timeout = 5000")
    return connection


@contextmanager
def database_connection(
    database_path: Path = DATABASE_PATH,
) -> Iterator[sqlite3.Connection]:
    """Provide a transaction that commits or rolls back safely."""
    connection = open_connection(database_path)
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def initialize_database(
    database_path: Path = DATABASE_PATH,
) -> None:
    """Create all tables, indexes, seed data, and upload directories."""
    POLICIES_DIRECTORY.mkdir(parents=True, exist_ok=True)
    EVIDENCE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    with database_connection(database_path) as connection:
        connection.executescript(SCHEMA_SQL)


def list_database_tables(
    database_path: Path = DATABASE_PATH,
) -> list[str]:
    """Return application table names for verification."""
    with database_connection(database_path) as connection:
        rows = connection.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name
            """
        ).fetchall()
    return [row["name"] for row in rows]


if __name__ == "__main__":
    initialize_database()
    print(f"Database initialized: {DATABASE_PATH.resolve()}")
    for t in list_database_tables():
        print(f"  - {t}")
