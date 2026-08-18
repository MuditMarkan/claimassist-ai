import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

DATABASE_PATH = Path("data/claimassist.db")
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS claims (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    claim_reference TEXT,
    raw_claim_text TEXT NOT NULL,
    processing_status TEXT NOT NULL DEFAULT 'RECEIVED'
        CHECK (
            processing_status IN (
                'RECEIVED',
                'PROCESSING',
                'COMPLETED',
                'REVIEW',
                'FAILED'
            )
        ),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS claim_documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    claim_id INTEGER NOT NULL,
    original_filename TEXT NOT NULL,
    stored_path TEXT NOT NULL,
    mime_type TEXT,
    file_size INTEGER NOT NULL,
    sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (claim_id)
        REFERENCES claims(id)
        ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS extracted_facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    claim_id INTEGER NOT NULL UNIQUE,
    extracted_claim_reference TEXT NOT NULL,
    claim_amount REAL NOT NULL,
    policy_limit REAL NOT NULL,
    required_documents_json TEXT NOT NULL,
    submitted_documents_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (claim_id)
        REFERENCES claims(id)
        ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    claim_id INTEGER NOT NULL,
    decision TEXT NOT NULL
        CHECK (decision IN ('APPROVE', 'PEND', 'REVIEW')),
    reasons_json TEXT NOT NULL,
    missing_documents_json TEXT NOT NULL,
    rule_version TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (claim_id)
        REFERENCES claims(id)
        ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS human_reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    claim_id INTEGER NOT NULL,
    review_status TEXT NOT NULL DEFAULT 'OPEN'
        CHECK (review_status IN ('OPEN', 'RESOLVED')),
    reviewer_action TEXT,
    reviewer_notes TEXT,
    corrected_facts_json TEXT,
    created_at TEXT NOT NULL,
    resolved_at TEXT,
    FOREIGN KEY (claim_id)
        REFERENCES claims(id)
        ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS policy_documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    policy_name TEXT NOT NULL,
    policy_version TEXT NOT NULL,
    original_filename TEXT NOT NULL,
    stored_path TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    index_status TEXT NOT NULL DEFAULT 'NOT_INDEXED'
        CHECK (
            index_status IN (
                'NOT_INDEXED',
                'INDEXING',
                'INDEXED',
                'FAILED'
            )
        ),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS workflow_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    claim_id INTEGER NOT NULL,
    workflow_status TEXT NOT NULL,
    current_node TEXT,
    error_message TEXT,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    FOREIGN KEY (claim_id)
        REFERENCES claims(id)
        ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS evaluation_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    total_cases INTEGER NOT NULL,
    passed_cases INTEGER NOT NULL,
    failed_cases INTEGER NOT NULL,
    extraction_accuracy REAL NOT NULL,
    decision_accuracy REAL NOT NULL,
    report_path TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS audit_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    entity_id INTEGER,
    details_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_claims_status
    ON claims(processing_status);

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

def open_connection(
    database_path: Path = DATABASE_PATH,
) -> sqlite3.Connection:
    """Open a configured SQLite connection."""

    database_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    connection = sqlite3.connect(database_path)
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
    """Create every required table and index."""

    with database_connection(database_path) as connection:
        connection.executescript(SCHEMA_SQL)

def list_database_tables(
    database_path: Path = DATABASE_PATH,
) -> list[str]:
    """Return application table names for verification."""

    with database_connection(database_path) as connection:
        rows = connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
              AND name NOT LIKE 'sqlite_%'
            ORDER BY name
            """
        ).fetchall()

    return [row["name"] for row in rows]
def main() -> None:
    initialize_database()

    print(f"Database initialized: {DATABASE_PATH.resolve()}")
    print("Tables:")

    for table_name in list_database_tables():
        print(f"- {table_name}")

if __name__ == "__main__":
    main()
