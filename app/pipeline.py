from pathlib import Path

from pydantic import ValidationError

from app.business_rules import (
    ClaimData,
    RuleDecision,
    evaluate_claim,
)
from app.claim_analyzer import extract_claim_data
from app.database import DATABASE_PATH
from app.repositories import (
    create_claim,
    mark_claim_failed,
    mark_claim_processing,
    save_decision,
    save_extracted_facts,
)

def create_review_decision(reason: str) -> RuleDecision:
    """Create a safe deterministic REVIEW result."""

    return RuleDecision(
        decision="REVIEW",
        reasons=[reason],
        missing_documents=[],
    )

def run_claim_pipeline(
    document_text: str,
) -> tuple[ClaimData | None, RuleDecision]:
    """
    Run extraction, validation, and deterministic rules.

    This function does not write to the database, making it
    suitable for unit tests.
    """

    try:
        claim = extract_claim_data(document_text)

    except ValidationError as error:
        print("Extraction validation error:")
        print(error)

        return None, create_review_decision(
            "The extracted claim information failed validation."
        )

    except ValueError as error:
        print("Claim input error:")
        print(error)

        return None, create_review_decision(str(error))

    except Exception as error:
        print("Claim-processing error:")
        print(f"{type(error).__name__}: {error}")

        return None, create_review_decision(
            "Claim processing failed and requires human review."
        )

    decision = evaluate_claim(claim)
    return claim, decision
def process_and_persist_claim(
    document_text: str,
    database_path: Path = DATABASE_PATH,
) -> tuple[int, ClaimData | None, RuleDecision]:
    """
    Run a claim and persist its complete processing history.

    The database transaction is not kept open while Qwen runs,
    preventing a long-running model request from locking SQLite.
    """

    claim_record_id = create_claim(
        document_text,
        database_path,
    )

    try:
        mark_claim_processing(
            claim_record_id,
            database_path,
        )

        claim, decision = run_claim_pipeline(document_text)

        if claim is not None:
            save_extracted_facts(
                claim_record_id,
                claim,
                database_path,
            )

        save_decision(
            claim_record_id,
            decision,
            database_path,
        )

        return claim_record_id, claim, decision

    except Exception as error:
        # Preserve the original failure while making a best effort
        # to record that processing did not complete.
        try:
            mark_claim_failed(
                claim_record_id,
                type(error).__name__,
                database_path,
            )
        except Exception:
            pass

        raise

def main() -> None:
    """Run and persist one synthetic ClaimAssist example."""

    document_text = """
Claim ID: CLM-001
The customer is requesting $12,000 for vehicle repairs.
The policy coverage limit is $10,000.

Required documents:
- Claim form
- Police report
- Repair estimate

Submitted documents:
- Claim form
- Police report
"""

    claim_record_id, claim, decision = (
        process_and_persist_claim(document_text)
    )
    print(f"Internal database record: {claim_record_id}")
    print()

    if claim is not None:
        print("Validated extracted facts:")
        print(claim.model_dump_json(indent=2))
        print()

    print("Deterministic ClaimAssist decision:")
    print(decision.model_dump_json(indent=2))
    print()
    print("The claim, decision, and audit events were saved.")

if __name__ == "__main__":
    main()