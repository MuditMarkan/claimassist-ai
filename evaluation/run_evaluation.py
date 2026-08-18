import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from app.business_rules import ClaimData, normalize_document_name
from app.pipeline import run_claim_pipeline

DEFAULT_DATASET = Path("evaluation/synthetic_claims.json")
DEFAULT_OUTPUT = Path("evaluation/results/latest.json")

def load_dataset(dataset_path: Path) -> list[dict[str, Any]]:
    """Load and validate the top-level evaluation dataset."""

    with dataset_path.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    if not isinstance(dataset, list):
        raise ValueError("The evaluation dataset must contain a JSON list.")
    if not dataset:
        raise ValueError("The evaluation dataset is empty.")

    return dataset

def normalized_documents(documents: list[str]) -> list[str]:
    """Normalize and sort document names before comparison."""

    return sorted(
        normalize_document_name(document)
        for document in documents
    )

def facts_match(
    actual: ClaimData | None,
    expected: dict[str, Any] | None,
) -> bool:
    """Compare extracted facts with the expected facts."""

    if expected is None:
        return actual is None

    if actual is None:
        return False

    return (
        actual.claim_id == expected["claim_id"]
        and math.isclose(
            actual.claim_amount,
            float(expected["claim_amount"]),
        )
        and math.isclose(
            actual.policy_limit,
            float(expected["policy_limit"]),
        )
        and normalized_documents(actual.required_documents)
        == normalized_documents(expected["required_documents"])
        and normalized_documents(actual.submitted_documents)
        == normalized_documents(expected["submitted_documents"])
    )

def run_case(case: dict[str, Any]) -> dict[str, Any]:
    """Run and score one evaluation case."""

    case_id = case["case_id"]
    category = case["category"]

    print(f"Running {case_id}: {category}", flush=True)

    claim, decision = run_claim_pipeline(case["claim_text"])

    extraction_correct = facts_match(
        claim,
        case["expected_facts"],
    )

    decision_correct = (
        decision.decision == case["expected_decision"]
    )
    missing_documents_correct = (
        normalized_documents(decision.missing_documents)
        == normalized_documents(
            case["expected_missing_documents"]
        )
    )

    case_passed = (
        extraction_correct
        and decision_correct
        and missing_documents_correct
    )

    print(
        f"  Actual decision: {decision.decision} | "
        f"Expected: {case['expected_decision']} | "
        f"{'PASS' if case_passed else 'FAIL'}",
        flush=True,
    )

    return {
        "case_id": case_id,
        "category": category,
        "passed": case_passed,
        "extraction_correct": extraction_correct,
        "decision_correct": decision_correct,
        "missing_documents_correct": missing_documents_correct,
        "expected_decision": case["expected_decision"],
        "actual_decision": decision.decision,
        "actual_reasons": decision.reasons,
        "expected_missing_documents": case[
            "expected_missing_documents"
        ],
        "actual_missing_documents": decision.missing_documents,
        "expected_facts": case["expected_facts"],
        "actual_facts": (
            claim.model_dump(mode="json")
            if claim is not None
            else None
        ),
    }

def percentage(correct: int, total: int) -> float:
    """Calculate a percentage rounded to two decimal places."""
    if total == 0:
        return 0.0

    return round((correct / total) * 100, 2)

def build_report(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Create summary metrics and preserve case-level evidence."""

    total = len(results)
    passed = sum(result["passed"] for result in results)

    extraction_correct = sum(
        result["extraction_correct"]
        for result in results
    )
    decision_correct = sum(
        result["decision_correct"]
        for result in results
    )

    missing_documents_correct = sum(
        result["missing_documents_correct"]
        for result in results
    )

    return {
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "summary": {
            "total_cases": total,
            "passed_cases": passed,
            "failed_cases": total - passed,
            "overall_pass_rate_percent": percentage(
                passed,
                total,
            ),
            "extraction_accuracy_percent": percentage(
                extraction_correct,
                total,
            ),
            "decision_accuracy_percent": percentage(
                decision_correct,
                total,
            ),
            "missing_document_accuracy_percent": percentage(
                missing_documents_correct,
                total,
            ),
        },
        "results": results,
    }

def save_report(
    report: dict[str, Any],
    output_path: Path,
) -> None:
    """Save the complete evaluation report."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the ClaimAssist evaluation dataset."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Path to the synthetic evaluation dataset.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path where the JSON report will be saved.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first specified number of cases.",
    )

    return parser.parse_args()

def main() -> None:
    args = parse_arguments()
    dataset = load_dataset(args.dataset)

    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be at least 1.")
        dataset = dataset[: args.limit]

    results = [
        run_case(case)
        for case in dataset
    ]

    report = build_report(results)
    save_report(report, args.output)

    summary = report["summary"]

    print()
    print("ClaimAssist evaluation complete")
    print(f"Cases: {summary['total_cases']}")
    print(
        "Passed: "
        f"{summary['passed_cases']}/"
        f"{summary['total_cases']}"
    )
    print(
        "Extraction accuracy: "
        f"{summary['extraction_accuracy_percent']}%"
    )
    print(
        "Decision accuracy: "
        f"{summary['decision_accuracy_percent']}%"
    )
    print(f"Report saved to: {args.output}")
if __name__ == "__main__":
    main()