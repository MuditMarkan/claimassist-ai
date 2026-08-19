from pathlib import Path
from typing import Literal

from langgraph.graph import END, START, StateGraph
from typing_extensions import NotRequired, TypedDict

from app.business_rules import (
    ClaimData,
    RuleDecision,
    evaluate_claim,
)
from app.claim_analyzer import (
    GroundedClaimExtraction,
    extract_grounded_claim_data,
)
from app.rag.retriever import (
    EvidenceBundle,
    retrieve_claim_evidence,
)
from app.rag.vector_store import CHROMA_DIRECTORY
from app.rag_pipeline import RagPipelineResult
from app.services.evidence_validator import (
    EvidenceValidationResult,
    validate_grounded_extraction,
)
class ClaimWorkflowState(TypedDict):
    """Shared state passed between LangGraph nodes."""

    claim_text: str
    persist_directory: str
    top_k: int

    evidence_bundle: NotRequired[EvidenceBundle | None]
    extraction: NotRequired[GroundedClaimExtraction | None]
    validation: NotRequired[EvidenceValidationResult | None]
    claim: NotRequired[ClaimData | None]
    decision: NotRequired[RuleDecision | None]

    citations: NotRequired[list[str]]
    issues: NotRequired[list[str]]
def retrieve_evidence_node(
    state: ClaimWorkflowState,
) -> dict:
    """Retrieve relevant policy passages from Chroma."""

    print(
        "LangGraph node: retrieve_evidence",
        flush=True,
    )

    try:
        bundle = retrieve_claim_evidence(
            state["claim_text"],
            persist_directory=Path(
                state["persist_directory"]
            ),
            top_k=state["top_k"],
        )
    except Exception as error:
        return {
            "evidence_bundle": None,
            "citations": [],
            "issues": [
                "Policy retrieval failed: "
                f"{type(error).__name__}"
            ],
        }

    if not bundle.has_evidence:
        return {
            "evidence_bundle": bundle,
            "citations": [],
            "issues": [
                "No relevant policy evidence was retrieved."
            ],
        }

    return {
        "evidence_bundle": bundle,
        "citations": bundle.citations,
        "issues": [],
    }

def route_after_retrieval(
    state: ClaimWorkflowState,
) -> Literal["extract_facts", "human_review"]:
    """Continue only when usable policy evidence exists."""

    bundle = state.get("evidence_bundle")
    issues = state.get("issues", [])

    if bundle is None or not bundle.has_evidence or issues:
        return "human_review"

    return "extract_facts"

def extract_facts_node(
    state: ClaimWorkflowState,
) -> dict:
    """Ask Qwen for nullable, source-linked facts."""

    print(
        "LangGraph node: extract_facts",
        flush=True,
    )

    bundle = state.get("evidence_bundle")

    if bundle is None:
        return {
            "extraction": None,
            "issues": [
                "Policy evidence is unavailable."
            ],
        }

    try:
        extraction = extract_grounded_claim_data(
            state["claim_text"],
            bundle.formatted_context,
        )
    except Exception as error:
        return {
            "extraction": None,
            "issues": [
                "Grounded extraction failed: "
                f"{type(error).__name__}"
            ],
        }
    return {
        "extraction": extraction,
        "issues": [],
    }

def route_after_extraction(
    state: ClaimWorkflowState,
) -> Literal["validate_evidence", "human_review"]:
    """Reject invalid or unavailable model extraction."""

    if (
        state.get("extraction") is None
        or state.get("issues")
    ):
        return "human_review"

    return "validate_evidence"

def validate_evidence_node(
    state: ClaimWorkflowState,
) -> dict:
    """Independently verify Qwen’s evidence references."""

    print(
        "LangGraph node: validate_evidence",
        flush=True,
    )

    extraction = state.get("extraction")
    bundle = state.get("evidence_bundle")

    if extraction is None or bundle is None:
        return {
            "validation": None,
            "claim": None,
            "issues": [
                "Evidence validation inputs are unavailable."
            ],
        }

    validation = validate_grounded_extraction(
        extraction,
        bundle,
    )
    return {
        "validation": validation,
        "claim": validation.claim,
        "citations": validation.citations,
        "issues": validation.issues,
    }

def route_after_validation(
    state: ClaimWorkflowState,
) -> Literal["apply_rules", "human_review"]:
    """Send unsupported facts to human review."""

    validation = state.get("validation")
    if (
        validation is None
        or not validation.is_valid
        or validation.claim is None
        or state.get("issues")
    ):
        return "human_review"

    return "apply_rules"

def apply_rules_node(
    state: ClaimWorkflowState,
) -> dict:
    """Apply deterministic Python business rules."""
    print(
        "LangGraph node: apply_rules",
        flush=True,
    )

    claim = state.get("claim")

    if claim is None:
        return {
            "decision": RuleDecision(
                decision="REVIEW",
                reasons=[
                    "Validated claim facts are unavailable."
                ],
                missing_documents=[],
            ),
            "issues": [
                "Validated claim facts are unavailable."
            ],
        }

    decision = evaluate_claim(claim)

    return {
        "decision": decision,
        "issues": [],
    }

def human_review_node(
    state: ClaimWorkflowState,
) -> dict:
    """Produce a safe REVIEW result for later human handling."""

    print(
        "LangGraph node: human_review",
        flush=True,
    )

    reasons = state.get("issues", [])

    if not reasons:
        reasons = [
            "The claim requires human review."
        ]
    return {
        "claim": None,
        "decision": RuleDecision(
            decision="REVIEW",
            reasons=list(dict.fromkeys(reasons)),
            missing_documents=[],
        ),
    }

def build_claim_graph():
    """Build and compile the ClaimAssist workflow."""

    builder = StateGraph(ClaimWorkflowState)

    builder.add_node(
        "retrieve_evidence",
        retrieve_evidence_node,
    )
    builder.add_node(
        "extract_facts",
        extract_facts_node,
    )
    builder.add_node(
        "validate_evidence",
        validate_evidence_node,
    )
    builder.add_node(
        "apply_rules",
        apply_rules_node,
    )
    builder.add_node(
        "human_review",
        human_review_node,
    )

    builder.add_edge(
        START,
        "retrieve_evidence",
    )

    builder.add_conditional_edges(
        "retrieve_evidence",
        route_after_retrieval,
        {
            "extract_facts": "extract_facts",
            "human_review": "human_review",
        },
    )

    builder.add_conditional_edges(
        "extract_facts",
        route_after_extraction,
        {
            "validate_evidence": "validate_evidence",
            "human_review": "human_review",
        },
    )

    builder.add_conditional_edges(
        "validate_evidence",
        route_after_validation,
        {
            "apply_rules": "apply_rules",
            "human_review": "human_review",
        },
    )

    builder.add_edge(
        "apply_rules",
        END,
    )

    builder.add_edge(
        "human_review",
        END,
    )

    return builder.compile()

CLAIM_GRAPH = build_claim_graph()

def run_claim_graph(
    claim_text: str,
    *,
    persist_directory: Path = CHROMA_DIRECTORY,
    top_k: int = 4,
) -> RagPipelineResult:
    """Invoke the compiled LangGraph workflow."""

    if not claim_text.strip():
        return RagPipelineResult(
            claim=None,
            decision=RuleDecision(
                decision="REVIEW",
                reasons=["The claim document is empty."],
                missing_documents=[],
            ),
            citations=[],
            issues=["The claim document is empty."],
            retrieved_evidence_count=0,
        )

    final_state = CLAIM_GRAPH.invoke(
        {
            "claim_text": claim_text,
            "persist_directory": str(
                persist_directory
            ),
            "top_k": top_k,
            "citations": [],
            "issues": [],
        }
    )

    bundle = final_state.get("evidence_bundle")
    decision = final_state.get("decision")

    if decision is None:
        decision = RuleDecision(
            decision="REVIEW",
            reasons=[
                "The workflow ended without a decision."
            ],
            missing_documents=[],
        )

    return RagPipelineResult(
        claim=final_state.get("claim"),
        decision=decision,
        citations=final_state.get(
            "citations",
            [],
        ),
        issues=final_state.get(
            "issues",
            [],
        ),
        retrieved_evidence_count=(
            len(bundle.evidence)
            if bundle is not None
            else 0
        ),
    )

def main() -> None:
    claim_text = """
Claim ID: CLM-GRAPH-001
The customer requests $12,000 for collision repairs.

Submitted documents:
- Claim form
- Police report
"""

    result = run_claim_graph(claim_text)

    print()
    print("LangGraph result:")
    print(result.model_dump_json(indent=2))

if __name__ == "__main__":
    main()