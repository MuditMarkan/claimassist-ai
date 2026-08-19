import sqlite3
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, ConfigDict

from app.business_rules import (
    ClaimData,
    RuleDecision,
    evaluate_claim,
)
from app.rag_pipeline import RagPipelineResult
from app.workflow.claim_graph import (
    ClaimWorkflowState,
    apply_rules_node,
    extract_facts_node,
    retrieve_evidence_node,
    route_after_extraction,
    route_after_retrieval,
    route_after_validation,
    validate_evidence_node,
)

CHECKPOINT_PATH = Path(
    "data/langgraph_checkpoints.sqlite"
)
class WorkflowInvocation(BaseModel):
    """Result returned when starting or resuming a workflow."""

    model_config = ConfigDict(extra="forbid")

    thread_id: str
    status: Literal[
        "COMPLETED",
        "WAITING_FOR_REVIEW",
    ]
    result: RagPipelineResult | None
    review_request: dict[str, Any] | None

def human_review_interrupt_node(
    state: ClaimWorkflowState,
) -> dict:
    """
    Pause execution and wait for a human response.

    Code before interrupt() performs no database writes, so safely
    running it again when the graph resumes is idempotent.
    """

    review_payload = {
        "type": "CLAIM_REVIEW_REQUIRED",
        "issues": state.get(
            "issues",
            ["The claim requires human review."],
        ),
        "citations": state.get("citations", []),
        "instructions": (
            "Submit corrected validated facts, or keep the "
            "claim in REVIEW."
        ),
        "allowed_actions": [
            "submit_corrected_facts",
            "keep_review",
        ],
    }

    response = interrupt(review_payload)

    if not isinstance(response, dict):
        return {
            "claim": None,
            "decision": RuleDecision(
                decision="REVIEW",
                reasons=[
                    "The human-review response was invalid."
                ],
                missing_documents=[],
            ),
            "issues": [
                "The human-review response was invalid."
            ],
        }

    action = response.get("action")

    if action == "keep_review":
        reviewer_notes = str(
            response.get("notes", "")
        ).strip()

        reasons = list(
            state.get(
                "issues",
                ["The claim requires human review."],
            )
        )

        if reviewer_notes:
            reasons.append(
                f"Reviewer note: {reviewer_notes}"
            )
        return {
            "claim": None,
            "decision": RuleDecision(
                decision="REVIEW",
                reasons=list(dict.fromkeys(reasons)),
                missing_documents=[],
            ),
            "issues": reasons,
        }

    if action == "submit_corrected_facts":
        corrected_data = response.get("corrected_claim")

        try:
            corrected_claim = ClaimData.model_validate(
                corrected_data
            )
        except Exception:
            return {
                "claim": None,
                "decision": RuleDecision(
                    decision="REVIEW",
                    reasons=[
                        "The reviewer-submitted claim facts "
                        "failed validation."
                    ],
                    missing_documents=[],
                ),
                "issues": [
                    "The reviewer-submitted claim facts "
                    "failed validation."
                ],
            }

        corrected_decision = evaluate_claim(
            corrected_claim
        )

        return {
            "claim": corrected_claim,
            "decision": corrected_decision,
            "issues": [],
        }
    return {
        "claim": None,
        "decision": RuleDecision(
            decision="REVIEW",
            reasons=[
                "The reviewer selected an unsupported action."
            ],
            missing_documents=[],
        ),
        "issues": [
            "The reviewer selected an unsupported action."
        ],
    }

def build_resumable_graph(
    checkpointer: SqliteSaver,
):
    """Build the workflow using a durable SQLite checkpointer."""

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
        human_review_interrupt_node,
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

    builder.add_edge("apply_rules", END)
    builder.add_edge("human_review", END)

    return builder.compile(
        checkpointer=checkpointer
    )

class ResumableClaimWorkflow:
    """Manage graph lifetime, checkpoints, starts, and resumes."""

    def __init__(
        self,
        checkpoint_path: Path = CHECKPOINT_PATH,
    ) -> None:
        checkpoint_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.connection = sqlite3.connect(
            str(checkpoint_path),
            check_same_thread=False,
        )

        self.checkpointer = SqliteSaver(
            self.connection
        )

        self.graph = build_resumable_graph(
            self.checkpointer
        )
    def close(self) -> None:
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(
        self,
        exception_type,
        exception_value,
        traceback,
    ) -> None:
        self.close()
    def start(
        self,
        claim_text: str,
        *,
        persist_directory: Path,
        top_k: int = 4,
        thread_id: str | None = None,
    ) -> WorkflowInvocation:
        """Start a new resumable claim workflow."""

        if not claim_text.strip():
            raise ValueError(
                "The claim document is empty."
            )
        active_thread_id = (
            thread_id or str(uuid4())
        )

        config = {
            "configurable": {
                "thread_id": active_thread_id,
            }
        }

        state = self.graph.invoke(
            {
                "claim_text": claim_text,
                "persist_directory": str(
                    persist_directory
                ),
                "top_k": top_k,
                "citations": [],
                "issues": [],
            },
            config=config,
        )

        return self._build_invocation(
            active_thread_id,
            state,
        )

    def resume(
        self,
        thread_id: str,
        response: dict[str, Any],
    ) -> WorkflowInvocation:
        """Resume a workflow using the same persistent thread ID."""

        if not thread_id.strip():
            raise ValueError(
                "A thread ID is required."
            )

        config = {
            "configurable": {
                "thread_id": thread_id,
            }
        }
        state = self.graph.invoke(
            Command(resume=response),
            config=config,
        )

        return self._build_invocation(
            thread_id,
            state,
        )

    def _build_invocation(
        self,
        thread_id: str,
        state: dict,
    ) -> WorkflowInvocation:
        interrupts = state.get(
            "__interrupt__",
            [],
        )

        if interrupts:
            first_interrupt = interrupts[0]
            payload = getattr(
                first_interrupt,
                "value",
                first_interrupt,
            )

            return WorkflowInvocation(
                thread_id=thread_id,
                status="WAITING_FOR_REVIEW",
                result=None,
                review_request=payload,
            )

        decision = state.get("decision")

        if decision is None:
            decision = RuleDecision(
                decision="REVIEW",
                reasons=[
                    "The workflow completed without "
                    "a decision."
                ],
                missing_documents=[],
            )

        bundle = state.get("evidence_bundle")

        result = RagPipelineResult(
            claim=state.get("claim"),
            decision=decision,
            citations=state.get(
                "citations",
                [],
            ),
            issues=state.get(
                "issues",
                [],
            ),
            retrieved_evidence_count=(
                len(bundle.evidence)
                if bundle is not None
                else 0
            ),
        )

        return WorkflowInvocation(
            thread_id=thread_id,
            status="COMPLETED",
            result=result,
            review_request=None,
        )