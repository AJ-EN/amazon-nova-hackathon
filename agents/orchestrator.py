from __future__ import annotations

from typing import Callable

from agents.browser_agent import BrowserAutomationAgent
from agents.reasoning_agent import ClinicalReasoningAgent
from agents.retrieval_agent import PayerPolicyRetrievalAgent
from agents.types import PriorAuthWorkflowResult, WorkflowTraceStep
from agents.voice_agent import VoiceIntakeAgent


class PriorAuthOrchestrator:
    """Coordinates the voice-to-submission prior authorization workflow."""

    def __init__(
        self,
        voice_agent: VoiceIntakeAgent | None = None,
        retrieval_agent: PayerPolicyRetrievalAgent | None = None,
        reasoning_agent: ClinicalReasoningAgent | None = None,
        browser_agent: BrowserAutomationAgent | None = None,
    ) -> None:
        self.voice_agent = voice_agent or VoiceIntakeAgent()
        self.retrieval_agent = retrieval_agent or PayerPolicyRetrievalAgent()
        self.reasoning_agent = reasoning_agent or ClinicalReasoningAgent()
        self.browser_agent = browser_agent or BrowserAutomationAgent()

    def run(
        self,
        transcript: str,
        auto_approve: bool = False,
        reviewer_approved: bool | None = None,
        trace_hook: Callable[[WorkflowTraceStep], None] | None = None,
    ) -> PriorAuthWorkflowResult:
        trace: list[WorkflowTraceStep] = []
        result = PriorAuthWorkflowResult(trace=trace)

        def emit(step: str, status: str, detail: str) -> None:
            self._add_trace(
                trace=trace,
                step=step,
                status=status,
                detail=detail,
                trace_hook=trace_hook,
            )

        emit("Voice Intake", "in_progress", "Parsing clinician transcript.")
        extracted = self.voice_agent.ingest(transcript)
        result.extracted_data = extracted
        emit(
            "Voice Intake",
            "completed",
            f"Patient={extracted.patient_name or 'unknown'} Payer={extracted.payer_name}.",
        )

        emit("Eligibility Verification", "in_progress", "Checking required identifiers.")
        if not extracted.member_id:
            emit(
                "Eligibility Verification",
                "failed",
                "Missing member ID from intake transcript.",
            )
            result.next_action = "collect_missing_member_id"
            return result
        emit(
            "Eligibility Verification",
            "completed",
            f"Member ID {extracted.member_id} is present.",
        )

        emit("Clinical Coding", "in_progress", "Mapping ICD-10 and CPT codes.")
        coding = self.reasoning_agent.map_codes(extracted)
        result.coding = coding
        emit(
            "Clinical Coding",
            "completed",
            f"Mapped diagnosis={coding.diagnosis_code}, procedure={coding.procedure_code}.",
        )

        emit("Knowledge Retrieval", "in_progress", "Fetching payer policy criteria.")
        policy = self.retrieval_agent.retrieve(
            payer_name=extracted.payer_name,
            member_id=extracted.member_id,
            procedure_code=coding.procedure_code,
            requested_service=extracted.requested_service,
        )
        result.policy = policy
        emit(
            "Knowledge Retrieval",
            "completed",
            f"Selected policy {policy.policy_id}.",
        )

        emit("Medical Necessity Analysis", "in_progress", "Evaluating policy criteria.")
        necessity = self.reasoning_agent.evaluate_medical_necessity(extracted, coding, policy)
        result.necessity = necessity
        emit(
            "Medical Necessity Analysis",
            "completed",
            f"meets_criteria={necessity.meets_criteria}, denial_risk={necessity.denial_risk_score:.2f}.",
        )

        emit("Form Population", "in_progress", "Building PA submission payload.")
        payload = self.reasoning_agent.build_form_payload(extracted, coding, necessity, policy)
        emit("Form Population", "completed", "Payload built for browser automation.")

        emit("Human Review", "in_progress", "Preparing approval snapshot.")
        review_snapshot = self.browser_agent.generate_review_snapshot(payload)
        approved = reviewer_approved if reviewer_approved is not None else auto_approve
        emit(
            "Human Review",
            "completed",
            f"approved={approved}.",
        )

        emit("Portal Submission", "in_progress", "Executing portal form submission.")
        submission = self.browser_agent.submit(
            payload=payload,
            approved=approved,
            review_snapshot=review_snapshot,
        )
        result.submission = submission
        if submission.status == "submitted":
            emit(
                "Portal Submission",
                "completed",
                f"Submitted successfully with reference {submission.reference}.",
            )
            result.next_action = "notify_clinician"
        elif submission.status == "needs_approval":
            emit(
                "Portal Submission",
                "blocked",
                "Awaiting clinician approval before submission.",
            )
            result.next_action = "human_review_required"
        else:
            emit(
                "Portal Submission",
                "failed",
                submission.message,
            )
            result.next_action = "retry_submission"
        return result

    @staticmethod
    def _add_trace(
        trace: list[WorkflowTraceStep],
        step: str,
        status: str,
        detail: str,
        trace_hook: Callable[[WorkflowTraceStep], None] | None = None,
    ) -> None:
        step_record = WorkflowTraceStep(step=step, status=status, detail=detail)
        trace.append(step_record)
        if trace_hook is not None:
            trace_hook(step_record)
