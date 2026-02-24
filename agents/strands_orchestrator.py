from __future__ import annotations

import json
import os
from typing import Any, Callable, TypeVar, cast

from opentelemetry import trace

from agents.browser_agent import BrowserAutomationAgent
from agents.reasoning_agent import ClinicalReasoningAgent
from agents.retrieval_agent import PayerPolicyRetrievalAgent
from agents.types import (
    CodingResult,
    ExtractedClinicalData,
    NecessityDecision,
    PolicyMatch,
    PriorAuthWorkflowResult,
    SubmissionResult,
    WorkflowTraceStep,
)
from agents.voice_agent import VoiceIntakeAgent

try:
    from strands import Agent, tool as _strands_tool
    from strands.models import BedrockModel

    _STRANDS_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - exercised by runtime fallback.
    Agent = None  # type: ignore[assignment]
    BedrockModel = None  # type: ignore[assignment]
    _strands_tool = None

    _STRANDS_IMPORT_ERROR = exc

_ToolFunc = TypeVar("_ToolFunc", bound=Callable[..., Any])


def _decorate_tool(func: _ToolFunc) -> _ToolFunc:
    if _strands_tool is None:
        return func
    return cast(_ToolFunc, _strands_tool(func))


def strands_available() -> bool:
    return _STRANDS_IMPORT_ERROR is None and Agent is not None and BedrockModel is not None


class _WorkflowTools:
    """Tool wrappers around existing domain agents."""

    def __init__(
        self,
        voice_agent: VoiceIntakeAgent,
        retrieval_agent: PayerPolicyRetrievalAgent,
        reasoning_agent: ClinicalReasoningAgent,
        browser_agent: BrowserAutomationAgent,
    ) -> None:
        self.voice_agent = voice_agent
        self.retrieval_agent = retrieval_agent
        self.reasoning_agent = reasoning_agent
        self.browser_agent = browser_agent

    @_decorate_tool
    def extract_clinical_data(self, transcript: str) -> str:
        """Extract structured patient and clinical details from transcript text."""
        extracted = self.voice_agent.ingest(transcript)
        return json.dumps(extracted.to_dict())

    @_decorate_tool
    def map_clinical_codes(self, extracted_data: dict[str, Any]) -> str:
        """Map ICD-10 and CPT codes from extracted clinical context."""
        extracted = ExtractedClinicalData(**extracted_data)
        coding = self.reasoning_agent.map_codes(extracted)
        return json.dumps(coding.to_dict())

    @_decorate_tool
    def retrieve_payer_policy(
        self,
        payer_name: str,
        member_id: str,
        procedure_code: str,
        requested_service: str,
    ) -> str:
        """Retrieve payer policy criteria using KB retrieval with local fallback."""
        policy = self.retrieval_agent.retrieve(
            payer_name=payer_name,
            member_id=member_id,
            procedure_code=procedure_code,
            requested_service=requested_service,
        )
        return json.dumps(policy.to_dict())

    @_decorate_tool
    def evaluate_necessity(
        self,
        extracted_data: dict[str, Any],
        coding_data: dict[str, Any],
        policy_data: dict[str, Any],
    ) -> str:
        """Evaluate medical necessity against payer criteria."""
        extracted = ExtractedClinicalData(**extracted_data)
        coding = CodingResult(**coding_data)
        policy = PolicyMatch(**policy_data)
        necessity = self.reasoning_agent.evaluate_medical_necessity(extracted, coding, policy)
        return json.dumps(necessity.to_dict())

    @_decorate_tool
    def build_submission_payload(
        self,
        extracted_data: dict[str, Any],
        coding_data: dict[str, Any],
        necessity_data: dict[str, Any],
        policy_data: dict[str, Any],
    ) -> str:
        """Build portal submission payload from workflow outputs."""
        extracted = ExtractedClinicalData(**extracted_data)
        coding = CodingResult(**coding_data)
        necessity = NecessityDecision(**necessity_data)
        policy = PolicyMatch(**policy_data)
        payload = self.reasoning_agent.build_form_payload(extracted, coding, necessity, policy)
        return json.dumps(payload)

    @_decorate_tool
    def generate_review_snapshot(self, payload: dict[str, str]) -> str:
        """Generate human-review summary for HITL approval."""
        snapshot = self.browser_agent.generate_review_snapshot(payload)
        return json.dumps({"review_snapshot": snapshot})

    @_decorate_tool
    def submit_form(
        self,
        payload: dict[str, str],
        approved: bool,
        review_snapshot: str,
    ) -> str:
        """Submit the prior-auth form through selected browser mode."""
        submission = self.browser_agent.submit(
            payload=payload,
            approved=approved,
            review_snapshot=review_snapshot,
        )
        return json.dumps(submission.to_dict())


class StrandsPriorAuthOrchestrator:
    """Strands-backed wrapper for prior-auth orchestration."""

    def __init__(
        self,
        voice_agent: VoiceIntakeAgent | None = None,
        retrieval_agent: PayerPolicyRetrievalAgent | None = None,
        reasoning_agent: ClinicalReasoningAgent | None = None,
        browser_agent: BrowserAutomationAgent | None = None,
        model_id: str | None = None,
    ) -> None:
        if not strands_available():
            raise RuntimeError(
                "Strands SDK is unavailable in this environment."
                f" import_error={_STRANDS_IMPORT_ERROR!r}"
            )

        self.voice_agent = voice_agent or VoiceIntakeAgent()
        self.retrieval_agent = retrieval_agent or PayerPolicyRetrievalAgent()
        self.reasoning_agent = reasoning_agent or ClinicalReasoningAgent()
        self.browser_agent = browser_agent or BrowserAutomationAgent()
        self.model_id = model_id or os.getenv("STRANDS_MODEL_ID", "amazon.nova-lite-v1:0")

        self._tools = _WorkflowTools(
            voice_agent=self.voice_agent,
            retrieval_agent=self.retrieval_agent,
            reasoning_agent=self.reasoning_agent,
            browser_agent=self.browser_agent,
        )

        self.intake_stage = self._build_stage_agent(
            name="Voice Intake Agent",
            description="Extracts structured patient and request data from clinician transcript.",
            tools=[self._tools.extract_clinical_data],
        )
        self.coding_stage = self._build_stage_agent(
            name="Clinical Coding Agent",
            description="Maps ICD-10 and CPT codes for the request.",
            tools=[self._tools.map_clinical_codes],
        )
        self.retrieval_stage = self._build_stage_agent(
            name="Policy Retrieval Agent",
            description="Retrieves payer policy criteria from Bedrock KB or local fallback.",
            tools=[self._tools.retrieve_payer_policy],
        )
        self.necessity_stage = self._build_stage_agent(
            name="Necessity Agent",
            description="Evaluates medical necessity and denial risk from policy criteria.",
            tools=[self._tools.evaluate_necessity],
        )
        self.payload_stage = self._build_stage_agent(
            name="Payload Agent",
            description="Builds structured portal payload from workflow artifacts.",
            tools=[self._tools.build_submission_payload],
        )
        self.review_stage = self._build_stage_agent(
            name="Human Review Agent",
            description="Generates human-review snapshot for clinician approval.",
            tools=[self._tools.generate_review_snapshot],
        )
        self.submission_stage = self._build_stage_agent(
            name="Submission Agent",
            description="Submits approved payload using selected browser automation mode.",
            tools=[self._tools.submit_form],
        )

    def _build_stage_agent(
        self,
        name: str,
        description: str,
        tools: list[Any],
    ) -> Any:
        if Agent is None or BedrockModel is None:
            raise RuntimeError("Strands SDK is unavailable in this environment.")
        return Agent(
            model=BedrockModel(model_id=self.model_id),
            tools=tools,
            callback_handler=None,
            name=name,
            description=description,
            record_direct_tool_call=False,
        )

    @staticmethod
    def _extract_tool_text(tool_result: dict[str, Any]) -> str:
        status = str(tool_result.get("status", "")).lower()
        blocks = tool_result.get("content", [])
        text = "\n".join(
            str(block.get("text", ""))
            for block in blocks
            if isinstance(block, dict) and block.get("text")
        ).strip()

        if status != "success":
            raise RuntimeError(text or "Strands tool call failed.")
        if not text:
            raise ValueError("Strands tool call returned empty content.")
        return text

    @classmethod
    def _tool_json(cls, tool_result: dict[str, Any]) -> dict[str, Any]:
        text = cls._extract_tool_text(tool_result)
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Strands tool response was not valid JSON: {text}") from exc
        if not isinstance(parsed, dict):
            raise ValueError("Strands tool response JSON must be an object.")
        return parsed

    def run(
        self,
        transcript: str,
        auto_approve: bool = False,
        reviewer_approved: bool | None = None,
        trace_hook: Callable[[WorkflowTraceStep], None] | None = None,
    ) -> PriorAuthWorkflowResult:
        tracer = trace.get_tracer("priorauth.strands.orchestrator")
        workflow_trace: list[WorkflowTraceStep] = []
        result = PriorAuthWorkflowResult(trace=workflow_trace)

        def emit(step: str, status: str, detail: str) -> None:
            self._add_trace(
                trace=workflow_trace,
                step=step,
                status=status,
                detail=detail,
                trace_hook=trace_hook,
            )

        emit("Voice Intake", "in_progress", "Parsing clinician transcript.")
        with tracer.start_as_current_span("voice_intake") as span:
            span.set_attribute("transcript_length", len(transcript))
            extracted_payload = self._tool_json(
                self.intake_stage.tool.extract_clinical_data(transcript=transcript)
            )
            span.set_attribute("patient_name", str(extracted_payload.get("patient_name", "")))
            span.set_attribute("payer_name", str(extracted_payload.get("payer_name", "")))
        extracted = ExtractedClinicalData(**extracted_payload)
        result.extracted_data = extracted
        emit(
            "Voice Intake",
            "completed",
            f"Patient={extracted.patient_name or 'unknown'} Payer={extracted.payer_name}.",
        )

        emit("Eligibility Verification", "in_progress", "Checking required identifiers.")
        with tracer.start_as_current_span("eligibility_verification") as span:
            member_id_present = bool(extracted.member_id)
            span.set_attribute("member_id_present", member_id_present)
            if not member_id_present:
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
        with tracer.start_as_current_span("clinical_coding") as span:
            coding_payload = self._tool_json(
                self.coding_stage.tool.map_clinical_codes(extracted_data=extracted.to_dict())
            )
            span.set_attribute("diagnosis_code", str(coding_payload.get("diagnosis_code", "")))
            span.set_attribute("procedure_code", str(coding_payload.get("procedure_code", "")))
        coding = CodingResult(**coding_payload)
        result.coding = coding
        emit(
            "Clinical Coding",
            "completed",
            f"Mapped diagnosis={coding.diagnosis_code}, procedure={coding.procedure_code}.",
        )

        emit("Knowledge Retrieval", "in_progress", "Fetching payer policy criteria.")
        with tracer.start_as_current_span("knowledge_retrieval") as span:
            policy_payload = self._tool_json(
                self.retrieval_stage.tool.retrieve_payer_policy(
                    payer_name=extracted.payer_name,
                    member_id=extracted.member_id,
                    procedure_code=coding.procedure_code,
                    requested_service=extracted.requested_service,
                )
            )
            span.set_attribute("policy_id", str(policy_payload.get("policy_id", "")))
        policy = PolicyMatch(**policy_payload)
        result.policy = policy
        emit(
            "Knowledge Retrieval",
            "completed",
            f"Selected policy {policy.policy_id}.",
        )

        emit("Medical Necessity Analysis", "in_progress", "Evaluating policy criteria.")
        with tracer.start_as_current_span("medical_necessity") as span:
            necessity_payload = self._tool_json(
                self.necessity_stage.tool.evaluate_necessity(
                    extracted_data=extracted.to_dict(),
                    coding_data=coding.to_dict(),
                    policy_data=policy.to_dict(),
                )
            )
            span.set_attribute("meets_criteria", bool(necessity_payload.get("meets_criteria", False)))
            span.set_attribute(
                "extended_thinking_used",
                bool(necessity_payload.get("extended_thinking_used", False)),
            )
        necessity = NecessityDecision(**necessity_payload)
        result.necessity = necessity
        emit(
            "Medical Necessity Analysis",
            "completed",
            f"meets_criteria={necessity.meets_criteria}, denial_risk={necessity.denial_risk_score:.2f}.",
        )

        emit("Form Population", "in_progress", "Building PA submission payload.")
        with tracer.start_as_current_span("form_population") as span:
            payload = self._tool_json(
                self.payload_stage.tool.build_submission_payload(
                    extracted_data=extracted.to_dict(),
                    coding_data=coding.to_dict(),
                    necessity_data=necessity.to_dict(),
                    policy_data=policy.to_dict(),
                )
            )
            span.set_attribute("payload_fields", len(payload))
        emit("Form Population", "completed", "Payload built for browser automation.")

        emit("Human Review", "in_progress", "Preparing approval snapshot.")
        approved = reviewer_approved if reviewer_approved is not None else auto_approve
        with tracer.start_as_current_span("human_review") as span:
            span.set_attribute("approved", approved)
            snapshot_payload = self._tool_json(
                self.review_stage.tool.generate_review_snapshot(payload=payload)
            )
            review_snapshot = str(snapshot_payload.get("review_snapshot", "")).strip()
        emit("Human Review", "completed", f"approved={approved}.")

        emit("Portal Submission", "in_progress", "Executing portal form submission.")
        with tracer.start_as_current_span("portal_submission") as span:
            submission_payload = self._tool_json(
                self.submission_stage.tool.submit_form(
                    payload=payload,
                    approved=approved,
                    review_snapshot=review_snapshot,
                )
            )
            span.set_attribute("submission_status", str(submission_payload.get("status", "")))
            span.set_attribute("submission_reference", str(submission_payload.get("reference", "")))
        submission = SubmissionResult(**submission_payload)
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
