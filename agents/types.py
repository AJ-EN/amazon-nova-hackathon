from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ExtractedClinicalData:
    transcript: str
    patient_name: str = ""
    date_of_birth: str = ""
    member_id: str = ""
    payer_name: str = ""
    provider_npi: str = ""
    requested_service: str = ""
    facility_name: str = ""
    urgency: str = "routine"
    conservative_therapy_weeks: int | None = None
    imaging_evidence: str = ""
    clinical_findings: list[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CodingResult:
    diagnosis_code: str
    procedure_code: str
    confidence: float
    rationale: str
    source: str = "heuristic"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PolicyMatch:
    policy_id: str
    payer_name: str
    title: str
    criteria: list[dict[str, str]]
    minimum_criteria: int
    required_documents: list[str]
    denial_patterns: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class NecessityDecision:
    meets_criteria: bool
    satisfied_criteria: list[str]
    missing_criteria: list[str]
    missing_documents: list[str]
    denial_risk_score: float
    justification: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SubmissionResult:
    status: str
    message: str
    reference: str = ""
    review_snapshot: str = ""
    payload: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class WorkflowTraceStep:
    step: str
    status: str
    detail: str
    timestamp: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PriorAuthWorkflowResult:
    trace: list[WorkflowTraceStep]
    extracted_data: ExtractedClinicalData | None = None
    coding: CodingResult | None = None
    policy: PolicyMatch | None = None
    necessity: NecessityDecision | None = None
    submission: SubmissionResult | None = None
    next_action: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace": [step.to_dict() for step in self.trace],
            "extracted_data": self.extracted_data.to_dict() if self.extracted_data else None,
            "coding": self.coding.to_dict() if self.coding else None,
            "policy": self.policy.to_dict() if self.policy else None,
            "necessity": self.necessity.to_dict() if self.necessity else None,
            "submission": self.submission.to_dict() if self.submission else None,
            "next_action": self.next_action,
        }
