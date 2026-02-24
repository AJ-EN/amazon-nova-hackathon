from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Protocol, cast

from agents.types import (
    CodingResult,
    ExtractedClinicalData,
    NecessityDecision,
    PolicyMatch,
)
from utils.bedrock_client import get_bedrock_client

ICD10_PATTERN = re.compile(r"^[A-TV-Z][0-9][0-9A-Z](?:\.[0-9A-Z]{1,4})?$")
CPT_PATTERN = re.compile(r"^\d{5}$")
logger = logging.getLogger(__name__)


class BedrockRuntimeClient(Protocol):
    def converse(self, **kwargs: Any) -> dict[str, Any]:
        ...


class ClinicalReasoningAgent:
    """Maps clinical context to coding and medical necessity decisions."""

    def __init__(
        self,
        use_model: bool | None = None,
        model_id: str | None = None,
        use_model_justification: bool | None = None,
        justification_model_id: str | None = None,
        prefer_extended_thinking: bool | None = None,
        require_model_success: bool = False,
    ) -> None:
        env_toggle = os.getenv("USE_NOVA_REASONING", "0").lower() in {"1", "true", "yes"}
        self.use_model = env_toggle if use_model is None else use_model
        self.model_id = model_id or os.getenv("NOVA_REASONING_MODEL_ID", "amazon.nova-lite-v1:0")
        justification_env_default = "1" if self.use_model else "0"
        justification_toggle = os.getenv(
            "USE_NOVA_JUSTIFICATION", justification_env_default
        ).lower() in {"1", "true", "yes"}
        self.use_model_justification = (
            justification_toggle if use_model_justification is None else use_model_justification
        )
        self.justification_model_id = justification_model_id or os.getenv(
            "NOVA_JUSTIFICATION_MODEL_ID", self.model_id
        )
        extended_thinking_toggle = os.getenv(
            "USE_NOVA_EXTENDED_THINKING", "1"
        ).lower() in {"1", "true", "yes"}
        self.prefer_extended_thinking = (
            extended_thinking_toggle if prefer_extended_thinking is None else prefer_extended_thinking
        )
        self.require_model_success = require_model_success
        self._runtime_client: BedrockRuntimeClient | None = None

    def map_codes(self, extracted: ExtractedClinicalData) -> CodingResult:
        if self.use_model:
            try:
                nova_coding = self._map_codes_with_nova(extracted)
                return self._apply_coding_guardrails(extracted, nova_coding)
            except Exception as exc:
                if self.require_model_success:
                    raise RuntimeError(f"Nova reasoning call failed: {exc}") from exc

                fallback = self._map_codes_heuristic(extracted)
                fallback.source = "heuristic_fallback"
                fallback.rationale = (
                    f"{fallback.rationale} "
                    f"Fell back from Nova call due to {exc.__class__.__name__}."
                )
                return self._apply_coding_guardrails(extracted, fallback)

        heuristic_coding = self._map_codes_heuristic(extracted)
        return self._apply_coding_guardrails(extracted, heuristic_coding)

    def evaluate_medical_necessity(
        self,
        extracted: ExtractedClinicalData,
        coding: CodingResult,
        policy: PolicyMatch,
    ) -> NecessityDecision:
        satisfied: list[str] = []
        missing: list[str] = []

        for criterion in policy.criteria:
            criterion_id = criterion.get("id", "")
            criterion_description = criterion.get("description", criterion_id)
            if self._criterion_met(criterion_id, extracted):
                satisfied.append(criterion_description)
            else:
                missing.append(criterion_description)

        meets_criteria = len(satisfied) >= policy.minimum_criteria
        available_documents = self._infer_documents(extracted)
        missing_documents = [
            document
            for document in policy.required_documents
            if document not in available_documents
        ]
        denial_risk_score = self._calculate_denial_risk(
            meets_criteria=meets_criteria,
            missing_criteria_count=len(missing),
            missing_documents_count=len(missing_documents),
            coding_confidence=coding.confidence,
            coding_source=coding.source,
        )
        justification, extended_thinking_used = self._build_justification_with_resilience(
            extracted=extracted,
            coding=coding,
            policy=policy,
            meets_criteria=meets_criteria,
            satisfied=satisfied,
            missing=missing,
            missing_documents=missing_documents,
            denial_risk_score=denial_risk_score,
        )

        return NecessityDecision(
            meets_criteria=meets_criteria,
            satisfied_criteria=satisfied,
            missing_criteria=missing,
            missing_documents=missing_documents,
            denial_risk_score=denial_risk_score,
            justification=justification,
            extended_thinking_used=extended_thinking_used,
        )

    @staticmethod
    def build_form_payload(
        extracted: ExtractedClinicalData,
        coding: CodingResult,
        necessity: NecessityDecision,
        policy: PolicyMatch,
    ) -> dict[str, str]:
        return {
            "patient_name": extracted.patient_name or "Unknown Patient",
            "date_of_birth": extracted.date_of_birth,
            "member_id": extracted.member_id,
            "payer_name": extracted.payer_name,
            "provider_npi": extracted.provider_npi,
            "requested_service": extracted.requested_service,
            "facility_name": extracted.facility_name,
            "diagnosis_code": coding.diagnosis_code,
            "procedure_code": coding.procedure_code,
            "clinical_justification": necessity.justification,
            "urgency": extracted.urgency,
            "policy_id": policy.policy_id,
            "denial_risk_score": f"{necessity.denial_risk_score:.2f}",
        }

    def _map_codes_with_nova(self, extracted: ExtractedClinicalData) -> CodingResult:
        self._ensure_runtime_client()
        runtime_client = self._runtime_client
        if runtime_client is None:
            raise RuntimeError("Bedrock runtime client failed to initialize.")

        system_prompt = (
            "You are a medical coding assistant for prior authorization workflows. "
            "Return only JSON with keys diagnosis_code, procedure_code, confidence, rationale. "
            "Use a single ICD-10-CM diagnosis code and a single CPT procedure code. "
            "Confidence must be a float between 0 and 1."
        )
        user_prompt = (
            "Map the following clinical request to one ICD-10-CM code and one CPT code.\n"
            f"Patient findings: {', '.join(extracted.clinical_findings) or 'Not specified'}\n"
            f"Requested service: {extracted.requested_service}\n"
            f"Clinical note: {extracted.transcript}\n"
            "Respond with JSON only."
        )

        response = runtime_client.converse(
            modelId=self.model_id,
            system=[{"text": system_prompt}],
            messages=[{"role": "user", "content": [{"text": user_prompt}]}],
            inferenceConfig={"maxTokens": 240, "temperature": 0.0},
        )

        content = response.get("output", {}).get("message", {}).get("content", [])
        text_parts = [part.get("text", "") for part in content if isinstance(part, dict)]
        raw_text = "\n".join(part for part in text_parts if part).strip()
        payload = self._parse_model_json(raw_text)

        diagnosis_code = self._normalize_icd(payload.get("diagnosis_code", ""))
        procedure_code = self._normalize_cpt(payload.get("procedure_code", ""))
        confidence = self._normalize_confidence(payload.get("confidence", 0.75))
        rationale = str(payload.get("rationale", "")).strip() or "Model-provided coding output."

        if not ICD10_PATTERN.match(diagnosis_code):
            raise ValueError(f"Invalid ICD-10 code returned by model: {diagnosis_code}")
        if not CPT_PATTERN.match(procedure_code):
            raise ValueError(f"Invalid CPT code returned by model: {procedure_code}")

        return CodingResult(
            diagnosis_code=diagnosis_code,
            procedure_code=procedure_code,
            confidence=confidence,
            rationale=rationale,
            source="nova",
        )

    def _build_justification_with_resilience(
        self,
        extracted: ExtractedClinicalData,
        coding: CodingResult,
        policy: PolicyMatch,
        meets_criteria: bool,
        satisfied: list[str],
        missing: list[str],
        missing_documents: list[str],
        denial_risk_score: float,
    ) -> tuple[str, bool]:
        if self.use_model_justification:
            try:
                return self._build_justification_with_nova(
                    extracted=extracted,
                    coding=coding,
                    policy=policy,
                    meets_criteria=meets_criteria,
                    satisfied=satisfied,
                    missing=missing,
                    missing_documents=missing_documents,
                    denial_risk_score=denial_risk_score,
                )
            except Exception as exc:
                if self.require_model_success:
                    raise RuntimeError(f"Nova justification call failed: {exc}") from exc
                template = self._build_justification(
                    extracted=extracted,
                    coding=coding,
                    policy=policy,
                    meets_criteria=meets_criteria,
                    satisfied=satisfied,
                    missing=missing,
                )
                return (
                    f"{template} "
                    f"Justification fallback used due to {exc.__class__.__name__}."
                ), False

        return self._build_justification(
            extracted=extracted,
            coding=coding,
            policy=policy,
            meets_criteria=meets_criteria,
            satisfied=satisfied,
            missing=missing,
        ), False

    def _build_justification_with_nova(
        self,
        extracted: ExtractedClinicalData,
        coding: CodingResult,
        policy: PolicyMatch,
        meets_criteria: bool,
        satisfied: list[str],
        missing: list[str],
        missing_documents: list[str],
        denial_risk_score: float,
    ) -> tuple[str, bool]:
        self._ensure_runtime_client()
        runtime_client = self._runtime_client
        if runtime_client is None:
            raise RuntimeError("Bedrock runtime client failed to initialize.")
        status = "meets" if meets_criteria else "partially meets"
        satisfied_text = "; ".join(satisfied) if satisfied else "None"
        missing_text = "; ".join(missing) if missing else "None"
        missing_doc_text = "; ".join(missing_documents) if missing_documents else "None"
        findings = ", ".join(extracted.clinical_findings) if extracted.clinical_findings else "None"

        system_prompt = (
            "You draft prior authorization medical necessity narratives for U.S. providers. "
            "Write concise payer-ready clinical prose in one paragraph. "
            "Do not mention AI, model confidence, or uncertainty language."
        )
        user_prompt = (
            "Draft the medical necessity justification.\n"
            f"Policy: {policy.title}\n"
            f"Case status: {status} policy criteria\n"
            f"Patient: {extracted.patient_name or 'Unknown patient'}\n"
            f"DOB: {extracted.date_of_birth or 'Unknown'}\n"
            f"Requested service: {extracted.requested_service}\n"
            f"ICD-10: {coding.diagnosis_code}\n"
            f"CPT: {coding.procedure_code}\n"
            f"Clinical findings: {findings}\n"
            f"Clinical history: {extracted.transcript}\n"
            f"Satisfied criteria: {satisfied_text}\n"
            f"Missing criteria: {missing_text}\n"
            f"Missing documents: {missing_doc_text}\n"
            f"Denial risk score: {denial_risk_score:.2f}\n"
            "Output only the justification paragraph."
        )

        request_payload: dict[str, Any] = {
            "modelId": self.justification_model_id,
            "system": [{"text": system_prompt}],
            "messages": [{"role": "user", "content": [{"text": user_prompt}]}],
            "inferenceConfig": {"maxTokens": 420, "temperature": 0.2},
        }

        response = None
        extended_thinking_fired = False
        if self.prefer_extended_thinking:
            extended_payload = dict(request_payload)
            extended_payload["additionalModelRequestFields"] = {
                "reasoningConfig": {
                    "type": "enabled",
                    "maxReasoningEffort": "medium",
                }
            }
            try:
                response = runtime_client.converse(**extended_payload)
                extended_thinking_fired = True
            except Exception as exc:
                logger.warning(
                    "Extended thinking fell back to standard inference: %s",
                    exc,
                )
                response = None

        if response is None:
            response = runtime_client.converse(**request_payload)

        text = self._extract_text_from_converse(response)
        if not text:
            raise ValueError("Model returned empty justification text.")
        return text, extended_thinking_fired

    @staticmethod
    def _extract_text_from_converse(response: dict[str, Any]) -> str:
        content = response.get("output", {}).get("message", {}).get("content", [])
        text_parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str) and text.strip():
                    text_parts.append(text.strip())
        return " ".join(text_parts).strip()

    def _ensure_runtime_client(self) -> None:
        if self._runtime_client is None:
            self._runtime_client = cast(BedrockRuntimeClient, get_bedrock_client())

    def _apply_coding_guardrails(
        self,
        extracted: ExtractedClinicalData,
        coding: CodingResult,
    ) -> CodingResult:
        allowed_icd, preferred_icd = self._allowed_diagnosis_codes(extracted)
        allowed_cpt, preferred_cpt = self._allowed_procedure_codes(extracted)

        diagnosis_code = coding.diagnosis_code
        procedure_code = coding.procedure_code
        notes: list[str] = []
        changed = False

        if allowed_cpt and procedure_code not in allowed_cpt and preferred_cpt:
            notes.append(f"Procedure {procedure_code} replaced with {preferred_cpt}.")
            procedure_code = preferred_cpt
            changed = True

        if allowed_icd and diagnosis_code not in allowed_icd and preferred_icd:
            notes.append(f"Diagnosis {diagnosis_code} replaced with {preferred_icd}.")
            diagnosis_code = preferred_icd
            changed = True

        if not changed:
            return coding

        source = coding.source
        if not source.endswith("_guardrailed"):
            source = f"{source}_guardrailed"

        confidence = max(0.0, min(coding.confidence * 0.95, 1.0))
        rationale = f"{coding.rationale} Guardrail adjustments: {' '.join(notes)}"
        return CodingResult(
            diagnosis_code=diagnosis_code,
            procedure_code=procedure_code,
            confidence=confidence,
            rationale=rationale,
            source=source,
        )

    @staticmethod
    def _map_codes_heuristic(extracted: ExtractedClinicalData) -> CodingResult:
        transcript = extracted.transcript.lower()
        findings = " ".join(extracted.clinical_findings).lower()
        signal = f"{transcript} {findings} {extracted.requested_service.lower()}"

        diagnosis_code = "R52"
        diagnosis_rationale = "Unspecified pain fallback due to limited diagnostic context."
        confidence = 0.55

        if "radiculopathy" in signal and "lumbar" in signal:
            diagnosis_code = "M54.17"
            diagnosis_rationale = "Lumbar/lumbosacral radiculopathy identified from findings."
            confidence = 0.92
        elif "disc herniation" in signal and "lumbar" in signal:
            diagnosis_code = "M51.26"
            diagnosis_rationale = "Lumbar disc displacement inferred from herniation finding."
            confidence = 0.88
        elif "back pain" in signal:
            diagnosis_code = "M54.50"
            diagnosis_rationale = "Low back pain identified from transcript."
            confidence = 0.8

        procedure_code = "72148"
        procedure_rationale = "MRI lumbar spine without contrast inferred from requested service."
        if "without and with contrast" in signal:
            procedure_code = "72158"
            procedure_rationale = (
                "MRI lumbar spine without/with contrast inferred from requested service."
            )
        elif "with contrast" in signal:
            procedure_code = "72149"
            procedure_rationale = "MRI lumbar spine with contrast inferred from requested service."
        elif "ct" in signal and "lumbar" in signal:
            procedure_code = "72131"
            procedure_rationale = "CT lumbar spine inferred from requested service."

        return CodingResult(
            diagnosis_code=diagnosis_code,
            procedure_code=procedure_code,
            confidence=confidence,
            rationale=f"{diagnosis_rationale} {procedure_rationale}",
            source="heuristic",
        )

    @staticmethod
    def _parse_model_json(raw_text: str) -> dict[str, Any]:
        if not raw_text:
            raise ValueError("Model returned empty output.")

        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(f"Model output did not contain JSON: {raw_text}")

        json_blob = raw_text[start : end + 1]
        parsed = json.loads(json_blob)
        if not isinstance(parsed, dict):
            raise ValueError("Model JSON output is not an object.")
        return parsed

    @staticmethod
    def _normalize_icd(value: Any) -> str:
        return str(value).strip().upper().replace(" ", "")

    @staticmethod
    def _normalize_cpt(value: Any) -> str:
        digits = "".join(ch for ch in str(value) if ch.isdigit())
        return digits[:5]

    @staticmethod
    def _normalize_confidence(value: Any) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            parsed = 0.75
        return max(0.0, min(parsed, 1.0))

    @staticmethod
    def _allowed_diagnosis_codes(extracted: ExtractedClinicalData) -> tuple[set[str], str | None]:
        signal = f"{extracted.transcript} {' '.join(extracted.clinical_findings)}".lower()
        service = extracted.requested_service.lower()
        lumbar_context = "lumbar" in signal or "lumbar" in service

        allowed: set[str] = set()
        preferred: str | None = None

        if lumbar_context and ("radiculopathy" in signal or "sciatica" in signal):
            allowed.update({"M54.16", "M54.17"})
            preferred = "M54.16"

        if lumbar_context and "disc herniation" in signal:
            allowed.add("M51.26")
            if preferred is None:
                preferred = "M51.26"

        if "back pain" in signal:
            allowed.add("M54.50")
            if preferred is None:
                preferred = "M54.50"

        if not allowed:
            allowed.add("R52")
            preferred = "R52"

        return allowed, preferred

    @staticmethod
    def _allowed_procedure_codes(extracted: ExtractedClinicalData) -> tuple[set[str], str | None]:
        signal = f"{extracted.transcript} {extracted.requested_service}".lower()
        lumbar_context = "lumbar" in signal

        if "mri" in signal and lumbar_context:
            allowed = {"72148", "72149", "72158"}
            if "without and with contrast" in signal:
                return allowed, "72158"
            if "with contrast" in signal:
                return allowed, "72149"
            return allowed, "72148"

        if "ct" in signal and lumbar_context:
            allowed = {"72131", "72132", "72133"}
            if "without and with contrast" in signal:
                return allowed, "72133"
            if "with contrast" in signal:
                return allowed, "72132"
            return allowed, "72131"

        return set(), None

    @staticmethod
    def _criterion_met(criterion_id: str, extracted: ExtractedClinicalData) -> bool:
        transcript = extracted.transcript.lower()
        findings = " ".join(extracted.clinical_findings).lower()
        signal = f"{transcript} {findings}"

        if criterion_id == "conservative_therapy_6w":
            return (extracted.conservative_therapy_weeks or 0) >= 6
        if criterion_id in {"radicular_symptoms", "red_flag_or_neuro_deficit"}:
            neuro_terms = ("radiculopathy", "weakness", "numbness", "sciatica")
            return any(term in signal for term in neuro_terms)
        if criterion_id == "objective_imaging_or_exam":
            objective_terms = ("herniation", "confirmed on", "seen on", "x-ray", "mri")
            return bool(extracted.imaging_evidence) or any(term in signal for term in objective_terms)
        if criterion_id == "persistent_pain":
            return "pain" in signal and (extracted.conservative_therapy_weeks or 0) >= 4
        if criterion_id == "clinical_indication":
            return bool(extracted.clinical_findings)
        return False

    @staticmethod
    def _infer_documents(extracted: ExtractedClinicalData) -> set[str]:
        docs = {"progress_notes"}
        if (extracted.conservative_therapy_weeks or 0) >= 1:
            docs.add("conservative_therapy_notes")
        if extracted.imaging_evidence:
            docs.add("imaging_report")
        return docs

    @staticmethod
    def _calculate_denial_risk(
        meets_criteria: bool,
        missing_criteria_count: int,
        missing_documents_count: int,
        coding_confidence: float,
        coding_source: str,
    ) -> float:
        risk = 0.12
        if not meets_criteria:
            risk += 0.35
        risk += 0.08 * missing_criteria_count
        risk += 0.10 * missing_documents_count
        risk += 0.12 * (1.0 - max(0.0, min(coding_confidence, 1.0)))
        source_lower = coding_source.lower()
        if "guardrailed" in source_lower:
            risk += 0.08
        elif "fallback" in source_lower:
            risk += 0.05
        return round(min(risk, 0.99), 2)

    @staticmethod
    def _build_justification(
        extracted: ExtractedClinicalData,
        coding: CodingResult,
        policy: PolicyMatch,
        meets_criteria: bool,
        satisfied: list[str],
        missing: list[str],
    ) -> str:
        patient = extracted.patient_name or "the patient"
        findings = ", ".join(extracted.clinical_findings[:3]) or "documented clinical findings"
        therapy = (
            f"{extracted.conservative_therapy_weeks} weeks of conservative therapy"
            if extracted.conservative_therapy_weeks
            else "documented conservative management"
        )
        status = "meets" if meets_criteria else "partially meets"
        satisfied_text = "; ".join(satisfied[:2]) if satisfied else "clinical indication documented"
        missing_text = "; ".join(missing[:2]) if missing else "none"

        return (
            f"Per {policy.title}, {patient} {status} criteria for "
            f"{coding.procedure_code} due to {findings}. "
            f"Case history includes {therapy}. "
            f"Primary diagnosis mapped to {coding.diagnosis_code}. "
            f"Satisfied criteria: {satisfied_text}. Missing criteria: {missing_text}."
        )
