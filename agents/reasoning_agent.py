from __future__ import annotations

from agents.types import (
    CodingResult,
    ExtractedClinicalData,
    NecessityDecision,
    PolicyMatch,
)


class ClinicalReasoningAgent:
    """Maps clinical context to coding and medical necessity decisions."""

    def map_codes(self, extracted: ExtractedClinicalData) -> CodingResult:
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
        )

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
        )
        justification = self._build_justification(
            extracted=extracted,
            coding=coding,
            policy=policy,
            meets_criteria=meets_criteria,
            satisfied=satisfied,
            missing=missing,
        )

        return NecessityDecision(
            meets_criteria=meets_criteria,
            satisfied_criteria=satisfied,
            missing_criteria=missing,
            missing_documents=missing_documents,
            denial_risk_score=denial_risk_score,
            justification=justification,
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
            objective_terms = ("herniation", "confirmed on",
                               "seen on", "x-ray", "mri")
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
        meets_criteria: bool, missing_criteria_count: int, missing_documents_count: int
    ) -> float:
        risk = 0.12
        if not meets_criteria:
            risk += 0.35
        risk += 0.08 * missing_criteria_count
        risk += 0.10 * missing_documents_count
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
        findings = ", ".join(
            extracted.clinical_findings[:3]) or "documented clinical findings"
        therapy = (
            f"{extracted.conservative_therapy_weeks} weeks of conservative therapy"
            if extracted.conservative_therapy_weeks
            else "documented conservative management"
        )
        status = "meets" if meets_criteria else "partially meets"
        satisfied_text = "; ".join(
            satisfied[:2]) if satisfied else "clinical indication documented"
        missing_text = "; ".join(missing[:2]) if missing else "none"

        return (
            f"Per {policy.title}, {patient} {status} criteria for "
            f"{coding.procedure_code} due to {findings}. "
            f"Case history includes {therapy}. "
            f"Primary diagnosis mapped to {coding.diagnosis_code}. "
            f"Satisfied criteria: {satisfied_text}. Missing criteria: {missing_text}."
        )
