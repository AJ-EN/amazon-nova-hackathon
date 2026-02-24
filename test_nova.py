from __future__ import annotations

import os
import unittest

from agents.browser_agent import BrowserAutomationAgent
from agents.orchestrator import PriorAuthOrchestrator
from agents.reasoning_agent import ClinicalReasoningAgent
from agents.retrieval_agent import PayerPolicyRetrievalAgent
from agents.types import CodingResult, SubmissionResult
from agents.voice_agent import VoiceIntakeAgent
from knowledge_base.setup_kb import bootstrap_local_policy_store

SAMPLE_TRANSCRIPT = (
    "I need a prior auth for Jane Doe, date of birth March 15 1965, "
    "member ID UHC-4429871. She needs an MRI of the lumbar spine, outpatient, "
    "at our facility. She has been through six weeks of physical therapy with no improvement, "
    "has radiculopathy with L4-L5 disc herniation confirmed on X-ray."
)


KB_POLICY_DOC_UHC = """Policy ID: UHC-LUMBAR-MRI-2026
Payer: UnitedHealthcare
Member ID Prefixes: UHC, UHG
Title: Lumbar Spine MRI Medical Necessity Policy

Covered Procedure Codes: 72148 (MRI lumbar spine without contrast), 72149 (MRI lumbar spine with contrast), 72158 (MRI lumbar spine without and with contrast)

Service Keywords: lumbar spine mri, mri lumbar, lumbar mri

Medical Necessity Criteria (minimum 2 of 3 must be met):
1. conservative_therapy_6w: Failure of at least 6 weeks of conservative therapy.
2. radicular_symptoms: Radicular symptoms or neurologic deficit are documented.
3. objective_imaging_or_exam: Objective imaging or exam findings support lumbar pathology.

Required Documents:
- progress_notes
- conservative_therapy_notes
- imaging_report

Common Denial Patterns:
- Missing conservative treatment duration
- Insufficient neurologic findings documentation
"""


class FakeBedrockKBClient:
    def __init__(
        self,
        retrieval_results: list[dict] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.retrieval_results = retrieval_results or []
        self.error = error

    def retrieve(self, **_: object) -> dict:
        if self.error is not None:
            raise self.error
        return {"retrievalResults": self.retrieval_results}


class FakeBrowserAgent(BrowserAutomationAgent):
    def __init__(self) -> None:
        super().__init__(portal_base_url="http://localhost:9999")

    def submit(
        self,
        payload: dict[str, str],
        approved: bool,
        review_snapshot: str,
    ) -> SubmissionResult:
        if not approved:
            return SubmissionResult(
                status="needs_approval",
                message="Paused for human review",
                reference="",
                review_snapshot=review_snapshot,
                payload=payload,
            )
        return SubmissionResult(
            status="submitted",
            message="Submitted via fake browser",
            reference="PA-TEST1234",
            review_snapshot=review_snapshot,
            payload=payload,
        )


class FakeInvalidNovaReasoningAgent(ClinicalReasoningAgent):
    def __init__(self) -> None:
        super().__init__(use_model=True)

    def _map_codes_with_nova(self, extracted):  # type: ignore[override]
        return CodingResult(
            diagnosis_code="A00.0",
            procedure_code="99999",
            confidence=0.9,
            rationale="Synthetic invalid model output for guardrail testing.",
            source="nova",
        )


class TestPriorAuthPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        bootstrap_local_policy_store(overwrite=False)

    def test_voice_intake_extracts_core_fields(self) -> None:
        agent = VoiceIntakeAgent()
        extracted = agent.ingest(SAMPLE_TRANSCRIPT)

        self.assertEqual(extracted.patient_name, "Jane Doe")
        self.assertEqual(extracted.date_of_birth, "1965-03-15")
        self.assertEqual(extracted.member_id, "UHC-4429871")
        self.assertEqual(extracted.payer_name, "UnitedHealthcare")
        self.assertEqual(extracted.conservative_therapy_weeks, 6)
        self.assertIn("Radiculopathy", extracted.clinical_findings)

    def test_voice_intake_parses_comma_date_of_birth(self) -> None:
        transcript = (
            "I need a prior auth for Jane Doe, date of birth March 15, 1965, "
            "member ID UHC-4429871."
        )
        extracted = VoiceIntakeAgent().ingest(transcript)
        self.assertEqual(extracted.date_of_birth, "1965-03-15")

    def test_reasoning_and_retrieval_are_policy_grounded(self) -> None:
        voice = VoiceIntakeAgent()
        retrieval = PayerPolicyRetrievalAgent()
        reasoning = ClinicalReasoningAgent(
            use_model=False,
            use_model_justification=False,
        )

        extracted = voice.ingest(SAMPLE_TRANSCRIPT)
        coding = reasoning.map_codes(extracted)
        policy = retrieval.retrieve(
            payer_name=extracted.payer_name,
            member_id=extracted.member_id,
            procedure_code=coding.procedure_code,
            requested_service=extracted.requested_service,
        )
        necessity = reasoning.evaluate_medical_necessity(
            extracted, coding, policy)

        self.assertEqual(coding.diagnosis_code, "M54.17")
        self.assertEqual(coding.procedure_code, "72148")
        self.assertTrue(policy.policy_id.startswith("UHC-"))
        self.assertTrue(necessity.meets_criteria)
        self.assertLess(necessity.denial_risk_score, 0.6)
        self.assertFalse(necessity.extended_thinking_used)

    def test_retrieval_agent_parses_policy_from_bedrock_kb(self) -> None:
        retrieval = PayerPolicyRetrievalAgent(
            kb_id="kb-123456",
            kb_client=FakeBedrockKBClient(
                retrieval_results=[
                    {"content": {"text": KB_POLICY_DOC_UHC}, "score": 0.89},
                ]
            ),
        )

        policy = retrieval.retrieve(
            payer_name="UnitedHealthcare",
            member_id="UHC-4429871",
            procedure_code="72148",
            requested_service="MRI lumbar spine",
        )

        self.assertEqual(policy.policy_id, "UHC-LUMBAR-MRI-2026")
        self.assertEqual(policy.payer_name, "UnitedHealthcare")
        self.assertEqual(policy.minimum_criteria, 2)
        self.assertEqual(policy.required_documents, ["progress_notes", "conservative_therapy_notes", "imaging_report"])
        self.assertEqual(retrieval.retrieval_source, "bedrock_kb")

    def test_retrieval_agent_falls_back_to_local_when_kb_unavailable(self) -> None:
        retrieval = PayerPolicyRetrievalAgent(
            kb_id="kb-123456",
            kb_client=FakeBedrockKBClient(error=RuntimeError("kb unavailable")),
        )

        policy = retrieval.retrieve(
            payer_name="UnitedHealthcare",
            member_id="UHC-4429871",
            procedure_code="72148",
            requested_service="MRI lumbar spine",
        )

        self.assertTrue(policy.policy_id.startswith("UHC-"))
        self.assertEqual(retrieval.retrieval_source, "local_fallback")

    def test_orchestrator_blocks_without_approval(self) -> None:
        orchestrator = PriorAuthOrchestrator(browser_agent=FakeBrowserAgent())
        result = orchestrator.run(SAMPLE_TRANSCRIPT, auto_approve=False)

        self.assertIsNotNone(result.submission)
        self.assertEqual(
            result.submission.status if result.submission else "", "needs_approval")
        self.assertEqual(result.next_action, "human_review_required")

    def test_orchestrator_submits_with_approval(self) -> None:
        orchestrator = PriorAuthOrchestrator(browser_agent=FakeBrowserAgent())
        result = orchestrator.run(SAMPLE_TRANSCRIPT, auto_approve=True)

        self.assertIsNotNone(result.submission)
        self.assertEqual(
            result.submission.status if result.submission else "", "submitted")
        self.assertEqual(
            result.submission.reference if result.submission else "", "PA-TEST1234")
        self.assertEqual(result.next_action, "notify_clinician")

    def test_nova_output_is_guardrailed_when_codes_are_off_policy(self) -> None:
        extracted = VoiceIntakeAgent().ingest(SAMPLE_TRANSCRIPT)
        reasoning = FakeInvalidNovaReasoningAgent()

        coding = reasoning.map_codes(extracted)

        self.assertEqual(coding.procedure_code, "72148")
        self.assertIn(coding.diagnosis_code, {"M54.16", "M54.17", "M51.26"})
        self.assertEqual(coding.source, "nova_guardrailed")
        self.assertIn("Guardrail adjustments:", coding.rationale)

    def test_denial_risk_increases_for_guardrailed_low_confidence_coding(self) -> None:
        extracted = VoiceIntakeAgent().ingest(SAMPLE_TRANSCRIPT)
        retrieval = PayerPolicyRetrievalAgent()
        policy = retrieval.retrieve(
            payer_name=extracted.payer_name,
            member_id=extracted.member_id,
            procedure_code="72148",
            requested_service=extracted.requested_service,
        )
        reasoning = ClinicalReasoningAgent()
        coding_high_conf = CodingResult(
            diagnosis_code="M54.16",
            procedure_code="72148",
            confidence=0.96,
            rationale="High confidence coding",
            source="nova",
        )
        coding_guardrailed = CodingResult(
            diagnosis_code="M54.16",
            procedure_code="72148",
            confidence=0.62,
            rationale="Guardrailed coding",
            source="nova_guardrailed",
        )

        high_conf_decision = reasoning.evaluate_medical_necessity(
            extracted, coding_high_conf, policy
        )
        guardrailed_decision = reasoning.evaluate_medical_necessity(
            extracted, coding_guardrailed, policy
        )

        self.assertGreater(
            guardrailed_decision.denial_risk_score,
            high_conf_decision.denial_risk_score,
        )

    @unittest.skipUnless(
        os.getenv("RUN_BEDROCK_INTEGRATION_TESTS") == "1",
        "Set RUN_BEDROCK_INTEGRATION_TESTS=1 to run live Nova integration checks.",
    )
    def test_reasoning_agent_nova_integration(self) -> None:
        extracted = VoiceIntakeAgent().ingest(SAMPLE_TRANSCRIPT)
        agent = ClinicalReasoningAgent(
            use_model=True,
            require_model_success=True,
        )
        coding = agent.map_codes(extracted)

        self.assertIn(coding.source, {"nova", "nova_guardrailed"})
        if coding.source.endswith("guardrailed"):
            self.assertIn("Guardrail adjustments:", coding.rationale)
        self.assertRegex(coding.diagnosis_code, r"^[A-Z]")
        self.assertRegex(coding.procedure_code, r"^\d{5}$")
        self.assertGreaterEqual(coding.confidence, 0.0)
        self.assertLessEqual(coding.confidence, 1.0)


if __name__ == "__main__":
    unittest.main()
