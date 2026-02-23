from __future__ import annotations

import os
import unittest

from agents.browser_agent import BrowserAutomationAgent
from agents.orchestrator import PriorAuthOrchestrator
from agents.reasoning_agent import ClinicalReasoningAgent
from agents.retrieval_agent import PayerPolicyRetrievalAgent
from agents.types import SubmissionResult
from agents.voice_agent import VoiceIntakeAgent
from knowledge_base.setup_kb import bootstrap_local_policy_store

SAMPLE_TRANSCRIPT = (
    "I need a prior auth for Jane Doe, date of birth March 15 1965, "
    "member ID UHC-4429871. She needs an MRI of the lumbar spine, outpatient, "
    "at our facility. She has been through six weeks of physical therapy with no improvement, "
    "has radiculopathy with L4-L5 disc herniation confirmed on X-ray."
)


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


class TestPriorAuthPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        bootstrap_local_policy_store(overwrite=False)

    def test_voice_intake_extracts_core_fields(self) -> None:
        agent = VoiceIntakeAgent()
        extracted = agent.ingest(SAMPLE_TRANSCRIPT)

        self.assertEqual(extracted.patient_name, "Jane Doe")
        self.assertEqual(extracted.member_id, "UHC-4429871")
        self.assertEqual(extracted.payer_name, "UnitedHealthcare")
        self.assertEqual(extracted.conservative_therapy_weeks, 6)
        self.assertIn("Radiculopathy", extracted.clinical_findings)

    def test_reasoning_and_retrieval_are_policy_grounded(self) -> None:
        voice = VoiceIntakeAgent()
        retrieval = PayerPolicyRetrievalAgent()
        reasoning = ClinicalReasoningAgent()

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

        self.assertEqual(coding.source, "nova")
        self.assertRegex(coding.diagnosis_code, r"^[A-Z]")
        self.assertRegex(coding.procedure_code, r"^\d{5}$")
        self.assertGreaterEqual(coding.confidence, 0.0)
        self.assertLessEqual(coding.confidence, 1.0)


if __name__ == "__main__":
    unittest.main()
