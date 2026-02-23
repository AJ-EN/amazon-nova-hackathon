from __future__ import annotations

import uuid

import requests

from agents.types import SubmissionResult


class BrowserAutomationAgent:
    """
    Nova Act-compatible submission adapter.
    For the local demo it submits to the mock payer portal HTTP endpoint.
    """

    def __init__(
        self,
        portal_base_url: str = "http://127.0.0.1:5000",
        timeout_seconds: int = 10,
        max_attempts: int = 3,
    ) -> None:
        self.portal_base_url = portal_base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max_attempts

    def generate_review_snapshot(self, payload: dict[str, str]) -> str:
        return (
            "Human Review Snapshot\n"
            f"- Patient: {payload.get('patient_name', '')}\n"
            f"- Member ID: {payload.get('member_id', '')}\n"
            f"- Payer: {payload.get('payer_name', '')}\n"
            f"- Diagnosis: {payload.get('diagnosis_code', '')}\n"
            f"- Procedure: {payload.get('procedure_code', '')}\n"
            f"- Denial Risk: {payload.get('denial_risk_score', '')}"
        )

    def submit(
        self,
        payload: dict[str, str],
        approved: bool,
        review_snapshot: str,
    ) -> SubmissionResult:
        if not approved:
            return SubmissionResult(
                status="needs_approval",
                message="Submission paused until clinician approves the pre-filled form.",
                reference="",
                review_snapshot=review_snapshot,
                payload=payload,
            )

        last_exception = None
        endpoint = f"{self.portal_base_url}/submit"
        for _ in range(self.max_attempts):
            try:
                response = requests.post(
                    endpoint,
                    data=payload,
                    timeout=self.timeout_seconds,
                )
                response.raise_for_status()
                body = response.json()
                return SubmissionResult(
                    status=body.get("status", "submitted"),
                    message="Prior authorization submitted successfully.",
                    reference=body.get("reference", self._local_reference()),
                    review_snapshot=review_snapshot,
                    payload=payload,
                )
            except requests.RequestException as exc:
                last_exception = exc

        return SubmissionResult(
            status="failed",
            message=f"Submission failed after retries: {last_exception}",
            reference="",
            review_snapshot=review_snapshot,
            payload=payload,
        )

    @staticmethod
    def _local_reference() -> str:
        return f"PA-{uuid.uuid4().hex[:8].upper()}"
